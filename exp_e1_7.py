"""E1-7: 반영 재검사 타당성 — 고친 게 반영됐는지 판별할 수 있는가.

왜 이 실험이 필요한가:
M5의 4단계는 "새 응답이 정정을 반영했는가"를 판정한다. 이 판정이 틀리면
P3 전체가 무의미하다. 메모리식 기준선과 갈리는 지점이 바로 여기이므로,
판정 도구가 맞는지 먼저 보여야 한다.

성공 기준(feedback2.md E1-7): 일부러 반영되지 않게 만든 응답과 반영된 응답을
구분하는 정확도 ≥ .85

설계:
    1. 전제 p가 깔린 계획을 생성한다 (예: 예산 30만원)
    2. p를 p'로 수정한다 (예산 15만원)
    3. 두 응답을 준비한다
         반영된 응답     = 수정된 상태로 재생성 (M4 컨텍스트 사용)
         반영 안 된 응답 = 원래 응답 그대로 (옛 값이 살아 있다)
    4. 각각에 do(p')를 걸어 유의하게 흔들리는 세그먼트를 찾는다
         흔들림 없음 → "반영 확인"
         흔들림 있음 → "반영 안 됨"
    5. 두 경우를 올바르게 구분했는지 센다

4단계 논리: 어떤 문장이 p'를 제대로 반영했다면 p'를 다시 단언해도 흔들리지
않는다. 이미 그렇게 쓰여 있으니까. 옛 p에 기대고 있으면 p'를 단언하는 순간
모순이 생겨 확률이 떨어진다.

주의: "반영 안 된 응답"을 원래 응답으로 쓰는 것은 가장 깨끗한 음성 사례다.
연구자가 손으로 꾸민 응답이 아니라 모델이 실제로 낸 것이고, 옛 값이 들어
있다는 사실이 정의상 보장된다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe exp_e1_7.py [사례수] [위약수]
"""
from __future__ import annotations

import json
import statistics as stats
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.behavior_test import BehaviorTester
from app.context_builder import build_context
from app.propagate import reflection_verdict, significant_segments
from app.segmenter import segment
from app.state_store import Evidence, Premise, StateStore
from exp_m3b import make_placebos

TASK = (
    "사용자의 조건을 반영한 계획을 4섹션(목표 / 주간 일정 / 주차별 마일스톤 / 예산)으로 "
    "제시하세요. 주간 일정은 월부터 일까지 일곱 줄을 오전·오후·저녁 칸으로 채우세요. "
    "각 항목은 명사구로만 쓰고, 이유나 근거를 설명하는 문장은 쓰지 마세요."
)

# 각 사례: 처음 발화, 깔아 둘 전제, 수정될 값
CASES = [
    {
        "id": "budget_cut",
        "turn": "7월 한 달 코딩테스트 준비랑 자소서를 같이 할 계획이야. "
                "평일 5시간 가능하고 토요일은 알바 8시간. 계획 만들어줘.",
        "premise": "전체 예산은 30만원이고 유료 인터넷 강의를 구매한다",
        "new": "전체 예산은 3만원이고 무료 자료만 사용한다",
    },
    {
        "id": "time_cut",
        "turn": "8월 한 달 포트폴리오 웹앱을 만들고 기술블로그도 쓰려고 해. "
                "예산은 25만원이야. 계획 만들어줘.",
        "premise": "평일에는 하루 6시간씩 작업할 수 있다",
        "new": "평일에는 하루 1시간밖에 작업할 수 없다",
    },
    {
        "id": "priority_flip",
        "turn": "1월 한 달 토익과 데이터분석을 같이 준비할 계획이야. "
                "하루 6시간 정도 가능하고 예산은 40만원. 계획 만들어줘.",
        "premise": "토익이 1순위이고 데이터분석은 여유가 있을 때 한다",
        "new": "데이터분석이 1순위이고 토익은 하지 않는다",
    },
    {
        "id": "scope_cut",
        "turn": "7월 한 달 자격증 세 개를 따려고 해. 하루 5시간 가능하고 "
                "예산은 50만원이야. 계획 만들어줘.",
        "premise": "자격증 세 개를 모두 7월 안에 취득한다",
        "new": "자격증은 한 개만 7월에 취득하고 나머지는 미룬다",
    },
    {
        "id": "weekend_add",
        "turn": "2월 한 달 알고리즘 공부와 영어 회화를 같이 하려고 해. "
                "하루 4시간 가능하고 예산은 20만원. 계획 만들어줘.",
        "premise": "주말에도 평일과 같이 공부한다",
        "new": "주말에는 어떤 일정도 넣지 않고 완전히 쉰다",
    },
    {
        "id": "method_switch",
        "turn": "8월 한 달 포트폴리오를 만들려고 해. 하루 5시간 가능하고 "
                "예산은 30만원이야. 계획 만들어줘.",
        "premise": "혼자 독학으로 진행하며 스터디는 하지 않는다",
        "new": "매일 스터디 그룹에 참여해 함께 진행한다",
    },
]


# 전제가 응답에 반영됐는지 보는 조작 점검. 키워드 기반이라 거칠지만 투명하다.
# 라벨이 틀린 사례로 검출기를 검증할 수는 없으므로 포함 여부를 이걸로 가른다.
MANIP_CHECK = {
    "budget_cut":    (["무료", "0원", "3만"], ["30만", "유료", "구매"]),
    "time_cut":      (["1시간"], []),
    "priority_flip": (["데이터"], ["토익"]),
    "scope_cut":     (["한 개", "1개", "하나"], ["세 개", "3개"]),
    "weekend_add":   (["휴식", "쉬", "없음"], []),
    "method_switch": (["스터디", "그룹", "함께"], ["독학", "혼자"]),
}


def manip_ok(case_id: str, text: str) -> bool:
    """응답이 새 값을 반영했는가 (정답 양성 자격)."""
    want, avoid = MANIP_CHECK.get(case_id, ([], []))
    if not want:
        return True
    return sum(1 for k in want if k in text) > sum(1 for k in avoid if k in text)


def reflection_check(bt, msgs, response, segments, old_value, new_value, placebos):
    """응답이 new_value를 반영했는지 짝지어 판정한다.

    통계량은 Δ평균 차(옛 값 단언 − 새 값 단언)다. 진단에서 후보 일곱 개를
    비교한 결과 이게 가장 잘 갈랐다(75%, 차선은 62%).

    max z를 쓰지 않는 이유: z는 세그먼트별 위약 분산으로 나눈 값이라, 제목처럼
    위약에 거의 안 흔들리는 세그먼트는 분산이 0에 가까워 z가 100 단위로 폭발한다.
    max를 쓰면 그 한 개가 통계를 지배한다(실측 −98, −123).

    절대 기준(흔들림 0개)도 쓰지 않는다. p'를 제대로 반영한 응답에도 p'를
    단언하면 세그먼트가 흔들린다(실측 평균 0.7개). 같은 주제의 문장을 덧붙이는
    행위 자체가 지역 분포를 바꾸기 때문이다. 위약은 '무관한 삽입'을 통제하지만
    '주제가 같은 삽입'을 통제하지 못한다. 옛 값과 새 값을 둘 다 단언해
    비교하면 그 교란이 상쇄된다.

    반환: (반영됨, 통계값, 새 값에 충돌하는 세그먼트)
    """
    base = bt.score_segments(msgs, response, segments)
    P = [bt.delta(base, bt.score_segments(msgs, response, segments, assumption=pl))
         for pl in placebos]
    d_new = bt.delta(base, bt.score_segments(msgs, response, segments,
                                             assumption=new_value))
    d_old = bt.delta(base, bt.score_segments(msgs, response, segments,
                                             assumption=old_value))
    common = sorted(set(d_new) & set(d_old))
    if not common:
        return False, 0.0, []
    stat = (stats.mean(d_old[s] for s in common)
            - stats.mean(d_new[s] for s in common))
    stale, _z, _thr = significant_segments(d_new, P)
    return stat > 0.0, stat, list(stale)


def gen_with_premise(bt, turn: str, premise: str):
    """전제 하나를 확인된 상태로 깔고 계획을 생성한다."""
    store = StateStore()
    store.add(Premise(id="p1", text=premise))
    store.ground("p1", Evidence(type="user_statement", turn=1, pointer="c1"))
    sys_p, msgs = build_context(store, [], turn, mode="H-state", task_instruction=TASK)
    resp = bt.generate(msgs, system=sys_p, max_new_tokens=900)
    segs = [s for s in segment(resp, turn=1) if s.start >= 0]
    return resp, segs, msgs


def main() -> None:
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else len(CASES)
    n_pl = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    placebos = make_placebos(n_pl)

    print("=" * 78)
    print("E1-7: 반영 재검사 타당성")
    print("=" * 78)
    print(f"사례 {n_cases} | 위약 {n_pl}개 | 기준: 구분 정확도 >= .85")
    print()
    print("A단계  정답을 아는 두 응답을 구분하는가 (도구 검증)")
    print("       양성 = p'를 깔고 생성한 응답, 음성 = p를 깔고 생성한 응답")
    print("       조작 점검을 통과한 사례만 센다. 모델이 전제를 무시한 응답은")
    print("       양성 라벨이 틀린 것이므로 도구 검증에 쓸 수 없다.")
    print("B단계  정정 후 재생성이 실제로 반영하는가 (결과, 도구 검증 아님)")

    bt = BehaviorTester()
    print(f"\n모델 {Path(bt.model_dir).name}")

    rows = []
    for ci in range(n_cases):
        c = CASES[ci % len(CASES)]
        print("\n" + "-" * 78)
        print(f"[{c['id']}]")
        print(f"  p  : {c['premise']}")
        print(f"  p' : {c['new']}")

        t0 = time.time()
        resp_p, segs_p, msgs = gen_with_premise(bt, c["turn"], c["premise"])
        resp_q, segs_q, _ = gen_with_premise(bt, c["turn"], c["new"])
        print(f"  생성 {time.time()-t0:.0f}s | 세그먼트 p={len(segs_p)} q={len(segs_q)}")
        if len(segs_p) < 8 or len(segs_q) < 8:
            print("  세그먼트 부족 - 건너뜀")
            continue

        # 조작 점검: p'응답이 실제로 새 값을 반영했는가
        eligible = manip_ok(c["id"], resp_q)
        print(f"  조작 점검: p'응답이 새 값을 반영 = {'O' if eligible else 'X (라벨 무효)'}")

        t0 = time.time()
        ok_q, stat_q, stale_q = reflection_check(
            bt, msgs, resp_q, segs_q, c["premise"], c["new"], placebos)
        ok_p, stat_p, stale_p = reflection_check(
            bt, msgs, resp_p, segs_p, c["premise"], c["new"], placebos)
        print(f"  채점 {time.time()-t0:.0f}s")
        a_pos = ok_q is True
        a_neg = ok_p is False
        print(f"  A [{'O' if a_pos else 'X'}] p'응답 -> "
              f"{'반영 확인' if ok_q else '반영 안 됨'} (통계 {stat_q:+.3f})")
        print(f"  A [{'O' if a_neg else 'X'}] p 응답 -> "
              f"{'반영 확인' if ok_p else '반영 안 됨'} (통계 {stat_p:+.3f})")

        # B단계: 정정 후 재생성
        store = StateStore()
        store.add(Premise(id="p1", text=c["premise"]))
        store.ground("p1", Evidence(type="user_statement", turn=1, pointer="c1"))
        store.correct("p1", c["new"],
                      Evidence(type="ui_correction", turn=2, pointer="op1"),
                      new_id="p1b")
        sys_c, msgs_c = build_context(store, [], c["turn"],
                                      mode="H-state", task_instruction=TASK)
        resp_c = bt.generate(msgs_c, system=sys_c, max_new_tokens=900)
        segs_c = [s for s in segment(resp_c, turn=2) if s.start >= 0]
        b_surface = manip_ok(c["id"], resp_c)
        if len(segs_c) >= 8:
            ok_c, stat_c, _st = reflection_check(
                bt, msgs, resp_c, segs_c, c["premise"], c["new"], placebos)
        else:
            ok_c, stat_c = None, 0.0
        print(f"  B      정정 후 재생성 -> 표면 반영 {'O' if b_surface else 'X'}"
              f" | 검사 판정 {'반영 확인' if ok_c else '반영 안 됨'} (통계 {stat_c:+.3f})")

        rows.append({
            "case": c["id"], "premise": c["premise"], "new": c["new"],
            "eligible": eligible,
            "a_pos_correct": a_pos, "a_neg_correct": a_neg,
            "stat_q": stat_q, "stat_p": stat_p,
            "b_surface": b_surface, "b_reflected": ok_c, "stat_c": stat_c,
            "n_segs": {"p": len(segs_p), "q": len(segs_q), "c": len(segs_c)},
        })

    if not rows:
        print("\n유효 사례 없음")
        return

    n = len(rows)
    elig = [r for r in rows if r["eligible"]]
    # 음성 사례는 조작 점검과 무관하게 유효하다 (p응답은 항상 옛 값을 담는다)
    neg_ok = sum(1 for r in rows if r["a_neg_correct"])
    pos_ok = sum(1 for r in elig if r["a_pos_correct"])
    acc_all = (sum(1 for r in rows if r["a_pos_correct"]) + neg_ok) / (2 * n)
    acc_elig = ((pos_ok + neg_ok) / (len(elig) + n)) if elig else 0.0

    print("\n" + "=" * 78)
    print("집계")
    print("=" * 78)
    print(f"  사례 {n} | 조작 점검 통과 {len(elig)}")
    print()
    print("  A단계 - 도구 검증")
    print(f"    음성(p응답을 '반영 안 됨')        {neg_ok}/{n} = {neg_ok/n:.0%}")
    if elig:
        print(f"    양성(p'응답을 '반영 확인')        {pos_ok}/{len(elig)} = {pos_ok/len(elig):.0%}"
              f"   [조작 통과분만]")
    print(f"    구분 정확도 (조작 통과분)         {acc_elig:.0%}   (기준 >=85%)")
    print(f"    구분 정확도 (전체, 참고)          {acc_all:.0%}")
    print()
    print("  B단계 - 모델이 정정을 반영하는가")
    bs = sum(1 for r in rows if r["b_surface"])
    print(f"    표면 반영 {bs}/{n} = {bs/n:.0%}")
    print()
    if acc_elig >= 0.85:
        print("  O A단계 통과. 반영 재검사 도구가 작동한다. E1-7 기준 충족.")
    else:
        print(f"  X A단계 미달 ({acc_elig:.0%} < 85%).")
        print("    조작 점검을 통과한 사례로도 기준에 못 미친다면 통계량을 더 봐야 한다.")
    if bs / n < 0.6:
        print()
        print(f"  * B단계: 정정 후 재생성이 {bs}/{n}만 표면 반영했다. 상태 블록에")
        print("    '옛 값 사용 금지'를 명시해도 모델이 자주 무시한다는 뜻이고,")
        print("    이것이 M5 반영 재검사가 필요한 직접적 근거다.")

    out = ROOT / ".runtime" / "e1_7_result.json"
    out.write_text(json.dumps({
        "n_cases": n, "n_eligible": len(elig), "n_placebos": n_pl,
        "accuracy_eligible": acc_elig, "accuracy_all": acc_all,
        "acc_negative": neg_ok / n,
        "acc_positive_eligible": (pos_ok / len(elig)) if elig else None,
        "surface_compliance": bs / n,
        "cases": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  저장: {out.name}")


if __name__ == "__main__":
    main()
