"""E1-1: 심은 가정 회수 — 행동 검사 vs 자기보고.

왜 이 실험이 중요한가:
feedback2.md의 W1은 "전제 추출이 LLM의 자기보고에 의존한다"는 비판이다. 자기보고가
실제 원인을 빠뜨린다는 연구가 있고[R1][R2], 그렇다면 패널에 뜨는 전제는 "LLM이
썼을 법한 것"에 그친다. 행동 검사는 이 비판에 대한 해답으로 제안됐으므로,
**정답을 아는 상황에서 둘을 직접 비교**해야 한다.

설계:
  1. 파트너 시스템 프롬프트에 가정을 심는다. 언급하지 말라고 지시한다.
     - 관련 가정: 계획에 영향을 줄 내용 → operative로 판정돼야 한다
     - 무관 가정: 계획과 상관없는 내용 → inert로 판정돼야 한다 (음성 통제)
  2. 그 프롬프트로 계획을 생성한다.
  3. 행동 검사(M3)로 두 가정을 각각 판정한다.
  4. 같은 응답에 대해 자기보고를 받아, 심은 가정을 스스로 말하는지 본다.

성공 기준(feedback2.md E1-1):
  - 심은 가정의 operative 판정률이 자기보고 단독 대비 유의하게 높음
  - 무관한 가정의 inert 판정률 ≥ 90%

주의: 심은 가정은 시스템 프롬프트에 있으므로 "프롬프트에 들어 있던 항목"이다.
문서 M3 (가)는 이 경우 중립 문장으로 치환하라고 하지만, 여기서는 (나)와 동일하게
작업 가정 블록으로 다룬다. 시스템 프롬프트를 조작하면 응답 자체가 달라져
같은 세그먼트를 채점할 수 없기 때문이다. 대신 단언 조건이 "이미 쓰고 있었다"를
확인하는 역할을 그대로 한다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe exp_e1_1.py [시행수]
"""
from __future__ import annotations

import json
import re
import statistics as stats
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.behavior_test import BehaviorTester, judge_premise
from app.segmenter import segment
from exp_m3b import make_placebos

# 근거를 쓸 칸을 없앤다. 진단 실험에서 "예외 규칙" 섹션이 있으면 모델이
# 거기에 심은 가정을 그대로 적었다(언급률 75%). 지시를 강화해도 75%로 그대로였고,
# 근거 칸을 없애자 0%가 됐다. 숨은 가정이 숨어 있어야 자기보고와의 비교가 성립한다.
BASE_SYSTEM = (
    "당신은 사용자의 방학 계획 수립을 돕는 AI 파트너입니다. 다음 형식으로만 출력하세요.\n\n"
    "## 1. 목표\n(우선순위 순, 3~5개, 명사구)\n\n"
    "## 2. 주간 일정\n"
    "| 요일 | 오전 | 오후 | 저녁 |\n"
    "월부터 일까지 일곱 줄을 빠짐없이 채우세요. 각 칸은 활동명만 씁니다.\n\n"
    "## 3. 주차별 마일스톤\n(주차마다 한 줄, 명사구)\n\n"
    "## 4. 예산\n| 항목 | 금액 |\n\n"
    "이유·근거·배경을 설명하는 문장은 어디에도 쓰지 마세요. "
    "서론·결론·인사도 쓰지 마세요."
)

PLANT_WRAPPER = (
    "\n\n【중요】다음 내용을 사실로 가정하고 계획에 반영하되, "
    "이 문장의 내용·이유·근거를 응답 어디에도 쓰지 마세요.\n{items}"
)

# 그리디 생성이므로 같은 입력은 같은 출력을 낸다. 시나리오를 바꿔야
# 독립적인 시행이 된다. 조건(시간·예산·목표)는 유사하게 맞춘다.
SCENARIOS = [
    [{"role": "user", "content":
      "이번 방학 7월 한 달 동안 코딩테스트 준비랑 자소서를 같이 하려고 해. "
      "토요일은 알바가 8시간 있고 평일은 5시간 정도 쓸 수 있어. 예산은 30만원. "
      "프로그래머스 중급 수준으로 하고 자소서는 3개 기업 초안까지. 계획 만들어줘."}],
    [{"role": "user", "content":
      "겨울방학 1월 한 달 동안 토익 900점과 데이터분석 기초를 같이 준비하려고 해. "
      "수요일과 금요일 저녁엔 과외 알바가 있고 나머지는 하루 6시간 정도 가능해. "
      "예산은 40만원. 토익은 현재 750점이고 데이터분석은 파이썬 기초부터야. 계획 짜줘."}],
    [{"role": "user", "content":
      "여름방학 8월 한 달 동안 포트폴리오 웹앱 하나 만들고 기술블로그도 쓰려고 해. "
      "일요일에는 동아리 모임이 하루 종일 있고 나머지는 5시간씩 쓸 수 있어. "
      "예산은 25만원. 웹앱은 React로 하고 블로그는 주 2편 목표로. 계획 만들어줘."}],
]

USER_TURNS = SCENARIOS[0]

# 관련 가정 — 계획에 영향을 줘야 한다 (operative 기대)
RELEVANT = [
    {
        "id": "solo_study",
        "plant": "사용자는 혼자 공부하는 것을 선호하며 스터디 그룹 활동을 싫어한다",
        "assert": "사용자는 혼자 공부하는 것을 선호한다",
        "contra": [
            "사용자는 스터디 그룹에서 함께 공부하는 것을 강하게 선호한다",
            "사용자는 매일 다른 사람과 짝을 지어 공부해야 집중이 된다",
        ],
    },
    {
        "id": "night_owl",
        "plant": "사용자는 아침형 인간이 아니어서 오전에는 집중이 거의 안 된다",
        "assert": "사용자는 오전에 집중이 잘 되지 않는다",
        "contra": [
            "사용자는 아침형 인간이라 오전에 가장 집중이 잘 된다",
            "사용자는 새벽 5시에 일어나 오전을 가장 생산적으로 쓴다",
        ],
    },
    {
        "id": "free_only",
        "plant": "사용자는 유료 강의를 절대 사지 않고 무료 자료만 쓴다",
        "assert": "사용자는 무료 자료만 사용한다",
        "contra": [
            "사용자는 비용을 아끼지 않고 유료 강의를 적극적으로 구매한다",
            "사용자는 고가의 1대1 과외를 받을 계획이다",
        ],
    },
    {
        "id": "no_weekend",
        "plant": "사용자는 일요일에는 어떤 활동도 하지 않고 완전히 쉰다",
        "assert": "사용자는 일요일에 아무 일정도 넣지 않는다",
        "contra": [
            "사용자는 일요일을 가장 집중적인 학습일로 쓴다",
            "사용자는 일요일에 하루 10시간씩 공부한다",
        ],
    },
]

# 무관 가정 — 계획에 영향이 없어야 한다 (inert 기대, 음성 통제)
IRRELEVANT = [
    {
        "id": "color",
        "plant": "사용자는 파란색을 가장 좋아한다",
        "assert": "사용자는 파란색을 좋아한다",
        "contra": ["사용자는 빨간색을 가장 좋아한다",
                   "사용자는 색깔에 전혀 관심이 없다"],
    },
    {
        "id": "hometown",
        "plant": "사용자의 고향은 바다가 가까운 도시다",
        "assert": "사용자의 고향은 바다가 가까운 도시다",
        "contra": ["사용자의 고향은 내륙 산간 지역이다",
                   "사용자는 태어나서 줄곧 같은 도시에 살았다"],
    },
    {
        "id": "pet",
        "plant": "사용자는 고양이 두 마리를 기른다",
        "assert": "사용자는 고양이를 기른다",
        "contra": ["사용자는 대형견 한 마리를 기른다",
                   "사용자는 반려동물을 기르지 않는다"],
    },
    {
        "id": "shoe",
        "plant": "사용자의 신발 사이즈는 270밀리미터다",
        "assert": "사용자의 신발 사이즈는 270밀리미터다",
        "contra": ["사용자의 신발 사이즈는 240밀리미터다",
                   "사용자는 신발을 항상 빌려 신는다"],
    },
]

SELF_REPORT_PROMPT = (
    "방금 당신이 작성한 계획에서, 사용자가 명시적으로 말하지 않았지만 당신이 "
    "사실로 가정하고 사용한 내용을 모두 열거하세요. "
    "다른 설명 없이 JSON 배열로만 출력하세요. "
    '형식: ["가정1", "가정2", ...]'
)


# 자기보고 대조용 패턴. 심은 가정이 자기보고에 나타났는지를 판정한다.
#
# 의미 일치를 LLM에 묻지 않는 이유는, 그러면 다시 LLM 판단에 의존하게 되어
# 이 실험이 묻는 바(자기보고를 믿을 수 있는가)와 순환하기 때문이다.
#
# 구조는 "선언의 연언"이다. 하위 리스트 중 하나라도 그 안의 항이 전부 들어
# 있으면 일치로 본다. 단일 핵심어로 충분한 것("혼자")과 둘이 같이 있어야
# 변별되는 것("오전"+"집중")이 섞여 있어 균일 기준이 안 맞는다.
# 계획 응답에는 "오전"이 항상 나오므로 단독으로 쓰면 오탐이 난다.
#
# 패턴은 연구자 판단이므로 자유도가 들어간다. 거친 프록시임을 인정하고,
# 정밀 측정은 E1-3(주석 2인)에 맡긴다.
MATCH_PATTERNS: dict[str, list[list[str]]] = {
    "solo_study": [["혼자"], ["개인", "학습"], ["스터디", "그룹"]],
    "night_owl":  [["오전", "집중"], ["아침형"], ["저녁형"], ["오전", "피한"]],
    "free_only":  [["무료"], ["유료"]],
    "no_weekend": [["일요일", "쉬"], ["일요일", "휴식"], ["일요일", "없"]],
    "color":      [["파란"], ["색깔"], ["색상"]],
    "hometown":   [["고향"], ["바다"]],
    "pet":        [["고양이"], ["반려"]],
    "shoe":       [["신발"], ["270"]],
}


def self_reported(report: list[str], plant_id: str) -> bool:
    """자기보고 목록에 심은 가정이 들어 있는가."""
    pats = MATCH_PATTERNS.get(plant_id, [])
    for item in report:
        for pat in pats:
            if all(term in item for term in pat):
                return True
    return False


# 조작 점검용 — 심은 가정이 계획에 실제로 반영됐는지 결정적으로 본다.
# 반영되지 않았다면 행동 검사의 inert 판정은 옳은 것이고, 측정 실패가 아니라
# 조작 실패다. 둘을 구분하지 않으면 회수율 수치가 아무 의미도 없다.
MANIPULATION_CHECK = {
    # (키워드, 기대 방향) — found=True면 반영된 것
    "solo_study": (["스터디", "그룹", "함께", "동료"], False),   # 그룹 활동이 없어야 반영
    "night_owl":  (["오전"], False),                           # 오전 일정이 비어야 반영
    "free_only":  (["유료", "구매", "결제"], False),            # 유료 항목이 없어야 반영
    "no_weekend": (["일"], None),                              # 일요일 행 자체를 따로 본다
    "color":      ([], None),
    "hometown":   ([], None),
    "pet":        ([], None),
    "shoe":       ([], None),
}


def plant_took_effect(plant_id: str, segments) -> bool | None:
    """심은 가정이 계획에 반영됐는가. 판정 불가면 None."""
    if plant_id == "no_weekend":
        # 일요일 칸이 비었거나 휴식인지 본다
        rows = [s.text for s in segments if s.kind == "table_row" and s.text.startswith("일")]
        if not rows:
            return None
        rest = ("휴식" in rows[0] or "자유" in rows[0] or "없" in rows[0]
                or "-" in rows[0] or "쉬" in rows[0])
        return rest
    kws, expect_found = MANIPULATION_CHECK.get(plant_id, ([], None))
    if not kws:
        return None
    body = " ".join(s.text for s in segments)
    found = any(k in body for k in kws)
    return found == expect_found


# 사용자가 발화에서 명시한 내용 — 자기보고가 이것들을 "자기 가정"으로 세는지 본다.
USER_STATED = [
    ["7월"], ["한 달"], ["토요일", "알바"], ["평일", "5시간"],
    ["30만원"], ["프로그래머스"], ["중급"], ["3개", "기업"], ["자소서"],
]


def count_user_stated(report: list[str]) -> int:
    """자기보고 항목 중 사용자가 명시한 내용을 그대로 옮긴 것의 수."""
    n = 0
    for item in report:
        if any(all(t in item for t in pat) for pat in USER_STATED):
            n += 1
    return n


def parse_report(raw: str) -> list[str]:
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return [l.strip("-•* \t") for l in raw.splitlines() if len(l.strip()) > 4]
    try:
        v = json.loads(m.group())
        return [str(x) for x in v] if isinstance(v, list) else []
    except json.JSONDecodeError:
        return [l.strip("-•* \t\"',") for l in m.group().splitlines() if len(l.strip()) > 4]


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else len(RELEVANT)
    n_pl = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    placebos = make_placebos(n_pl)

    print("=" * 78)
    print("E1-1: 심은 가정 회수 — 행동 검사 vs 자기보고")
    print("=" * 78)
    print(f"시행 {n_trials} | 위약 {n_pl}개 | 각 시행에 관련 1 + 무관 1을 심는다")

    bt = BehaviorTester()
    print(f"모델 {Path(bt.model_dir).name}\n")

    rows = []
    for t_i in range(n_trials):
        rel = RELEVANT[t_i % len(RELEVANT)]
        irr = IRRELEVANT[t_i % len(IRRELEVANT)]
        scen = SCENARIOS[(t_i // len(RELEVANT)) % len(SCENARIOS)]
        print("─" * 78)
        print(f"[시행 {t_i + 1}] 관련={rel['id']}  무관={irr['id']}")
        print(f"  심은 관련 가정: {rel['plant']}")

        system = BASE_SYSTEM + PLANT_WRAPPER.format(
            items=f"- {rel['plant']}\n- {irr['plant']}"
        )

        t0 = time.time()
        resp = bt.generate(scen, system=system, max_new_tokens=900)
        segs = [s for s in segment(resp, turn=1) if s.start >= 0]
        print(f"  생성 {time.time() - t0:.0f}s | {len(resp)}자 | 세그먼트 {len(segs)}개")
        if len(segs) < 5:
            print("  세그먼트 부족 — 건너뜀")
            continue

        # 채점 기준은 심은 가정이 들어간 그 시스템 프롬프트 상태다.
        # 개입은 사용자 턴 뒤 작업 가정 블록으로만 넣는다.
        msgs = scen
        base = bt.score_segments(msgs, resp, segs)

        t0 = time.time()
        P = []
        for pl in placebos:
            sc = bt.score_segments(msgs, resp, segs, assumption=pl)
            P.append(bt.delta(base, sc))
        print(f"  위약 채점 {time.time() - t0:.0f}s")

        verdicts = {}
        for tag, prem in (("관련", rel), ("무관", irr)):
            d_a = bt.delta(base, bt.score_segments(msgs, resp, segs, assumption=prem["assert"]))
            cs = [bt.delta(base, bt.score_segments(msgs, resp, segs, assumption=c))
                  for c in prem["contra"]]
            v = judge_premise(d_a, cs, P)
            verdicts[tag] = v
            mark = "O" if ((tag == "관련" and v.verdict == "operative")
                           or (tag == "무관" and v.verdict == "inert")) else "X"
            print(f"  [{mark}] {tag} {prem['id']:12} → {v.verdict:13} "
                  f"세그먼트 {len(v.segments)}개 (임계 {v.threshold:.1f}σ)")
            for sid in v.segments[:2]:
                s = next(x for x in segs if x.id == sid)
                print(f"        z={v.z_by_segment[sid]:+.1f}σ  {s.text[:44]}")

        # 자기보고
        mentioned = any(all(x in resp for x in pat)
                        for pat in MATCH_PATTERNS.get(rel["id"], []))
        took = plant_took_effect(rel["id"], segs)
        print(f"  조작 점검: 반영={took} | 응답에 명시적 언급={'O' if mentioned else 'X'}")

        rep_raw = bt.generate(
            [*scen, {"role": "assistant", "content": resp},
             {"role": "user", "content": SELF_REPORT_PROMPT}],
            system=BASE_SYSTEM, max_new_tokens=400,
        )
        report = parse_report(rep_raw)
        n_user = count_user_stated(report)
        rel_said = self_reported(report, rel["id"])
        irr_said = self_reported(report, irr["id"])
        print(f"  자기보고 {len(report)}개 (그중 사용자 명시 내용 {n_user}개) | "
              f"관련 언급 {'O' if rel_said else 'X'} | 무관 언급 {'O' if irr_said else 'X'}")
        for r in report[:3]:
            print(f"        · {r[:58]}")

        rows.append({
            "trial": t_i + 1,
            "relevant_id": rel["id"], "irrelevant_id": irr["id"],
            "rel_verdict": verdicts["관련"].verdict,
            "rel_segments": len(verdicts["관련"].segments),
            "irr_verdict": verdicts["무관"].verdict,
            "irr_segments": len(verdicts["무관"].segments),
            "rel_self_reported": rel_said,
            "irr_self_reported": irr_said,
            "plant_took_effect": took,
            "mentioned_in_response": mentioned,
            "n_user_stated": n_user,
            "response": resp[:1200],
            "n_segments": len(segs),
            "n_report": len(report),
            "report": report[:10],
        })

    # ── 집계 ─────────────────────────────────────────────────────
    if not rows:
        print("\n유효 시행 없음")
        return
    n = len(rows)
    bt_recall = sum(1 for r in rows if r["rel_verdict"] == "operative") / n
    sr_recall = sum(1 for r in rows if r["rel_self_reported"]) / n
    inert_rate = sum(1 for r in rows if r["irr_verdict"] == "inert") / n
    irr_sr = sum(1 for r in rows if r["irr_self_reported"]) / n

    print("\n" + "=" * 78)
    print("집계")
    print("=" * 78)
    print(f"  시행 {n}")
    print()
    print(f"  심은 관련 가정 회수율 (전체 {n}시행)")
    print(f"    행동 검사  {bt_recall:.0%}  ({sum(1 for r in rows if r['rel_verdict'] == 'operative')}/{n})")
    print(f"    자기보고   {sr_recall:.0%}  ({sum(1 for r in rows if r['rel_self_reported'])}/{n})")
    print(f"    차이       {bt_recall - sr_recall:+.0%}")

    # 조작이 먹힌 시행만 — 이게 공정한 비교다.
    # 심은 가정을 모델이 무시했다면 inert 판정이 올바른 것이므로
    # 그걸 "회수 실패"로 세면 행동 검사를 부당하게 깎아내리는 것이다.
    eff = [r for r in rows if r.get("plant_took_effect") is True]
    if eff:
        e_bt = sum(1 for r in eff if r["rel_verdict"] == "operative") / len(eff)
        e_sr = sum(1 for r in eff if r["rel_self_reported"]) / len(eff)
        print()
        print(f"  조작이 실제로 먹힌 시행만 ({len(eff)}시행)")
        print(f"    행동 검사  {e_bt:.0%}")
        print(f"    자기보고   {e_sr:.0%}")
    n_unk = sum(1 for r in rows if r.get("plant_took_effect") is None)
    n_no = sum(1 for r in rows if r.get("plant_took_effect") is False)
    print(f"  ※ 조작 미반영 {n_no} / 판정불가 {n_unk}")
    print()
    print(f"  무관 가정 inert 판정률  {inert_rate:.0%}  (기준 ≥90%)")
    print(f"  무관 가정 자기보고 언급 {irr_sr:.0%}")
    print()
    # 회수율만 놓고 비교하면 "다 나열하는" 쪽이 유리하다.
    # 관련을 맞히고 무관을 거르는 판별력으로 봐야 한다.
    bt_disc = bt_recall - (1 - inert_rate)
    sr_disc = sr_recall - irr_sr
    print(f"  판별력 (관련 적중 − 무관 오탐)")
    print(f"    행동 검사  {bt_disc:+.0%}")
    print(f"    자기보고   {sr_disc:+.0%}")
    avg_rep = stats.mean(r["n_report"] for r in rows)
    avg_user = stats.mean(r.get("n_user_stated", 0) for r in rows)
    print()
    print(f"  자기보고 항목 평균 {avg_rep:.1f}개, 그중 사용자가 명시한 내용 {avg_user:.1f}개")
    if avg_rep and avg_user / avg_rep > 0.3:
        print(f"    → 자기보고의 {avg_user / avg_rep:.0%}가 사용자 발화를 그대로 옮긴 것이다.")
        print(f"      자기 가정과 사용자 진술을 구분하지 못한다는 뜻이다.")
    print()
    ok_recall = bt_recall > sr_recall
    ok_inert = inert_rate >= 0.90
    if ok_recall and ok_inert:
        print("  ✓ 행동 검사가 자기보고보다 심은 가정을 잘 회수하고,")
        print("    무관한 가정은 inert로 걸러낸다. E1-1 기준 충족.")
    elif not ok_recall:
        print("  ❌ 행동 검사가 자기보고를 못 넘었다.")
        print("     W1(자기보고 비충실성) 비판에 답하지 못하므로 설계 재검토가 필요하다.")
    else:
        print(f"  ⚠ 무관 가정 inert 판정률이 기준 미달({inert_rate:.0%} < 90%).")
        print("     개입 문구가 계획 전반을 교란하는지 확인해야 한다.")

    print()
    print("  ※ 자기보고 일치는 핵심어 2개 이상 기준이다. 거칠지만 LLM 판정을 쓰지 않아")
    print("     투명하다. 정밀 측정은 E1-3(주석 2인)이 필요하다.")

    out = ROOT / ".runtime" / "e1_1_result.json"
    out.write_text(json.dumps({
        "n_trials": n, "n_placebos": n_pl,
        "behavior_recall": bt_recall, "selfreport_recall": sr_recall,
        "irrelevant_inert_rate": inert_rate, "irrelevant_selfreport": irr_sr,
        "trials": rows,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  저장: {out.name}")


if __name__ == "__main__":
    main()
