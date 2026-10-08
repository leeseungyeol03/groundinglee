"""M5 수정 전파·반영 검증 — 모델 없이 도는 로직 검증.

왜 모델 없이 되는가:
M5의 핵심 판정은 세 가지다. (a) 어떤 세그먼트가 옛 전제에 기댔나,
(b) 새 응답이 정정을 반영했나, (c) 무관한 부분이 함께 바뀌었나. (a)와 (b)는
Δ 행렬만 있으면 되고, (c)는 M0 정렬만 있으면 된다. Δ를 주입하면 모델 없이
전수 검증할 수 있고, 그래야 회귀로 묶을 수 있다.

M5는 메모리식 기준선과 가장 크게 갈리는 모듈이라(문서 M5) 여기가 틀리면
E2의 주 결과가 무의미해진다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe test_propagate.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.propagate import (
    PropagationResult,
    find_collateral,
    significant_segments,
)
from app.segmenter import align, segment
from app.state_store import Evidence, Premise, StateStore

_passed = 0
_failed = 0


def check(cond: bool, name: str, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"   ({detail})" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


SEGS = [f"s{i}" for i in range(20)]
# 실측에서 세그먼트별 위약 민감도가 5배 차이났다. 그 이질성을 재현한다.
SENS = {s: (0.01 if i % 3 else 0.15) for i, s in enumerate(SEGS)}


def draw(rng: random.Random) -> dict[str, float]:
    return {s: rng.gauss(0.0, SENS[s]) for s in SEGS}


PLAN_V1 = """## 1. 목표
1. 프로그래머스 중급 100문제
2. 자소서 3개 기업

## 2. 주간 일정
| 월 | 알고리즘 학습 | 문제풀이 |
| 화 | 알고리즘 학습 | 자소서 |

## 4. 예산
- 인터넷 강의: 15만원
- 교재: 10만원
"""


def main() -> None:
    print("=" * 78)
    print("M5 수정 전파·반영 검증")
    print("=" * 78)

    rng = random.Random(20261008)
    B = 200
    placebos = [draw(rng) for _ in range(B)]

    # ── 1. 유의성 판정 교정 ──────────────────────────────────────
    section("1. 유의성 판정 — 귀무가설에서 거짓양성이 통제되는가")
    fp = 0
    trials = 400
    for _ in range(trials):
        hits, _z, _t = significant_segments(draw(rng), placebos)
        if hits:
            fp += 1
    rate = fp / trials
    print(f"  거짓양성 {fp}/{trials} = {rate:.2%}  (이론 상한 1/{B + 1} = {1 / (B + 1):.2%})")
    check(rate <= 0.05, "거짓양성 ≤ 5%", f"{rate:.2%}")

    # ── 2. 위약 풀이 바뀌어도 ────────────────────────────────────
    section("2. 위약 풀이 바뀌어도 기준을 지키는가")
    worst = 0.0
    rates = []
    for k in range(4):
        r2 = random.Random(9100 + k)
        pool = [draw(r2) for _ in range(B)]
        f = sum(1 for _ in range(150) if significant_segments(draw(r2), pool)[0]) / 150
        rates.append(f)
        worst = max(worst, f)
    print(f"  풀 4개: {' '.join(f'{x:.1%}' for x in rates)}  (최대 {worst:.1%})")
    check(worst <= 0.05, "최악의 풀에서도 ≤ 5%", f"{worst:.1%}")

    # ── 3. 검출력 ────────────────────────────────────────────────
    section("3. 실제로 흔들린 세그먼트를 잡는가")
    det = cor = 0
    for _ in range(200):
        tgt = rng.choice(SEGS)
        d = draw(rng)
        d[tgt] = SENS[tgt] * 12
        hits, _z, _t = significant_segments(d, placebos)
        if hits:
            det += 1
            if tgt in hits:
                cor += 1
    print(f"  검출 {det}/200 = {det / 200:.0%} | 대상 적중 {cor}/{max(det, 1)} = {cor / max(det, 1):.0%}")
    check(det / 200 >= 0.90, "검출력 ≥ 90%", f"{det / 200:.0%}")
    check(cor / max(det, 1) >= 0.95, "대상 적중 ≥ 95%")

    # ── 4. 반영 판정의 방향 ──────────────────────────────────────
    section("4. 반영 판정 — 흔들리면 '반영 안 됨'이어야")
    # 새 값을 단언해도 조용하면 이미 반영된 것이다
    quiet = {s: 0.0 for s in SEGS}
    hits_q, _z, _t = significant_segments(quiet, placebos)
    check(not hits_q, "단언이 조용하면 반영 확인(유의 세그먼트 0)",
          f"{len(hits_q)}개")

    # 어떤 세그먼트가 옛 값에 기대 있으면 새 값 단언에 흔들린다
    stale = {s: 0.0 for s in SEGS}
    stale["s7"] = SENS["s7"] * 15
    hits_s, _z, _t = significant_segments(stale, placebos)
    check(hits_s == ["s7"], "옛 값에 기댄 세그먼트만 지목", f"{hits_s}")

    # ── 5. 함께 바뀐 부분 ────────────────────────────────────────
    section("5. 함께 바뀐 부분 — 영향 범위 밖의 변경을 잡는가")
    v1 = segment(PLAN_V1, turn=1)
    # 예산만 바꾼 개정 (의도된 변경)
    v2_text = PLAN_V1.replace("인터넷 강의: 15만원", "무료 강의 활용: 0원")
    v2 = segment(v2_text, turn=2)
    r = align(v1, v2) if False else None  # align은 find_collateral 내부에서 호출된다
    budget_seg = next(s for s in v2 if "무료" in s.text or "0원" in s.text) \
        if any("무료" in s.text or "0원" in s.text for s in v2) else None
    affected = [budget_seg.id] if budget_seg else []
    ids, texts = find_collateral(v1, v2, affected)
    print(f"  의도된 변경만 있을 때 부수 변경 {len(ids)}개")
    check(len(ids) == 0, "영향 범위 내 변경은 부수 변경으로 세지 않음",
          f"{texts}")

    # 무관한 부분까지 바뀐 개정
    v3_text = (v2_text
               .replace("알고리즘 학습 | 자소서", "알고리즘 학습 | 휴식")
               .replace("자소서 3개 기업", "자소서 5개 기업"))
    v3 = segment(v3_text, turn=3)
    ids3, texts3 = find_collateral(v1, v3, affected)
    print(f"  무관한 변경이 있을 때 부수 변경 {len(ids3)}개")
    for t in texts3[:3]:
        print(f"    · {t}")
    check(len(ids3) >= 2, "영향 범위 밖 변경을 잡아냄", f"{len(ids3)}개")
    check(all("강의" not in t for t in texts3),
          "의도된 예산 변경은 부수 변경에 포함되지 않음", f"{texts3}")

    # ── 6. 상태 전이와 원칙 2 ────────────────────────────────────
    section("6. 상태 전이 — 행동 검사가 CG 승격 근거가 되지 않는가")
    s = StateStore()
    s.add(Premise(id="p1", text="예산 30만원은 강의와 교재에 나눈다"))
    s.set_verdict("p1", "operative", turn=3, segments=["s1", "s2"])
    p1 = s.get("p1")
    check(p1.status == "proposed",
          "operative 판정이 status를 바꾸지 않음",
          "행동 검사를 승격 근거로 쓰면 원칙 2 위반")
    check(p1.verdict == "operative", "판정은 verdict에 기록됨")
    check(not p1.is_grounded, "여전히 미확인 전제")

    s.ground("p1", Evidence(type="ui_ack", turn=4, pointer="op_1"))
    check(s.get("p1").is_grounded, "사용자 증거로는 승격됨")

    # ── 7. 수정 시 옛 값 보존 ────────────────────────────────────
    section("7. 수정 — 옛 값을 지우지 않는가")
    new = s.correct("p1", "예산 15만원은 무료 자료로 쓴다",
                    Evidence(type="ui_correction", turn=5, pointer="op_2"))
    old = s.get("p1")
    check(old.status == "corrected", "옛 전제는 corrected로 남음")
    check(old.text == "예산 30만원은 강의와 교재에 나눈다",
          "옛 값 문구가 보존됨", "옛 값을 지우면 반영 재검사를 할 수 없다")
    check(old.superseded_by == new.id, "새 전제로 연결됨")
    check(new.status == "corrected" and new.text.startswith("예산 15만원"),
          "새 전제가 생성됨")
    pairs = s.superseded_pairs()
    check(len(pairs) == 1 and pairs[0][0].id == "p1", "정정 쌍으로 조회됨")

    # ── 8. 파생 전제 재검토 ──────────────────────────────────────
    section("8. 파생 전제 — 자동 수정 없이 재검토 표시만")
    s.add(Premise(id="p2", text="교재는 중고로 구한다", derived_from=["p1"]))
    s.add(Premise(id="p3", text="중고 서점 세 곳을 돈다", derived_from=["p2"]))
    s.add(Premise(id="p4", text="4단계 파생", derived_from=["p3"]))
    s.add(Premise(id="p9", text="무관한 전제"))
    flagged = s.flag_derived("p1", depth=2)
    print(f"  표시된 전제 {flagged}")
    check("p2" in flagged, "1단계 파생이 표시됨")
    check("p3" in flagged, "2단계 파생이 표시됨")
    check("p4" not in flagged, "깊이 제한 2가 지켜짐", "깊이 제한이 없으면 전부 표시된다")
    check("p9" not in flagged, "무관한 전제는 표시되지 않음")
    check(s.get("p2").status == "proposed",
          "재검토 표시가 status를 바꾸지 않음", "자동으로 고치면 안 된다")

    # ── 9. 결과 객체 ─────────────────────────────────────────────
    section("9. 결과 객체")
    res = PropagationResult(premise_id="p1", old_text="옛", new_text="새")
    check(res.status == "반영 안 됨", "기본 상태는 반영 안 됨",
          "반영을 가정하면 추정 그라운딩이다")
    res.reflected = True
    check(res.status == "반영 확인", "반영 시 상태 전환")
    d = res.to_dict()
    for k in ["premise_id", "affected", "unreflected", "collateral", "status"]:
        check(k in d, f"직렬화에 {k} 포함")

    print("\n" + "=" * 78)
    print(f"총계: {_passed} PASS / {_failed} FAIL")
    print("=" * 78)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
