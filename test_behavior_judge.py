"""M3 판정 로직 검증 — 스튜던트화 max-T.

모델 없이 도는 순수 통계 로직이라 빠르게 돌릴 수 있다. 행동 검사의 판정이
무너지면 P2(사용 추적)와 P3(반영 검증)가 전부 무의미해지므로 여기를 고정한다.

검증 대상:
  - 귀무가설에서 거짓양성이 기준(5%) 이내인가
  - 세그먼트별 민감도가 달라도 편향되지 않는가 (스튜던트화의 존재 이유)
  - 단언 조건이 "개입 자체의 교란"을 걸러내는가
  - feedback2.md 원안(세그먼트별 순열 + BH)이 왜 안 되는지

실행: C:/seungyeol/vscode/GL/.runtime/python.exe test_behavior_judge.py
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

from app.behavior_test import judge_premise

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


SEGS = [f"s{i}" for i in range(19)]
# 실측에서 세그먼트별 위약 표준편차가 0.010~0.163으로 5.4배 차이났다.
# 그 이질성을 재현한다.
SENS = {s: (0.01 if i % 3 else 0.16) for i, s in enumerate(SEGS)}


def draw(rng: random.Random, scale: float = 1.0) -> dict[str, float]:
    return {s: rng.gauss(0.0, SENS[s] * scale) for s in SEGS}


def main() -> None:
    print("=" * 78)
    print("M3 판정 로직 — 스튜던트화 max-T")
    print("=" * 78)

    rng = random.Random(20261007)
    # B=200은 위약 수 민감도 측정에서 정한 값이다. B=100이면 위약 풀에 따라
    # 조건부 거짓양성이 6%까지 튀고, B=200이면 최대 3.2%로 안정된다.
    B = 200
    placebos = [draw(rng) for _ in range(B)]
    print(f"위약 {len(placebos)}개 | 세그먼트 {len(SEGS)}개 | 반대 2개 기준 이론 상한 {2 / (B + 1):.2%}")

    # ── 1. 영분포 구성 ───────────────────────────────────────────
    section("1. 판정 함수가 영분포를 올바르게 쓰는가")
    v0 = judge_premise(draw(rng), [draw(rng)], placebos)
    check(v0.threshold > 0, "임계값이 산출됨", f"{v0.threshold:.2f}")
    check(abs(v0.alpha - 1 / (B + 1)) < 1e-9, "단일 반대의 FWER 상한 = 1/(B+1)",
          f"{v0.alpha:.4f}")
    v0b = judge_premise(draw(rng), [draw(rng), draw(rng)], placebos)
    check(abs(v0b.alpha - 2 / (B + 1)) < 1e-9, "반대 2개면 union bound로 2배",
          f"{v0b.alpha:.4f}")
    check(len(v0b.z_by_segment) == len(SEGS), "모든 세그먼트에 z 산출",
          f"{len(v0b.z_by_segment)}/{len(SEGS)}")

    # ── 2. 귀무가설 거짓양성 ─────────────────────────────────────
    section("2. 귀무가설 — 효과 없는 전제는 inert여야")
    fp = 0
    trials = 400
    for _ in range(trials):
        if judge_premise(draw(rng), [draw(rng), draw(rng)], placebos).verdict == "operative":
            fp += 1
    rate = fp / trials
    print(f"  거짓양성 {fp}/{trials} = {rate:.2%}  (이론 상한 {2 / (B + 1):.2%})")
    check(rate <= 0.05, "거짓양성 ≤ 5%", f"{rate:.2%}")

    # ── 3. 위약 풀에 따른 조건부 변동 ────────────────────────────
    section("3. 위약 풀이 바뀌어도 기준을 지키는가")
    # 임계값이 영분포의 최대값이라 풀마다 흔들린다. B를 키운 이유가 이것이다.
    worst = 0.0
    rates = []
    for k in range(4):
        r2 = random.Random(5000 + k)
        pool = [draw(r2) for _ in range(B)]
        f = sum(
            1 for _ in range(150)
            if judge_premise(draw(r2), [draw(r2), draw(r2)], pool).verdict == "operative"
        ) / 150
        rates.append(f)
        worst = max(worst, f)
    print(f"  풀 4개 거짓양성: {' '.join(f'{x:.1%}' for x in rates)}  (최대 {worst:.1%})")
    check(worst <= 0.05, "최악의 풀에서도 ≤ 5%", f"{worst:.1%}")

    # ── 4. 스튜던트화가 편향을 없애는가 ──────────────────────────
    section("4. 스튜던트화 — 불안정한 세그먼트가 판정을 독점하지 않는가")
    hits_by_seg: dict[str, int] = {s: 0 for s in SEGS}
    for _ in range(400):
        for sid in judge_premise(draw(rng), [draw(rng), draw(rng)], placebos).segments:
            hits_by_seg[sid] += 1
    noisy = [s for s in SEGS if SENS[s] > 0.1]
    quiet = [s for s in SEGS if SENS[s] <= 0.1]
    n_hits = sum(hits_by_seg[s] for s in noisy)
    q_hits = sum(hits_by_seg[s] for s in quiet)
    print(f"  불안정({len(noisy)}개) 오탐 {n_hits} / 안정({len(quiet)}개) 오탐 {q_hits}")
    check(n_hits <= max(3, q_hits * 3 + 3),
          "불안정 세그먼트에 오탐이 쏠리지 않음", f"{n_hits} vs {q_hits}")

    # ── 5. 대립가설 검출력 ───────────────────────────────────────
    section("5. 대립가설 — 실제로 기댄 세그먼트를 잡는가")
    detected = correct = 0
    trials2 = 200
    for _ in range(trials2):
        target = rng.choice(SEGS)
        cs = [draw(rng), draw(rng)]
        cs[rng.randrange(2)][target] = SENS[target] * 12
        v = judge_premise(draw(rng), cs, placebos)
        if v.verdict == "operative":
            detected += 1
            if target in v.segments:
                correct += 1
    print(f"  검출 {detected}/{trials2} = {detected / trials2:.0%}"
          f" | 대상 적중 {correct}/{max(detected, 1)} = {correct / max(detected, 1):.0%}")
    check(detected / trials2 >= 0.90, "검출력 ≥ 90%", f"{detected / trials2:.0%}")
    check(correct / max(detected, 1) >= 0.95, "검출 시 대상 적중 ≥ 95%")

    # ── 6. 단언 조건 ─────────────────────────────────────────────
    section("6. 단언 조건 — 개입 자체의 교란을 걸러내는가")
    target = "s4"
    cs = [draw(rng), draw(rng)]
    cs[0][target] = SENS[target] * 12
    quiet_assert = {s: -99.0 for s in SEGS}
    v_ok = judge_premise(quiet_assert, cs, placebos)
    check(v_ok.verdict == "operative", "단언이 조용하면 operative", v_ok.verdict)
    check(target in v_ok.segments, "대상 세그먼트를 지목")

    loud_assert = {s: 0.0 for s in SEGS}
    loud_assert[target] = SENS[target] * 12
    v_dis = judge_premise(loud_assert, cs, placebos)
    check(v_dis.verdict == "inconclusive",
          "단언도 크게 흔들리면 inconclusive", v_dis.verdict)
    check(target not in v_dis.segments, "교란 시 기댐으로 집계하지 않음")

    # ── 7. 원안이 왜 안 되는가 ───────────────────────────────────
    section("7. 대조 — feedback2.md 원안의 다중비교 미통제")
    # 원안: 위약 19개를 쓰고, 세그먼트별로 "위약 전부보다 큼"을 판정 기준으로 삼는다.
    # 세그먼트가 19개면 시도가 19번이라 다중비교가 통제되지 않는다.
    small = placebos[:19]
    fp_perseg = 0
    for _ in range(300):
        c = draw(rng)
        if any(all(c[s] > p[s] for p in small) for s in SEGS):
            fp_perseg += 1
    r_perseg = fp_perseg / 300
    print(f"  세그먼트별 순열(위약 19개) 거짓양성 {r_perseg:.1%}")
    print(f"  스튜던트화 max-T(위약 {B}개) 거짓양성 {rate:.2%}")
    check(r_perseg > rate * 3, "원안이 실제로 더 많이 오판함 — 교체 근거",
          f"{r_perseg:.1%} vs {rate:.2%}")
    print(f"  ※ 위약 19개면 순열 p 하한이 1/20 = 0.050이라")
    print(f"     BH가 요구하는 (1/m)·q = {0.05 / len(SEGS):.4f}에 도달할 수 없다")

    print("\n" + "=" * 78)
    print(f"총계: {_passed} PASS / {_failed} FAIL")
    print("=" * 78)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
