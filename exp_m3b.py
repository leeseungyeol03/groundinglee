"""M3 판정 보정 — 위약 100개 + 스튜던트화 max-T.

직전 실행(exp_m3.py)에서 작동 판정이 0/3이었다. 전제 수준에서는 반대 조건의
최대 Δ(0.665)가 임계값(0.564)을 넘겼는데도 세그먼트 단위로는 0건이었다.
원인 후보가 둘이다.

  (a) 세그먼트별 분산 이질성
      짧거나 위치에 민감한 세그먼트는 아무 삽입에나 크게 흔들린다.
      max-T 영분포가 그런 세그먼트에 끌려가 임계값이 위약 평균보다 5.5σ 위로 뜬다.
      → 세그먼트마다 위약 분포로 표준화(스튜던트화)하면 해소된다.

  (b) "두 반대 모두 임계 초과" 요구가 과도
      반대 값 2개는 의미가 서로 다르므로(예: "예산 제약 없음" vs "예산 3만원")
      같은 세그먼트를 똑같이 흔들 이유가 없다. 둘 다 요구하면 교집합만 남는다.

둘을 구분해서 측정하고, 동시에 위약을 19 → 100개로 늘린다.
로그확률 채점은 0.47초라 100개도 47초면 끝나고, 턴당 한 번만 계산해
모든 전제가 공유한다. 순열 p의 하한이 1/20 = 0.05에서 1/101 ≈ 0.0099로 내려가
반대 조건 2개에 Bonferroni를 걸 여지가 생긴다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe exp_m3b.py
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

from app.behavior_test import BehaviorTester, judge_premise
from app.segmenter import segment
from exp_m3 import MESSAGES, PREMISES, RESPONSE

# --------------------------------------------------------------------------- #
# 위약 100개 — 과업과 무관하되 형식·길이가 비슷해야 한다.
# 조합으로 만들어 수를 확보하되, 전부 "사용자에 대한 무관한 사실" 형태로 통일한다.
# --------------------------------------------------------------------------- #
_SUBJECTS = [
    "사용자는", "사용자의 가족은", "사용자의 친구는", "사용자의 이웃은",
]
_FACTS = [
    "파란색을 가장 좋아한다", "왼손잡이다", "매운 음식을 잘 먹는다",
    "안경을 쓰지 않는다", "지하철보다 버스를 자주 탄다", "겨울보다 여름을 좋아한다",
    "반려식물을 세 개 기른다", "수영을 할 줄 안다", "클래식 음악을 자주 듣는다",
    "빵보다 밥을 선호한다", "등산을 연 2회 정도 간다", "손목시계를 차지 않는다",
    "라면에 계란을 넣지 않는다", "커피보다 차를 마신다", "영화관보다 집에서 본다",
    "고양이보다 강아지를 좋아한다", "신발 사이즈가 270밀리미터다",
    "바다가 가까운 도시에서 자랐다", "형제가 두 명이다", "자전거를 탈 줄 안다",
    "우산을 자주 잃어버린다", "단 음식을 즐기지 않는다", "사진 찍는 것을 좋아한다",
    "대중교통 정기권을 쓴다", "휴대폰 배경을 바꾸지 않는다",
]


def make_placebos(n: int = 100) -> list[str]:
    out: list[str] = []
    for i in range(n):
        s = _SUBJECTS[i % len(_SUBJECTS)]
        f = _FACTS[(i // len(_SUBJECTS)) % len(_FACTS)]
        out.append(f"{s} {f}")
    # 중복 제거 후 부족하면 접미 변형으로 채운다
    uniq = list(dict.fromkeys(out))
    k = 0
    while len(uniq) < n:
        uniq.append(f"{_SUBJECTS[k % len(_SUBJECTS)]} {_FACTS[k % len(_FACTS)]} 편이다")
        k += 1
    return uniq[:n]


def perm_p(observed: float, null: list[float]) -> float:
    """단측 순열 p값. 하한은 1/(len(null)+1)."""
    ge = sum(1 for v in null if v >= observed)
    return (1 + ge) / (1 + len(null))


def main() -> None:
    n_pl = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    placebos = make_placebos(n_pl)

    print("=" * 78)
    print(f"M3 행동 검사 — 스튜던트화 max-T, 위약 {len(placebos)}개")
    print("=" * 78)
    print("판정: 반대 중 하나라도 임계 초과 && 단언은 임계 이하")
    print("임계값은 그 조건을 위약 풀에 넣고 전부 LOO로 채점한 뒤 위약 쪽 최대 z")

    bt = BehaviorTester()
    segs = [s for s in segment(RESPONSE, turn=1) if s.start >= 0]
    base = bt.score_segments(MESSAGES, RESPONSE, segs)
    seg_by_id = {s.id: s for s in segs}
    print(f"\n모델 {Path(bt.model_dir).name} | 세그먼트 {len(base)}개")

    # ── 위약 Δ 행렬 (턴당 1회, 모든 전제가 공유) ────────────────
    t0 = time.time()
    P: list[dict[str, float]] = []
    for pl in placebos:
        sc = bt.score_segments(MESSAGES, RESPONSE, segs, assumption=pl)
        P.append(bt.delta(base, sc))
    dt = time.time() - t0
    print(f"위약 채점 {dt:.0f}s ({dt / len(placebos):.2f}s/개)")

    # 세그먼트별 민감도 이질성 — 스튜던트화가 필요한 이유
    per_seg = {sid: [d[sid] for d in P if sid in d] for sid in base}
    sds = sorted(stats.pstdev(v) for v in per_seg.values() if len(v) > 2)
    print(f"세그먼트별 위약 표준편차 {sds[0]:.4f} ~ {sds[-1]:.4f} "
          f"(중앙 {sds[len(sds) // 2]:.4f}, 최대/중앙 {sds[-1] / sds[len(sds) // 2]:.1f}배)")

    # ── 전제별 판정 ──────────────────────────────────────────────
    print(f"\n전제 {len(PREMISES)}개 판정")
    results = []
    for prem in PREMISES:
        sc_a = bt.score_segments(MESSAGES, RESPONSE, segs, assumption=prem["assert"])
        d_a = bt.delta(base, sc_a)
        contras = []
        for c in prem["contra"]:
            sc_c = bt.score_segments(MESSAGES, RESPONSE, segs, assumption=c)
            contras.append(bt.delta(base, sc_c))

        v = judge_premise(d_a, contras, P)
        on = sum(1 for sid in v.segments
                 if prem["expect"] in seg_by_id[sid].path
                 or prem["expect"] in seg_by_id[sid].text)
        prec = on / len(v.segments) if v.segments else 0.0

        print(f"\n  ── {prem['slot']}  (관련: {prem['expect']})")
        print(f"     {v.verdict} | 기댄 세그먼트 {len(v.segments)}개 | "
              f"임계 {v.threshold:.2f}σ | FWER≤{v.alpha:.3f}")
        for sid in v.segments[:5]:
            s = seg_by_id[sid]
            tag = "O" if (prem["expect"] in s.path or prem["expect"] in s.text) else " "
            print(f"       [{tag}] z={v.z_by_segment[sid]:+.1f}σ  "
                  f"{s.path[:14]:16} {s.text[:34]}")
        if v.segments:
            print(f"     관련 섹션 적중 {on}/{len(v.segments)} = {prec:.0%}")

        results.append({
            "slot": prem["slot"], "expect": prem["expect"],
            "verdict": v.verdict, "n_segments": len(v.segments),
            "precision": prec, "threshold": v.threshold, "alpha": v.alpha,
            "segments": [seg_by_id[s].text[:44] for s in v.segments],
        })

    # ── 위약 오탐 (위약 하나를 대상 전제로 간주) ─────────────────
    print(f"\n위약 오탐 — 위약 {min(40, len(P))}개를 대상 전제로 간주")
    quiet = {sid: -99.0 for sid in base}
    fp = 0
    n_try = min(40, len(P))
    for i in range(n_try):
        others = [P[j] for j in range(len(P)) if j != i]
        if judge_premise(quiet, [P[i]], others).verdict == "operative":
            fp += 1
    print(f"  {fp}/{n_try} = {fp / n_try:.1%}  (기준 ≤5%)")

    # ── 결론 ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    n_op = sum(1 for r in results if r["verdict"] == "operative")
    tot = sum(r["n_segments"] for r in results)
    precs = [r["precision"] for r in results if r["n_segments"]]
    print(f"작동 {n_op}/{len(results)} | 기댄 세그먼트 총 {tot} | "
          f"적중률 평균 {stats.mean(precs) if precs else 0:.0%}")
    print(f"위약 오탐 {fp / n_try:.1%} | 위약 채점 {dt:.0f}s (턴당 1회)")
    print()
    ok = n_op == len(results) and fp / n_try <= 0.05
    if ok:
        print("✓ 전제 전부가 작동으로 판정됐고 오탐이 기준 이내다.")
        print("  로그확률 기반 행동 검사가 성립한다. G1 통과 조건 충족.")
    elif n_op == 0:
        print("❌ 작동 판정 0건. 위약을 늘리거나 세그먼트를 잘게 잘라야 한다.")
    else:
        print("~ 일부만 판정됐다. 전제별 원인을 봐야 한다.")

    out = ROOT / ".runtime" / "m3b_result.json"
    out.write_text(json.dumps({
        "n_placebos": len(P), "n_segments": len(base),
        "sd_min": sds[0], "sd_med": sds[len(sds) // 2], "sd_max": sds[-1],
        "fp_rate": fp / n_try, "placebo_seconds": dt,
        "premises": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out.name}")


if __name__ == "__main__":
    main()
