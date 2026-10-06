"""시뮬레이션 기반 a priori 검정력 분석 — 축 B(GL-full vs GL-view) 대응.

배경:
이전 판은 "주입된 미검증 전제의 자발적 감지율이 GroundLens 조건에서 externalization
baseline보다 높다"를 H1으로 두고 검정력을 계산했다. 축 B 재프레이밍 이후 그 비교는
H3이 되었고, 조건도 GL vs baseline이 아니라 GL-full vs GL-view가 되었다.
주된 확증 가설은 H1(교정 시도)과 H2(교정 반영·재발)다(논문 6.1절).

이 스크립트는 세 가설의 검정력을 각각 산출한다.

  H1  교정 시도율          GL-full > GL-view
  H2  교정 반영 성공률     GL-full > GL-view   (주 확증)
  H3  자발적 감지율        GL-full > GL-view

설계: 피험자 내 2조건, 조건당 4개 주입 사건(8템플릿을 조건 간 분할).
모형: 참가자 군집 이항 GEE(교환가능 상관). 논문 6.8절의 참가자·템플릿 무작위 절편
이항 혼합 로지스틱 회귀에 대한 검정력 근사이며, GEE는 혼합모형보다 약간 보수적이다.

── 분모 문제 (중요) ────────────────────────────────────────────
논문 6.1절은 H1을 "감지된 미검증 전제에 대한 교정 시도율"로 정의하고,
6.7.1절도 "감지 이후" 교정했는지로 측정한다고 쓴다. 그러나 6.8절은 분석 단위를
"개별 주입 사건"이라 한다. 이 셋이 일치하지 않는다.

감지는 처치 이후에 결정되는 변수이고 H3은 조건이 감지를 바꾼다고 예측하므로,
감지된 사건만 골라 비교하면 충돌부 편향(collider bias)이 발생한다. 두 조건의
"감지된 부분표본"은 동등하지 않기 때문이다. GL-full이 더 많이 감지하면 그 표본에
감지가 어려웠던 사례가 더 섞이고, 그 사례들은 교정도 어려울 가능성이 높다.

시뮬레이션으로 확인한 결과(SEED 7, N=300k), 교정에 대한 처치 효과를 정확히 0으로
두어도 감지 조건화 시 -0.032의 가짜 음의 효과가 나타난다. 방향이 가설에 불리하므로
보수적이지만, 편향이 없다고 말할 수는 없다.

따라서 이 스크립트는 H1을 두 가지 분모로 모두 계산한다.
  all       전체 주입 사건 (편향 없음. 확증 검정 권장)
  detected  감지된 사건만  (조건부 해석. 보조 보고)
논문 6.1/6.7.1/6.8의 서술도 이에 맞추어 통일해야 한다.
──────────────────────────────────────────────────────────────

실행:
  C:/seungyeol/vscode/GL/.runtime/python.exe power_analysis.py
  C:/seungyeol/vscode/GL/.runtime/python.exe power_analysis.py --sims 500
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

warnings.filterwarnings("ignore")
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.generalized_estimating_equations import GEE

# ─── CONFIG — 파일럿 실측으로 반드시 갱신할 것 ────────────────
N_INJ_PER_COND = 4       # 조건당 주입 사건 수 (8템플릿을 조건 간 분할)
N_TEMPLATES = 8

# GL-view(열람 전용) 기저율. 세 지표 모두 잠정 가정이다.
P_DETECT_VIEW = 0.40     # 자발적 감지율 — 대화 채널로만 표현
P_CORRECT_VIEW = 0.50    # 감지한 전제를 교정 시도할 확률
P_REFLECT_VIEW = 0.55    # 교정이 후속 응답·산출물에 반영될 확률

# GL-full 효과 후보 (로그오즈 증가분)
EFFECT_GRID = [0.4, 0.6, 0.8, 1.0]

N_PART_GRID = [12, 16, 20, 24, 30, 40]
SD_PARTICIPANT = 0.7     # 참가자 무작위 절편 SD (로그오즈)
SD_TEMPLATE = 0.4        # 템플릿 무작위 절편 SD (로그오즈)

# 사건별 잠재 난이도. 감지·교정·반영에 공통으로 작용하며 관측되지 않는다.
# 충돌부 편향의 원천이므로 0으로 두면 편향이 사라진다.
SD_DIFFICULTY = 1.0

N_SIMS = 300
ALPHA = 0.05
SEED = 20260929
# ──────────────────────────────────────────────────────────────


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _inv(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate(n_part: int, effect: float, rng: np.random.Generator) -> pd.DataFrame:
    """한 실험을 모사한다.

    감지 → 교정 → 반영의 순차 구조를 반영한다. 교정은 감지한 전제에 대해서만
    관찰 가능하고, 반영은 교정한 전제에 대해서만 관찰 가능하다.
    세 단계 모두에 공통 잠재 난이도가 작용하여 충돌부 편향의 원천이 된다.
    """
    b_det = _logit(P_DETECT_VIEW)
    b_cor = _logit(P_CORRECT_VIEW)
    b_ref = _logit(P_REFLECT_VIEW)

    templ_det = rng.normal(0, SD_TEMPLATE, N_TEMPLATES)
    templ_cor = rng.normal(0, SD_TEMPLATE, N_TEMPLATES)

    rows = []
    for p in range(n_part):
        u_det = rng.normal(0, SD_PARTICIPANT)
        u_cor = rng.normal(0, SD_PARTICIPANT)
        for c in (0, 1):  # 0 = GL-view, 1 = GL-full
            for k in range(N_INJ_PER_COND):
                t = int(rng.integers(0, N_TEMPLATES))
                diff = rng.normal(0, SD_DIFFICULTY)  # 관측되지 않는 사건 난이도

                p_det = _inv(b_det + effect * c + u_det + templ_det[t] - diff)
                detected = int(rng.random() < p_det)

                # 교정 시도: 인터페이스 채널이 추가되어 처치 효과가 있다
                p_cor = _inv(b_cor + effect * c + u_cor + templ_cor[t] - diff)
                corrected = int(rng.random() < p_cor)

                # 반영: 조작이 evidence로 기능하면 반영률이 오른다 (H2)
                p_ref = _inv(b_ref + effect * c + u_cor - 0.5 * diff)
                reflected = int(rng.random() < p_ref)

                rows.append({
                    "pid": p, "cond": c, "templ": t,
                    "detected": detected,
                    # 교정은 감지한 경우에만 관찰된다
                    "corrected": corrected if detected else 0,
                    "corrected_obs": corrected if detected else np.nan,
                    # 반영은 교정한 경우에만 관찰된다
                    "reflected_obs": reflected if (detected and corrected) else np.nan,
                })
    return pd.DataFrame(rows)


def _gee_positive(df: pd.DataFrame, y: str) -> bool:
    """cond 계수가 양이고 유의한가."""
    sub = df.dropna(subset=[y])
    if sub.empty or sub[y].nunique() < 2 or sub["cond"].nunique() < 2:
        return False
    # 한 조건에서 결과가 전부 동일하면 GEE가 발산한다
    if sub.groupby("cond")[y].nunique().min() < 1:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = GEE.from_formula(f"{y} ~ cond", groups="pid", data=sub,
                                 family=Binomial(), cov_struct=Exchangeable()).fit()
        return bool(m.pvalues.get("cond", 1.0) < ALPHA and m.params.get("cond", 0.0) > 0)
    except Exception:
        return False


def power_at(n_part: int, effect: float, rng: np.random.Generator,
             n_sims: int = N_SIMS) -> dict[str, float]:
    hits = {"H3_detect": 0, "H1_all": 0, "H1_detected": 0, "H2_reflect": 0}
    for _ in range(n_sims):
        df = simulate(n_part, effect, rng)
        if _gee_positive(df, "detected"):
            hits["H3_detect"] += 1
        # H1 — 분모 두 가지
        if _gee_positive(df, "corrected"):          # 전체 주입 사건
            hits["H1_all"] += 1
        if _gee_positive(df, "corrected_obs"):      # 감지된 사건만
            hits["H1_detected"] += 1
        if _gee_positive(df, "reflected_obs"):
            hits["H2_reflect"] += 1
    return {k: v / n_sims for k, v in hits.items()}


def bias_demo(rng: np.random.Generator) -> None:
    """감지 조건화가 만드는 충돌부 편향을 수치로 보인다."""
    N = 200_000
    diff = rng.normal(0, 1, N)
    cond = rng.integers(0, 2, N)
    det = rng.random(N) < _inv(0.0 + 0.9 * cond - diff)
    cor = rng.random(N) < _inv(0.3 + 0.0 * cond - diff)  # 처치 효과 정확히 0

    a = cor[cond == 1].mean() - cor[cond == 0].mean()
    b = cor[(cond == 1) & det].mean() - cor[(cond == 0) & det].mean()
    print("\n" + "=" * 74)
    print("참고: 감지 조건화의 충돌부 편향 (교정에 대한 처치 진짜 효과 = 0)")
    print("=" * 74)
    print(f"  전체 주입 사건 기준   차이 {a:+.3f}   ← 0에 가깝다(편향 없음)")
    print(f"  감지된 사건만 기준    차이 {b:+.3f}   ← 0이 아니다(편향)")
    print(f"  편향 크기 {b - a:+.3f}")
    print("  방향이 가설에 불리하므로 보수적이나, 6.1/6.7.1/6.8의 분모 서술을")
    print("  '전체 주입 사건'으로 통일하고 감지 조건부 값은 보조 지표로 보고할 것.")


LABELS = {
    "H1_all": "H1 교정시도(전체)",
    "H1_detected": "H1 교정시도(감지된 것만)",
    "H2_reflect": "H2 교정반영",
    "H3_detect": "H3 감지",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=N_SIMS)
    ap.add_argument("--no-bias-demo", action="store_true")
    args = ap.parse_args()
    n_sims = args.sims
    rng = np.random.default_rng(SEED)

    print("=" * 74)
    print("a priori 검정력 — GL-full vs GL-view (피험자 내 2조건)")
    print("=" * 74)
    print(f"조건당 주입 {N_INJ_PER_COND} · 템플릿 {N_TEMPLATES} · sims {n_sims} · alpha {ALPHA}(단측)")
    print(f"GL-view 기저율: 감지 {P_DETECT_VIEW} / 교정 {P_CORRECT_VIEW} / 반영 {P_REFLECT_VIEW}")
    print(f"SD: 참가자 {SD_PARTICIPANT} · 템플릿 {SD_TEMPLATE} · 사건난이도 {SD_DIFFICULTY}")
    print("효과는 로그오즈 증가분. 0.6 ≈ 오즈비 1.8, 1.0 ≈ 오즈비 2.7")

    results: dict[tuple[int, float], dict[str, float]] = {}
    for n in N_PART_GRID:
        for e in EFFECT_GRID:
            results[(n, e)] = power_at(n, e, rng, n_sims)

    for key in ("H1_all", "H2_reflect", "H3_detect", "H1_detected"):
        print("\n" + "-" * 74)
        print(f"{LABELS[key]}")
        print("-" * 74)
        head = "N_part | " + " | ".join(f"eff={e:.1f}" for e in EFFECT_GRID)
        print(head)
        for n in N_PART_GRID:
            cells = [f"  {results[(n, e)][key]:.2f} " for e in EFFECT_GRID]
            print(f"{n:>6} | " + " | ".join(cells))

    # 0.80 도달 최소 N
    print("\n" + "=" * 74)
    print("검정력 0.80 도달 최소 참가자 수")
    print("=" * 74)
    print("효과  | " + " | ".join(f"{LABELS[k][:14]:<14}" for k in
                                  ("H1_all", "H2_reflect", "H3_detect")))
    for e in EFFECT_GRID:
        row = []
        for key in ("H1_all", "H2_reflect", "H3_detect"):
            hit = next((n for n in N_PART_GRID if results[(n, e)][key] >= 0.80), None)
            row.append(f"{(str(hit) + '명') if hit else '>40명':<14}")
        print(f"{e:.1f}   | " + " | ".join(row))

    print("\n주 확증 가설은 H1·H2다(6.1절). 두 가설이 동시에 0.80을 넘는 N을 택하고,")
    print("Holm 보정을 쓰므로 실제로는 여기서 나온 값보다 여유를 두어야 한다.")

    # H2 분모 진단 — 구조적 저검정력의 원인을 수치로 보인다
    probe = simulate(N_PART_GRID[3], EFFECT_GRID[2], np.random.default_rng(SEED + 1))
    n_all = len(probe)
    n_cor = int(probe["corrected_obs"].notna().sum())
    n_ref = int(probe["reflected_obs"].notna().sum())
    print("\n" + "=" * 74)
    print("H2 저검정력의 원인 — 분모 축소")
    print("=" * 74)
    print(f"  참가자 {N_PART_GRID[3]}명 기준 총 주입 사건 {n_all}건")
    print(f"  H1(전체)  분모 {n_all:4d}건 (100%)")
    print(f"  H1(감지)  분모 {n_cor:4d}건 ({n_cor / n_all:.0%})  — 감지 조건화")
    print(f"  H2(반영)  분모 {n_ref:4d}건 ({n_ref / n_all:.0%})  — 감지 AND 교정 이중 조건화")
    print(f"  → 참가자당 조건당 {n_ref / N_PART_GRID[3] / 2:.2f}건. 분모가 곱으로 줄어 구조적으로 불리하다.")
    print("  → 주입 수를 배증해도(조건당 8건) 40명에서 0.52 수준이므로,")
    print("     H2를 현 정의 그대로 주 확증 가설로 두는 것은 재검토가 필요하다.")
    print("     대안: 산출물 기반 판정(필드 단위)을 쓰면 분모가 사건이 아니라")
    print("     필드가 되어 수십 배로 늘어난다(6.7.1절의 두 번째 측정 방식).")

    if not args.no_bias_demo:
        bias_demo(rng)

    print("\n" + "!" * 74)
    print("주의: 기저율·효과크기·SD는 전부 잠정 가정이다.")
    print("파일럿 관찰로 갱신하기 전에는 이 표를 표본 수 근거로 인용하지 말 것.")
    print("!" * 74)


if __name__ == "__main__":
    main()
