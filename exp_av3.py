"""A-V3: 위약 오탐률과 재생성 잡음 — 제안서가 "가장 큰 가정"이라 부른 것의 검증.

제안서 5.4의 A-V3은 "위약을 대상 전제로 취급했을 때 작동으로 판정되는 비율 ≤ 5%"다.
그런데 판정 임계값 자체가 위약 분포의 95% 분위이므로, 이 수치는 거의 정의상 5%가
나온다. 그것만 재면 아무것도 확인하지 못한다.

실제로 확인해야 하는 것은 제안서 말미가 밝힌 전제다:

  "가장 큰 가정은 재생성 잡음이 위약 기준선으로 통제될 만큼 작다는 것입니다.
   이 가정이 틀리면 A의 범위가 줄어듭니다."

즉 **위약 Δ가 작은가, 그리고 실제 전제의 Δ가 그보다 뚜렷하게 큰가**가 관건이다.
분리가 없으면 어떤 전제든 작동으로 보이거나 아무것도 작동으로 안 보인다.

그래서 네 가지를 함께 잰다.
  1) 잡음 바닥   개입 없이 같은 컨텍스트로 반복 생성 → Δ가 0인가
  2) 위약 분포   무관한 가정 삽입 → Δ가 잡음 바닥 수준인가
  3) 실제 전제   대체·단언 → 대체는 크고 단언은 작은가
  4) 분리·오탐   위약 p95 기준으로 판정했을 때의 분리폭과 오탐률

실행: C:/seungyeol/vscode/GL/.runtime/python.exe exp_av3.py [반복수]
"""
from __future__ import annotations

import json
import os
import statistics as stats
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["MOCK_LLM"] = "false"

import logging

import yaml

logging.disable(logging.CRITICAL)

from app.config import assemble_partner_system_prompt
from app.intervention import (
    build_intervention_prompt,
    plan_delta,
    section_delta,
    structured_plan_delta,
)

# 분석 모드. "text"는 토큰 Jaccard, "struct"는 의미 단위 환원.
# 1차 실토에서 text 모드는 자유 텍스트 환언이 위약 기준선을 0.4까지
# 끌어올려 분리가 사라진다. 제안서 5.6의 대안을 기본값으로 둔다.
DELTA_MODE = os.environ.get("AV3_MODE", "struct")
_DELTA = structured_plan_delta if DELTA_MODE == "struct" else plan_delta
from app.llm import get_assistant_response
from app.plan_parser import has_plan, parse_plan

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

# 고정 전사. 매 시행 생성하면 조건 간 비교가 오염되므로 하드코딩한다.
HISTORY = [
    {"role": "user", "content":
     "이번 방학 7월 한 달 동안 코딩테스트 준비랑 자소서를 같이 하려고 해. "
     "토요일은 알바가 8시간 있고 평일은 5시간 정도 쓸 수 있어. 예산은 30만원."},
    {"role": "assistant", "content":
     "조건을 정리하겠습니다. 7월 한 달, 평일 5시간 가용, 토요일 알바 8시간, 예산 30만원으로 "
     "코딩테스트 준비와 자소서 작성을 병행하는 계획을 세우겠습니다."},
    {"role": "user", "content":
     "코테는 프로그래머스 중급 수준으로 하고 자소서는 3개 기업 초안까지. 계획 만들어줘."},
]

# 위약 — 과업과 무관하되 길이가 비슷한 가정.
# "작업 가정" 블록에 들어가는 텍스트 자체가 계획을 흔드는지를 재는 기준선이다.
PLACEBOS = [
    "사용자는 파란색을 가장 좋아한다",
    "사용자는 왼손잡이다",
    "사용자의 고향은 바다가 가까운 도시다",
    "사용자는 고양이보다 강아지를 좋아한다",
    "사용자는 아침에 커피보다 차를 마신다",
    "사용자는 영화관보다 집에서 영화를 본다",
    "사용자의 신발 사이즈는 270밀리미터다",
    "사용자는 매운 음식을 잘 먹는다",
    "사용자는 안경을 쓰지 않는다",
    "사용자는 지하철보다 버스를 자주 탄다",
]

# 실제 전제 — 계획에 영향을 줄 것으로 예상되는 것.
# 각 항목은 (슬롯 이름, 단언할 값, 대체할 값 2개, 영향받을 주소 접두사)
REAL_PREMISES = [
    {
        "slot": "budget.level",
        "assert": "숙박·교재 등 비용은 저렴한 쪽을 택해 총 30만원 안에서 해결한다",
        "substitutes": [
            "비용은 신경 쓰지 않고 가장 효과적인 유료 강의와 교재를 쓴다",
            "예산은 10만원까지만 쓰고 나머지는 전부 무료 자료로 해결한다",
        ],
        "expect_prefix": "budget",
    },
    {
        "slot": "schedule.density",
        "assert": "평일 5시간은 오전과 오후에 나누어 배치하고 저녁은 비워 둔다",
        "substitutes": [
            "평일 가용 시간은 전부 저녁에 몰아서 배치한다",
            "평일에는 오전만 공부하고 오후와 저녁은 모두 휴식으로 둔다",
        ],
        "expect_prefix": "schedule",
    },
    {
        "slot": "priority.order",
        "assert": "코딩테스트 준비가 자소서보다 우선순위가 높다",
        "substitutes": [
            "자소서 작성이 코딩테스트 준비보다 우선순위가 높다",
            "두 목표의 우선순위는 완전히 동등하며 시간을 정확히 반으로 나눈다",
        ],
        "expect_prefix": "priority",
    },
]


def generate_plan(assumption: str | None, temperature: float = 0.0) -> list[dict]:
    """개입을 적용해 계획만 생성하고 필드로 파싱한다."""
    base = assemble_partner_system_prompt(CFG, [], [], [])
    prompt = build_intervention_prompt(base, assumption)
    cfg = dict(CFG)
    cfg["temperature"] = temperature
    r = get_assistant_response(HISTORY, prompt, cfg)
    if not has_plan(r.text):
        return []
    return parse_plan(r.text, turn=1)

def p95(values: list[float]) -> float:
    """95% 분위. 표본이 적으면 최대값으로 대체한다."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) < 20:
        return s[-1]
    return s[int(0.95 * (len(s) - 1))]


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print("=" * 78)
    print("A-V3: 위약 기준선과 필드별 인과 판정")
    print(f"모델 {CFG['model']} · greedy · 계획 전용 생성 · 모드 {DELTA_MODE}")
    print("=" * 78)
    print("판정은 제안서 5.2대로 **필드별**로 한다. 계획 전체 평균으로 판정하면")
    print("한 섹션만 바꾸는 전제가 안 바뀐 나머지 필드에 희석돼 비작동으로 보인다.")

    # ── 1. 기준 계획과 잡음 바닥 ──────────────────────────────────
    print("\n[1] 잡음 바닥 — 개입 없이 반복")
    ref = generate_plan(None)
    if not ref:
        print("  기준 생성 실패 — 중단")
        return
    print(f"  기준: {len(ref)}필드")

    noise_runs = []
    for i in range(reps):
        p = generate_plan(None)
        if not p:
            continue
        d = _DELTA(ref, p)
        noise_runs.append(d)
        print(f"  시행{i + 1}: 평균Δ {d['mean_delta']:.3f} | 변경 {d['changed']}/{d['n_common']}")

    # ── 2. 위약 — 필드별 분포를 만든다 ────────────────────────────
    print(f"\n[2] 위약 {len(PLACEBOS)}개 — 필드별 잡음 분포 구축")
    placebo_runs = []
    for i, pl in enumerate(PLACEBOS, 1):
        p = generate_plan(pl)
        if not p:
            print(f"  {i:2d}. 실패")
            continue
        d = _DELTA(ref, p)
        placebo_runs.append(d)
        print(f"  {i:2d}. 평균Δ {d['mean_delta']:.3f} | {pl[:24]}")

    if len(placebo_runs) < 3:
        print("  위약 표본 부족 — 중단")
        return

    # 필드별 위약 Δ 분포 → 필드별 임계값
    field_thr: dict[str, float] = {}
    field_samples: dict[str, list[float]] = {}
    for d in placebo_runs:
        for addr, delta in d["per_field"].items():
            field_samples.setdefault(addr, []).append(delta)
    for addr, vals in field_samples.items():
        field_thr[addr] = p95(vals)

    stable = [a for a, v in field_samples.items() if max(v) == 0.0]
    print(f"\n  필드 {len(field_thr)}개의 임계값 산출")
    print(f"  위약에 전혀 안 흔들린 필드: {len(stable)}/{len(field_thr)} "
          f"({len(stable) / len(field_thr):.0%})")
    by_sec: dict[str, list[float]] = {}
    for addr, t in field_thr.items():
        by_sec.setdefault(addr.split(".")[0], []).append(t)
    for sec, vals in sorted(by_sec.items()):
        zero = sum(1 for v in vals if v == 0.0)
        print(f"    {sec:10} 필드 {len(vals):3d} | 임계 평균 {stats.mean(vals):.3f} "
              f"| 임계 0인 필드 {zero:3d}")

    # ── 3. 실제 전제 — 필드별 작동 판정 ──────────────────────────
    print("\n[3] 실제 전제 — 필드별 판정")
    results = []
    for prem in REAL_PREMISES:
        print(f"\n  ── {prem['slot']}  (영향 예상: {prem['expect_prefix']}.*)")
        pa = generate_plan(prem["assert"])
        if not pa:
            print("     단언 실패")
            continue
        d_assert = _DELTA(ref, pa)

        sub_deltas = []
        for j, sub in enumerate(prem["substitutes"], 1):
            ps = generate_plan(sub)
            if not ps:
                continue
            sub_deltas.append(_DELTA(ref, ps))
        if not sub_deltas:
            print("     대체 전부 실패")
            continue

        # 필드별 판정: 모든 대체에서 임계 초과 && 단언은 임계 이하
        operative_fields = []
        for addr, thr in field_thr.items():
            if addr not in d_assert["per_field"]:
                continue
            a_d = d_assert["per_field"][addr]
            s_ds = [sd["per_field"].get(addr) for sd in sub_deltas]
            s_ds = [x for x in s_ds if x is not None]
            if not s_ds:
                continue
            if min(s_ds) > thr and a_d <= thr:
                operative_fields.append((addr, min(s_ds), thr))

        sec_hits = {}
        for addr, _, _ in operative_fields:
            sec_hits[addr.split(".")[0]] = sec_hits.get(addr.split(".")[0], 0) + 1

        verdict = "작동(operative)" if operative_fields else "비작동(inert)"
        print(f"     작동 필드 {len(operative_fields)}개 → {verdict}")
        if sec_hits:
            print(f"     섹션 분포: {sec_hits}")
            expected = sec_hits.get(prem["expect_prefix"], 0)
            total = sum(sec_hits.values())
            print(f"     예상 섹션 적중 {expected}/{total} ({expected / total:.0%})")
        for addr, sd, thr in operative_fields[:5]:
            print(f"       {addr:28} 대체Δ {sd:.2f} > 임계 {thr:.2f}")

        results.append({
            "slot": prem["slot"],
            "expect_prefix": prem["expect_prefix"],
            "n_operative": len(operative_fields),
            "sections": sec_hits,
            "precision": (sec_hits.get(prem["expect_prefix"], 0) / sum(sec_hits.values()))
                         if sec_hits else 0.0,
            "fields": [a for a, _, _ in operative_fields],
            "verdict": verdict,
        })

    # ── 4. 위약 오탐 — 위약을 대상 전제로 취급 ────────────────────
    print(f"\n[4] 위약 오탐 — 위약 {len(placebo_runs)}개를 대상 전제로 취급")
    fp_counts = []
    for k, d in enumerate(placebo_runs):
        # 자기 자신을 분포에서 제외한 leave-one-out 임계값
        hits = 0
        for addr, delta in d["per_field"].items():
            others = [v for j, dd in enumerate(placebo_runs) if j != k
                      for v in [dd["per_field"].get(addr)] if v is not None]
            if not others:
                continue
            if delta > p95(others):
                hits += 1
        fp_counts.append(hits)
        n = len(d["per_field"])
        print(f"  {k + 1:2d}. 임계 초과 필드 {hits:3d}/{n:3d} ({hits / max(n, 1):.1%})")

    mean_fp = stats.mean(c / max(len(placebo_runs[i]['per_field']), 1)
                         for i, c in enumerate(fp_counts))

    print("\n" + "=" * 78)
    print("판정")
    print("=" * 78)
    print(f"  위약 오탐률(필드 단위, leave-one-out) {mean_fp:.1%}")
    print(f"  기준: ≤ 5%  →  {'충족' if mean_fp <= 0.05 else '미달'}")
    print()
    if results:
        print("  전제별 작동 필드 수와 섹션 정확도:")
        for r in results:
            print(f"    {r['slot']:18} 작동필드 {r['n_operative']:3d} | "
                  f"예상섹션 적중 {r['precision']:.0%} | {r['verdict']}")
        n_op = sum(1 for r in results if r["n_operative"] > 0)
        mean_prec = stats.mean(r["precision"] for r in results if r["n_operative"] > 0) \
            if n_op else 0.0
        print()
        print(f"  작동 판정 {n_op}/{len(results)} | 평균 섹션 정확도 {mean_prec:.0%}")
        print()
        if n_op == 0:
            print("  ❌ 어떤 전제도 작동으로 판정되지 않았다.")
            print("     재생성 Δ로는 작동 전제를 가려낼 수 없다 → 제안서 5.6 대안 필요.")
        elif mean_fp > 0.05:
            print("  ⚠ 작동 판정은 나왔으나 위약 오탐률이 기준을 넘는다.")
            print("     임계값이 느슨해 아무 삽입이나 통과한다는 뜻이므로,")
            print("     이 상태의 작동 판정은 신뢰할 수 없다.")
        elif mean_prec < 0.5:
            print("  ⚠ 작동 판정은 나왔으나 엉뚱한 섹션을 가리킨다.")
            print("     귀속으로 쓸 수 없다. 개입 문구가 계획 전반을 교란하는지 확인 필요.")
        else:
            print("  ✓ 위약 오탐이 기준 이내이고 작동 필드가 예상 섹션에 집중됐다.")
            print("    필드별 인과 판정이 성립한다. A2~A3 설계를 유지한다.")

    out = ROOT / ".runtime" / "av3_v2_result.json"
    out.write_text(json.dumps({
        "mode": DELTA_MODE,
        "noise_mean": stats.mean(d["mean_delta"] for d in noise_runs) if noise_runs else None,
        "placebo_mean": stats.mean(d["mean_delta"] for d in placebo_runs),
        "n_fields_thresholded": len(field_thr),
        "n_fields_stable": len(stable),
        "fp_rate_field": mean_fp,
        "premises": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  저장: {out.name}")


if __name__ == "__main__":
    main()
