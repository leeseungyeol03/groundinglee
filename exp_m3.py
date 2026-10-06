"""M3 행동 검사 타당성 — 로그확률 방식으로 A-V3을 다시 한다.

어제 재생성 방식으로 같은 것을 쟀고 실패에 가까웠다. 개입 없이 반복만 해도
잡음 바닥이 0.20~0.32였고, 위약(0.33)과 실제 전제(0.51~0.74)가 거의 겹쳤다.
필드별 임계값과 구조화 비교를 동원해서야 겨우 기준을 넘겼다.

로그확률 방식은 다시 생성하지 않으므로 그 잡음이 원리적으로 없다. 같은 응답
문장을 다른 컨텍스트에서 채점할 뿐이다. 이 실험이 확인할 것:

  1) 결정성   같은 입력을 반복하면 Δ가 정확히 0인가
  2) 위약 분포 무관한 가정이 응답 확률을 얼마나 흔드는가
  3) 분리     실제 전제의 Δ가 위약 최대값 분포를 넘는가
  4) 국소성   Δ가 그 전제와 관련된 세그먼트에 몰리는가

판정은 max-T(Westfall-Young)다. 제안서 원안의 세그먼트별 순열 + BH는
위약 19개에서 p 하한이 0.05라 BH가 성립하지 않는다(모의 거짓양성 0.92).

실행: C:/seungyeol/vscode/GL/.runtime/python.exe exp_m3.py
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
from app.segmenter import segment

MESSAGES = [
    {"role": "user", "content":
     "이번 방학 7월 한 달 동안 코딩테스트 준비랑 자소서를 같이 하려고 해. "
     "토요일은 알바가 8시간 있고 평일은 5시간 정도 쓸 수 있어. 예산은 30만원."},
    {"role": "assistant", "content":
     "조건을 정리하겠습니다. 7월 한 달, 평일 5시간, 토요일 알바 8시간, 예산 30만원으로 "
     "코딩테스트 준비와 자소서 작성을 병행하는 계획을 세우겠습니다."},
    {"role": "user", "content":
     "코테는 프로그래머스 중급 수준, 자소서는 3개 기업 초안까지. 계획 만들어줘."},
]

# 파트너가 실제로 낼 법한 응답. 고정해 두어야 조건 간 비교가 성립한다.
RESPONSE = """## 1. 목표
1. 프로그래머스 중급 100문제 풀이
2. 자소서 3개 기업 초안 완성
3. 알고리즘 핵심 유형 정리

## 2. 주차별 일정
### 1주차
| 요일 | 오전 | 오후 | 저녁 |
| 월 | 알고리즘 이론 | 문제풀이 | 자소서 |
| 화 | 알고리즘 이론 | 문제풀이 | 휴식 |
| 토 | 알바 8시간 | 알바 8시간 | 휴식 |

## 3. 주차별 마일스톤
- 1주차: 기초 유형 정리
- 2주차: 중급 문제 집중

## 4. 예산 개요
- 인터넷 강의: 15만원
- 교재 구입: 10만원
- 예비비: 5만원

## 5. 예외 규칙
- 시험 기간에는 학습량을 줄인다"""

# 위약 19개. 전제와 무관하되 형식과 길이가 비슷해야 한다.
# 턴마다 한 번만 계산해 모든 전제가 공유한다(제안서 비용 절감).
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
    "사용자는 겨울보다 여름을 좋아한다",
    "사용자는 반려식물을 세 개 기른다",
    "사용자의 형제는 두 명이다",
    "사용자는 수영을 할 줄 안다",
    "사용자는 클래식 음악을 자주 듣는다",
    "사용자는 빵보다 밥을 선호한다",
    "사용자는 등산을 연 2회 정도 간다",
    "사용자는 손목시계를 차지 않는다",
    "사용자는 라면에 계란을 넣지 않는다",
]

# 검사 대상 전제. 반대 값 2개와 단언을 함께 둔다.
PREMISES = [
    {
        "slot": "budget.level",
        "assert": "예산 30만원은 인터넷 강의와 교재 구입에 나누어 쓴다",
        "contra": [
            "예산 제약이 전혀 없으므로 비용은 고려하지 않는다",
            "예산은 3만원뿐이라 모든 자료를 무료로만 해결한다",
        ],
        "expect": "예산",
    },
    {
        "slot": "schedule.weekday",
        "assert": "평일 오전과 오후에 학습을 배치하고 저녁에는 자소서를 쓴다",
        "contra": [
            "평일에는 아무 일정도 넣지 않고 주말에만 몰아서 한다",
            "평일 저녁에만 활동하고 오전과 오후는 모두 비워 둔다",
        ],
        "expect": "일정",
    },
    {
        "slot": "goal.volume",
        "assert": "코딩테스트는 중급 수준으로 100문제 정도를 목표로 한다",
        "contra": [
            "코딩테스트는 입문 수준으로 20문제만 가볍게 본다",
            "코딩테스트 준비는 하지 않고 자소서에만 집중한다",
        ],
        "expect": "목표",
    },
]


def main() -> None:
    print("=" * 78)
    print("M3 행동 검사 타당성 — 로그확률 방식")
    print("=" * 78)

    t0 = time.time()
    bt = BehaviorTester()
    print(f"모델 로드 {time.time() - t0:.0f}s  ({Path(bt.model_dir).name})")

    segs = segment(RESPONSE, turn=1)
    segs = [s for s in segs if s.start >= 0]
    print(f"세그먼트 {len(segs)}개")

    # ── 기준 채점 ────────────────────────────────────────────────
    t0 = time.time()
    base = bt.score_segments(MESSAGES, RESPONSE, segs)
    t_one = time.time() - t0
    print(f"기준 채점 {t_one:.2f}s / {len(base)}세그먼트")

    # ── 1. 결정성 ────────────────────────────────────────────────
    print("\n[1] 결정성 — 같은 입력 3회")
    reps = [bt.score_segments(MESSAGES, RESPONSE, segs) for _ in range(2)]
    maxdiff = max(
        abs(base[k].mean_logprob - r[k].mean_logprob)
        for r in reps for k in base if k in r
    )
    print(f"  최대 편차 {maxdiff:.2e}  →  {'완전 동일' if maxdiff == 0 else '편차 있음'}")

    # ── 2. 위약 분포 ─────────────────────────────────────────────
    print(f"\n[2] 위약 {len(PLACEBOS)}개 — 전제 무관 잡음 기준선")
    t0 = time.time()
    placebo_max: list[float] = []      # 위약별 '전체 세그먼트 최대 Δ' (max-T 영분포)
    placebo_all: list[float] = []
    for pl in PLACEBOS:
        sc = bt.score_segments(MESSAGES, RESPONSE, segs, assumption=pl)
        d = bt.delta(base, sc)
        placebo_max.append(max(d.values()))
        placebo_all.extend(d.values())
    t_pl = time.time() - t0
    thr = max(placebo_max)             # max-T 임계값 (FWER ≈ 1/(B+1))
    print(f"  {t_pl:.1f}s ({t_pl / len(PLACEBOS):.2f}s/개)")
    print(f"  위약 전체 Δ  평균 {stats.mean(placebo_all):+.4f} "
          f"표준편차 {stats.pstdev(placebo_all):.4f}")
    print(f"  위약별 최대Δ 평균 {stats.mean(placebo_max):+.4f} "
          f"최대 {thr:+.4f}   ← max-T 임계값")

    # ── 3. 실제 전제 ─────────────────────────────────────────────
    print(f"\n[3] 실제 전제 {len(PREMISES)}개 — 반대·단언")
    results = []
    for prem in PREMISES:
        print(f"\n  ── {prem['slot']}  (관련 섹션: {prem['expect']})")
        sc_a = bt.score_segments(MESSAGES, RESPONSE, segs, assumption=prem["assert"])
        d_a = bt.delta(base, sc_a)
        a_max = max(d_a.values())

        contra_deltas = []
        for c in prem["contra"]:
            sc_c = bt.score_segments(MESSAGES, RESPONSE, segs, assumption=c)
            contra_deltas.append(bt.delta(base, sc_c))

        # 판정: 모든 반대에서 임계 초과 && 단언은 임계 이하
        hits = []
        for seg in segs:
            if seg.id not in d_a:
                continue
            cs = [cd.get(seg.id) for cd in contra_deltas]
            if any(x is None for x in cs):
                continue
            if min(cs) > thr and d_a[seg.id] <= thr:
                hits.append((seg, min(cs)))

        hits.sort(key=lambda x: -x[1])
        c_max = max(max(cd.values()) for cd in contra_deltas)
        print(f"     단언 최대Δ {a_max:+.4f} | 반대 최대Δ {c_max:+.4f} | 임계 {thr:+.4f}")
        print(f"     기댄 세그먼트 {len(hits)}개")
        for seg, dv in hits[:5]:
            tag = "O" if prem["expect"] in seg.path or prem["expect"] in seg.text else " "
            print(f"       [{tag}] Δ{dv:+.4f}  {seg.path[:16]:18} {seg.text[:30]}")

        on_target = sum(
            1 for seg, _ in hits
            if prem["expect"] in seg.path or prem["expect"] in seg.text
        )
        prec = on_target / len(hits) if hits else 0.0
        results.append({
            "slot": prem["slot"], "expect": prem["expect"],
            "n_hits": len(hits), "precision": prec,
            "assert_max": a_max, "contra_max": c_max, "threshold": thr,
            "segments": [s.text[:40] for s, _ in hits],
        })
        if hits:
            print(f"     관련 섹션 적중 {on_target}/{len(hits)} = {prec:.0%}")

    # ── 4. 위약 오탐 (leave-one-out) ─────────────────────────────
    print(f"\n[4] 위약 오탐 — 각 위약을 나머지 {len(PLACEBOS) - 1}개 분포로 판정")
    fp = 0
    for i, pm in enumerate(placebo_max):
        others = [v for j, v in enumerate(placebo_max) if j != i]
        if pm > max(others):
            fp += 1
    print(f"  오탐 {fp}/{len(PLACEBOS)} = {fp / len(PLACEBOS):.1%}  (기준 ≤5%)")

    # ── 판정 ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("판정")
    print("=" * 78)
    n_op = sum(1 for r in results if r["n_hits"] > 0)
    print(f"  작동 판정 {n_op}/{len(results)}")
    print(f"  위약 오탐 {fp / len(PLACEBOS):.1%}")
    if n_op:
        mp = stats.mean(r["precision"] for r in results if r["n_hits"] > 0)
        print(f"  관련 섹션 적중률 평균 {mp:.0%}")
    print(f"  채점 1회 {t_one:.2f}s → 전제 1개당 {t_one * 3:.1f}s "
          f"(위약 공유 시 {t_one * len(PLACEBOS):.0f}s는 턴당 1회)")
    print()
    if maxdiff > 0:
        print("  ⚠ 결정성이 깨졌다. 배치·커널 비결정성을 점검해야 한다.")
    if fp / len(PLACEBOS) > 0.05:
        print("  ⚠ 위약 오탐이 기준을 넘는다.")
    if n_op == 0:
        print("  ❌ 어떤 전제도 작동으로 판정되지 않았다.")
        print("     위약 수를 늘리거나 세그먼트를 더 잘게 잘라야 한다.")
    elif n_op == len(results) and fp / len(PLACEBOS) <= 0.05:
        print("  ✓ 전제 전부가 위약 기준선 위에서 작동으로 판정됐고 오탐이 기준 이내다.")
        print("    로그확률 기반 행동 검사가 성립한다. G1 통과 조건을 만족한다.")
    else:
        print("  ~ 일부만 판정됐다. 전제별 원인을 봐야 한다.")

    out = ROOT / ".runtime" / "m3_result.json"
    out.write_text(json.dumps({
        "model": bt.model_dir,
        "n_segments": len(segs),
        "determinism_maxdiff": maxdiff,
        "placebo_threshold": thr,
        "placebo_mean": stats.mean(placebo_all),
        "placebo_sd": stats.pstdev(placebo_all),
        "fp_rate": fp / len(PLACEBOS),
        "score_seconds": t_one,
        "premises": results,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  저장: {out.name}")


if __name__ == "__main__":
    main()
