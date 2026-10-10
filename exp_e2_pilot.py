"""E2 파일럿 — B0(대화만) vs B1(메모리식 기준선). 게이트 G2 판정.

왜 이 파일럿인가 (feedback2.md 7장 G2):
    "B1에서 잔존, 파생 생존, 표면 수용이 측정 가능한 수준으로 발생
     (개선 여지 확인)"

기준선에서 문제가 거의 안 생기면 GroundLens가 고칠 것이 없다는 뜻이고,
그러면 연구 문제의 범위를 다시 봐야 한다. 그래서 B4(제안 방식)를 만들기 전에
B0·B1부터 돌려 문제가 실제로 발생하는지 확인한다.

흐름
    턴1      페르소나의 모호한 첫 발화 → 파트너가 계획 초안
    턴2~T-1  시뮬레이터가 프로필을 조금씩 공개 → 파트너가 개정
    턴T      조건 변경 제시 (자원 축소 / 제약 추가 / 우선순위 반전)
    턴T+     최대 2턴 동안 개정

측정 (최종 계획 기준)
    낡은 세그먼트 잔존율   바뀐 조건과 모순되는 세그먼트 비율 (주 지표)
    표면 수용률            응답은 받아들였다고 하지만 세부는 그대로인가
    숨은 프로필 충족도     최종 계획이 체크리스트를 얼마나 만족하는가
    부수 변경률            변경과 무관한데 바뀐 세그먼트 비율
    사용자 부담            시뮬레이터 조작 수, 턴 수

판정자는 파트너(로컬 Qwen)·시뮬레이터(Claude Haiku)와 또 다른 호출(Claude
Sonnet)이다. 같은 모델이 만들고 같은 모델이 채점하면 편향이 들어간다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe exp_e2_pilot.py [에피소드수]
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

from app.behavior_test import BehaviorTester
from app.usage_meter import METER
from app.episode import EpisodeRunner
from app.propagate import find_collateral
from app import results
from app.conditions import (
    CONDITIONS,
    build_system,
    extract_memory_items,
    review_memory_items,
)
from app.segmenter import align, segment
from app.simulator import PERSONAS, UserSimulator
from exp_m3b import make_placebos

JUDGE_MODEL = "claude-sonnet-4-5-20250929"

_STALE_PROMPT = """사용자가 조건을 바꿨습니다. 아래 계획 항목들 중 바뀐 조건과
모순되는 것을 찾으세요.

[바뀐 조건]
{change}

[이 조건이 참이 되게 하는 것]
{implies}

[이 조건 때문에 더 이상 유효하지 않은 것]
{invalidates}

[계획 항목]
{segments}

각 항목에 대해 바뀐 조건과 모순되면 1, 아니면 0을 매기고 JSON 배열로만
답하세요. 항목 수와 배열 길이가 같아야 합니다."""

_SURFACE_PROMPT = """사용자가 조건을 바꿨고 AI가 계획을 다시 냈습니다.

[바뀐 조건]
{change}

[AI의 응답]
{plan}

AI의 응답이 바뀐 조건을 **말로는** 받아들였다고 표현하면서 **실제 계획 내용은**
바뀐 조건을 반영하지 않았습니까?

- 말로도 반영하고 내용도 반영 -> 0
- 말로는 반영했으나 내용은 그대로 -> 1
- 말로도 반영 안 함 -> 2

숫자 하나만 답하세요."""


def judge_stale(client, segments, persona) -> list[int]:
    """각 세그먼트가 바뀐 조건과 모순되는지 판정한다."""
    if not segments:
        return []
    numbered = "\n".join(f"{i+1}. {s.text}" for i, s in enumerate(segments))
    r = client.messages.create(
        model=JUDGE_MODEL, max_tokens=900,
        messages=[{"role": "user", "content": _STALE_PROMPT.format(
            change=persona.change_utterance,
            implies="\n".join(f"- {x}" for x in persona.change_implies) or "- (없음)",
            invalidates="\n".join(f"- {x}" for x in persona.change_invalidates) or "- (없음)",
            segments=numbered)}],
        extra_body={"temperature": 0.0},
    )
    METER.record("judge.stale", JUDGE_MODEL, r)
    m = re.search(r"\[[^\]]*\]", r.content[0].text or "", re.S)
    vals: list[int] = []
    if m:
        try:
            vals = [int(bool(v)) for v in json.loads(m.group())]
        except Exception:
            vals = []
    return (vals + [0] * len(segments))[:len(segments)]


def judge_surface(client, plan: str, persona) -> int:
    r = client.messages.create(
        model=JUDGE_MODEL, max_tokens=20,
        messages=[{"role": "user", "content": _SURFACE_PROMPT.format(
            change=persona.change_utterance, plan=plan[:3500])}],
        extra_body={"temperature": 0.0},
    )
    METER.record("judge.surface", JUDGE_MODEL, r)
    m = re.search(r"[012]", r.content[0].text or "")
    return int(m.group()) if m else 2


def run_episode(bt, persona, condition: str, seed: int = 0,
                attention: int | None = 5, placebos=None) -> dict:
    """한 에피소드를 돌리고 지표를 낸다."""
    runner = EpisodeRunner(bt, persona, condition, seed=seed,
                           attention=attention, placebos=placebos)
    t = runner.run()
    client = runner.client
    sim = runner.sim

    final_plan = t.plans[-1]
    final_segs = [s for s in segment(final_plan, turn=9) if s.start >= 0]
    pre_segs = [s for s in segment(t.pre_change_plan, turn=0) if s.start >= 0]

    stale = judge_stale(client, final_segs, persona)
    stale_rate = (sum(stale) / len(stale)) if stale else 0.0
    surface = judge_surface(client, t.first_after_change, persona)
    checklist = sim.checklist_report(final_plan)

    stale_ids = {s.id for s, f in zip(final_segs, stale) if f}
    coll_ids, _ = find_collateral(pre_segs, final_segs, stale_ids)
    r = align(pre_segs, final_segs)
    collateral_rate = (len(coll_ids) / len(r["pairs"])) if r["pairs"] else 0.0

    # 파생 전제 생존율 (B2 이상에서만 상태가 있다)
    derived_alive = None
    if runner.store is not None:
        flagged = [p for p in runner.store.premises.values() if p.flagged_review]
        if flagged:
            alive = sum(1 for p in flagged if p.is_active)
            derived_alive = alive / len(flagged)

    return {
        "persona": persona.id, "condition": condition, "seed": seed,
        "change_type": persona.change_type,
        "n_turns": len(t.plans),
        "n_final_segments": len(final_segs),
        "stale_rate": stale_rate, "n_stale": sum(stale),
        "stale_texts": [s.text[:60] for s, f in zip(final_segs, stale) if f][:5],
        "surface_code": surface,
        "surface_acceptance": int(surface == 1),
        "checklist_rate": checklist["rate"],
        "checklist_satisfied": checklist["satisfied"],
        "checklist_total": checklist["total"],
        "collateral_rate": collateral_rate,
        "user_ops": t.panel_ops,
        "panel_shown_mean": (sum(t.panel_shown) / len(t.panel_shown)) if t.panel_shown else 0,
        "n_corrections": len(t.corrections),
        "n_propagations": len(t.propagations),
        "derived_alive_rate": derived_alive,
        "status_counts": t.status_counts,
        "verdict_counts": t.verdict_counts,
        "propagations": t.propagations,
        "n_disclosed": len(sim.state.disclosed),
        "final_plan": final_plan[:1500],
    }


def main() -> None:
    conds = sys.argv[1].split(",") if len(sys.argv) > 1 else list(CONDITIONS)
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    n_pl = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    attention = 5
    placebos = make_placebos(n_pl)

    print("=" * 78)
    print("E2 파일럿 — 조건 비교")
    print("=" * 78)
    print(f"조건 {conds} | 페르소나 {len(PERSONAS)} | 반복 {reps} | 위약 {n_pl}")
    print("파트너 로컬 Qwen3-8B / 시뮬레이터 Claude Haiku / 판정자 Claude Sonnet")

    bt = BehaviorTester()
    rows = []
    t_all = time.time()
    total = len(PERSONAS) * len(conds) * reps
    k = 0
    for rep in range(reps):
        for persona in PERSONAS:
            for cond in conds:
                k += 1
                print("\n" + "-" * 78)
                print(f"[{k}/{total}] {persona.id} / {cond} / {persona.change_type}")
                t0 = time.time()
                try:
                    r = run_episode(bt, persona, cond, seed=rep * 10 + k,
                                    attention=attention, placebos=placebos)
                except Exception as e:
                    print(f"  실패: {type(e).__name__}: {e}")
                    continue
                r["seconds"] = time.time() - t0
                rows.append(r)
                # 증분 저장 — 중단돼도 앞선 에피소드를 잃지 않는다.
                # 19/20을 끝내고 타임아웃으로 전부 날린 적이 있다.
                results.save("e2_partial", {"episodes": rows}, note="in-progress")
                print(f"  {r['seconds']:.0f}s | 세그먼트 {r['n_final_segments']}"
                      f" | 낡은 {r['n_stale']}/{r['n_final_segments']}"
                      f" = {r['stale_rate']:.0%}")
                print(f"  표면수용 {'O' if r['surface_acceptance'] else 'X'}"
                      f" | 충족 {r['checklist_satisfied']}/{r['checklist_total']}"
                      f" | 부수 {r['collateral_rate']:.0%}"
                      f" | 조작 {r['user_ops']} | 수정 {r['n_corrections']}")
                if r["verdict_counts"]:
                    agg = {}
                    for d in r["verdict_counts"]:
                        for kk, vv in d.items():
                            agg[kk] = agg.get(kk, 0) + vv
                    print(f"  행동 검사 누적 {agg}")
                if r["propagations"]:
                    refl = [p for p in r["propagations"] if "reflected" in p]
                    if refl:
                        print(f"  반영 재검사 {['O' if p['reflected'] else 'X' for p in refl]}"
                              f" 재시도 {sum(1 for p in refl if p.get('retried'))}")

    if not rows:
        print("\n유효 에피소드 없음")
        return

    print("\n" + "=" * 78)
    print("집계")
    print("=" * 78)
    print(f"  에피소드 {len(rows)} | 총 {time.time()-t_all:.0f}s")
    print()
    print(f"  {'조건':5} {'n':>3} {'낡은잔존':>9} {'표면수용':>9} {'충족':>7} "
          f"{'부수변경':>9} {'조작':>5} {'초':>5}")
    agg = {}
    for cond in conds:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        a = {
            "n": len(sub),
            "stale": stats.mean(r["stale_rate"] for r in sub),
            "surface": stats.mean(r["surface_acceptance"] for r in sub),
            "checklist": stats.mean(r["checklist_rate"] for r in sub),
            "collateral": stats.mean(r["collateral_rate"] for r in sub),
            "ops": stats.mean(r["user_ops"] for r in sub),
            "seconds": stats.mean(r["seconds"] for r in sub),
        }
        agg[cond] = a
        print(f"  {cond:5} {a['n']:>3} {a['stale']:>8.0%} {a['surface']:>9.0%} "
              f"{a['checklist']:>7.0%} {a['collateral']:>9.0%} {a['ops']:>5.1f} "
              f"{a['seconds']:>5.0f}")

    # 주 대비: B1 vs B4
    if "B1" in agg and "B4" in agg:
        d = agg["B1"]["stale"] - agg["B4"]["stale"]
        print()
        print("  주 대비 (HB1: 낡은 세그먼트 잔존율 B4 < B1)")
        print(f"    B1 {agg['B1']['stale']:.0%} -> B4 {agg['B4']['stale']:.0%}"
              f"  차이 {d:+.0%}")
        print(f"    사용자 부담 (HB5: B4 <= B1)  "
              f"{agg['B1']['ops']:.1f} -> {agg['B4']['ops']:.1f}")
        print()
        if d > 0:
            print("  O G3 경향 확인. B4가 B1보다 낡은 내용을 덜 남긴다.")
            print("    본 실행으로 넘어갈 근거가 된다.")
        else:
            print("  X G3 미달. B4가 B1을 넘지 못한다.")
            print("    B2·B3로 어느 단계에서 이득이 사라지는지 진단해야 한다.")
            if "B2" in agg and "B3" in agg:
                print(f"    단계별: B1 {agg['B1']['stale']:.0%}"
                      f" -> B2 {agg['B2']['stale']:.0%}"
                      f" -> B3 {agg['B3']['stale']:.0%}"
                      f" -> B4 {agg['B4']['stale']:.0%}")

    print()
    print("  API 사용량")
    print(METER.report(len(rows)))
    print()
    print(METER.project(len(rows), 3800))

    path = results.save("e2_pilot", {"attention": attention, "conditions": conds,
                                     "aggregate": agg, "episodes": rows},
                        n_placebos=n_pl, reps=reps)
    METER.save(results.RUNTIME / "e2_usage_latest.json", len(rows))
    print(f"\n  저장: {path.name}")


if __name__ == "__main__":
    main()
