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
from app.conditions import (
    CONDITIONS,
    build_system,
    extract_memory_items,
    review_memory_items,
)
from app.segmenter import align, segment
from app.simulator import PERSONAS, UserSimulator

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
    m = re.search(r"[012]", r.content[0].text or "")
    return int(m.group()) if m else 2


def run_episode(bt, persona, condition: str, seed: int = 0,
                build_turns: int = 3, revise_turns: int = 2,
                attention: int | None = 5) -> dict:
    """한 에피소드를 돌리고 지표를 낸다."""
    sim = UserSimulator(persona, attention_budget=attention, seed=seed)
    client = sim._client
    history: list[dict] = []
    memory: list[str] = []
    ops_total = 0
    plans: list[str] = []

    def partner_turn(user_text: str) -> str:
        history.append({"role": "user", "content": user_text})
        system = build_system(condition, memory)
        resp = bt.generate(history, system=system, max_new_tokens=900)
        history.append({"role": "assistant", "content": resp})
        plans.append(resp)
        return resp

    # ── 1턴: 모호한 첫 발화 ─────────────────────────────────────
    plan = partner_turn(sim.first_turn())

    # ── 2~T-1턴: 프로필을 조금씩 공개 ──────────────────────────
    for _ in range(build_turns):
        if condition == "B1":
            items = extract_memory_items(client, history, plan)
            memory, ops = review_memory_items(client, items, persona, attention)
            ops_total += ops
        plan = partner_turn(sim.next_turn(plan))

    pre_change_plan = plan
    pre_segs = [s for s in segment(pre_change_plan, turn=0) if s.start >= 0]

    # ── T턴: 조건 변경 ──────────────────────────────────────────
    if condition == "B1":
        items = extract_memory_items(client, history, plan)
        memory, ops = review_memory_items(client, items, persona, attention)
        ops_total += ops
    plan = partner_turn(sim.change_turn())
    first_after_change = plan

    # ── T+턴: 개정 ──────────────────────────────────────────────
    for _ in range(revise_turns):
        if condition == "B1":
            items = extract_memory_items(client, history, plan)
            memory, ops = review_memory_items(client, items, persona, attention)
            ops_total += ops
        plan = partner_turn(sim.next_turn(plan))

    final_plan = plan
    final_segs = [s for s in segment(final_plan, turn=9) if s.start >= 0]

    # ── 지표 ────────────────────────────────────────────────────
    stale = judge_stale(client, final_segs, persona)
    stale_rate = (sum(stale) / len(stale)) if stale else 0.0
    surface = judge_surface(client, first_after_change, persona)
    checklist = sim.checklist_report(final_plan)

    # 부수 변경률: 변경 전후 정렬에서, 낡지 않았는데 바뀐 세그먼트
    r = align(pre_segs, final_segs)
    stale_ids = {s.id for s, f in zip(final_segs, stale) if f}
    changed_unrelated = [p for p in r["pairs"]
                         if p["changed"] and p["curr"] not in stale_ids]
    collateral_rate = (len(changed_unrelated) / len(r["pairs"])) if r["pairs"] else 0.0

    return {
        "persona": persona.id, "condition": condition, "seed": seed,
        "change_type": persona.change_type,
        "n_turns": len(plans),
        "n_final_segments": len(final_segs),
        "stale_rate": stale_rate, "n_stale": sum(stale),
        "stale_texts": [s.text[:60] for s, f in zip(final_segs, stale) if f][:5],
        "surface_code": surface,
        "surface_acceptance": int(surface == 1),
        "checklist_rate": checklist["rate"],
        "checklist_satisfied": checklist["satisfied"],
        "checklist_total": checklist["total"],
        "collateral_rate": collateral_rate,
        "user_ops": ops_total,
        "n_disclosed": len(sim.state.disclosed),
        "memory_size": len(memory),
        "final_plan": final_plan[:1500],
    }


def main() -> None:
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else len(PERSONAS) * 2
    attention = 5

    print("=" * 78)
    print("E2 파일럿 — B0(대화만) vs B1(메모리식 기준선)")
    print("=" * 78)
    print(f"에피소드 {n_ep} | 주의 예산 {attention} | 게이트 G2 판정")
    print("파트너 로컬 Qwen3-8B / 시뮬레이터 Claude Haiku / 판정자 Claude Sonnet")

    bt = BehaviorTester()
    rows = []
    t_all = time.time()
    for i in range(n_ep):
        persona = PERSONAS[(i // len(CONDITIONS)) % len(PERSONAS)]
        cond = CONDITIONS[i % len(CONDITIONS)]
        print("\n" + "-" * 78)
        print(f"[{i+1}/{n_ep}] {persona.id} / {cond} / 변경={persona.change_type}")
        t0 = time.time()
        try:
            r = run_episode(bt, persona, cond, seed=i, attention=attention)
        except Exception as e:
            print(f"  실패: {type(e).__name__}: {e}")
            continue
        r["seconds"] = time.time() - t0
        rows.append(r)
        print(f"  {r['seconds']:.0f}s | 최종 세그먼트 {r['n_final_segments']}")
        print(f"  낡은 세그먼트 잔존 {r['n_stale']}/{r['n_final_segments']} "
              f"= {r['stale_rate']:.0%}")
        print(f"  표면 수용 {'O' if r['surface_acceptance'] else 'X'} (코드 {r['surface_code']})"
              f" | 프로필 충족 {r['checklist_satisfied']}/{r['checklist_total']}"
              f" | 부수 변경 {r['collateral_rate']:.0%}"
              f" | 조작 {r['user_ops']}")
        for t in r["stale_texts"][:2]:
            print(f"      낡음: {t}")

    if not rows:
        print("\n유효 에피소드 없음")
        return

    print("\n" + "=" * 78)
    print("집계")
    print("=" * 78)
    print(f"  에피소드 {len(rows)} | 총 {time.time()-t_all:.0f}s")
    print()
    hdr = f"  {'조건':6} {'n':>3} {'낡은잔존':>9} {'표면수용':>9} {'프로필충족':>10} {'부수변경':>9} {'조작':>5}"
    print(hdr)
    agg = {}
    for cond in CONDITIONS:
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
        }
        agg[cond] = a
        print(f"  {cond:6} {a['n']:>3} {a['stale']:>8.0%} {a['surface']:>9.0%} "
              f"{a['checklist']:>10.0%} {a['collateral']:>9.0%} {a['ops']:>5.1f}")

    print()
    print("  게이트 G2 — 기준선에서 문제가 측정 가능한 수준으로 발생하는가")
    b1 = agg.get("B1")
    if b1:
        checks = [
            ("낡은 세그먼트 잔존율 > 0", b1["stale"] > 0.0, f"{b1['stale']:.0%}"),
            ("표면 수용 발생", b1["surface"] > 0.0, f"{b1['surface']:.0%}"),
            ("프로필 충족도 < 100%", b1["checklist"] < 1.0, f"{b1['checklist']:.0%}"),
        ]
        for name, ok, val in checks:
            print(f"    [{'O' if ok else 'X'}] {name:26} {val}")
        passed = sum(1 for _, ok, _ in checks if ok)
        print()
        if passed >= 2:
            print("  O G2 통과. 메모리식 기준선에서 문제가 실제로 발생한다.")
            print("    개선 여지가 있으므로 B2~B4 구현으로 넘어갈 수 있다.")
        else:
            print("  X G2 미달. 기준선에서 문제가 거의 안 생긴다.")
            print("    시나리오의 모호성·사적 의미·변경 강도를 높여야 한다.")

    out = ROOT / ".runtime" / "e2_pilot_result.json"
    out.write_text(json.dumps({"attention": attention, "aggregate": agg,
                               "episodes": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n  저장: {out.name}")


if __name__ == "__main__":
    main()
