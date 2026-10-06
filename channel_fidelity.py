"""채널 충실도 실험 — 사용자 연구 없이 §3.2의 이론적 주장을 검증한다.

검증 대상 주장(§3.2):
  "두 채널은 동등하지만 동일하지는 않다. 대화 채널에서는 사용자의 evidence가
   자연어 발화로 표현된 뒤 시스템의 추출과 LLM의 재해석을 거쳐야 하며,
   그 과정에서 어느 전제에 대한 evidence인지가 불분명해질 수 있다.
   반면 인터페이스 채널에서는 사용자가 대상 전제를 직접 지시하므로
   evidence의 귀속 대상이 모호하지 않으며, 해석 손실 없이 시스템 상태로 기록된다."

이 주장은 사람의 행동에 관한 것이 아니라 **채널의 정보 전달 특성**에 관한 것이므로
참가자 없이 측정할 수 있다. 동일한 교정 내용을 두 채널로 투입하고 결과 상태를 비교한다.

설계:
  1) 공통 전사(prefix)로 계획이 산출된 상태 S를 만든다
  2) S를 복제해 두 팔로 나눈다
     - 인터페이스 팔: correction_node로 대상 전제를 직접 지시해 교정
     - 대화 팔      : 동일 교정 문구를 사용자 발화로 전달
  3) 양 팔에서 다음 턴을 실행하고 아래를 측정한다

측정 지표:
  attribution_precision  교정이 의도한 전제에만 반영됐는가 (귀속 정확도)
  collateral             의도하지 않은 전제·필드가 함께 바뀌었는가 (부수 피해)
  propagation            대상 전제에 귀속된 필드가 실제로 재구성됐는가 (§8 반증 조건)
  unverified_load_delta  미검증 부하 변화

§8이 정한 반증 조건: 교정이 상태를 바꿨다고 보고되지만 계획 필드가 그대로면
조작이 evidence로 기능하지 못하고 표시상의 행위에 그친 것이다.
"""
from __future__ import annotations

import copy
import json
import os
import sys
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

from app.graph import initial_state, run_correction_graph, run_message_graph

CFG = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

SEED_TURNS = [
    "이번 방학에 코딩테스트 준비랑 자소서를 같이 하고 싶어. 7월 한 달이고 주말엔 알바가 있어.",
    "알바는 토요일만 8시간. 중급이고 프로그래머스로 할래. 자소서는 3개 기업 초안. 평일 5시간 가능.",
    "좋아, 이 조건으로 5섹션 계획 초안을 만들어줘.",
]


def step(state: dict, turn: int, msg: str) -> dict:
    state["current_turn"] = turn
    state["C_t"] = msg
    state.setdefault("conversation_history", []).append(
        {"role": "user", "content": msg, "turn": turn}
    )
    return run_message_graph(state, CFG)


def field_map(state: dict) -> dict[str, str]:
    return {f["address"]: f.get("content", "") for f in state.get("plan_fields", [])}


def unverified_load(state: dict) -> tuple[int, int]:
    """(미검증 전제에 뿌리를 둔 필드 수, 전체 필드 수)"""
    cg_ids = {i["id"] for i in state.get("common_ground", []) if i.get("id")}
    pf = state.get("plan_fields", [])
    unv = sum(
        1
        for f in pf
        if f.get("premise_ids") and any(i not in cg_ids for i in f["premise_ids"])
    )
    return unv, len(pf)


def build_common_prefix() -> dict:
    """계획이 산출될 때까지 공통 전사를 진행한다."""
    st = initial_state("chanfid", "코딩테스트 준비와 자소서 작성 병행 방학 계획")
    for i, msg in enumerate(SEED_TURNS, start=1):
        st = step(st, i, msg)
        print(
            f"  [전사 턴{i}] CG {len(st['common_ground']):2d} "
            f"LG {len(st['llm_ground']):2d} 필드 {len(st.get('plan_fields', [])):2d}"
        )
        if st.get("plan_fields"):
            return st
    return st


def pick_target(state: dict) -> dict | None:
    """귀속된 필드를 가진 LG 항목 중 필드 수가 가장 많은 것을 고른다.

    전파 범위가 0이면 채널 비교가 성립하지 않으므로 가장 영향이 큰 전제를 택한다.
    """
    pf = state.get("plan_fields", [])
    best, best_n = None, 0
    for item in state.get("llm_ground", []):
        n = sum(1 for f in pf if item.get("id") in f.get("premise_ids", []))
        if n > best_n:
            best, best_n = item, n
    return best


def arm_interface(base: dict, target: dict, correction: str, turn: int) -> dict:
    """인터페이스 채널: 대상 전제를 직접 지시해 교정한다."""
    st = copy.deepcopy(base)
    st["target_item_id"] = target["id"]
    st["correction_text"] = correction
    st = run_correction_graph(st)
    staged = [f["address"] for f in st["plan_fields"] if f.get("status") == "stale"]
    st = step(st, turn, "방금 수정한 내용 반영해서 계획 다시 정리해줘.")
    return {"state": st, "staged": staged}


def arm_conversation(base: dict, correction: str, turn: int) -> dict:
    """대화 채널: 동일한 교정 내용을 자연어 발화로만 전달한다."""
    st = copy.deepcopy(base)
    st = step(st, turn, correction + " 이거 반영해서 계획 다시 정리해줘.")
    return {"state": st, "staged": []}


def measure(label: str, base: dict, arm: dict, target: dict) -> dict:
    """§8 반증 조건에 따라 필드 단위로 전파를 확인한다."""
    before, after = field_map(base), field_map(arm["state"])
    tgt_addrs = {
        f["address"]
        for f in base.get("plan_fields", [])
        if target.get("id") in f.get("premise_ids", [])
    }
    other = set(before) - tgt_addrs

    changed_t = {a for a in tgt_addrs if a in after and after[a] != before[a]}
    gone_t = {a for a in tgt_addrs if a not in after}
    changed_o = {a for a in other if a in after and after[a] != before[a]}

    u0, n0 = unverified_load(base)
    u1, n1 = unverified_load(arm["state"])
    reached = len(changed_t | gone_t)
    prop = reached / len(tgt_addrs) if tgt_addrs else 0.0
    coll = len(changed_o) / len(other) if other else 0.0

    print(f"\n  [{label}]")
    print(f"    대상 전제 귀속 필드 {len(tgt_addrs)}개 → 재구성 {reached}개 (전파율 {prop:.0%})")
    print(f"    무관 필드 {len(other)}개 → 변경 {len(changed_o)}개 (부수변경 {coll:.0%})")
    print(f"    미검증 부하 {u0}/{n0} ({u0/max(n0,1):.0%}) → {u1}/{n1} ({u1/max(n1,1):.0%})")
    if arm["staged"]:
        print(f"    stale 선표시 {len(arm['staged'])}개 (인터페이스 채널만 가능)")
    return {
        "channel": label,
        "target_fields": len(tgt_addrs),
        "propagated": reached,
        "propagation_rate": round(prop, 3),
        "other_fields": len(other),
        "collateral_changed": len(changed_o),
        "collateral_rate": round(coll, 3),
        "unverified_before": [u0, n0],
        "unverified_after": [u1, n1],
        "staged_stale": len(arm["staged"]),
    }


def main() -> None:
    print("=" * 74)
    print("채널 충실도 실험 — 동일 교정을 대화 채널 vs 인터페이스 채널로 투입")
    print("검증: §3.2 '두 채널은 동등하지만 동일하지 않다' / §8 반증 조건")
    print("=" * 74)

    print("\n[1] 공통 전사")
    base = build_common_prefix()
    if not base.get("plan_fields"):
        print("  계획 미산출 — 중단")
        return

    target = pick_target(base)
    if target is None:
        print("  귀속된 LG 전제 없음 — 중단")
        return

    n_fields = sum(
        1 for f in base["plan_fields"] if target["id"] in f.get("premise_ids", [])
    )
    print(f"\n[2] 교정 대상 전제 (귀속 필드 {n_fields}개)")
    print(f"    {target.get('content', '')[:88]}")

    correction = "이론 학습은 빼고 문제 풀이량 위주로 진행한다"
    print(f"    교정 내용: {correction}")

    turn = int(base.get("current_turn", 3)) + 1
    print("\n[3] 두 팔 실행")
    iface = arm_interface(base, target, correction, turn)
    conv = arm_conversation(base, correction, turn)

    print("\n[4] 측정")
    r_i = measure("인터페이스 채널", base, iface, target)
    r_c = measure("대화 채널", base, conv, target)

    print("\n" + "=" * 74)
    print("판정")
    print("=" * 74)
    dp = r_i["propagation_rate"] - r_c["propagation_rate"]
    dc = r_i["collateral_rate"] - r_c["collateral_rate"]
    print(f"  전파율   인터페이스 {r_i['propagation_rate']:.0%} vs 대화 {r_c['propagation_rate']:.0%}  (차 {dp:+.0%})")
    print(f"  부수변경 인터페이스 {r_i['collateral_rate']:.0%} vs 대화 {r_c['collateral_rate']:.0%}  (차 {dc:+.0%})")
    print()
    if r_i["propagated"] == 0:
        print("  ❌ §8 반증 조건 해당: 인터페이스 조작이 필드를 바꾸지 못했다.")
        print("     조작이 evidence로 기능하지 못하고 표시상의 행위에 그친다는 징후다.")
    elif dp > 0 or dc < 0:
        print("  ✓ §3.2 주장과 정합: 인터페이스 채널이 귀속 대상을 더 정확히 지시한다.")
        print("     (전파율이 높거나 부수변경이 낮음 = 해석 손실이 적음)")
    else:
        print("  ~ 두 채널의 차이가 관찰되지 않았다. 표본 1건이므로 반복 필요.")
        print("     지속되면 '동등하지만 동일하지 않다'는 주장의 후반부가 약해진다.")

    out = ROOT / ".runtime" / "channel_fidelity_result.json"
    out.write_text(
        json.dumps(
            {"target": target.get("content", ""), "correction": correction,
             "interface": r_i, "conversation": r_c},
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\n  결과 저장: {out.name}")


if __name__ == "__main__":
    main()
