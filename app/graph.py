from __future__ import annotations

import concurrent.futures
import copy
import logging
import uuid
from typing import Any, TypedDict

from app.config import assemble_partner_system_prompt
from app.database import now_iso
from app.llm import get_assistant_response, get_probe_response
from classification.module import (
    step1_1_preprocess_utterance,
    step1_2_grounding_process,
    step2_1_preprocess_thinking,
    step2_2_reflection_process,
)

logger = logging.getLogger("uvicorn.error")


class GroundLensState(TypedDict, total=False):
    session_id: str
    conversation_history: list[dict]
    user_intent: str
    current_turn: int
    current_stage: str       # 'S1' | 'S2' | 'S3' | 'done'
    turn_in_stage: int       # turns elapsed in current stage
    revision_count: int      # revision requests in S2
    stage_gate_ready: bool   # True when stage advance condition met
    F: list  # list[dict] with {premise, evidence_type[]}
    common_ground: list[dict]
    llm_ground: list[dict]
    grounding_corrections: list[dict]
    llm_ground_deletions: list[dict]
    C_t: str
    D_t: str
    B_t: str
    A_t: dict
    system_prompt_used: str
    errors: list[str]
    target_item_id: str
    correction_text: str


_REVISION_KEYWORDS = {
    "바꿔", "바꿔줘", "수정", "수정해줘", "대신", "빼줘", "빼", "넣어줘", "넣어",
    "다시", "변경", "고쳐", "고쳐줘", "아니", "아닌데", "틀렸", "잘못",
}


def _is_revision(C_t: str) -> bool:
    return any(kw in C_t for kw in _REVISION_KEYWORDS)


def _advance_stage_fields(state: GroundLensState) -> GroundLensState:
    stage = state.get("current_stage", "S1")
    turn_in = int(state.get("turn_in_stage", 0)) + 1
    rev = int(state.get("revision_count", 0))

    if stage == "S2" and _is_revision(state.get("C_t", "")):
        rev += 1

    gate = False
    if stage == "S1" and turn_in >= 3:
        gate = True
    elif stage == "S2" and rev >= 2:
        gate = True
    elif stage == "S3":
        gate = True

    next_state = dict(state)
    next_state["turn_in_stage"] = turn_in
    next_state["revision_count"] = rev
    next_state["stage_gate_ready"] = gate
    return next_state


def initial_state(session_id: str, user_intent: str) -> GroundLensState:
    return {
        "session_id": session_id,
        "conversation_history": [],
        "user_intent": user_intent,
        "current_turn": 0,
        "current_stage": "S1",
        "turn_in_stage": 0,
        "revision_count": 0,
        "stage_gate_ready": False,
        "F": [{"premise": f"초기 사용자 의도: {user_intent}", "evidence_type": ["direct_statement"]}],
        "common_ground": [],
        "llm_ground": [],
        "grounding_corrections": [],
        "llm_ground_deletions": [],
        "C_t": "",
        "D_t": "",
        "B_t": "",
        "A_t": {},
        "system_prompt_used": "",
        "errors": [],
    }


def partner_llm_node(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    next_state = copy.deepcopy(state)
    prompt = assemble_partner_system_prompt(
        cfg,
        next_state.get("grounding_corrections", []),
        next_state.get("llm_ground_deletions", []),
    )
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in next_state.get("conversation_history", [])
    ]
    result = get_assistant_response(history, prompt, cfg)
    next_state["D_t"] = result.text
    next_state["B_t"] = (result.thinking or "")[:1500]
    next_state["system_prompt_used"] = prompt
    next_state.setdefault("conversation_history", []).append(
        {
            "role": "assistant",
            "content": result.text,
            "turn": next_state["current_turn"],
            "timestamp": now_iso(),
        }
    )
    logger.info("[GroundLens] partner prompt for %s turn %s:\n%s", state["session_id"], state["current_turn"], prompt)
    return next_state


def shadow_probe_node(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    next_state = copy.deepcopy(state)
    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in next_state.get("conversation_history", [])
    ]
    probe = get_probe_response(history, str(cfg.get("shadow_probe_prompt", "")), cfg)
    next_state["A_t"] = probe.data
    return next_state


def grounding_process_node(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    next_state = copy.deepcopy(state)
    errors = next_state.setdefault("errors", [])
    model = str(cfg.get("model"))
    elements = step1_1_preprocess_utterance(
        next_state.get("C_t", ""),
        next_state.get("D_t", ""),
        errors,
        model,
    )
    result = step1_2_grounding_process(
        F_prev=next_state.get("F", []),
        C_t=next_state.get("C_t", ""),
        preprocessed_elements=elements,
        common_ground=next_state.get("common_ground", []),
        llm_ground=next_state.get("llm_ground", []),
        turn_num=int(next_state.get("current_turn", 0)),
        errors=errors,
        model=model,
    )
    next_state["F"] = result.get("F_t", next_state.get("F", []))
    cg = _with_ids(result.get("common_ground", []), "cg", next_state["current_turn"])
    lg = _with_ids(result.get("llm_ground", []), "llm", next_state["current_turn"])
    next_state["common_ground"], next_state["llm_ground"] = _enforce_classification_rules(
        next_state.get("F", []), cg, lg, next_state.get("llm_ground_deletions")
    )
    return next_state


def reflection_process_node(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    next_state = copy.deepcopy(state)
    errors = next_state.setdefault("errors", [])
    model = str(cfg.get("model"))
    thinking_elements, _ = step2_1_preprocess_thinking(
        next_state.get("A_t", {}),
        next_state.get("B_t", ""),
        errors,
        model,
    )
    result = step2_2_reflection_process(
        F_t=next_state.get("F", []),
        common_ground=next_state.get("common_ground", []),
        llm_ground=next_state.get("llm_ground", []),
        thinking_elements=thinking_elements,
        turn_num=int(next_state.get("current_turn", 0)),
        errors=errors,
        model=model,
    )
    cg = _with_ids(result.get("common_ground", next_state.get("common_ground", [])), "cg", next_state["current_turn"])
    lg = _with_ids(result.get("llm_ground", next_state.get("llm_ground", [])), "llm", next_state["current_turn"])
    next_state["common_ground"], next_state["llm_ground"] = _enforce_classification_rules(
        next_state.get("F", []), cg, lg, next_state.get("llm_ground_deletions")
    )
    return next_state


def deletion_node(state: GroundLensState) -> GroundLensState:
    next_state = copy.deepcopy(state)
    target_id = next_state.get("target_item_id", "")
    deleted_item = None
    kept_llm = []
    for item in next_state.get("llm_ground", []):
        if item.get("id") == target_id:
            deleted_item = item
        else:
            kept_llm.append(item)
    if deleted_item:
        next_state["llm_ground"] = kept_llm
        next_state.setdefault("llm_ground_deletions", []).append({
            "content": deleted_item.get("content", ""),
            "source_turn": deleted_item.get("source_turn", 0),
            "deleted_at_turn": next_state.get("current_turn", 0),
        })
    return next_state


def run_deletion_graph(state: GroundLensState) -> GroundLensState:
    return deletion_node(state)


def correction_node(state: GroundLensState) -> GroundLensState:
    next_state = copy.deepcopy(state)
    target_id = next_state.get("target_item_id", "")
    correction_text = next_state.get("correction_text", "").strip()
    source_turn = int(next_state.get("current_turn", 0))
    kept_llm = []
    for item in next_state.get("llm_ground", []):
        if item.get("id") == target_id:
            source_turn = int(item.get("source_turn", source_turn))
        else:
            kept_llm.append(item)
    corrected = {
        "id": f"cg_{uuid.uuid4().hex[:10]}",
        "content": correction_text,
        "first_turn": source_turn,
        "source_turn": source_turn,
        "corrected_at_turn": next_state.get("current_turn", 0),
    }
    next_state["llm_ground"] = kept_llm
    next_state.setdefault("common_ground", []).append(corrected)
    next_state.setdefault("grounding_corrections", []).append(
        {
            "content": correction_text,
            "source_turn": source_turn,
            "corrected_at_turn": next_state.get("current_turn", 0),
        }
    )
    # 사용자 인터페이스 조작은 interface_action + correction evidence
    next_state.setdefault("F", []).append({
        "premise": f"사용자 확인: {correction_text}",
        "evidence_type": ["interface_action", "correction"],
    })
    return next_state


def run_message_graph(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    # partner_llm and shadow_probe both read from the same input state → run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_llm = executor.submit(partner_llm_node, state, cfg)
        future_probe = executor.submit(shadow_probe_node, state, cfg)
        llm_state = future_llm.result()
        probe_state = future_probe.result()

    merged = dict(llm_state)
    merged["A_t"] = probe_state.get("A_t", {})

    ground_state = grounding_process_node(merged, cfg)
    final_state = reflection_process_node(ground_state, cfg)
    return _advance_stage_fields(final_state)


def run_correction_graph(state: GroundLensState) -> GroundLensState:
    return correction_node(state)


_ZONE_TYPES = ("구조적 결정", "암묵적 전제", "용어 해석")


def _enforce_classification_rules(
    F: list[str],
    common_ground: list[dict],
    llm_ground: list[dict],
    deletions: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    두 가지 하드 룰을 강제한다 (LLM 출력이 위반하는 경향이 있음):
    1. '용어 해석:' 항목은 사용자가 확인하기 전까지 CG 불가 → LLM Ground로 이동.
    2. F_t 항목과 content가 완전히 동일한 CG 항목은 삭제 (직접 복사 차단).
    """
    f_set = {(item["premise"] if isinstance(item, dict) else item) for item in F}
    clean_cg: list[dict] = []
    promoted: list[dict] = []

    for item in common_ground:
        content = item.get("content", "")
        if content.startswith("용어 해석"):
            entry = dict(item)
            entry["type"] = "용어 해석"
            promoted.append(entry)
        elif content in f_set:
            pass  # F_t 직접 복사본 — 제거
        else:
            clean_cg.append(item)

    existing_ids = {item.get("id") for item in llm_ground}
    merged_llm = list(llm_ground)
    for item in promoted:
        if item.get("id") not in existing_ids:
            merged_llm.append(item)
            existing_ids.add(item.get("id"))

    # 사용자가 삭제한 항목이 LLM 출력에 재등장하면 제거
    if deletions:
        deleted_contents = {d.get("content", "") for d in deletions}
        merged_llm = [item for item in merged_llm if item.get("content") not in deleted_contents]

    return clean_cg, merged_llm


def _zone_type_from_content(content: str) -> str:
    if content.startswith("구조적 결정"):
        return "구조적 결정"
    if content.startswith("용어 해석"):
        return "용어 해석"
    return "암묵적 전제"


def _with_ids(items: list[dict], prefix: str, fallback_turn: int) -> list[dict]:
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            item = {"content": str(item)}
        entry = dict(item)
        entry.setdefault("source_turn", entry.get("first_turn", fallback_turn))
        if prefix == "cg":
            entry.setdefault("first_turn", entry.get("source_turn", fallback_turn))
        else:
            # Ensure type is a zone category, not a status label like "신규추가"
            if entry.get("type") not in _ZONE_TYPES:
                entry["type"] = _zone_type_from_content(entry.get("content", ""))
        entry.setdefault("id", f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, entry.get('content', '') + str(entry.get('source_turn', ''))).hex[:10]}")
        if entry["id"] not in seen:
            normalized.append(entry)
            seen.add(entry["id"])
    return normalized
