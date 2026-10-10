from __future__ import annotations

import concurrent.futures
import copy
import logging
import re
import uuid
from typing import Any, TypedDict

from app.config import assemble_partner_system_prompt
from app.database import now_iso
from app.llm import get_assistant_response, get_probe_response
from app.plan_parser import has_plan, merge_plan_fields, parse_plan
from classification.module import (
    step1_1_preprocess_utterance,
    step1_2_grounding_process,
    step2_1_preprocess_thinking,
    step2_2_reflection_process,
    step3_attribute_plan_fields,
)

logger = logging.getLogger("uvicorn.error")


class GroundLensState(TypedDict, total=False):
    session_id: str
    conversation_history: list[dict]
    user_intent: str
    current_turn: int
    current_stage: str       # 'S1' | 'S2' | 'S3' | 'done'
    task_phase: int          # 1 = 계획 수립, 2 = 요구사항 변경 후 개정 (논문 6.3절)
    requirement_change: dict  # 2단계에서 제시된 변경 (6.5절)
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
    plan_fields: list[dict]  # 계획 산출물의 주소 가능한 필드 + 전제 귀속


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
        "task_phase": 1,
        "requirement_change": {},
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
        "plan_fields": [],
    }


def partner_llm_node(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    next_state = copy.deepcopy(state)
    prompt = assemble_partner_system_prompt(
        cfg,
        next_state.get("grounding_corrections", []),
        next_state.get("llm_ground_deletions", []),
        [f for f in next_state.get("plan_fields", []) if f.get("status") == "stale"],
        next_state.get("requirement_change") or {},
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


def grounding_process_node(
    state: GroundLensState,
    cfg: dict[str, Any],
    precomputed_elements: list[str] | None = None,
) -> GroundLensState:
    next_state = copy.deepcopy(state)
    errors = next_state.setdefault("errors", [])
    model = str(cfg.get("model"))
    elements = precomputed_elements if precomputed_elements is not None else step1_1_preprocess_utterance(
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
    prev_cg = next_state.get("common_ground", [])
    prev_lg = next_state.get("llm_ground", [])
    prev_all = prev_cg + prev_lg
    cg = _with_ids(result.get("common_ground", []), "cg", next_state["current_turn"], restore_from=prev_all)
    lg = _with_ids(result.get("llm_ground", []), "llm", next_state["current_turn"], restore_from=prev_all)
    # Defensive: LLM이 기존 llm_ground 항목을 누락하면 복원 (CG로 이동한 것은 제외)
    new_ids = {item.get("id") for item in cg} | {item.get("id") for item in lg}
    restored = []
    for prev_item in prev_lg:
        if prev_item.get("id") and prev_item["id"] not in new_ids:
            lg.append(prev_item)
            new_ids.add(prev_item["id"])
            restored.append(prev_item.get("id"))
    if restored:
        logger.warning("[GroundLens] grounding dropped %d llm_ground items — restored: %s", len(restored), restored[:5])
    next_state["common_ground"], next_state["llm_ground"] = _enforce_classification_rules(
        next_state.get("F", []), cg, lg, next_state.get("llm_ground_deletions")
    )
    return next_state


def reflection_process_node(
    state: GroundLensState,
    cfg: dict[str, Any],
    precomputed_thinking: list[str] | None = None,
) -> GroundLensState:
    next_state = copy.deepcopy(state)
    errors = next_state.setdefault("errors", [])
    model = str(cfg.get("model"))
    thinking_elements = precomputed_thinking if precomputed_thinking is not None else (
        step2_1_preprocess_thinking(
            next_state.get("A_t", {}),
            next_state.get("B_t", ""),
            errors,
            model,
        )[0]
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
    prev_cg_r = next_state.get("common_ground", [])
    prev_lg_r = next_state.get("llm_ground", [])
    prev_all = prev_cg_r + prev_lg_r
    cg = _with_ids(result.get("common_ground", prev_cg_r), "cg", next_state["current_turn"], restore_from=prev_all)
    lg = _with_ids(result.get("llm_ground", prev_lg_r), "llm", next_state["current_turn"], restore_from=prev_all)
    # Defensive: LLM이 기존 llm_ground 항목을 누락하면 복원 (CG로 이동한 것은 제외)
    new_ids_r = {item.get("id") for item in cg} | {item.get("id") for item in lg}
    restored_r = []
    for prev_item in prev_lg_r:
        if prev_item.get("id") and prev_item["id"] not in new_ids_r:
            lg.append(prev_item)
            new_ids_r.add(prev_item["id"])
            restored_r.append(prev_item.get("id"))
    if restored_r:
        logger.warning("[GroundLens] reflection dropped %d llm_ground items — restored: %s", len(restored_r), restored_r[:5])
    next_state["common_ground"], next_state["llm_ground"] = _enforce_classification_rules(
        next_state.get("F", []), cg, lg, next_state.get("llm_ground_deletions")
    )
    return next_state


def plan_attribution_node(state: GroundLensState, cfg: dict[str, Any]) -> GroundLensState:
    """계획 산출물을 필드로 분해하고 각 필드를 전제에 귀속시킨다.

    평평한 전제 리스트만으로는 "전제 A가 계획의 어느 부분을 만들었는가"를 표현할 수 없어
    A를 교정해도 A가 만든 계획 내용이 그대로 남는다. 이 노드가 그 연결을 만든다.
    """
    next_state = copy.deepcopy(state)
    if not has_plan(next_state.get("D_t", "")):
        return next_state  # 계획 응답이 아니면 이전 plan_fields 유지

    turn = int(next_state.get("current_turn", 0))
    parsed = parse_plan(next_state["D_t"], turn=turn)
    if not parsed:
        return next_state

    unchanged, to_attribute = merge_plan_fields(parsed, next_state.get("plan_fields", []))

    if to_attribute:
        attribution = step3_attribute_plan_fields(
            to_attribute,
            next_state.get("common_ground", []),
            next_state.get("llm_ground", []),
            next_state.setdefault("errors", []),
            str(cfg.get("model")),
        )
        for field in to_attribute:
            field["premise_ids"] = attribution.get(field["address"], [])
            field["status"] = "active"

    # 사라진 필드는 뒤따르지 않는다 — 계획은 매 턴 재생성되므로 현재 산출물이 기준이다
    merged = unchanged + to_attribute
    merged.sort(key=lambda f: [p["address"] for p in parsed].index(f["address"]))
    next_state["plan_fields"] = merged

    logger.info(
        "[GroundLens] 귀속 turn=%s — 필드 %d (재사용 %d / 신규·변경 %d), 미귀속 %d",
        turn, len(merged), len(unchanged), len(to_attribute),
        sum(1 for f in merged if not f.get("premise_ids")),
    )
    return next_state


def _mark_dependent_fields_stale(
    state: GroundLensState,
    premise_id: str,
    reason: str,
    replacement_id: str | None = None,
) -> list[str]:
    """전제가 교정·거부되면 그 전제가 만든 계획 필드를 stale로 표시한다.

    지금까지는 교정이 전제 목록만 바꾸고 계획 내용은 그대로 두었다. 그래서 사용자가
    전제를 고쳐도 그 전제가 만든 루틴·예산·마일스톤은 남아 누적이 되돌려지지 않았다.
    stale 표시된 필드는 다음 프롬프트에 재검토 대상으로 명시된다(5.3.6절).

    replacement_id가 있으면(교정) 귀속을 새 전제로 갈아끼우고,
    없으면(거부) 해당 귀속을 제거한다.
    """
    turn = int(state.get("current_turn", 0))
    touched: list[str] = []
    for field in state.get("plan_fields", []):
        ids = field.get("premise_ids", [])
        if premise_id not in ids:
            continue
        field["premise_ids"] = [
            replacement_id if i == premise_id else i
            for i in ids
            if replacement_id is not None or i != premise_id
        ]
        field["status"] = "stale"
        field["stale_reason"] = reason
        field["stale_at_turn"] = turn
        touched.append(field["address"])
    if touched:
        logger.info(
            "[GroundLens] stale 전파 — %s로 %d개 필드 무효화: %s",
            reason, len(touched), ", ".join(touched[:6]),
        )
    return touched


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
        _mark_dependent_fields_stale(next_state, target_id, "rejected")
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
    _mark_dependent_fields_stale(next_state, target_id, "corrected", replacement_id=corrected["id"])
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
    model = str(cfg.get("model"))

    # Phase A — partner_llm + shadow_probe in parallel (both read the same input state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_llm = executor.submit(partner_llm_node, state, cfg)
        future_probe = executor.submit(shadow_probe_node, state, cfg)
        llm_state = future_llm.result()
        probe_state = future_probe.result()

    merged = dict(llm_state)
    merged["A_t"] = probe_state.get("A_t", {})
    errors: list[str] = merged.setdefault("errors", [])

    # Phase B — step1_1 + step2_1 in parallel (both depend only on Phase A output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_elements = executor.submit(
            step1_1_preprocess_utterance,
            merged.get("C_t", ""), merged.get("D_t", ""), errors, model,
        )
        future_thinking = executor.submit(
            step2_1_preprocess_thinking,
            merged.get("A_t", {}), merged.get("B_t", ""), errors, model,
        )
        elements = future_elements.result()
        thinking_elements, _ = future_thinking.result()

    # Phase C/D — sequential: step1_2 then step2_2 (each depends on the previous)
    ground_state = grounding_process_node(merged, cfg, precomputed_elements=elements)
    final_state = reflection_process_node(ground_state, cfg, precomputed_thinking=thinking_elements)
    # Phase E — 계획 필드 귀속. CG/LG가 확정된 뒤에야 귀속 대상 전제 집합이 정해진다.
    attributed_state = plan_attribution_node(final_state, cfg)
    return _advance_stage_fields(attributed_state)


def run_correction_graph(state: GroundLensState) -> GroundLensState:
    return correction_node(state)


_ZONE_TYPES = ("구조적 결정", "암묵적 전제", "용어 해석")


# ──────────────────────────────────────────
# CG 승격 게이트 (false common ground 차단)
#
# 배경: 승격 판정을 LLM에게 맡기고 키워드 문자열 일치만 요구하면,
# 도메인 일반어가 겹치는 것만으로 근거 없는 항목이 Common Ground로 올라간다.
# CG 오분류는 사용자가 해당 전제를 검토 대상에서 제외하게 만들어
# 미검증 전제를 오히려 은폐하므로(논문 5.2절), 코드 수준에서 결정적으로 차단한다.
# ──────────────────────────────────────────

# 도메인·담화 일반어: 거의 모든 항목에 등장하므로 지지 근거가 될 수 없다
_GENERIC_TOKENS = {
    "사용자", "전제", "목표", "계획", "준비", "포함", "수준", "이상", "이하", "기준",
    "배치", "활동", "구성", "제시", "제안", "진행", "필요", "가능", "경우", "내용",
    "방식", "상태", "결과", "설정", "대한", "관련", "위해", "통해", "하는", "있는",
    "한다", "이다", "된다", "그리고", "또는", "혹은", "각각", "모두", "등을", "등이",
    # 기능어·형식명사: 그라운딩 내용을 실질적으로 담지 않으면서 자연문에 항상 등장한다.
    # 이것을 분모에 넣으면 지지 비율이 부당하게 희석되어 정당한 항목도 승격하지 못한다.
    "것이", "것을", "것은", "것으", "핵심", "내에", "동시", "여부", "정도", "위주",
    "따라", "그러", "이러", "저러", "어떤", "무엇", "다음", "이번", "현재", "주요",
    "고려", "반영", "적용", "사용", "수행", "완료", "작성", "확보", "유지", "존재",
}

# LLM이 사용자에게 제시한 선택지·질문 — 확인된 전제가 아니다
_UNRESOLVED_PAT = re.compile(
    r"여부\s*$"                      # "... 국내 또는 해외 여부"
    r"|(또는|혹은)[^,.]{0,20}(여부|인지)"
    r"|\?"                           # 질문문
    r"|예\s*\)"                      # "예) 7월 초 ~ 8월 말"
    r"|[-–]\s*[^-–]{0,30}(또는|혹은)"  # "여행 스타일 - 국내 또는 해외"
)

# LLM이 스스로 미확인 가정임을 선언한 항목 — 정의상 LLM Ground
_SELF_ASSUMED_PAT = re.compile(
    r"암묵적|임의로|가정한다|가정하는|가정함|추정한다|추정하는|미파악|미확인|불명확|명시되지\s*않"
)

_SUPPORT_THRESHOLD = 0.5  # CG 항목 내용어 중 F_t가 지지해야 하는 최소 비율


def _is_generic(w: str) -> bool:
    """교착어 대응: 어간이 일반어면 조사·어미가 붙어도 일반어로 본다.
    ("사용자는", "제시해야", "계획을" → 모두 일반어)"""
    return any(w.startswith(g) for g in _GENERIC_TOKENS)


def _content_tokens(text: str) -> set[str]:
    """일반어를 제거한 내용어 토큰 집합."""
    return {
        w for w in re.findall(r"[가-힣A-Za-z0-9]+", text or "")
        if len(w) >= 2 and not _is_generic(w)
    }


# 발화와 전제문에 붙는 조사. 길이 역정렬로 달아야 "에서"를 "에"보다 먼저 떼어낸다.
_PARTICLES = (
    "에서도", "에게는", "에게도", "으로는", "으로도", "부터는", "까지는",
    "에서", "에게", "에겐", "으로", "부터", "까지", "보다", "처럼",
    "마다", "라도", "이나", "라는", "이라", "하고", "이랑", "정도",
    "은", "는", "이", "가", "을", "를", "에", "의", "도", "만", "와", "과",
    "로", "랑", "야", "인", "에도",
)


def _strip_particle(w: str) -> str:
    """어말 조사를 한 번 떼어낸다. 어간이 2자 미만으로 줄면 그대로 둔다."""
    for q in _PARTICLES:
        if len(w) > len(q) + 1 and w.endswith(q):
            return w[: -len(q)]
    return w


def _tok_match(a: str, b: str) -> bool:
    """한국어 교착 대응: 같은 어간이면 같은 단어로 본다.

    접두사 관계만 보면 한쪽에만 조사가 붙은 경우만 잡힌다.
    ("코딩테스트" <-> "코딩테스트는" 잡힘, "토요일에" <-> "토요일은" 놓침)
    둘 다 조사를 벗긴 뒤 비교하면 후자도 잡힌다. 의역과 유의어는 그래도
    인정하지 않는다(문서 M2: "의역은 지지로 인정하지 않는다").
    """
    if a == b:
        return True
    sa, sb = _strip_particle(a), _strip_particle(b)
    if sa == sb and len(sa) >= 2:
        return True
    lo, hi = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(lo) >= 2 and hi.startswith(lo)


def _support_ratio(content: str, f_tokens: set[str]) -> float:
    """CG 항목의 내용어 중 F_t에 의해 지지되는 비율."""
    ct = _content_tokens(content)
    if not ct:
        return 0.0
    hit = sum(1 for t in ct if any(_tok_match(t, ft) for ft in f_tokens))
    return hit / len(ct)


def _cg_reject_reason(content: str, f_tokens: set[str]) -> str | None:
    """CG 유지 불가 사유. None이면 CG 유지 가능."""
    if _UNRESOLVED_PAT.search(content):
        return "unresolved_option"   # LLM이 물은 선택지/질문
    if _SELF_ASSUMED_PAT.search(content):
        return "self_assumed"        # LLM이 스스로 가정이라 선언
    ratio = _support_ratio(content, f_tokens)
    if ratio < _SUPPORT_THRESHOLD:
        return f"unsupported({ratio:.2f})"
    return None


def _enforce_classification_rules(
    F: list,
    common_ground: list[dict],
    llm_ground: list[dict],
    deletions: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    하드 룰을 강제한다 (LLM 출력이 위반하는 경향이 있음):
    1. '용어 해석:' 항목은 사용자가 확인하기 전까지 CG 불가 → LLM Ground로 이동.
    2. F_t 항목과 content가 완전히 동일한 CG 항목은 삭제 (직접 복사 차단).
    3. F_t의 지지 근거가 없는 CG 항목은 LLM Ground로 강등 (false common ground 차단).
    """
    f_texts = [(item["premise"] if isinstance(item, dict) else item) for item in F]
    f_set = set(f_texts)
    f_tokens: set[str] = set()
    for ft in f_texts:
        f_tokens |= _content_tokens(ft)

    clean_cg: list[dict] = []
    promoted: list[dict] = []
    demoted_log: list[str] = []

    for item in common_ground:
        content = item.get("content", "")
        if content.startswith("용어 해석"):
            entry = dict(item)
            entry["type"] = "용어 해석"
            promoted.append(entry)
        elif content in f_set:
            pass  # F_t 직접 복사본 — 제거
        else:
            reason = _cg_reject_reason(content, f_tokens)
            if reason is None:
                clean_cg.append(item)
            else:
                entry = dict(item)
                if entry.get("type") not in _ZONE_TYPES:
                    entry["type"] = _zone_type_from_content(content)
                entry.pop("evidence_type", None)  # 지지 근거 없음 — evidence 표기 제거
                promoted.append(entry)
                demoted_log.append(f"{reason}: {content[:40]}")

    if demoted_log:
        logger.info(
            "[GroundLens] CG 승격 게이트 — %d건 강등: %s",
            len(demoted_log), " | ".join(demoted_log[:5]),
        )

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


def _with_ids(
    items: list[dict],
    prefix: str,
    fallback_turn: int,
    restore_from: list[dict] | None = None,
) -> list[dict]:
    def _clean(c: str) -> str:
        return re.sub(r"^\[후보\d+\]\s*", "", c or "").strip()

    # content 기준으로 이전 항목의 source_turn/first_turn/type을 복원
    prior: dict[str, dict] = {_clean(i.get("content", "")): i for i in (restore_from or [])}
    normalized = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            item = {"content": str(item)}
        entry = dict(item)
        entry["content"] = _clean(entry.get("content", ""))
        p = prior.get(entry["content"])
        if p:
            entry.setdefault("source_turn", p.get("source_turn"))
            entry.setdefault("first_turn", p.get("first_turn"))
        entry.setdefault("source_turn", entry.get("first_turn", fallback_turn))
        if prefix == "cg":
            entry.setdefault("first_turn", entry.get("source_turn", fallback_turn))
        else:
            # 세부유형(zone): 프롬프트가 지정한 값이 유효하면 유지,
            # 아니면 이전 항목에서 복원, 그래도 없으면 content 휴리스틱
            if entry.get("type") not in _ZONE_TYPES:
                if p and p.get("type") in _ZONE_TYPES:
                    entry["type"] = p["type"]
                else:
                    entry["type"] = _zone_type_from_content(entry.get("content", ""))
        entry.setdefault("id", f"{prefix}_{uuid.uuid5(uuid.NAMESPACE_URL, entry.get('content', '') + str(entry.get('source_turn', ''))).hex[:10]}")
        if entry["id"] not in seen:
            normalized.append(entry)
            seen.add(entry["id"])
    return normalized
