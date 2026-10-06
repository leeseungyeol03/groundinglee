from __future__ import annotations

import json
import os
import re
from typing import Any

from classification.prompts import (
    ATTRIBUTION_PROMPT,
    GROUNDING_PROCESS_PROMPT,
    PREPROCESS_THINKING_PROMPT,
    PREPROCESS_UTTERANCE_PROMPT,
    REFLECTION_PROCESS_PROMPT,
)

B_T_MAX_LEN = 1500
CLASSIFICATION_MAX_TOKENS = int(os.environ.get("GL_CLASSIFICATION_MAX_TOKENS", "16384"))


def _is_mock() -> bool:
    return os.environ.get("MOCK_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
            pass
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group())
        except json.JSONDecodeError:
            pass
    return None


def call_llm(messages: list[dict[str, str]], model: str, max_tokens: int = 800) -> str:
    import anthropic

    from app.llm import get_api_key

    client = anthropic.Anthropic(api_key=get_api_key())
    # 분류 재현성을 위해 temperature=0 고정.
    # SDK 1.7.0은 상위 인자를 받지 않아 extra_body로 전달한다.
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        extra_body={"temperature": 0},
    )
    return "".join(b.text for b in response.content if b.type == "text")


def _parse_json_with_retry(
    messages: list[dict[str, str]],
    step_name: str,
    errors: list[str],
    model: str,
    max_tokens: int,
) -> Any:
    for attempt in range(1, 3):
        try:
            parsed = _extract_json(call_llm(messages, model=model, max_tokens=max_tokens))
            if parsed is not None:
                return parsed
            errors.append(f"[{step_name}] attempt {attempt}: JSON parse failed")
        except Exception as exc:
            errors.append(f"[{step_name}] attempt {attempt}: {exc}")
    return None


def step1_1_preprocess_utterance(
    C_t: str,
    D_t: str,
    errors: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> list[str]:
    if _is_mock():
        elements = []
        if C_t.strip():
            elements.append(f"사용자가 방학 목표와 제약을 제시했다: {C_t.strip()[:90]}")
        elements.append("AI가 방학 계획 구조와 우선순위 배분을 제안했다")
        return elements

    prompt = PREPROCESS_UTTERANCE_PROMPT.format(C_t=C_t, D_t=D_t)
    result = _parse_json_with_retry(
        [{"role": "user", "content": prompt}],
        "Preprocess Utterance",
        errors,
        model,
        500,
    )
    elements = result.get("elements", []) if isinstance(result, dict) else []
    return [e for e in elements if "[확인 필요]" not in e]


def step1_2_grounding_process(
    F_prev: list,
    C_t: str,
    preprocessed_elements: list[str],
    common_ground: list[dict],
    llm_ground: list[dict],
    turn_num: int,
    errors: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    if _is_mock():
        F_t = [_normalise_f_item(item) for item in F_prev]
        user_premise = f"사용자 발화 {turn_num}: {C_t.strip()[:120]}"
        existing_premises = {item["premise"] for item in F_t}
        if C_t.strip() and user_premise not in existing_premises:
            F_t.append({"premise": user_premise, "evidence_type": ["direct_statement"]})
        return {
            "F_t": F_t,
            "common_ground": common_ground,
            "llm_ground": llm_ground,
            "grounding_log": "mock grounding completed",
        }

    prompt = GROUNDING_PROCESS_PROMPT.format(
        F_prev=json.dumps(F_prev, ensure_ascii=False),
        C_t=C_t,
        preprocessed_elements=json.dumps(preprocessed_elements, ensure_ascii=False),
        common_ground=json.dumps(common_ground, ensure_ascii=False),
        llm_ground=json.dumps(llm_ground, ensure_ascii=False),
        current_turn=turn_num,
    )
    result = _parse_json_with_retry(
        [{"role": "user", "content": prompt}],
        "Grounding Process",
        errors,
        model,
        CLASSIFICATION_MAX_TOKENS,
    )
    if not isinstance(result, dict):
        return {
            "F_t": F_prev,
            "common_ground": common_ground,
            "llm_ground": llm_ground,
            "grounding_log": "failed; previous state kept",
        }
    return result


def step2_1_preprocess_thinking(
    A_t: dict,
    B_t: str,
    errors: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> tuple[list[str], bool]:
    truncated = len(B_t) > B_T_MAX_LEN
    if _is_mock():
        return (
            [
                f"구조적 결정: {turn_label(A_t)} 방학 계획은 목표·루틴·마일스톤·예산·예외규칙 5섹션으로 구성해야 한다",
                f"암묵적 전제: {turn_label(A_t)} 사용자는 고정 일정을 방해하는 계획을 원하지 않는다",
                f"용어 해석: {turn_label(A_t)} 방학 계획은 주차별 마일스톤과 일일 루틴을 포함하는 구조적 산출물이다",
            ],
            truncated,
        )

    prompt = PREPROCESS_THINKING_PROMPT.format(
        A_t=json.dumps(A_t, ensure_ascii=False),
        B_t_excerpt=B_t[:B_T_MAX_LEN],
    )
    result = _parse_json_with_retry(
        [{"role": "user", "content": prompt}],
        "Preprocess Thinking",
        errors,
        model,
        800,
    )
    elements = result.get("elements", []) if isinstance(result, dict) else []
    return [e for e in elements if "[확인 필요]" not in e], truncated


def step2_2_reflection_process(
    F_t: list,
    common_ground: list[dict],
    llm_ground: list[dict],
    thinking_elements: list[str],
    turn_num: int,
    errors: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    if _is_mock():
        existing = {item.get("content") for item in llm_ground}
        next_llm = list(llm_ground)
        for content in thinking_elements:
            content = f"{content} [turn {turn_num}]"
            if content not in existing:
                next_llm.append(
                    {
                        "content": content,
                        "type": _classify_type(content),
                        "source_turn": turn_num,
                    }
                )
        return {
            "common_ground": common_ground,
            "llm_ground": next_llm,
            "reflection_log": "mock reflection completed",
        }

    prompt = REFLECTION_PROCESS_PROMPT.format(
        F_t=json.dumps(F_t, ensure_ascii=False),
        common_ground=json.dumps(common_ground, ensure_ascii=False),
        llm_ground=json.dumps(llm_ground, ensure_ascii=False),
        thinking_elements=json.dumps(thinking_elements, ensure_ascii=False),
        current_turn=turn_num,
    )
    result = _parse_json_with_retry(
        [{"role": "user", "content": prompt}],
        "Reflection Process",
        errors,
        model,
        CLASSIFICATION_MAX_TOKENS,
    )
    if not isinstance(result, dict):
        return {
            "common_ground": common_ground,
            "llm_ground": llm_ground,
            "reflection_log": "failed; previous state kept",
        }
    return result


# 한 필드에 묶을 수 있는 전제 수 상한. 넘치면 교정 시 무효화 범위가 과도하게 번진다.
MAX_PREMISES_PER_FIELD = 3


def step3_attribute_plan_fields(
    plan_fields: list[dict[str, Any]],
    common_ground: list[dict],
    llm_ground: list[dict],
    errors: list[str],
    model: str = "claude-haiku-4-5-20251001",
) -> dict[str, list[str]]:
    """계획 필드 → 그 필드를 만든 전제 id 목록.

    반환: {address: [premise_id, ...]}. 귀속이 없는 필드는 빈 리스트로 남는다
    ("근거 없이 들어간 계획 내용"이라는 유의미한 신호이므로 억지로 채우지 않는다).

    전제는 CG·LG를 모두 넣는다. 필드가 검증된 전제에서 나왔는지 미검증 전제에서
    나왔는지를 구분해야 미검증 부하(unverified load)를 산출할 수 있다.
    """
    if not plan_fields:
        return {}

    premises: list[dict[str, str]] = []
    for item in common_ground:
        if item.get("id") and item.get("content"):
            premises.append({"id": item["id"], "content": item["content"], "state": "검증됨"})
    for item in llm_ground:
        if item.get("id") and item.get("content"):
            premises.append({"id": item["id"], "content": item["content"], "state": "미검증"})
    if not premises:
        return {f["address"]: [] for f in plan_fields}

    valid_ids = {p["id"] for p in premises}

    if _is_mock():
        # 결정적 mock: 내용어 겹침으로 귀속. 실제 판정과 다르지만 배선을 동작시킨다.
        out: dict[str, list[str]] = {}
        for field in plan_fields:
            ft = _tokens(_field_text(field))
            scored = []
            for p in premises:
                overlap = len(ft & _tokens(p["content"]))
                if overlap >= 2:
                    scored.append((overlap, p["id"]))
            scored.sort(reverse=True)
            out[field["address"]] = [pid for _, pid in scored[:MAX_PREMISES_PER_FIELD]]
        return out

    prompt = ATTRIBUTION_PROMPT.format(
        premises=json.dumps(premises, ensure_ascii=False, indent=1),
        fields=json.dumps(
            [{"address": f["address"], "content": _field_text(f)} for f in plan_fields],
            ensure_ascii=False,
            indent=1,
        ),
    )
    result = _parse_json_with_retry(
        [{"role": "user", "content": prompt}],
        "Attribution",
        errors,
        model,
        CLASSIFICATION_MAX_TOKENS,
    )
    if not isinstance(result, dict):
        errors.append("[Attribution] failed; no attribution recorded")
        return {f["address"]: [] for f in plan_fields}

    known = {f["address"] for f in plan_fields}
    out = {f["address"]: [] for f in plan_fields}
    for entry in result.get("attributions", []):
        if not isinstance(entry, dict):
            continue
        address = entry.get("address")
        if address not in known:
            continue  # 없는 필드는 무시
        ids = [i for i in entry.get("premise_ids", []) if i in valid_ids]  # 환각 id 제거
        out[address] = list(dict.fromkeys(ids))[:MAX_PREMISES_PER_FIELD]
    return out


def _field_text(field: dict[str, Any]) -> str:
    """귀속 판정에 넘길 필드 텍스트.

    예산·마일스톤은 표의 첫 열(항목명·주차)이 label로 분리되어 있어
    content만 보내면 "20만원"처럼 판단 근거가 잘린다. 둘을 합쳤다.
    """
    label = str(field.get("label", "")).strip()
    content = str(field.get("content", "")).strip()
    return f"{label}: {content}" if label else content


def _tokens(text: str) -> set[str]:
    """mock 귀속용 토큰. 조사를 대충 털어내기 위해 앞 2글자만 남긴다."""
    words = re.findall(r"[가-힣A-Za-z0-9]+", text or "")
    return {w[:2] for w in words if len(w) >= 2}


def _normalise_f_item(item) -> dict:
    """F_t 항목을 {premise, evidence_type} 객체로 정규화. 기존 문자열 항목도 처리."""
    if isinstance(item, dict) and "premise" in item:
        item.setdefault("evidence_type", ["direct_statement"])
        return item
    return {"premise": str(item), "evidence_type": ["direct_statement"]}


def _classify_type(content: str) -> str:
    if content.startswith("구조적 결정"):
        return "구조적 결정"
    if content.startswith("암묵적 전제"):
        return "암묵적 전제"
    if content.startswith("용어 해석"):
        return "용어 해석"
    return "암묵적 전제"


def turn_label(A_t: dict) -> str:
    goal = str(A_t.get("goal", "")) if isinstance(A_t, dict) else ""
    return "현재" if not goal else "현재"
