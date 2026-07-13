from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def is_mock() -> bool:
    return os.environ.get("MOCK_LLM", "true").strip().lower() in {"1", "true", "yes", "on"}


def _make_client():
    from openai import OpenAI
    return OpenAI(
        base_url="https://sam.soonsoon.ai/openai/v1",
        api_key=os.environ.get("SAM_API_KEY", ""),
    )


@dataclass
class AssistantResult:
    text: str
    thinking: str


@dataclass
class ProbeResult:
    data: dict[str, Any]
    raw: str


def get_assistant_response(
    history: list[dict[str, str]],
    system_prompt: str,
    cfg: dict[str, Any],
) -> AssistantResult:
    if is_mock():
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        correction_note = ""
        if "그라운딩 보정" in system_prompt or "명시적으로 확인·수정" in system_prompt:
            correction_note = " 이전에 수정해주신 조건도 반영해서 볼게요."
        return AssistantResult(
            text=(
                "[MOCK] 말씀하신 목표와 제약을 기준으로 방학 계획을 함께 만들어보겠습니다."
                f"{correction_note} 우선 핵심 조건을 정리하면: {last_user[:90]}"
            ),
            thinking=(
                "[MOCK THINKING] 사용자는 방학 중 달성할 목표와 제약 조건을 정리하길 원한다. "
                "5섹션 구조로 초안을 제시한 뒤 수정받는 방식을 취해야 한다."
            ),
        )

    model = str(cfg.get("model"))
    max_tokens = int(cfg.get("max_tokens", 4096))
    messages = [{"role": "system", "content": system_prompt}, *history]
    response = _make_client().chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    text = response.choices[0].message.content or ""
    return AssistantResult(text=text, thinking="")


def get_probe_response(
    history: list[dict[str, str]],
    probe_prompt: str,
    cfg: dict[str, Any],
) -> ProbeResult:
    if is_mock():
        raw = json.dumps(
            {
                "goal": "[MOCK] 방학 목표 달성 계획 수립",
                "constraints": ["시간", "예산", "고정 일정"],
                "term_interpretation": {"방학 계획": "목표·루틴·마일스톤·예산·예외규칙을 구조화하는 작업"},
            },
            ensure_ascii=False,
        )
        return ProbeResult(json.loads(raw), raw)

    response = _make_client().chat.completions.create(
        model=str(cfg.get("model")),
        max_tokens=512,
        messages=[*history, {"role": "user", "content": probe_prompt}],
    )
    raw = response.choices[0].message.content or ""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:]).rstrip("`").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        data = {"goal": "", "constraints": [], "term_interpretation": {}}
    return ProbeResult(data, raw)
