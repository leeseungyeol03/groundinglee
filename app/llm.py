from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def is_mock() -> bool:
    return os.environ.get("MOCK_LLM", "false").strip().lower() in {"1", "true", "yes", "on"}


def get_api_key() -> str:
    """Anthropic API 키를 반환한다.

    .env의 키 이름이 `Anthropic_API_KEY`라 표준 명칭과 대소문자가 다르므로
    둘 다 허용한다.
    """
    for name in ("ANTHROPIC_API_KEY", "Anthropic_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _make_client():
    import anthropic
    return anthropic.Anthropic(api_key=get_api_key())


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
    temperature = float(cfg.get("temperature", 1.0))
    # anthropic SDK 1.7.0은 temperature를 상위 인자에서 제거했지만
    # API 자체는 여전히 수용한다. 재현성을 위해 extra_body로 명시한다.
    response = _make_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": m["role"], "content": m["content"]} for m in history],
        extra_body={"temperature": temperature},
    )
    text = "".join(b.text for b in response.content if b.type == "text")
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

    response = _make_client().messages.create(
        model=str(cfg.get("model")),
        max_tokens=512,
        messages=[
            *({"role": m["role"], "content": m["content"]} for m in history),
            {"role": "user", "content": probe_prompt},
        ],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:]).rstrip("`").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        data = {"goal": "", "constraints": [], "term_interpretation": {}}
    return ProbeResult(data, raw)
