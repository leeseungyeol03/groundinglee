from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def assemble_partner_system_prompt(
    cfg: dict[str, Any],
    grounding_corrections: list[dict],
    llm_ground_deletions: list[dict] | None = None,
    stale_fields: list[dict] | None = None,
    requirement_change: dict | None = None,
) -> str:
    """Build the visible-partner system prompt. user_intent is intentionally absent.

    stale_fields: 교정·거부된 전제에 귀속되어 더 이상 유효하지 않은 계획 부분.
    교정을 문장으로 붙이는 것만으로는 이미 생성된 계획 내용이 바뀌지 않으므로,
    어느 부분을 다시 만들어야 하는지를 명시한다.

    requirement_change: 2단계에서 제시된 요구사항 변경(논문 6.5절).
    변경 사실 자체는 파트너 LLM도 알아야 하지만, **어느 계획 부분이
    영향받는지는 알려주지 않는다.** 그것을 찾는 것이 참가자의 과업이고
    H3의 측정 대상이므로, 범위를 알려주면 종속변수가 오염된다.
    """
    parts = [
        str(cfg.get("neutral_system_prompt", "")).strip(),
        str(cfg.get("task_context", "")).strip(),
    ]
    corrections = [
        str(item.get("content", "")).strip()
        for item in grounding_corrections
        if str(item.get("content", "")).strip()
    ]
    if corrections:
        correction_block = "\n".join(f"- {content}" for content in corrections)
        parts.append(
            "다음은 사용자가 명시적으로 확인·수정한 이해입니다. 반드시 반영하세요:\n"
            f"{correction_block}"
        )
    deletions = [
        str(item.get("content", "")).strip()
        for item in (llm_ground_deletions or [])
        if str(item.get("content", "")).strip()
    ]
    if deletions:
        deletion_block = "\n".join(f"- {content}" for content in deletions)
        parts.append(
            "다음은 사용자가 더 이상 고려하지 않도록 제거한 이해입니다. 이 내용은 이후 응답에서 사용하지 마세요:\n"
            f"{deletion_block}"
        )

    rc = requirement_change or {}
    if str(rc.get("text", "")).strip():
        parts.append(
            "과업 조건이 변경되었습니다. 사용자에게 아래 내용이 전달되었으므로 이후 응답은 "
            "변경된 조건을 기준으로 합니다:\n"
            f"- {str(rc['text']).strip()}\n"
            "사용자가 지시하지 않은 부분을 먼저 고쳐 제시하지 마세요. "
            "어느 부분을 어떻게 바꿀지는 사용자와 상의해 정합니다."
        )

    stale = [f for f in (stale_fields or []) if str(f.get("content", "")).strip()]
    if stale:
        stale_block = "\n".join(
            f"- [{f.get('address', '')}] 현재 내용: {str(f.get('content', '')).strip()}"
            for f in stale
        )
        parts.append(
            "다음 계획 부분은 사용자가 수정·제거한 전제를 근거로 만들어진 것입니다. "
            "위의 수정된 이해를 반영해 해당 부분을 다시 구성하세요. "
            "나머지 부분은 이유 없이 바꾸지 마세요:\n"
            f"{stale_block}"
        )
    return "\n\n".join(part for part in parts if part)
