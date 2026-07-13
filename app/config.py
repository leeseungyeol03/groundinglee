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
) -> str:
    """Build the visible-partner system prompt. user_intent is intentionally absent."""
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
    return "\n\n".join(part for part in parts if part)
