"""E2 조건 구현 — B0 대화만, B1 메모리식 기준선 (feedback2.md E2).

왜 B1이 중요한가:
논문의 주 비교 대상이다. 사용자가 편집할 수 있고 편집이 다음 주입에 반영되지만
P1(검증 상태)·P2(사용 추적)·P3(수정 전파와 반영 검증)는 없는 조건이다.
제품 메모리가 하는 일이 정확히 이것이고, 이 기준선을 넘지 못하면 논문이 되지
않는다.

B0  패널 없음. 교정은 대화로만. 일반 챗봇
B1  LLM이 요약한 전제 목록을 사용자가 보고 편집할 수 있고, 다음 턴 주입에
    반영된다. 상태 구분·사용 추적·전파·반영 검증은 없다

구현상 B1이 B0와 다른 점은 두 가지뿐이다.
    1. 매 턴 전제 목록을 뽑아 사용자에게 보여 주고 편집을 받는다
    2. 편집된 목록을 다음 프롬프트에 주입한다
이게 메모리 기능의 전부다. 고친 게 기존 산출물에 어떻게 퍼져 있었는지,
재생성 결과에 실제로 반영됐는지는 알려주지 않는다.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

CONDITIONS = ("B0", "B1")

PLAN_TASK = (
    "사용자의 조건을 반영한 계획을 4섹션(목표 / 주간 일정 / 주차별 마일스톤 / 예산)으로 "
    "제시하세요. 주간 일정은 월부터 일까지 일곱 줄을 오전·오후·저녁 칸으로 채우세요. "
    "각 항목은 명사구로만 쓰고, 이유나 근거를 설명하는 문장은 쓰지 마세요. "
    "사용자가 말하지 않은 조건이 필요하면 계획 끝에 질문 한 줄을 덧붙이세요."
)

_MEMORY_HEADER = "【기억하고 있는 내용】"


# --------------------------------------------------------------------------- #
# 전제 추출 — 메모리 기능이 하는 일
# --------------------------------------------------------------------------- #

_EXTRACT_PROMPT = """아래는 사용자와 AI의 대화, 그리고 AI가 만든 계획입니다.
AI가 이 계획을 만들면서 사용자에 대해 사실로 삼은 내용을 항목으로 뽑으세요.

사용자가 직접 말한 것과 AI가 추측한 것을 구분하지 말고, 계획의 근거가 된
내용이면 모두 적으세요. 제품 메모리가 대화를 요약해 저장하는 방식과 같습니다.

규칙
- 한 항목은 한 문장. 명제 형태로 쓰세요
- 6~10개
- 계획에 드러난 내용만. 없는 내용을 지어내지 마세요

JSON 배열로만 답하세요. 예: ["사용자는 평일 3시간을 쓸 수 있다", ...]

[대화]
{history}

[계획]
{plan}"""


def extract_memory_items(client, history: list[dict], plan: str,
                         model: str = "claude-haiku-4-5-20251001") -> list[str]:
    """계획의 근거가 된 전제를 뽑는다. 메모리 기능의 요약 단계에 해당한다."""
    hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-6:])
    r = client.messages.create(
        model=model, max_tokens=700,
        messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(
            history=hist, plan=plan[:3000])}],
        extra_body={"temperature": 0.0},
    )
    raw = r.content[0].text or ""
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        v = json.loads(m.group())
        return [str(x).strip() for x in v if str(x).strip()][:12]
    except json.JSONDecodeError:
        return []


_REVIEW_PROMPT = """당신은 AI와 계획을 세우는 사용자입니다. AI가 당신에 대해
기억하고 있다는 목록을 보고 있습니다.

[당신의 실제 프로필]
{profile}

[AI가 기억하고 있다는 목록]
{items}

당신이 검토할 수 있는 항목은 위에서 최대 {k}개까지입니다. 그중 당신의 실제
프로필과 어긋나는 항목이 있으면 고치세요.

각 항목에 대해 아래 중 하나로 답하세요.
  keep    그대로 둔다
  edit    고친다 (고친 문장을 함께 적는다)
  delete  지운다

JSON 배열로만 답하세요. 검토하지 않은 항목은 keep으로 두세요.
예: [{{"i":0,"op":"keep"}},{{"i":1,"op":"edit","text":"고친 문장"}},{{"i":2,"op":"delete"}}]"""


def review_memory_items(client, items: list[str], persona, k: int | None,
                        model: str = "claude-haiku-4-5-20251001") -> tuple[list[str], int]:
    """사용자가 메모리 목록을 검토하고 편집한다.

    주의 예산 k가 있으면 앞에서 k개까지만 본다. 사람이 긴 목록을 전부 읽지
    않는다는 사실을 반영한 것이고, 이 제약이 없으면 메모리 기준선이 비현실적으로
    강해진다.

    반환: (편집된 목록, 조작 수)
    """
    if not items:
        return [], 0
    budget = len(items) if k is None else min(k, len(items))
    profile = "\n".join(f"- {x}" for x in persona.checklist)
    for key, val in persona.private_meanings.items():
        profile += f'\n- "{key}"는 {val}라는 뜻이다'
    numbered = "\n".join(f"{i}. {x}" for i, x in enumerate(items))
    r = client.messages.create(
        model=model, max_tokens=900,
        messages=[{"role": "user", "content": _REVIEW_PROMPT.format(
            profile=profile, items=numbered, k=budget)}],
        extra_body={"temperature": 0.0},
    )
    raw = r.content[0].text or ""
    m = re.search(r"\[.*\]", raw, re.S)
    out = list(items)
    ops = 0
    if m:
        try:
            for d in json.loads(m.group()):
                i = int(d.get("i", -1))
                if not (0 <= i < len(out)) or i >= budget:
                    continue
                op = d.get("op", "keep")
                if op == "edit" and d.get("text"):
                    out[i] = str(d["text"]).strip()
                    ops += 1
                elif op == "delete":
                    out[i] = ""
                    ops += 1
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return [x for x in out if x], ops


# --------------------------------------------------------------------------- #
def build_system(condition: str, memory_items: list[str]) -> str:
    """조건별 시스템 프롬프트.

    B0은 메모리가 없다. B1은 편집된 목록을 그대로 주입한다. 상태 구분이 없으므로
    확인된 것과 추측한 것이 한 덩어리로 들어간다 — 인식론적으로 평평하다.
    """
    base = ("당신은 사용자의 계획 수립을 돕는 AI 파트너입니다.\n\n" + PLAN_TASK)
    if condition == "B0" or not memory_items:
        return base
    mem = "\n".join(f"- {x}" for x in memory_items)
    return f"{base}\n\n{_MEMORY_HEADER}\n{mem}"
