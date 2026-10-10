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
from app.usage_meter import METER
import os
import re
from dataclasses import dataclass, field
from typing import Any

CONDITIONS = ("B0", "B1", "B2", "B3", "B4")

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
    METER.record("memory.extract", model, r)
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
    METER.record("memory.review", model, r)
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


# --------------------------------------------------------------------------- #
# B2~B4 — 처리 층을 단계적으로 켠다
#
# 조건별로 뭐가 켜지는지 (feedback2.md E2)
#   B0  패널 없음
#   B1  평평한 메모리 목록 + 편집          <- 제품 메모리
#   B2  + P1 검증 상태 구분
#   B3  + P2 사용 추적 (쓰인 LG만 표시, 영향 세그먼트 표시)
#   B4  + P3 파생 재검토 + 반영 재검사와 재시도
#
# 레벨로 두는 이유는 어블레이션이 목적이기 때문이다. B2·B3는 어느 처리에서
# 이득이 생기는지 가르는 역할이고, 코드가 조건마다 갈라지면 그 비교가
# 오염된다. 한 경로에 스위치만 둔다.
# --------------------------------------------------------------------------- #

LEVEL = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4}


def assign_status(store, items: list[str], user_utterances: list[str],
                  turn: int) -> dict[str, int]:
    """전제 후보에 상태를 부여한다 (M2 / P1).

    사용자 발화에 근거가 있으면 grounded, 없으면 proposed(LG)다. 판정은 기존
    승격 게이트를 그대로 쓴다(형태소 기반 내용어 지지율, 선택지·질문 서술 배제,
    도메인 일반어 제외). 의역은 지지로 인정하지 않는다.

    **행동 검사 결과는 여기 들어오지 않는다**(원칙 2). "LLM이 쓰고 있다"는
    "사용자와 확인되었다"가 아니다.
    """
    from app.graph import (
        _SUPPORT_THRESHOLD,
        _cg_reject_reason,
        _content_tokens,
        _support_ratio,
    )
    from app.state_store import Evidence, Premise

    ft: set[str] = set()
    for u in user_utterances:
        ft |= _content_tokens(u)

    counts = {"grounded": 0, "proposed": 0}
    existing = {p.text for p in store.premises.values()}
    for i, text in enumerate(items):
        if text in existing:
            continue
        pid = f"p{turn:02d}_{i:02d}"
        reason = _cg_reject_reason(text, ft)
        ratio = _support_ratio(text, ft)
        ok = reason is None and ratio >= _SUPPORT_THRESHOLD
        p = Premise(id=pid, text=text, origin="session",
                    status="proposed", candidate_sources=["surface"])
        store.add(p)
        if ok:
            store.ground(pid, Evidence(type="user_statement", turn=turn,
                                       pointer=f"t{turn}", attribution="estimated"))
            counts["grounded"] += 1
        else:
            counts["proposed"] += 1
    return counts


def test_dependence(bt, msgs, plan: str, segments, store,
                    placebo_deltas, turn: int) -> dict[str, int]:
    """미확인 전제가 실제로 쓰였는지 행동 검사로 판정한다 (M3 / P2).

    operative인 LG만 패널에 올린다. 전부 보여 주면 사용자가 상시 점검을 해야
    하고, 최근 연구에서 사용자들이 그걸 가장 낮게 평가했다[R7]. 원칙 4대로
    실제로 쓰인 것만 보여 준다.

    반대 값은 LLM에게 만들게 하지 않고 고정 틀로 부정한다. 개입 문장 생성에
    또 LLM을 끼우면 호출이 배로 늘고 재현성도 떨어진다.
    """
    from app.behavior_test import judge_premise

    base = bt.score_segments(msgs, plan, segments)
    counts = {"operative": 0, "inert": 0, "inconclusive": 0}
    for p in store.by_status("proposed"):
        if p.verdict != "untested" and p.tested_turn == turn:
            continue
        contra = f"다음은 사실이 아니다: {p.text}"
        d_a = bt.delta(base, bt.score_segments(msgs, plan, segments,
                                               assumption=p.text))
        d_c = bt.delta(base, bt.score_segments(msgs, plan, segments,
                                               assumption=contra))
        v = judge_premise(d_a, [d_c], placebo_deltas)
        store.set_verdict(p.id, v.verdict, turn, v.segments)
        counts[v.verdict] = counts.get(v.verdict, 0) + 1
    return counts


_PANEL_PROMPT = """당신은 AI와 계획을 세우는 사용자입니다. AI가 당신에 대해
가지고 있는 전제 목록을 보고 있습니다.

[당신의 실제 프로필]
{profile}

[AI의 전제 목록]
{items}

당신이 검토할 수 있는 항목은 위에서 최대 {k}개까지입니다. 그중 당신의 실제
프로필과 어긋나는 항목이 있으면 고치세요.

각 항목에 대해 아래 중 하나로 답하세요.
  keep    그대로 둔다
  ack     맞다고 확인해 준다
  edit    고친다 (고친 문장을 함께 적는다)
  reject  틀렸으니 쓰지 말라고 한다

JSON 배열로만 답하세요. 검토하지 않은 항목은 keep으로 두세요.
예: [{{"i":0,"op":"ack"}},{{"i":1,"op":"edit","text":"고친 문장"}},{{"i":2,"op":"reject"}}]"""


def review_panel(client, store, persona, k: int | None, turn: int,
                 only_operative: bool = False,
                 model: str = "claude-haiku-4-5-20251001") -> dict:
    """사용자가 패널을 검토한다 (B2 이상).

    B1의 review_memory_items와 다른 점은 상태가 보이고 조작이 증거로 남는다는
    것이다. 승인(ack)은 ui_ack, 수정은 ui_correction, 거부는 ui_reject다.
    인터페이스 조작은 어느 전제를 가리키는지 분명하므로 attribution이
    designated다. 대화 교정은 LLM 추정이라 estimated로 남는다.

    only_operative(B3 이상): 행동 검사로 쓰인 것이 확인된 LG만 올린다.
    """
    from app.state_store import Evidence

    from app.usage_meter import METER

    shown = [p for p in store.by_status("grounded", "proposed", "corrected")
             if p.status != "corrected"]
    if only_operative:
        shown = [p for p in shown
                 if p.status == "grounded" or p.verdict == "operative"]
    if not shown:
        return {"ops": 0, "shown": 0, "corrections": []}

    budget = len(shown) if k is None else min(k, len(shown))
    profile = "\n".join(f"- {x}" for x in persona.checklist)
    for key, val in persona.private_meanings.items():
        profile += f'\n- "{key}"는 {val}라는 뜻이다'
    numbered = "\n".join(
        f"{i}. [{'확인' if p.is_grounded else '추측'}] {p.text}"
        for i, p in enumerate(shown))

    r = client.messages.create(
        model=model, max_tokens=900,
        messages=[{"role": "user", "content": _PANEL_PROMPT.format(
            profile=profile, items=numbered, k=budget)}],
        extra_body={"temperature": 0.0},
    )
    METER.record("panel.review", model, r)
    raw = r.content[0].text or ""
    m = re.search(r"\[.*\]", raw, re.S)
    ops = 0
    corrections: list[tuple[str, str]] = []
    if m:
        try:
            for d in json.loads(m.group()):
                i = int(d.get("i", -1))
                if not (0 <= i < len(shown)) or i >= budget:
                    continue
                p = shown[i]
                op = d.get("op", "keep")
                if op == "ack" and not p.is_grounded:
                    store.ground(p.id, Evidence(type="ui_ack", turn=turn,
                                                pointer=f"op{turn}_{i}"))
                    ops += 1
                elif op == "edit" and d.get("text"):
                    new = str(d["text"]).strip()
                    if new and new != p.text:
                        corrections.append((p.id, new))
                        ops += 1
                elif op == "reject":
                    store.reject(p.id, Evidence(type="ui_reject", turn=turn,
                                                pointer=f"op{turn}_{i}"))
                    ops += 1
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {"ops": ops, "shown": len(shown), "corrections": corrections}
