"""M4. 컨텍스트 구성기 — 상태에서 결정적으로 프롬프트를 만든다.

왜 필요한가:
지금 방식은 교정을 "고쳐줘"라는 지시문으로 시스템 프롬프트에 덧붙인다. 그러면
두 가지가 깨진다. 첫째, 같은 상태인데도 프롬프트가 매번 달라질 수 있다. 둘째,
긴 대화에서는 옛 전제가 그대로 적힌 과거 응답이 지시문을 이긴다.

M4는 **상태 S_t가 생성의 1차 입력**이고 대화 기록은 보조 자료라는 구조로 바꾼다.
상태가 같으면 프롬프트가 바이트 단위로 같아야 한다(문서 완료 기준).

대화 기록 처리 세 변형을 벤치마크에서 비교한다(feedback2.md M4).
    H-full   원문 그대로. 자연스럽지만 옛 전제가 새어 들어올 위험
    H-mark   정정된 세그먼트에 표시를 붙임
    H-state  상태 블록 + 최근 1턴만. 가장 깨끗하지만 뉘앙스 손실
"""
from __future__ import annotations

from typing import Any, Iterable

from app.state_store import Premise, StateStore

HISTORY_MODES = ("H-full", "H-mark", "H-state")

_FIXED_INSTRUCTION = (
    "당신은 사용자의 계획 수립을 돕는 AI 파트너입니다.\n"
    "아래 【현재 상태】는 지금까지 확인된 내용과 아직 확인되지 않은 내용을 구분한 것입니다.\n"
    "상태를 기준으로 답하고, 금지된 항목은 사용하지 마세요."
)

_SECTION_TITLES = {
    "grounded": "확인된 전제 (사용자 증거로 확립됨)",
    "proposed": "미확인 전제 (사용할 수는 있으나 사실로 단정하지 말 것)",
    "corrected": "정정된 전제 (옛 값 사용 금지)",
    "rejected": "거부된 전제 (사용 금지)",
}


def _fmt_premise(p: Premise) -> str:
    return f"- {p.text}"


def _fmt_corrected(old: Premise, new: Premise) -> str:
    return f"- {new.text}   (옛 값 사용 금지: {old.text})"


def build_state_block(store: StateStore) -> str:
    """상태 블록. 구역 순서와 항목 순서를 모두 고정한다.

    순서를 고정하지 않으면 같은 상태에서도 프롬프트가 달라져
    조건 간 비교가 오염된다. 정렬 기준은 id다(state_store.by_status).
    """
    lines: list[str] = ["【현재 상태】"]

    grounded = [p for p in store.by_status("grounded")]
    lines.append(f"\n[{_SECTION_TITLES['grounded']}]")
    lines.extend(_fmt_premise(p) for p in grounded) if grounded else lines.append("- (없음)")

    proposed = [p for p in store.by_status("proposed")]
    lines.append(f"\n[{_SECTION_TITLES['proposed']}]")
    if proposed:
        lines.extend(_fmt_premise(p) for p in proposed)
    else:
        lines.append("- (없음)")

    pairs = store.superseded_pairs()
    lines.append(f"\n[{_SECTION_TITLES['corrected']}]")
    if pairs:
        lines.extend(_fmt_corrected(o, n) for o, n in pairs)
    else:
        lines.append("- (없음)")

    rejected = [p for p in store.by_status("rejected")]
    lines.append(f"\n[{_SECTION_TITLES['rejected']}]")
    if rejected:
        lines.extend(_fmt_premise(p) for p in rejected)
    else:
        lines.append("- (없음)")

    flagged = sorted((p for p in store.premises.values() if p.flagged_review),
                     key=lambda p: p.id)
    if flagged:
        lines.append("\n[재검토 대상 (상위 전제가 바뀌어 유효성 불확실)]")
        lines.extend(_fmt_premise(p) for p in flagged)

    return "\n".join(lines)


def _mark_superseded(
    text: str,
    store: StateStore,
    superseded: list[tuple[str, str]],
) -> str:
    """H-mark: 과거 응답에서 더 이상 유효하지 않은 부분에 정정 표시를 붙인다.

    superseded는 (세그먼트 원문, 대체 내용) 쌍이고, M5의 "영향 찾기"가
    산출한다. 전제 문구를 과거 응답에서 문자열로 찾는 방식은 쓰지 않는다.
    전제는 "예산 30만원은 강의와 교재에 나눈다" 같은 명제인데 응답은
    "예산 30만원 기준으로" 처럼 다르게 쓰여 거의 안 걸리기 때문이다.

    전제 문구가 우연히 그대로 들어 있는 경우는 보조로 함께 잡는다.
    """
    out = text
    for seg_text, note in superseded:
        if seg_text and seg_text in out:
            out = out.replace(seg_text, f"{seg_text} [정정됨: 이후 '{note}'로 변경]")
    for old, new in store.superseded_pairs():
        if old.text and old.text in out and "[정정됨" not in out:
            out = out.replace(old.text, f"{old.text} [정정됨: 이후 '{new.text}'로 변경]")
    return out


def build_history_block(
    history: list[dict[str, Any]],
    store: StateStore,
    mode: str = "H-mark",
    superseded: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    """대화 기록을 변형에 따라 가공해 메시지 목록으로 반환한다."""
    if mode not in HISTORY_MODES:
        raise ValueError(f"알 수 없는 기록 변형: {mode}")

    msgs = [{"role": m["role"], "content": m["content"]} for m in history]
    if mode == "H-full":
        return msgs
    if mode == "H-mark":
        sup = sorted(superseded or [])
        return [{"role": m["role"],
                 "content": _mark_superseded(m["content"], store, sup)
                 if m["role"] == "assistant" else m["content"]}
                for m in msgs]
    # H-state: 최근 1턴만 남긴다
    return msgs[-1:] if msgs else []


def build_context(
    store: StateStore,
    history: list[dict[str, Any]],
    user_turn: str,
    mode: str = "H-mark",
    task_instruction: str = "",
    superseded: list[tuple[str, str]] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """(시스템 프롬프트, 메시지 목록)을 만든다.

    순서 고정: 고정 지시문 → 과업 지시 → 상태 블록 → (메시지로) 대화 기록 → 이번 발화

    상태 블록을 시스템 쪽에 두는 이유는, 대화 기록 뒤에 두면 긴 기록에서
    옛 전제가 상태 블록을 이기는 문제가 그대로 남기 때문이다.
    """
    parts = [_FIXED_INSTRUCTION]
    if task_instruction.strip():
        parts.append(task_instruction.strip())
    parts.append(build_state_block(store))
    system = "\n\n".join(parts)

    msgs = build_history_block(history, store, mode, superseded)
    msgs.append({"role": "user", "content": user_turn})
    return system, msgs


def context_fingerprint(system: str, msgs: list[dict[str, str]]) -> str:
    """프롬프트의 바이트 지문. 같은 상태면 같아야 한다(완료 기준 검사용)."""
    import hashlib

    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    for m in msgs:
        h.update(b"\x00")
        h.update(m["role"].encode("utf-8"))
        h.update(b"\x01")
        h.update(m["content"].encode("utf-8"))
    return h.hexdigest()
