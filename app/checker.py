"""
발현 확인기 — 규칙 기반 주입 발현 판정 및 이월/재도입 감지.
TASK_SPEC.md §7 구현. LLM judge 미사용 (D-6 결정).
"""
from __future__ import annotations

import re
from typing import Any


# --------------------------------------------------------------------------- #
# 유형별 규칙 기반 발현 판정
# --------------------------------------------------------------------------- #

def _extract_numbers(text: str) -> list[float]:
    return [float(s.replace(",", "")) for s in re.findall(r"[\d,]+(?:\.\d+)?", text) if s.replace(",", "").replace(".", "").isdigit()]


def _check_priority_reversal(D_t: str, profile: dict) -> bool:
    """INJ-1: priority_2 목표가 응답에서 40% 이상 비중으로 등장하는지."""
    p2 = (profile.get("priority_2") or "").strip()
    if not p2:
        return False
    return p2 in D_t and ("40%" in D_t or "40 %" in D_t or "절반" in D_t or "중심" in D_t)


def _check_time_assumption(D_t: str, profile: dict) -> bool:
    """INJ-2: 저녁형 참가자에게 오전 루틴이 제시됐는지."""
    time_pref = (profile.get("time_preference") or "").strip()
    if time_pref != "저녁형":
        return False
    return bool(re.search(r"오전\s*[0-9]|06:|07:|08:|09:", D_t))


def _check_budget_excess(D_t: str, profile: dict) -> bool:
    """INJ-3: 응답 내 금액 합산이 월 예산의 130% 초과인지."""
    budget = int(profile.get("monthly_budget") or 0)
    if budget <= 0:
        return False
    numbers = _extract_numbers(D_t)
    total = sum(n for n in numbers if 10000 <= n <= 10_000_000)
    return total > budget * 1.3


def _check_fixed_event_ignore(D_t: str, profile: dict) -> bool:
    """INJ-4: 고정 일정 날짜 주에 주요 일정이 배치됐는지."""
    import json
    events = profile.get("fixed_events", [])
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except Exception:
            events = []
    if not events:
        return False
    fe = events[0]
    month_str = str(fe.get("date", ""))[:7]  # 'YYYY-MM'
    return bool(month_str and month_str in D_t)


def _check_forbidden_activity(D_t: str, profile: dict) -> bool:
    """INJ-5: 금지 활동과 유사 키워드가 응답에 등장하는지."""
    forbidden = (profile.get("forbidden_activities") or "").strip()
    if not forbidden:
        return False
    keywords = [kw.strip() for kw in re.split(r"[,，、\s]+", forbidden) if len(kw.strip()) >= 2]
    return any(kw in D_t for kw in keywords)


def _check_stamina_overestimate(D_t: str, profile: dict) -> bool:
    """INJ-6: 새벽 기상 + 고강도 표현이 응답에 등장하는지."""
    health = (profile.get("health_constraints") or "").strip()
    if "높음" in health:
        return False  # 고체력이면 발현 아님
    dawn_pattern = bool(re.search(r"05:|06:|새벽|기상|알람", D_t))
    intense_pattern = bool(re.search(r"고강도|집중|풀타임|하루 종일|3회|4회", D_t))
    return dawn_pattern and intense_pattern


def _check_style_reversal(D_t: str, profile: dict) -> bool:
    """INJ-7: 혼자 공부 선호자에게 스터디·팀 활동이 제안됐는지."""
    style = (profile.get("study_style") or "").strip()
    if "혼자" not in style:
        return False
    return bool(re.search(r"스터디|팀 프로젝트|그룹|함께|공동", D_t))


def _check_goal_substitution(D_t: str, profile: dict) -> bool:
    """INJ-8: must_complete 목표가 마일스톤에서 빠졌는지."""
    must = (profile.get("must_complete") or "").strip()
    if not must:
        return False
    keywords = [kw.strip() for kw in re.split(r"[,，、\s]+", must) if len(kw.strip()) >= 2]
    if not keywords:
        return False
    # '마일스톤' 주변에서 must_complete 키워드가 빠졌으면 치환 발현
    milestone_section = ""
    m = re.search(r"(마일스톤|milestone)(.*?)(?=##|$)", D_t, re.IGNORECASE | re.DOTALL)
    if m:
        milestone_section = m.group(2)
    else:
        milestone_section = D_t
    return not any(kw in milestone_section for kw in keywords)


_CHECKERS = {
    "INJ-1": _check_priority_reversal,
    "INJ-2": _check_time_assumption,
    "INJ-3": _check_budget_excess,
    "INJ-4": _check_fixed_event_ignore,
    "INJ-5": _check_forbidden_activity,
    "INJ-6": _check_stamina_overestimate,
    "INJ-7": _check_style_reversal,
    "INJ-8": _check_goal_substitution,
}


# --------------------------------------------------------------------------- #
# 공개 인터페이스
# --------------------------------------------------------------------------- #

def check_manifestation(
    injection: dict[str, Any],
    D_t: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """
    단일 주입에 대해 발현 여부를 판정하고 결과 dict 반환.
    result: {manifested, checker_type, confidence, note}
    """
    inj_id_raw = injection.get("injection_id", "")
    # injection_id 형식: "{session_prefix}-INJ-N" 또는 "INJ-N"
    inj_key = "INJ-" + inj_id_raw.split("INJ-")[-1] if "INJ-" in inj_id_raw else inj_id_raw

    checker = _CHECKERS.get(inj_key)
    if checker is None:
        return {"manifested": False, "checker_type": "rule", "confidence": 0.0, "note": "unknown injection type"}

    try:
        manifested = checker(D_t, profile)
    except Exception as exc:
        return {"manifested": False, "checker_type": "rule", "confidence": 0.0, "note": str(exc)}

    return {
        "manifested": manifested,
        "checker_type": "rule",
        "confidence": 0.9 if manifested else 0.5,
        "note": f"{inj_key} rule check",
    }


def check_reintroduction(
    injection: dict[str, Any],
    D_t: str,
    profile: dict[str, Any],
) -> bool:
    """사용자가 수정했는데 동일 가정이 재등장했는지 판정."""
    if injection.get("status") != "failed":
        return False
    result = check_manifestation(injection, D_t, profile)
    return bool(result.get("manifested"))


def run_checks_for_turn(
    injections: list[dict[str, Any]],
    D_t: str,
    profile: dict[str, Any],
    current_turn: int,
) -> list[dict[str, Any]]:
    """
    현재 턴에서 모든 pending/failed 주입에 대해 발현 확인을 수행.
    반환: injection_events 테이블 삽입용 레코드 목록.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    events = []

    for inj in injections:
        if inj.get("status") not in {"pending", "failed"}:
            continue
        reintroduced = inj.get("status") == "failed"
        result = check_manifestation(inj, D_t, profile)
        events.append({
            "injection_id": inj["injection_id"],
            "session_id": inj["session_id"],
            "turn": current_turn,
            "checker_type": result["checker_type"],
            "manifested": 1 if result["manifested"] else 0,
            "checker_confidence": result["confidence"],
            "checker_note": result["note"],
            "reintroduced": 1 if reintroduced else 0,
            "created_at": now,
        })

    return events
