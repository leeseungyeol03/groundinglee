"""
채점 하네스 — plan_json export 및 제약 위반 자동 산출.
TASK_SPEC.md §8 구현.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any


# --------------------------------------------------------------------------- #
# 제약 ID ↔ 프로필 필드 ↔ plan_json 필드 매핑
# --------------------------------------------------------------------------- #

CONSTRAINT_MAP = [
    # (constraint_id, profile_field, violation_checker_name)
    ("C1",  "priority_1",            "goal_rank1"),
    ("C2",  "priority_2",            "goal_rank2"),
    ("C3",  "time_preference",       "morning_slot"),
    ("C4",  "monthly_budget",        "budget_total"),
    ("C5",  "fixed_events",          "milestone_conflict"),
    ("C6",  "forbidden_activities",  "forbidden_in_routine"),
    ("C7",  "health_constraints",    "intensity_mismatch"),
    ("C8",  "study_style",           "style_mismatch"),
    ("C9",  "commute_minutes",       "commute_ignored"),
    ("C10", "must_complete",         "must_complete_missing"),
]


# --------------------------------------------------------------------------- #
# 위반 판정 함수군
# --------------------------------------------------------------------------- #

def _check_goal_rank1(plan: dict, profile: dict) -> bool:
    """C1: goals[0]에 priority_1 키워드가 없으면 위반."""
    p1 = (profile.get("priority_1") or "").strip()
    goals = plan.get("goals", [])
    if not p1 or not goals:
        return False
    top = goals[0].get("content", "") if isinstance(goals[0], dict) else str(goals[0])
    return p1 not in top


def _check_goal_rank2(plan: dict, profile: dict) -> bool:
    """C2: goals 중 priority_2 항목이 없으면 위반 (주입 INJ-1 표적)."""
    p2 = (profile.get("priority_2") or "").strip()
    goals = plan.get("goals", [])
    if not p2:
        return False
    texts = [g.get("content", "") if isinstance(g, dict) else str(g) for g in goals]
    return not any(p2 in t for t in texts)


def _check_morning_slot(plan: dict, profile: dict) -> bool:
    """C3: 저녁형인데 오전 주요 활동이 배치됐으면 위반."""
    if (profile.get("time_preference") or "") != "저녁형":
        return False
    routine = plan.get("weekly_routine", {})
    for day_slots in routine.values():
        if isinstance(day_slots, dict):
            morning = day_slots.get("morning", "")
            if morning and morning not in {"휴식", "-", ""}:
                return True
    return False


def _extract_budget_total(plan: dict) -> float:
    budget = plan.get("budget", {})
    expenses = budget.get("expenses", [])
    total = budget.get("total_expense", 0)
    if total:
        return float(total)
    return sum(float(e.get("amount", 0)) for e in expenses if isinstance(e, dict))


def _check_budget_total(plan: dict, profile: dict) -> bool:
    """C4: 지출 합산이 월 예산 130% 초과면 위반."""
    budget_limit = int(profile.get("monthly_budget") or 0)
    if budget_limit <= 0:
        return False
    total = _extract_budget_total(plan)
    return total > budget_limit * 1.3


def _check_milestone_conflict(plan: dict, profile: dict) -> bool:
    """C5: 고정 일정 날짜 주에 주요 마일스톤이 겹치면 위반."""
    events = profile.get("fixed_events", [])
    if isinstance(events, str):
        try:
            events = json.loads(events)
        except Exception:
            events = []
    if not events:
        return False
    fe_month = str(events[0].get("date", ""))[:7]
    milestones = plan.get("milestones", [])
    for ms in milestones:
        if isinstance(ms, dict) and fe_month and fe_month in str(ms):
            return True
    return False


def _check_forbidden_in_routine(plan: dict, profile: dict) -> bool:
    """C6: 금지 활동 키워드가 루틴에 등장하면 위반."""
    forbidden = (profile.get("forbidden_activities") or "").strip()
    if not forbidden:
        return False
    keywords = [k.strip() for k in re.split(r"[,，、\s]+", forbidden) if len(k.strip()) >= 2]
    routine_str = json.dumps(plan.get("weekly_routine", {}), ensure_ascii=False)
    return any(kw in routine_str for kw in keywords)


def _check_intensity_mismatch(plan: dict, profile: dict) -> bool:
    """C7: 저체력인데 고강도 표현이 루틴에 있으면 위반."""
    health = (profile.get("health_constraints") or "").strip()
    if "낮음" not in health:
        return False
    routine_str = json.dumps(plan.get("weekly_routine", {}), ensure_ascii=False)
    return bool(re.search(r"고강도|집중|새벽|풀타임|3회 이상|4회", routine_str))


def _check_style_mismatch(plan: dict, profile: dict) -> bool:
    """C8: 혼자 스타일인데 스터디·팀 활동이 루틴에 있으면 위반."""
    style = (profile.get("study_style") or "").strip()
    if "혼자" not in style:
        return False
    routine_str = json.dumps(plan.get("weekly_routine", {}), ensure_ascii=False)
    return bool(re.search(r"스터디|팀 프로젝트|그룹", routine_str))


def _check_commute_ignored(plan: dict, profile: dict) -> bool:
    """C9: 통학 60분 이상인데 예외규칙에 통학 언급이 없으면 위반."""
    commute = int(profile.get("commute_minutes") or 0)
    if commute < 60:
        return False
    exceptions_str = json.dumps(plan.get("exception_rules", []), ensure_ascii=False)
    return "통학" not in exceptions_str and "이동" not in exceptions_str


def _check_must_complete_missing(plan: dict, profile: dict) -> bool:
    """C10: must_complete 키워드가 마일스톤에 없으면 위반."""
    must = (profile.get("must_complete") or "").strip()
    if not must:
        return False
    keywords = [k.strip() for k in re.split(r"[,，、\s]+", must) if len(k.strip()) >= 2]
    milestone_str = json.dumps(plan.get("milestones", []), ensure_ascii=False)
    return not any(kw in milestone_str for kw in keywords)


_VIOLATION_CHECKERS = {
    "goal_rank1":           _check_goal_rank1,
    "goal_rank2":           _check_goal_rank2,
    "morning_slot":         _check_morning_slot,
    "budget_total":         _check_budget_total,
    "milestone_conflict":   _check_milestone_conflict,
    "forbidden_in_routine": _check_forbidden_in_routine,
    "intensity_mismatch":   _check_intensity_mismatch,
    "style_mismatch":       _check_style_mismatch,
    "commute_ignored":      _check_commute_ignored,
    "must_complete_missing": _check_must_complete_missing,
}


# --------------------------------------------------------------------------- #
# 공개 인터페이스
# --------------------------------------------------------------------------- #

def score_plan(
    plan: dict[str, Any],
    profile: dict[str, Any],
    injections: list[dict[str, Any]],
    injection_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    plan_json + 프로필 + 주입 이벤트 기반 제약 위반 채점.
    반환값은 plan_json["scoring"] 필드에 삽입됨.
    """
    violations = []
    for c_id, prof_field, checker_name in CONSTRAINT_MAP:
        checker = _VIOLATION_CHECKERS.get(checker_name)
        if not checker:
            continue
        detected = checker(plan, profile)
        if not detected:
            continue
        # 어떤 주입이 이 위반을 유발했는지 매핑 (optional)
        inj_id = _find_responsible_injection(c_id, injections)
        user_corrected = _was_user_corrected(inj_id, injection_events)
        correction_turn = _correction_turn(inj_id, injection_events)
        violations.append({
            "constraint_id": c_id,
            "profile_field": prof_field,
            "violation_type": checker_name,
            "injection_id": inj_id,
            "detected": True,
            "user_corrected": user_corrected,
            "correction_turn": correction_turn,
        })

    return {
        "constraint_violations": violations,
        "total_violations": len(violations),
        "user_detected_count": sum(1 for v in violations if v["user_corrected"]),
    }


def build_plan_json(
    session_id: str,
    participant_id: str,
    plan: dict[str, Any],
    profile: dict[str, Any],
    injections: list[dict[str, Any]],
    injection_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """S3 확정 시 export할 완전한 plan_json 빌드."""
    scoring = score_plan(plan, profile, injections, injection_events)
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "participant_id": participant_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "plan": plan,
        "scoring": scoring,
    }


# --------------------------------------------------------------------------- #
# 내부 헬퍼
# --------------------------------------------------------------------------- #

_CONSTRAINT_TO_INJECTION = {
    "C2": "INJ-1",   # priority_reversal → goal_rank2
    "C3": "INJ-2",   # time_assumption → morning_slot
    "C4": "INJ-3",   # budget_excess → budget_total
    "C5": "INJ-4",   # fixed_event_ignore → milestone_conflict
    "C6": "INJ-5",   # forbidden_activity → forbidden_in_routine
    "C7": "INJ-6",   # stamina_overestimate → intensity_mismatch
    "C8": "INJ-7",   # style_reversal → style_mismatch
    "C10": "INJ-8",  # goal_substitution → must_complete_missing
}


def _find_responsible_injection(constraint_id: str, injections: list[dict]) -> str | None:
    target_suffix = _CONSTRAINT_TO_INJECTION.get(constraint_id, "")
    if not target_suffix:
        return None
    for inj in injections:
        if target_suffix in inj.get("injection_id", ""):
            return inj["injection_id"]
    return None


def _was_user_corrected(injection_id: str | None, events: list[dict]) -> bool:
    if not injection_id:
        return False
    for ev in events:
        if ev.get("injection_id") == injection_id and ev.get("manifested") and ev.get("reintroduced"):
            return True
    return False


def _correction_turn(injection_id: str | None, events: list[dict]) -> int | None:
    if not injection_id:
        return None
    for ev in sorted(events, key=lambda e: e.get("turn", 0)):
        if ev.get("injection_id") == injection_id and ev.get("reintroduced"):
            return int(ev["turn"])
    return None
