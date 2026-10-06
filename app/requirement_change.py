"""요구사항 변경 — 2단계 과업의 조작 (논문 6.5절).

배경:
이전 설계는 연구자가 잘못된 전제를 프롬프트로 주입하고 그 발현을 기다렸다.
그 방식은 두 문제를 낳았다. 검증하려는 문제를 연구자가 직접 만들어 넣었다는
비판을 피할 수 없었고, 주입 발현이 확률적이어서 종속변수가 조건부가 되었다.

본 모듈은 주입 대신 요구사항 변경을 사용한다. 참가자가 1단계에서 직접 쌓아
올린 계획의 전제 조건 하나를 바꾸어, 그전까지 무해했던 전제들이 동시에
무효가 되도록 만든다. 변경은 결정적으로 발생하므로 모든 참가자가 동일한 수의
개입 기회를 갖는다.

핵심 설계:
변경 문구는 사전에 작성하되 **참가자가 1단계에서 실제로 제시한 값을 채워 넣는다.**
예컨대 참가자가 예산을 30만원이라 밝혔다면 문구는 15만원을 명시한다.
이렇게 하면 형식과 강도는 참가자 간에 표준화되면서도 내용은 각자의 계획에
실제로 결속된다.

주의: 이 모듈은 파트너 LLM의 프롬프트를 조작하지 않는다. 변경은 과업 화면에
제시되는 사용자 대면 지시문이며, 참가자가 그것을 어떻게 대화·인터페이스로
반영하는지가 측정 대상이다.
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────
# 변경 4유형 (논문 6.5절)
#
# 각 유형은 참가자 프로필의 서로 다른 필드에 결속되며, 무효화하는 전제의
# 성격이 다르다. 한 참가자는 두 조건에서 서로 다른 유형을 받는다.
# ─────────────────────────────────────────────────────────────

CHANGE_TYPES: dict[str, dict[str, Any]] = {
    "resource_cut": {
        "label": "자원 축소",
        "description": "예산이나 가용 시간 같은 양적 제약을 줄인다",
        "profile_field": "monthly_budget",
        "template": (
            "사정이 생겨 이번 방학에 쓸 수 있는 예산이 {new_value}으로 줄었습니다. "
            "앞서 말씀하신 {old_value}의 절반입니다. "
            "바뀐 예산에 맞게 계획을 수정해 주세요."
        ),
    },
    "schedule_constraint": {
        "label": "일정 제약 추가",
        "description": "비어 있던 시간대에 고정 일정이 생긴다",
        "profile_field": "work_days",
        "template": (
            "방학 중에 {new_value}에 정기적인 일정이 새로 생겼습니다. "
            "그 시간에는 다른 활동을 할 수 없습니다. "
            "이 일정을 반영해 계획을 수정해 주세요."
        ),
    },
    "priority_reversal": {
        "label": "우선순위 반전",
        "description": "후순위였던 목표가 최우선이 된다",
        "profile_field": "priority_2",
        "template": (
            "상황이 바뀌어 {new_value}{josa_iga} 이번 방학의 가장 중요한 목표가 되었습니다. "
            "앞서 1순위로 두셨던 {old_value}보다 우선합니다. "
            "바뀐 우선순위에 맞게 계획을 수정해 주세요."
        ),
    },
    "scope_change": {
        "label": "범위 변경",
        "description": "목표 자체의 대상이나 형태가 바뀐다",
        "profile_field": "priority_1",
        "template": (
            "{old_value}의 목표가 바뀌었습니다. {new_value} "
            "바뀐 목표에 맞게 계획을 수정해 주세요."
        ),
    },
}

# 조건 배정: 참가자 index로 두 조건에 서로 다른 유형을 배정한다.
# 4유형을 2조건에 배정하므로 6가지 조합이 가능하나, 유형 간 균형을 위해
# 인접하지 않은 쌍을 순환 배정한다.
_ASSIGNMENT_PAIRS = [
    ("resource_cut", "priority_reversal"),
    ("schedule_constraint", "scope_change"),
    ("priority_reversal", "resource_cut"),
    ("scope_change", "schedule_constraint"),
]


def assign_change_types(participant_index: int) -> tuple[str, str]:
    """참가자에게 두 조건의 변경 유형을 배정한다.

    반환: (첫 번째 조건의 유형, 두 번째 조건의 유형)
    동일 참가자가 같은 유형을 두 번 받지 않는다.
    """
    return _ASSIGNMENT_PAIRS[participant_index % len(_ASSIGNMENT_PAIRS)]


def _has_final_consonant(word: str) -> bool:
    """마지막 글자에 받침이 있는가. 숫자로 끝나면 드음으로 판단한다."""
    word = (word or "").strip()
    if not word:
        return False
    last = word[-1]
    if last.isdigit():
        # 0,1,3,6,7,8 드음에 받침 있음 (영·일·삼·육·칠·팔)
        return last in "0136788"
    if "\uac00" <= last <= "\ud7a3":
        return (ord(last) - 0xAC00) % 28 != 0
    # 영문·기타는 받침 없는 것으로 취급
    return False


def _josa(word: str, with_final: str, without_final: str) -> str:
    """받침 여부에 따라 조사를 고른다. 예: _josa(w, "이", "가")"""
    return with_final if _has_final_consonant(word) else without_final


def _fmt_won(amount: int) -> str:
    """금액을 한국어 표기로. 300000 → '30만원'"""
    if amount >= 10000 and amount % 10000 == 0:
        return f"{amount // 10000}만원"
    if amount >= 10000:
        return f"{amount / 10000:.1f}만원".replace(".0만원", "만원")
    return f"{amount:,}원"


_FREE_DAY_ORDER = ["토", "일", "금", "수", "월", "목", "화"]


def _pick_free_slot(profile: dict) -> str:
    """프로필에서 비어 있는 요일을 골라 새 고정 일정을 넣을 자리를 정한다.

    work_days에 없는 요일 중 주말을 우선한다. 전부 차 있으면 평일 저녁을 쓴다.
    """
    busy = set(profile.get("work_days") or [])
    for day in _FREE_DAY_ORDER:
        if day not in busy:
            return f"매주 {day}요일 오후"
    return "매주 수요일 저녁"


def fill_template(change_type: str, profile: dict) -> dict[str, Any]:
    """변경 유형과 참가자 프로필로 실제 전달 문구를 만든다.

    반환 dict:
      change_type   유형 키
      label         한국어 라벨
      text          참가자에게 제시될 문구
      old_value     변경 전 값 (정답 판정 시 참조)
      new_value     변경 후 값
      profile_field 결속된 프로필 필드
    """
    if change_type not in CHANGE_TYPES:
        raise ValueError(f"알 수 없는 변경 유형: {change_type}")

    meta = CHANGE_TYPES[change_type]
    old_value = ""
    new_value = ""

    if change_type == "resource_cut":
        budget = int(profile.get("monthly_budget") or 0)
        if budget <= 0:
            # 예산을 밝히지 않았으면 이 유형을 쓸 수 없다
            raise ValueError("resource_cut: 프로필에 예산이 없다")
        old_value = _fmt_won(budget)
        new_value = _fmt_won(budget // 2)

    elif change_type == "schedule_constraint":
        # 일정 제약을 추가하려면 참가자가 무엇을 하려는지를 알아야 한다.
        # 목표가 없으면 새 일정이 어떤 전제도 무효화하지 못해 조작이 성립하지 않는다.
        if not (profile.get("priority_1") or "").strip():
            raise ValueError("schedule_constraint: 최소한 우선순위 1이 필요하다")
        old_value = ", ".join(profile.get("work_days") or []) or "기존 일정"
        new_value = _pick_free_slot(profile)

    elif change_type == "priority_reversal":
        p1 = (profile.get("priority_1") or "").strip()
        p2 = (profile.get("priority_2") or "").strip()
        if not p1 or not p2:
            raise ValueError("priority_reversal: 우선순위 1·2가 모두 필요하다")
        old_value = p1
        new_value = p2

    elif change_type == "scope_change":
        p1 = (profile.get("priority_1") or "").strip()
        if not p1:
            raise ValueError("scope_change: 우선순위 1이 필요하다")
        old_value = p1
        # 범위 변경은 목표의 대상·형태를 바꾸므로 구체 문구를 연구자가 지정한다.
        # 참가자 목표를 그대로 받아 "더 좁은 범위"로 재지정하는 형태를 쓴다.
        new_value = (
            f"이제 {p1} 전체가 아니라 그중 가장 우선순위가 높은 한 가지에만 "
            "집중하기로 했습니다."
        )

    text = meta["template"].format(
        old_value=old_value,
        new_value=new_value,
        josa_iga=_josa(new_value, "이", "가"),
        josa_eun=_josa(new_value, "은", "는"),
        josa_eul=_josa(new_value, "을", "를"),
    )
    return {
        "change_type": change_type,
        "label": meta["label"],
        "text": text,
        "old_value": old_value,
        "new_value": new_value,
        "profile_field": meta["profile_field"],
    }


def usable_change_types(profile: dict) -> list[str]:
    """이 프로필로 실제 생성 가능한 변경 유형 목록.

    프로필 필드가 비어 있으면 해당 유형은 문구를 만들 수 없다.
    배정 시 이 목록을 먼저 확인해야 한다.
    """
    usable = []
    for ct in CHANGE_TYPES:
        try:
            fill_template(ct, profile)
            usable.append(ct)
        except ValueError:
            pass
    return usable


def select_change(participant_index: int, condition_index: int, profile: dict) -> dict[str, Any]:
    """배정표에 따라 변경을 고르되, 프로필로 생성 불가하면 대체한다.

    condition_index: 0 = 첫 번째 조건, 1 = 두 번째 조건
    """
    assigned = assign_change_types(participant_index)[condition_index]
    usable = usable_change_types(profile)
    if not usable:
        raise ValueError("프로필이 비어 있어 어떤 변경도 생성할 수 없다")
    if assigned not in usable:
        # 배정된 유형을 쓸 수 없으면 사용 가능한 것 중 순환 배정으로 대체
        assigned = usable[participant_index % len(usable)]
    result = fill_template(assigned, profile)
    result["assigned_by"] = "table" if assigned == assign_change_types(participant_index)[condition_index] else "fallback"
    return result
