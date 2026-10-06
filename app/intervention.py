"""인과적 전제 검증 — 개입 실행과 필드 변화량 측정 (제안서 작업 A).

왜 필요한가:
현재 패널에 뜨는 전제는 LLM의 자기보고(Shadow Probe)에서 나온다. 자기보고가
실제로 응답을 만든 원인인지는 알 수 없다. 추출된 전제가 "LLM이 썼을 법한 것"에
그치면, 사용자는 실제로 산출물을 지탱하는 전제를 못 보고 엉뚱한 것을 검토한다.

해결 방식:
전제를 컨텍스트에 명시적으로 넣거나 다른 값으로 덮어쓰고 계획을 다시 생성해,
산출물이 어떻게 변하는지로 "실제로 쓰인 전제인가"를 판정한다.

세 개입:
  대체 do(p→p′)  같은 슬롯의 다른 값을 넣는다 → 작동 전제면 관련 필드가 크게 변함
  단언 do(p)     그 전제를 그대로 넣는다     → 이미 쓰고 있었으면 거의 안 변함
  위약 placebo   무관한 가정을 넣는다        → 잡음 기준선

단언 조건이 설계의 핵심이다. 대체만 보면 "아무 가정이나 넣으면 계획이 흔들린다"는
효과와 구분되지 않는다. 단언했을 때 변하지 않는다는 것이 "이미 그 전제를 쓰고
있었다"는 직접 증거가 된다.

주의: 인과 검증 결과는 "LLM이 이 전제를 쓴다"만 말한다. "사용자와 확인되었다"를
뜻하지 않으므로 **Common Ground 승격의 근거로 쓰면 안 된다.**
"""
from __future__ import annotations

import re
from typing import Any

# 개입 블록. 대화 기록 뒤, 생성 직전에 붙인다.
# prefix caching이 앞부분을 재사용할 수 있도록 맨 끝에 둔다.
ASSUMPTION_BLOCK = """

【작업 가정】
아래 내용을 사실로 전제하고 계획을 작성하세요.
- {text}
"""

# 계획만 생성시키는 전용 지시. 응답 전체를 만들지 않아 비용과 분산이 줄어든다.
PLAN_ONLY_INSTRUCTION = """
지금까지의 대화를 근거로 계획을 제시하세요.
대화체 설명·인사·질문 없이 아래 5섹션 형식만 출력하세요.

## 1. 목표
(우선순위 순, 최대 5개)

## 2. 주차별 일정
주차마다 소제목을 달고 그 아래에 요일×시간블록 표를 두세요.
### 1주차
| 요일 | 오전 | 오후 | 저녁 |
### 2주차
| 요일 | 오전 | 오후 | 저녁 |
(이후 주차도 같은 형식)

## 3. 주차별 마일스톤

## 4. 예산 개요
| 항목 | 금액 |

## 5. 예외 규칙
"""


def build_intervention_prompt(base_prompt: str, assumption: str | None) -> str:
    """시스템 프롬프트 + 계획 전용 지시 + (선택) 작업 가정 블록."""
    parts = [base_prompt.rstrip(), PLAN_ONLY_INSTRUCTION.strip()]
    if assumption and assumption.strip():
        parts.append(ASSUMPTION_BLOCK.format(text=assumption.strip()).strip())
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# A3. 필드 변화량 Δ_f
# --------------------------------------------------------------------------- #

_NUM_PAT = re.compile(r"[\d,]+(?:\.\d+)?")
_TOKEN_PAT = re.compile(r"[가-힣A-Za-z0-9]+")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM_PAT.finditer(text or ""):
        try:
            out.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return out


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_PAT.findall(text or "") if len(t) > 1}


def field_delta(before: str, after: str) -> float:
    """두 필드 내용의 변화량을 0~1로. 0 = 동일, 1 = 완전히 다름.

    필드 유형에 따라 다르게 센다(제안서 A3).
      수치가 있는 필드  상대 차이를 정규화
      그 외 텍스트      1 − 토큰 Jaccard

    자유 텍스트의 의미 동치를 LLM으로 판정하는 단계는 두지 않는다.
    판정자를 넣으면 다시 LLM 판단에 의존하게 되고, 위약 기준선이
    판정자 잡음까지 흡수해야 해서 측정이 흐려진다.
    """
    b = (before or "").strip()
    a = (after or "").strip()
    if b == a:
        return 0.0
    if not b or not a:
        return 1.0

    nb, na = _numbers(b), _numbers(a)
    if nb and na and len(nb) == len(na):
        # 수치형: 상대 차이의 평균
        diffs = []
        for x, y in zip(nb, na):
            scale = max(abs(x), abs(y), 1.0)
            diffs.append(min(abs(x - y) / scale, 1.0))
        num_delta = sum(diffs) / len(diffs)
        # 수치가 같아도 설명이 바뀔 수 있어 토큰 차이를 함께 본다
        tb, ta = _tokens(b), _tokens(a)
        jac = len(tb & ta) / len(tb | ta) if (tb | ta) else 1.0
        return max(num_delta, (1.0 - jac) * 0.5)

    tb, ta = _tokens(b), _tokens(a)
    if not (tb | ta):
        return 0.0
    return 1.0 - len(tb & ta) / len(tb | ta)


def plan_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    """두 계획의 필드별 Δ와 집계.

    반환:
      per_field    {주소: Δ}
      changed      Δ > 0 인 필드 수
      mean_delta   공통 주소에 대한 Δ 평균 (잡음 기준선 비교용 주 지표)
      max_delta    최대 Δ
      appeared     새로 생긴 주소
      vanished     사라진 주소
      jaccard_addr 주소 집합의 Jaccard (구조 안정성)
    """
    b = {f["address"]: f.get("content", "") for f in before}
    a = {f["address"]: f.get("content", "") for f in after}
    common = sorted(set(b) & set(a))

    per_field = {addr: field_delta(b[addr], a[addr]) for addr in common}
    deltas = list(per_field.values())
    addr_union = set(b) | set(a)

    return {
        "per_field": per_field,
        "n_common": len(common),
        "changed": sum(1 for d in deltas if d > 0),
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
        "appeared": sorted(set(a) - set(b)),
        "vanished": sorted(set(b) - set(a)),
        "jaccard_addr": (len(set(b) & set(a)) / len(addr_union)) if addr_union else 1.0,
    }


def section_delta(result: dict[str, Any], prefix: str) -> float:
    """특정 섹션(주소 접두사)에 한정한 평균 Δ.

    전제가 어느 부분에 영향을 주는지 보려면 전체 평균으로는 부족하다.
    예: 예산 전제를 바꾸면 budget.* 만 커지고 schedule.* 는 그대로여야 한다.
    """
    vals = [d for addr, d in result["per_field"].items() if addr.startswith(prefix)]
    return (sum(vals) / len(vals)) if vals else 0.0

# --------------------------------------------------------------------------- #
# 구조화 비교 — 자유 텍스트 환언을 변화로 세지 않는다
#
# A-V3 실측에서 전체 평균 Δ는 schedule 자유 텍스트가 지배했다(112필드 중 76개).
# "알고리즘 학습" 대 "알고리즘 복습"은 토큰 Jaccard로는 Δ=0.67이지만
# 계획의 의미로는 같은 활동이다. 이 환언 잡음이 위약 기준선을 0.4까지 끌어올려
# 실제 전제와의 분리를 지워 버렸다.
#
# 그래서 필드 유형별로 의미 단위를 비교한다.
#   budget.*    파싱된 금액 수치
#   meta.*      계산된 수치
#   schedule.*  활동 범주 (학습/작업/휴식/집필/기타)
#   priority.*  목표 토큰 집합
# --------------------------------------------------------------------------- #

ACTIVITY_CATEGORIES = {
    "study":   ("학습", "공부", "강의", "이론", "개념", "복습", "인강", "수강"),
    "practice": ("문제", "풀이", "코테", "알고리즘", "구현", "실전", "모의"),
    "writing": ("자소서", "작성", "초안", "첨삭", "지원서", "포트폴리오"),
    "work":    ("알바", "근무", "출근", "아르바이트"),
    "rest":    ("휴식", "자유", "예비", "공백", "운동", "수면", "재충전", "없음"),
}


def activity_category(content: str) -> str:
    """일정 셀 내용을 활동 범주로 환원한다. 환언·수식어 차이를 흡수한다."""
    c = (content or "").strip()
    if not c:
        return "empty"
    for cat, words in ACTIVITY_CATEGORIES.items():
        if any(w in c for w in words):
            return cat
    return "other"


def structured_value(field: dict[str, Any]) -> Any:
    """필드를 비교 가능한 의미 단위로 환원한다."""
    addr = field.get("address", "")
    content = field.get("content", "")
    if addr.startswith("budget.") or addr.startswith("meta."):
        nums = _numbers(content)
        return round(nums[0], 2) if nums else None
    if addr.startswith("schedule."):
        return activity_category(content)
    return frozenset(_tokens(content))


def _struct_delta(a: Any, b: Any) -> float:
    if a is None and b is None:
        return 0.0
    if a is None or b is None:
        return 1.0
    if isinstance(a, frozenset) and isinstance(b, frozenset):
        if not (a | b):
            return 0.0
        return 1.0 - len(a & b) / len(a | b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        scale = max(abs(a), abs(b), 1.0)
        return min(abs(a - b) / scale, 1.0)
    return 0.0 if a == b else 1.0


def structured_plan_delta(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> dict[str, Any]:
    """구조화 비교판 plan_delta. 반환 형식은 동일해 교체해 쓸 수 있다."""
    b = {f["address"]: structured_value(f) for f in before}
    a = {f["address"]: structured_value(f) for f in after}
    common = sorted(set(b) & set(a))
    per_field = {addr: _struct_delta(b[addr], a[addr]) for addr in common}
    deltas = list(per_field.values())
    union = set(b) | set(a)
    return {
        "per_field": per_field,
        "n_common": len(common),
        "changed": sum(1 for d in deltas if d > 0),
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "max_delta": max(deltas) if deltas else 0.0,
        "appeared": sorted(set(a) - set(b)),
        "vanished": sorted(set(b) - set(a)),
        "jaccard_addr": (len(set(b) & set(a)) / len(union)) if union else 1.0,
    }
