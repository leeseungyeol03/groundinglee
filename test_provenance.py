"""전제 → 계획 필드 귀속(provenance) 회귀 테스트.

검증 대상은 A 설계의 핵심 주장이다:
    사용자가 전제를 교정하면, 그 전제가 만든 계획 부분이 실제로 무효화되고
    다음 프롬프트에 재구성 대상으로 전달된다.

기존 구조에서는 교정이 전제 목록만 바꾸고 계획 내용은 그대로 남았다(누적 미해소).

실행:
  python test_provenance.py       # 종료코드 0 = 전부 통과
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

os.environ["MOCK_LLM"] = "true"

from app.config import assemble_partner_system_prompt  # noqa: E402
from app.graph import (  # noqa: E402
    _mark_dependent_fields_stale,
    correction_node,
    deletion_node,
    plan_attribution_node,
)
from app.plan_parser import merge_plan_fields, parse_plan  # noqa: E402

results: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> None:
    results.append((passed, label))
    print(f"{'PASS' if passed else '**FAIL**'}  {label}" + (f"   ({detail})" if detail else ""))


PLAN_T2 = """바로 초안 드릴게요.

## 1. 목표
1. 컴활 1급 합격
2. 국내 단기 여행

## 2. 주간 루틴

| 요일 | 오전 | 오후 | 저녁 |
|------|------|------|------|
| 월 | 알바 | 컴활 공부 | 휴식 |
| 화 | 컴활 공부 | 알바 | 휴식 |

## 3. 주차별 마일스톤

| 주차 | 핵심 목표 |
|------|----------|
| 1주차 | 컴활 필기 이론 완료 |

## 4. 예산 개요

| 항목 | 내용 |
|------|------|
| 여행 경비 | 20만원 |

## 5. 예외 규칙

- 알바 일정 변경 시 컴활 공부를 다음날로 순연
"""

# 전제: 하나는 사용자가 확인한 것(CG), 둘은 LLM이 도입한 것(LG)
CG = [
    {"id": "cg_alba", "content": "사용자의 알바는 주 3회이다", "first_turn": 1},
]
LG = [
    {"id": "lg_travel", "content": "여행 경비는 20만원 수준으로 가정한다",
     "type": "암묵적 전제", "source_turn": 2},
    {"id": "lg_study", "content": "컴활 공부는 알바가 없는 시간에 배치해야 한다",
     "type": "구조적 결정", "source_turn": 2},
]

print("=" * 78)
print("1. 귀속 — 계획 필드가 전제에 연결되는가")
print("=" * 78)

state = {
    "current_turn": 2,
    "D_t": PLAN_T2,
    "common_ground": CG,
    "llm_ground": LG,
    "plan_fields": [],
    "errors": [],
}
state = plan_attribution_node(state, {"model": "mock"})
fields = state["plan_fields"]

check(len(fields) > 0, f"계획이 필드로 분해됨 ({len(fields)}개)")
attributed = [f for f in fields if f.get("premise_ids")]
check(len(attributed) > 0, f"일부 필드가 전제에 귀속됨 ({len(attributed)}/{len(fields)})")
check(all("status" in f for f in fields), "모든 필드에 status 부여")
check(all(f["status"] == "active" for f in fields), "최초 귀속 시 전부 active")

valid_ids = {"cg_alba", "lg_travel", "lg_study"}
bad = [i for f in fields for i in f.get("premise_ids", []) if i not in valid_ids]
check(not bad, "존재하지 않는 전제 id로 귀속되지 않음", f"환각 {len(bad)}건")
check(all(len(f.get("premise_ids", [])) <= 3 for f in fields), "필드당 전제 3개 이하")

budget = next((f for f in fields if f["address"].startswith("budget.")), None)
check(budget is not None and "lg_travel" in budget.get("premise_ids", []),
      "예산 필드가 여행 경비 전제에 귀속", f"got {budget.get('premise_ids') if budget else None}")

print("\n" + "=" * 78)
print("2. 재귀속 비용 — 변경된 필드만 다시 묻는가")
print("=" * 78)

parsed_again = parse_plan(PLAN_T2, turn=3)
unchanged, to_attr = merge_plan_fields(parsed_again, fields)
check(len(to_attr) == 0, "내용이 같으면 재귀속하지 않음", f"재귀속 대상 {len(to_attr)}개")
check(len(unchanged) == len(fields), "이전 귀속이 보존됨")

PLAN_T3 = PLAN_T2.replace("| 여행 경비 | 20만원 |", "| 여행 경비 | 35만원 |")
parsed_changed = parse_plan(PLAN_T3, turn=3)
unchanged2, to_attr2 = merge_plan_fields(parsed_changed, fields)
# 예산 항목이 바뀜면 파생 필드 meta.budget_total도 함께 바뀜다.
# 파생 필드는 계산 결과이므로 이것이 정상 동작이다.
non_meta2 = [f for f in to_attr2 if f.get("section") != "meta"]
meta2 = [f for f in to_attr2 if f.get("section") == "meta"]
check(len(non_meta2) == 1, "내용이 바뀜 필드만 재귀속 대상", f"got {len(non_meta2)}")
check(any(f["address"] == "meta.budget_total" for f in meta2),
      "예산 변경이 파생 합계에 반영됨", f"meta {[f['address'] for f in meta2]}")
check(non_meta2 and non_meta2[0]["address"].startswith("budget."),
      "바뀐 필드가 예산 항목으로 특정됨")
check(all(not f.get("premise_ids") or f["address"] != to_attr2[0]["address"] for f in unchanged2),
      "변경된 필드의 옛 귀속은 재사용되지 않음")

print("\n" + "=" * 78)
print("3. 교정 전파 — 전제를 고치면 그 전제가 만든 계획이 무효화되는가")
print("=" * 78)

before = [f for f in state["plan_fields"] if "lg_travel" in f.get("premise_ids", [])]
check(len(before) > 0, f"교정 전: lg_travel에 걸린 필드 {len(before)}개")

corr_state = dict(state)
corr_state["target_item_id"] = "lg_travel"
corr_state["correction_text"] = "여행 경비는 10만원 이내로 한다"
corr_state = correction_node(corr_state)

stale = [f for f in corr_state["plan_fields"] if f.get("status") == "stale"]
check(len(stale) == len(before), f"교정된 전제의 필드가 stale로 표시됨 ({len(stale)}개)")
check(all(f.get("stale_reason") == "corrected" for f in stale), "stale 사유가 corrected")

new_cg_id = corr_state["common_ground"][-1]["id"]
check(all(new_cg_id in f["premise_ids"] for f in stale),
      "귀속이 교정된 새 전제로 갈아끼워짐")
check(all("lg_travel" not in f["premise_ids"] for f in stale), "옛 전제 귀속은 제거됨")

untouched = [f for f in corr_state["plan_fields"] if f.get("status") != "stale"]
check(len(untouched) == len(state["plan_fields"]) - len(before),
      "무관한 필드는 건드리지 않음", f"{len(untouched)}개 유지")

print("\n" + "=" * 78)
print("4. 거부 전파 — 거부된 전제의 필드도 무효화되는가")
print("=" * 78)

del_state = dict(state)
del_state["plan_fields"] = [dict(f) for f in state["plan_fields"]]
del_state["target_item_id"] = "lg_study"
del_state = deletion_node(del_state)

stale_del = [f for f in del_state["plan_fields"] if f.get("status") == "stale"]
check(len(stale_del) > 0, f"거부된 전제의 필드가 stale ({len(stale_del)}개)")
check(all(f.get("stale_reason") == "rejected" for f in stale_del), "stale 사유가 rejected")
check(all("lg_study" not in f["premise_ids"] for f in stale_del),
      "거부는 대체 전제 없이 귀속만 제거")

print("\n" + "=" * 78)
print("5. 프롬프트 반영 — stale이 실제로 LLM에 전달되는가")
print("=" * 78)

cfg = {"neutral_system_prompt": "당신은 방학 계획 파트너입니다.", "task_context": "과업: 방학 계획"}
prompt_before = assemble_partner_system_prompt(cfg, [], [], [])
prompt_after = assemble_partner_system_prompt(
    cfg,
    corr_state.get("grounding_corrections", []),
    [],
    [f for f in corr_state["plan_fields"] if f.get("status") == "stale"],
)

check("다시 구성" not in prompt_before, "stale 없으면 재구성 지시 없음")
check("다시 구성" in prompt_after, "stale 있으면 재구성 지시 포함")
check(any(f["address"] in prompt_after for f in stale), "stale 필드 주소가 프롬프트에 명시됨")
check("여행 경비는 10만원 이내로 한다" in prompt_after, "교정 내용도 함께 전달됨")
check(len(prompt_after) > len(prompt_before), "프롬프트가 실제로 확장됨")

print("\n" + "=" * 78)
print("6. 경계 조건")
print("=" * 78)

s = plan_attribution_node({"current_turn": 1, "D_t": "안녕하세요! 방학 기간이 어떻게 되나요?",
                           "common_ground": [], "llm_ground": [], "plan_fields": [], "errors": []},
                          {"model": "mock"})
check(s["plan_fields"] == [], "비계획 응답은 필드를 만들지 않음")

kept = [{"address": "goals[0]", "content": "x", "content_hash": "h", "premise_ids": ["p1"],
         "status": "active", "set_at_turn": 1, "section": "goals"}]
s2 = plan_attribution_node({"current_turn": 2, "D_t": "질문만 있는 응답입니다.",
                            "common_ground": [], "llm_ground": [], "plan_fields": kept, "errors": []},
                           {"model": "mock"})
check(s2["plan_fields"] == kept, "비계획 턴에도 이전 plan_fields 유지")

s3 = plan_attribution_node({"current_turn": 2, "D_t": PLAN_T2, "common_ground": [],
                            "llm_ground": [], "plan_fields": [], "errors": []}, {"model": "mock"})
check(len(s3["plan_fields"]) > 0 and all(not f["premise_ids"] for f in s3["plan_fields"]),
      "전제가 없으면 전부 미귀속으로 남김")

touched = _mark_dependent_fields_stale({"current_turn": 3, "plan_fields": []}, "nope", "corrected")
check(touched == [], "존재하지 않는 전제 id는 무해")

print("\n" + "=" * 78)
ok = sum(1 for p, _ in results if p)
fail = len(results) - ok
print(f"총계: {ok} PASS / {fail} FAIL")
print("=" * 78)
raise SystemExit(1 if fail else 0)
