"""계획 파서 회귀 테스트 — 실제 세션 로그로 검증.

data/sessions.db의 turn_artifacts에 저장된 실제 파트너 LLM 응답을 파싱한다.
DB가 없으면 내장 픽스처만으로 돌아간다(스키마 변형 회귀 방지용).

실행:
  python test_plan_parser.py       # 종료코드 0 = 전부 통과
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.plan_parser import (  # noqa: E402
    content_hash,
    has_plan,
    parse_plan,
    split_sections,
    to_plan_dict,
)

results: list[tuple[bool, str]] = []


def check(passed: bool, label: str, detail: str = "") -> None:
    results.append((passed, label))
    print(f"{'PASS' if passed else '**FAIL**'}  {label}" + (f"   ({detail})" if detail else ""))


# ── 픽스처: 실제 세션 739de2b5 turn 2 응답의 구조를 그대로 축약 ──
FIXTURE = """좋아요! 충분히 파악됐어요. 바로 초안 드릴게요 🗓️

---

## 1. 목표
1. **오픽 목표 등급 달성** (8월 초 응시)
2. **연구 및 논문 활동** 꾸준히 유지
3. **취업 서치 및 준비**

---

## 2. 주간 루틴

| 요일 | 오전 (9~12시) | 오후 (13~18시) | 저녁 (19~22시) |
|------|-------------|--------------|--------------|
| 월 | 연구 | 연구 | 오픽 공부 |
| 화 | 연구 | 연구 | 운동 |
| 토 | 자유 | **데이트** | 데이트 |

---

## 3. 주차별 마일스톤

| 주차 | 기간 | 핵심 목표 |
|------|------|---------|
| 1주차 | 7월 1주 | 루틴 세팅, 오픽 현재 수준 파악 |
| 5주차 | 8월 1주 | **오픽 응시**, 연구 집중 전환 |

---

## 4. 예산 개요

| 항목 | 내용 |
|------|------|
| 수입 | [확인 필요 — 연구비·장학금 등] |
| 오픽 응시료 | 8만원 |

---

## 5. 예외 규칙

- **데이트 일정 충돌 시** → 금요일 저녁으로 이동
- **연구실 급한 일 생길 시** → 해당 저녁 블록을 다음날로 순연

---

몇 가지 여쭤볼게요!

- **운동 종목**이 정해져 있나요?
"""

print("=" * 78)
print("계획 파서 — 픽스처 검증")
print("=" * 78)

sections = split_sections(FIXTURE)
check(len(sections) == 5, "5섹션 모두 인식", f"got {sorted(sections)}")

fields = parse_plan(FIXTURE, turn=2)
by_section: dict[str, list[dict]] = {}
for f in fields:
    by_section.setdefault(f["section"], []).append(f)

check(len(by_section.get("priority", [])) == 3, "목표 3개", f"got {len(by_section.get('priority', []))}")
# 주차 구분 없는 루틴은 반복 일정으로 보고 w1..w4로 전개된다 (3일×3블록×4주)
check(len(by_section.get("schedule", [])) == 36, "일정 3일×3블록×4주=36", f"got {len(by_section.get('schedule', []))}")
check(len(by_section.get("milestone", [])) == 2, "마일스톤 2개", f"got {len(by_section.get('milestone', []))}")
check(len(by_section.get("budget", [])) == 2, "예산 2항목", f"got {len(by_section.get('budget', []))}")
check(len(by_section.get("constraint", [])) == 2, "예외규칙 2개", f"got {len(by_section.get('constraint', []))}")
check(len(by_section.get("meta", [])) >= 2, "파생 메타 필드 생성", f"got {len(by_section.get('meta', []))}")

addrs = {f["address"] for f in fields}
check("priority.1" in addrs, "목표 주소 형식", "priority.1")
check("schedule.w1.mon.am" in addrs, "일정 주소 형식", "schedule.w1.mon.am")
check("schedule.w4.mon.am" in addrs, "반복 루틴이 전 주차로 전개", "schedule.w4.mon.am")
check("schedule.w1.sat.pm" in addrs, "시간 범위 헤더도 매핑됨", "schedule.w1.sat.pm")
check("budget.오픽 응시료" in addrs, "예산은 항목명 주소", "budget.오픽 응시료")

# 마크다운 강조가 제거되는가
date_field = next((f for f in fields if f["address"] == "schedule.w1.sat.pm"), None)
check(date_field is not None and date_field["content"] == "데이트",
      "마크다운 강조 제거", f"got {date_field['content'] if date_field else None}")

# 섹션 뒤 잡담이 필드로 새지 않는가 — 핵심: "운동 종목이 정해져 있나요?"
leaked = [f for f in fields if "운동 종목" in f["content"]]
check(not leaked, "섹션 뒤 질문은 필드로 새지 않음", f"leaked={len(leaked)}")

# 마일스톤: 첫 열은 라벨, 마지막 열이 내용
ms = by_section.get("milestone", [])
check(bool(ms) and ms[0].get("label") == "1주차" and "루틴 세팅" in ms[0]["content"],
      "마일스톤 3열에서 라벨/내용 분리")

# content_hash는 강조·공백 차이를 무시
check(content_hash("**데이트**") == content_hash("데이트"), "content_hash가 강조 무시")
check(content_hash("오픽 공부") != content_hash("운동"), "다른 내용은 다른 해시")

# plan dict 변환 — scoring.py가 기대하는 형태
plan = to_plan_dict(fields)
check(len(plan["goals"]) == 3, "plan.goals 변환")
check(plan["weekly_routine"].get("w1", {}).get("mon", {}).get("eve") == "오픽 공부",
      "plan.weekly_routine 변환")
check(len(plan["exception_rules"]) == 2, "plan.exception_rules 변환")
amounts = {e["item"]: e["amount"] for e in plan["budget"]["expenses"]}
check(amounts.get("오픽 응시료") == 80000, "금액 파싱 (8만원→80000)", f"got {amounts.get('오픽 응시료')}")
check(amounts.get("수입") == 0.0, "'[확인 필요]'는 금액 0")

# has_plan 판별
check(has_plan(FIXTURE), "계획 응답으로 판별")
check(not has_plan("안녕하세요! 몇 가지 여쭤볼게요. 방학 기간이 어떻게 되나요?"),
      "질문만 있는 응답은 계획 아님")
check(not has_plan("## 1. 목표\n- 취업 준비"), "섹션 1개뿐이면 계획 아님")

# ── 실제 DB 로그 검증 ──
print("\n" + "=" * 78)
print("계획 파서 — 실제 세션 로그")
print("=" * 78)

db = ROOT / "data" / "sessions.db"
if not db.exists():
    print("(data/sessions.db 없음 — 로그 검증 건너뜀)")
else:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT session_id, turn, D_t FROM turn_artifacts ORDER BY session_id, turn"
    ).fetchall()
    conn.close()

    plans = [r for r in rows if re.search(r"##\s*2\.\s*주간 루틴", r["D_t"] or "")]
    check(len(plans) > 0, f"5섹션 응답 발견 ({len(plans)}건)")

    all_ok = True
    empty_content = 0
    for r in plans:
        f = parse_plan(r["D_t"], turn=r["turn"])
        secs = {x["section"] for x in f}
        if not has_plan(r["D_t"]) or len(secs) < 4 or len(f) < 10:
            all_ok = False
            print(f"   빈약: {r['session_id'][:8]} t{r['turn']} — 섹션 {len(secs)}, 필드 {len(f)}")
        empty_content += sum(1 for x in f if not x["content"].strip())
    check(all_ok, f"{len(plans)}건 모두 4섹션 이상·10필드 이상 파싱")
    check(empty_content == 0, "빈 content 필드 없음", f"got {empty_content}")

    # 주소 유일성 — provenance 귀속의 전제 조건
    dup_total = 0
    for r in plans:
        f = parse_plan(r["D_t"], turn=r["turn"])
        addrs = [x["address"] for x in f]
        dup_total += len(addrs) - len(set(addrs))
    check(dup_total == 0, "주소가 응답 내에서 유일함", f"중복 {dup_total}건")

    # 비계획 응답은 걸러지는가
    non_plans = [r for r in rows if not re.search(r"##\s*2\.\s*주간 루틴", r["D_t"] or "")]
    false_pos = [r for r in non_plans if has_plan(r["D_t"] or "")]
    check(not false_pos, f"비계획 응답 {len(non_plans)}건 오인 없음", f"오인 {len(false_pos)}건")

print("\n" + "=" * 78)
ok = sum(1 for p, _ in results if p)
fail = len(results) - ok
print(f"총계: {ok} PASS / {fail} FAIL")
print("=" * 78)
raise SystemExit(1 if fail else 0)
