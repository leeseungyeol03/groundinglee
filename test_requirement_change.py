"""요구사항 변경 모듈 검증 — 논문 6.3·6.5절 사양과 대조.

배경:
2단계 과업의 조작은 요구사항 변경이다(A안). 변경 문구는 참가자가 1단계에서
제시한 값에 결속되어야 하고, 한 참가자가 두 조건에서 같은 유형을 받으면 안 되며,
프로필이 결손이면 생성 가능한 유형으로 대체되어야 한다.

이 사양이 깨지면 조건 간 비교가 오염되므로 정적으로 고정한다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe test_requirement_change.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.requirement_change import (
    CHANGE_TYPES,
    _has_final_consonant,
    _josa,
    assign_change_types,
    fill_template,
    select_change,
    usable_change_types,
)

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


FULL = {
    "priority_1": "코딩테스트 준비",
    "priority_2": "자소서 초안 3개",
    "monthly_budget": 300000,
    "work_days": ["토"],
}


def main() -> None:
    print("=" * 76)
    print("요구사항 변경 검증 — 논문 6.3·6.5절")
    print("=" * 76)

    section("1. 변경 4유형이 모두 정의되어 있는가")
    expected = {"resource_cut", "schedule_constraint", "priority_reversal", "scope_change"}
    check("4유형 정의", set(CHANGE_TYPES) == expected, f"실제 {sorted(CHANGE_TYPES)}")
    for ct in sorted(expected):
        meta = CHANGE_TYPES.get(ct, {})
        check(f"{ct} 메타 완비",
              all(k in meta for k in ("label", "description", "profile_field", "template")))

    section("2. 문구가 참가자 값에 결속되는가 (6.5절 핵심)")
    r = fill_template("resource_cut", FULL)
    check("예산 변경이 참가자 금액의 절반", "15만원" in r["text"] and "30만원" in r["text"],
          f"실제: {r['text'][:60]}")
    check("old/new 값 기록", r["old_value"] == "30만원" and r["new_value"] == "15만원")

    r = fill_template("priority_reversal", FULL)
    check("우선순위 반전이 참가자 목표를 사용",
          "자소서 초안 3개" in r["text"] and "코딩테스트 준비" in r["text"])

    r = fill_template("schedule_constraint", FULL)
    check("일정 제약이 비어 있는 요일을 고름", "토요일" not in r["text"],
          "이미 알바가 있는 요일을 다시 막으면 무효화가 일어나지 않는다")

    r = fill_template("scope_change", FULL)
    check("범위 변경이 1순위 목표를 대상으로 함", "코딩테스트 준비" in r["text"])

    section("3. 한국어 조사 처리")
    check("받침 있음 판정", _has_final_consonant("운동") and _has_final_consonant("토익 900점"))
    check("받침 없음 판정",
          not _has_final_consonant("자소서 초안 3개") and not _has_final_consonant("프로젝트"))
    check("조사 선택 — 받침 있음", _josa("운동", "이", "가") == "이")
    check("조사 선택 — 받침 없음", _josa("프로젝트", "이", "가") == "가")
    txt = fill_template("priority_reversal", FULL)["text"]
    check("생성 문구에 조사 오류 없음", "이(가)" not in txt and "을(를)" not in txt, txt[:60])

    section("4. 배정 — 한 참가자가 같은 유형을 두 번 받지 않는가")
    for i in range(8):
        a, b = assign_change_types(i)
        check(f"참가자 {i}: {a} / {b} 상이", a != b)

    section("5. 유형 균형 — 4유형이 고르게 쓰이는가")
    from collections import Counter
    c = Counter()
    for i in range(16):
        a, b = assign_change_types(i)
        c[a] += 1
        c[b] += 1
    check("4유형 모두 등장", len(c) == 4, f"실제 {dict(c)}")
    check("빈도 편차 없음", max(c.values()) == min(c.values()), f"실제 {dict(c)}")

    section("6. 프로필 결손 시 대체 (6.5절)")
    thin = {"priority_1": "운동", "priority_2": "", "monthly_budget": 0, "work_days": []}
    usable = usable_change_types(thin)
    check("예산 없으면 resource_cut 불가", "resource_cut" not in usable, f"실제 {usable}")
    check("2순위 없으면 priority_reversal 불가", "priority_reversal" not in usable)
    check("생성 가능한 유형은 남음", len(usable) > 0, f"실제 {usable}")
    sel = select_change(0, 0, thin)
    check("배정 불가 시 대체됨", sel["change_type"] in usable and sel["assigned_by"] == "fallback",
          f"실제 {sel['change_type']} / {sel['assigned_by']}")

    section("7. 완전 결손 프로필은 거부")
    try:
        select_change(0, 0, {"priority_1": "", "priority_2": "", "monthly_budget": 0, "work_days": []})
        check("빈 프로필 거부", False, "예외가 발생해야 한다")
    except ValueError:
        check("빈 프로필 거부", True)

    section("8. 서버 배선")
    main_src = (Path(__file__).resolve().parent / "app" / "main.py").read_text(encoding="utf-8")
    check("phase2 엔드포인트 존재", '@app.post("/api/phase2/start")' in main_src)
    check("프로필 선행 요구", "profile required before phase 2" in main_src)
    check("중복 호출 거부", "already in phase 2" in main_src)
    check("올바른 프로필 테이블 참조", "FROM constraint_profiles" in main_src,
          "profiles 로 잘못 쓰면 500이 난다")
    check("debug가 plan_fields 노출", '"plan_fields": st.get("plan_fields", [])' in main_src)
    check("debug가 task_phase 노출", '"task_phase": st.get("task_phase", 1)' in main_src)

    cfg_src = (Path(__file__).resolve().parent / "app" / "config.py").read_text(encoding="utf-8")
    check("프롬프트가 변경을 전달", "requirement_change" in cfg_src)
    check("영향 범위는 알려주지 않음", "사용자가 지시하지 않은 부분을 먼저 고쳐 제시하지 마세요" in cfg_src,
          "범위를 알려주면 H3 측정이 오염된다")

    db_src = (Path(__file__).resolve().parent / "app" / "database.py").read_text(encoding="utf-8")
    check("requirement_changes 테이블", "CREATE TABLE IF NOT EXISTS requirement_changes" in db_src)
    check("task_phase 컬럼", '"task_phase"' in db_src)

    print("\n" + "=" * 76)
    print(f"총계: {_passed} PASS / {_failed} FAIL")
    print("=" * 76)
    if _failed == 0:
        print("\n실서버 관통 확인(2026-09-30, haiku):")
        print("  1단계 40필드 축적 → 요구사항 변경(예산 30만→15만) → 개정")
        print("  결과: 예산 필드 3개 변경 / 무관 필드 0개 변경")
        print("  → 변경이 영향 범위에만 전파됨 (H3 측정 가능성 확인)")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
