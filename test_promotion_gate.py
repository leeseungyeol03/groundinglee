"""CG 승격 게이트 회귀 테스트.

배경: 이전 승격 규칙은 "핵심 키워드 2개 이상이 F_t 항목에 부분 문자열로 포함되면 CG 이동"
이었다. 방학 계획 도메인에서 "방학"·"기간"·"계획" 같은 일반어가 거의 모든 F_t 항목에
등장하므로 사실상 무조건 승격으로 작동했고, 시드 코퍼스에서 CG 7개 중 4개가
positive evidence 없이 올라갔다(false common ground).

이 테스트는 그 실패 사례를 고정해 두어 회귀를 막는다.
아래 케이스는 seed_corpus_output.txt의 seed_job_01 / seed_rest_01 턴2 상태에서 가져왔다.

실행:
  python test_promotion_gate.py      # 종료코드 0 = 전부 통과
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.graph import (  # noqa: E402
    _cg_reject_reason,
    _content_tokens,
    _enforce_classification_rules,
    _support_ratio,
)

# ── seed_job_01 턴2 시점의 F_t (사용자가 실제로 말한 것만) ──
# 사용자 발화: T1 "이번 방학엔 취업 준비에 집중하려고. 코딩테스트랑 자소서 준비를 하고 싶어."
#            T2 "개발 직무로 갈 거고, 코테는 파이썬으로. 자소서는 아직 기업을 안 정했어."
F_JOB = [
    {"premise": "사용자의 핵심 목표는 방학 기간 내 코딩테스트 실력 향상과 자기소개서 작성 준비를 병행하는 것이다",
     "evidence_type": ["direct_statement"]},
    {"premise": "두 준비 항목(코딩테스트, 자소서)은 단순 나열이 아니라 동시에 병행해야 하는 구조적 요구이다",
     "evidence_type": ["direct_statement"]},
    {"premise": "사용자는 개발 직무로 지원한다", "evidence_type": ["direct_statement"]},
    {"premise": "코딩테스트는 파이썬으로 준비한다", "evidence_type": ["direct_statement"]},
    {"premise": "자소서의 지원 기업은 아직 정하지 않았다", "evidence_type": ["direct_statement"]},
]

# seed_rest_01 턴2: "취미는 그림이랑 요리 정도. 여행은 부담 없이 국내로."
F_REST = [
    {"premise": "사용자의 핵심 목표는 이번 방학 동안 여행과 취미 활동을 통해 충분한 휴식을 취하는 것이다",
     "evidence_type": ["direct_statement"]},
    {"premise": "취미는 그림과 요리이다", "evidence_type": ["direct_statement"]},
    {"premise": "여행은 국내로만 계획하며 해외여행은 선택지에서 제외된다", "evidence_type": ["direct_statement"]},
    {"premise": "여행 및 일정은 부담 없고 가벼운 수준이어야 한다", "evidence_type": ["direct_statement"]},
]

# (설명, content, F, 기대 기각사유 prefix)  None = CG 유지되어야 함
CASES: list[tuple[str, str, list[dict], str | None]] = [
    ("LLM이 물은 방학 기간 예시",
     "방학 기간: 예) 7월 초 ~ 8월 말, 약 8주 등", F_JOB, "unresolved_option"),
    ("LLM이 물은 여행 스타일 선택지",
     "여행 스타일 - 국내 또는 해외 여부", F_REST, "unresolved_option"),
    ("자기선언 가정",
     "방학 기간의 구체적 길이가 명시되지 않았으므로, 일반적인 대학 방학 기간(4~8주)을 암묵적으로 전제한다",
     F_JOB, "self_assumed"),
    ("구조적 결정 (사용자 승인 없음)",
     "응답은 코딩테스트 준비와 자소서 준비를 각각 구분된 섹션으로 제시해야 한다", F_JOB, "unsupported"),
    ("LLM 독자 용어 해석",
     "방학이라는 기간은 무한정이 아닌 유한한 시간 자원으로, 우선순위와 일정 배분이 필요한 제약 조건이다",
     F_JOB, "unsupported"),
    ("LLM이 물은 코테 수준 선택지",
     "코딩테스트 수준 - 입문: 기초 문법은 알지만 알고리즘 문제 경험 적음", F_JOB, "unsupported"),

    ("[정당] 사용자가 실제 말한 목표를 반영",
     "사용자는 방학 기간 내에 파이썬 코딩테스트 역량 향상과 개발 직무 자소서 작성 준비를 동시에 진행하려 한다",
     F_JOB, None),
    ("[정당] 사용자가 실제 말한 제약",
     "여행은 국내로만 계획하며 해외여행은 선택지에서 제외된다", F_REST, None),
    ("[정당] 사용자가 실제 말한 취미",
     "사용자의 취미 활동은 그림과 요리이다", F_REST, None),
]


def _f_tokens(F: list[dict]) -> set[str]:
    out: set[str] = set()
    for item in F:
        out |= _content_tokens(item["premise"])
    return out


def run_cases() -> tuple[int, int]:
    print("=" * 78)
    print("CG 승격 게이트 — 시드 실패 사례 회귀")
    print("=" * 78)
    ok = fail = 0
    for desc, content, F, expect in CASES:
        ft = _f_tokens(F)
        reason = _cg_reject_reason(content, ft)
        ratio = _support_ratio(content, ft)
        passed = (reason is None) if expect is None else (reason or "").startswith(expect)
        ok, fail = (ok + 1, fail) if passed else (ok, fail + 1)
        print(f"\n{'PASS' if passed else '**FAIL**'}  {desc}")
        print(f"   기대={expect or '유지(CG)'}  실제={reason or '유지(CG)'}  지지비율={ratio:.2f}")
    return ok, fail


def run_integration() -> tuple[int, int]:
    """seed_job_01 턴2의 CG 7개 전체를 게이트에 통과시킨다."""
    print("\n" + "=" * 78)
    print("통합 — seed_job_01 턴2 CG 7개")
    print("=" * 78)
    cg_before = [
        {"id": "cg1", "content": F_JOB[0]["premise"]},   # F_t 직접 복사본 (룰2로 제거)
        {"id": "cg2", "content": F_JOB[1]["premise"]},   # F_t 직접 복사본 (룰2로 제거)
        {"id": "cg3", "content": "방학 기간: 예) 7월 초 ~ 8월 말, 약 8주 등"},
        {"id": "cg4", "content": "방학이라는 기간은 무한정이 아닌 유한한 시간 자원으로, 우선순위와 일정 배분이 필요한 제약 조건이다"},
        {"id": "cg5", "content": "응답은 코딩테스트 준비와 자소서 준비를 각각 구분된 섹션으로 제시해야 한다"},
        {"id": "cg6", "content": "방학 기간의 구체적 길이가 명시되지 않았으므로, 일반적인 대학 방학 기간(4~8주)을 암묵적으로 전제한다"},
        {"id": "cg7", "content": "사용자는 방학 기간 내에 파이썬 코딩테스트 역량 향상과 개발 직무 자소서 작성 준비를 동시에 진행하려 한다"},
    ]
    kept, lg = _enforce_classification_rules(F_JOB, cg_before, [], None)
    print(f"\nCG {len(cg_before)}개 → 유지 {len(kept)} / 강등 {len(lg)} / F_t 복사본 제거 2")
    for i in kept:
        print(f"  유지 · {i['content'][:64]}")
    for i in lg:
        print(f"  강등 · [{i.get('type')}] {i['content'][:58]}")

    ok = fail = 0
    if len(kept) == 1 and kept[0]["id"] == "cg7":
        ok += 1
        print("\nPASS  정당한 CG 1개만 유지됨")
    else:
        fail += 1
        print(f"\n**FAIL**  유지 항목이 예상과 다름: {[i['id'] for i in kept]}")
    if len(lg) == 4:
        ok += 1
        print("PASS  false CG 4개 전부 강등됨")
    else:
        fail += 1
        print(f"**FAIL**  강등 수가 예상과 다름: {len(lg)}")
    return ok, fail


def run_edge_cases() -> tuple[int, int]:
    print("\n" + "=" * 78)
    print("경계 조건")
    print("=" * 78)
    ok = fail = 0
    checks = [
        ("빈 입력", ([], [], []), lambda c, l: c == [] and l == []),
        ("F_t가 비면 근거 없음 → 전부 강등",
         ([], [{"id": "x", "content": "아무 내용이든"}], []), lambda c, l: len(c) == 0 and len(l) == 1),
        ("'용어 해석' 항목은 무조건 강등",
         (F_JOB, [{"id": "y", "content": "용어 해석: 방학은 자유 시간이다"}], []),
         lambda c, l: len(c) == 0 and l and l[0]["type"] == "용어 해석"),
    ]
    for name, (F, cg, lg_in), pred in checks:
        c, l = _enforce_classification_rules(F, cg, lg_in, None)
        passed = bool(pred(c, l))
        ok, fail = (ok + 1, fail) if passed else (ok, fail + 1)
        print(f"{'PASS' if passed else '**FAIL**'}  {name}")
    return ok, fail


def main() -> int:
    o1, f1 = run_cases()
    o2, f2 = run_integration()
    o3, f3 = run_edge_cases()
    ok, fail = o1 + o2 + o3, f1 + f2 + f3
    print("\n" + "=" * 78)
    print(f"총계: {ok} PASS / {fail} FAIL")
    print("=" * 78)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
