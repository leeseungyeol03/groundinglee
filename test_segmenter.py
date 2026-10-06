"""M0 분절기·버전 정렬 검증 — feedback2.md의 완료 기준 대조.

완료 기준(문서 M0):
  - 손으로 만든 정렬 정답 50쌍에서 정렬 정확도 ≥ .90
  - 같은 입력이면 출력이 같아야 한다(결정성)

왜 이걸 고정하는가:
M0은 도메인 스키마를 대신하는 유일한 구조다. 여기가 흔들리면 P2(사용 추적)와
P3(수정 전파)가 전부 무너진다. 이전 설계의 결정적 주소(schedule.w1.mon.am)를
포기한 대가로 정렬 신뢰도를 수치로 보증해야 한다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe test_segmenter.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.segmenter import (
    Segment,
    align,
    changed_segments,
    lexical_similarity,
    segment,
)

_passed = 0
_failed = 0


def check(cond: bool, name: str, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"PASS  {name}")
    else:
        _failed += 1
        print(f"FAIL  {name}" + (f"   ({detail})" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


SAMPLE = """# 방학 계획

조건을 정리하면 7월 한 달입니다. 평일 5시간을 쓸 수 있습니다.

## 1. 목표
1. 프로그래머스 중급 100문제
2. 자소서 3개 기업

## 2. 주차별 일정
### 1주차
| 요일 | 오전 | 오후 |
|---|---|---|
| 월 | 알고리즘 학습 | 문제풀이 |
| 화 | 복습 | 자소서 작성 |

## 4. 예산
- 인터넷 강의: 15만원
- 교재: 5만원
"""


def _mk_pair(kind: str, base: str) -> tuple[str, str, str]:
    """(이전 텍스트, 현재 텍스트, 기대 결과) 생성.

    기대 결과: "match_same" | "match_changed" | "replace"
    """
    if kind == "same":
        return base, base, "match_same"
    if kind == "paraphrase":
        # 어미·수식어만 바꾼다. 같은 내용이므로 매칭되되 changed여야 한다.
        return base, base + " 진행", "match_changed"
    if kind == "minor":
        return base, base.replace("학습", "공부") if "학습" in base else base + "요", "match_changed"
    if kind == "replace":
        return base, "완전히 다른 내용으로 대체된 항목입니다", "replace"
    raise ValueError(kind)


_BASES = [
    "프로그래머스 중급 100문제를 푼다",
    "자소서는 3개 기업 초안까지 쓴다",
    "월 | 알고리즘 학습 | 문제풀이",
    "화 | 복습 | 자소서 작성",
    "인터넷 강의에 15만원을 쓴다",
    "교재 구입비는 5만원이다",
    "토요일은 알바가 8시간 있다",
    "평일 가용 시간은 5시간이다",
    "1주차에는 기초 개념을 다진다",
    "2주차부터 실전 문제로 넘어간다",
    "시험 기간에는 일정을 축소한다",
    "예산 총액은 30만원을 넘지 않는다",
    "오전에는 집중이 필요한 작업을 배치한다",
]


def main() -> None:
    print("=" * 78)
    print("M0 분절기·버전 정렬 검증")
    print("=" * 78)

    # ── 1. 결정성 ────────────────────────────────────────────────
    section("1. 결정성 — 같은 입력이면 같은 출력")
    runs = [segment(SAMPLE, turn=1) for _ in range(5)]
    ids = [[s.id for s in r] for r in runs]
    check(all(x == ids[0] for x in ids), "5회 실행 ID 동일",
          f"고유 결과 {len({tuple(x) for x in ids})}개")
    texts = [[s.text for s in r] for r in runs]
    check(all(x == texts[0] for x in texts), "5회 실행 내용 동일")
    check(all(len(r) == len(runs[0]) for r in runs), "세그먼트 수 동일",
          f"{[len(r) for r in runs]}")

    # ── 2. 분절 정확도 ───────────────────────────────────────────
    section("2. 분절 — 구조 단위를 올바르게 인식하는가")
    segs = segment(SAMPLE, turn=1)
    kinds = {}
    for s in segs:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    print(f"  종류별: {kinds}")
    # 문서 제목 1 + 섹션 3 + 소제목(1주차) 1 = 5
    check(kinds.get("heading", 0) == 5, "제목 5개", f"got {kinds.get('heading', 0)}")
    check(kinds.get("list_item", 0) == 4, "목록 항목 4개", f"got {kinds.get('list_item', 0)}")
    check(kinds.get("table_row", 0) == 3, "표 행 3개(헤더 포함)", f"got {kinds.get('table_row', 0)}")
    check(kinds.get("sentence", 0) == 2, "문장 2개", f"got {kinds.get('sentence', 0)}")

    check(all("|" not in s.text or s.kind == "table_row" for s in segs),
          "표 구분선이 세그먼트로 남지 않음")
    check(not any(s.text.startswith("#") for s in segs), "제목 기호 제거됨")
    check(not any("**" in s.text for s in segs), "마크다운 강조 제거됨")

    deep = next((s for s in segs if s.text.startswith("월 |")), None)
    check(deep is not None and "1주차" in deep.path,
          "표 행이 구조 경로를 보존", deep.path if deep else "없음")

    check(all(len(s.text) >= 2 for s in segs), "빈 세그먼트 없음")

    # ── 3. 정렬 정확도 (정답 50쌍) ───────────────────────────────
    section("3. 정렬 정확도 — 손으로 만든 정답 쌍")
    cases: list[tuple[str, str, str]] = []
    kinds_cycle = ["same", "paraphrase", "minor", "replace"]
    for i, base in enumerate(_BASES):
        for k in kinds_cycle:
            cases.append(_mk_pair(k, base))
    cases = cases[:52]
    print(f"  정답 쌍 {len(cases)}개 (동일/환언/소폭수정/교체)")

    correct = 0
    errs: list[str] = []
    for prev_t, curr_t, expect in cases:
        p = [Segment(id="p0", index=0, text=prev_t, kind="sentence", path="", turn=1)]
        c = [Segment(id="c0", index=0, text=curr_t, kind="sentence", path="", turn=2)]
        r = align(p, c)
        if r["pairs"]:
            got = "match_changed" if r["pairs"][0]["changed"] else "match_same"
        else:
            got = "replace"
        if got == expect:
            correct += 1
        elif len(errs) < 5:
            errs.append(f"{expect}→{got}: {prev_t[:22]!r} / {curr_t[:22]!r}")

    acc = correct / len(cases)
    print(f"  정확도 {correct}/{len(cases)} = {acc:.3f}")
    for e in errs:
        print(f"    오분류 {e}")
    check(acc >= 0.90, "정렬 정확도 ≥ .90 (문서 완료 기준)", f"{acc:.3f}")

    # ── 4. 문서 수준 정렬 ────────────────────────────────────────
    section("4. 문서 수준 — 한 항목만 바꿨을 때")
    v1 = segment(SAMPLE, turn=1)
    v2_text = SAMPLE.replace("인터넷 강의: 15만원", "무료 강의 활용: 0원")
    v2 = segment(v2_text, turn=2)
    r = align(v1, v2)
    changed = changed_segments(r)
    print(f"  매칭 {len(r['pairs'])} | 신규 {len(r['appeared'])} | 소멸 {len(r['vanished'])}")
    print(f"  내용 변경 {len(changed)}개")
    affected = len(changed) + len(r["appeared"])
    check(affected <= 2, "한 항목 변경이 1~2개 세그먼트에만 영향",
          f"영향 {affected}개")
    check(len(r["pairs"]) >= len(v1) - 2, "나머지는 전부 안정적으로 매칭",
          f"매칭 {len(r['pairs'])}/{len(v1)}")

    # ── 5. 환언 견딤 ─────────────────────────────────────────────
    section("5. 환언 견딤 — 표현만 바뀌면 끊기지 않아야")
    v3_text = (SAMPLE
               .replace("알고리즘 학습", "알고리즘 이론 학습")
               .replace("문제풀이", "문제 풀이")
               .replace("자소서 작성", "자기소개서 작성"))
    v3 = segment(v3_text, turn=2)
    r3 = align(v1, v3)
    check(len(r3["vanished"]) == 0, "환언만으로 소멸이 생기지 않음",
          f"소멸 {len(r3['vanished'])}개")
    check(len(r3["appeared"]) == 0, "환언만으로 신규가 생기지 않음",
          f"신규 {len(r3['appeared'])}개")
    print(f"  평균 유사도 {r3['mean_sim']:.3f}")

    # ── 6. 반복 구조 — 헝가리안이 필요한 경우 ───────────────────
    section("6. 반복 구조 — 비슷한 행이 여러 개일 때 어긋나지 않는가")
    rep1 = segment("""## 일정
| 월 | 알고리즘 | 문제풀이 |
| 화 | 알고리즘 | 자소서 |
| 수 | 알고리즘 | 문제풀이 |
| 목 | 알고리즘 | 자소서 |""", turn=1)
    rep2 = segment("""## 일정
| 월 | 알고리즘 | 문제풀이 |
| 화 | 알고리즘 | 자소서 |
| 수 | 알고리즘 | 휴식 |
| 목 | 알고리즘 | 자소서 |""", turn=2)
    rr = align(rep1, rep2)
    m1 = {s.id: s for s in rep1}
    m2 = {s.id: s for s in rep2}
    wrong = [p for p in rr["pairs"]
             if m1[p["prev"]].text.split("|")[0] != m2[p["curr"]].text.split("|")[0]]
    check(not wrong, "같은 요일끼리 매칭됨", f"어긋난 쌍 {len(wrong)}개")
    ch = [p for p in rr["pairs"] if p["changed"]]
    check(len(ch) == 1 and "수" in m2[ch[0]["curr"]].text,
          "바뀐 행만 변경으로 표시", f"변경 {len(ch)}개")

    # ── 7. 유사도 함수 교체 가능성 ───────────────────────────────
    section("7. 유사도 함수 주입 — 임베딩으로 교체 가능한가")
    called = {"n": 0}

    def fake_sim(a: str, b: str) -> float:
        called["n"] += 1
        return lexical_similarity(a, b)

    align(rep1, rep2, sim_fn=fake_sim)
    check(called["n"] > 0, "주입한 유사도 함수가 호출됨", f"{called['n']}회")

    print("\n" + "=" * 78)
    print(f"총계: {_passed} PASS / {_failed} FAIL")
    print("=" * 78)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
