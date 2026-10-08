"""M4 컨텍스트 구성기 검증 — feedback2.md 완료 기준 대조.

완료 기준(문서 M4): 상태가 같으면 프롬프트가 바이트 단위로 같아야 한다.

왜 이게 중요한가:
M4의 존재 이유가 "고쳐줘를 덧붙이는 대신 상태에서 규칙대로 만든다"는 것이다.
같은 상태에서 프롬프트가 흔들리면 조건 간 비교(E2의 B1~B5)가 전부 오염된다.
dict 순회 순서나 집합 순서에 기대는 코드가 하나라도 있으면 깨진다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe test_context_builder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.context_builder import (
    HISTORY_MODES,
    build_context,
    build_history_block,
    build_state_block,
    context_fingerprint,
)
from app.state_store import Evidence, Premise, StateStore

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


HISTORY = [
    {"role": "user", "content": "코테와 자소서를 같이 준비하려고 해"},
    {"role": "assistant", "content": "예산 30만원 기준으로 계획을 짜겠습니다"},
    {"role": "user", "content": "평일 5시간 가능해"},
    {"role": "assistant", "content": "평일 오전은 알고리즘 이론에 배정하겠습니다"},
]


def make_store(shuffle: bool = False) -> StateStore:
    """같은 내용의 상태를 다른 삽입 순서로 만든다.

    삽입 순서가 달라도 프롬프트가 같아야 한다. 다르면 dict 순회에 기댄 것이다.
    """
    s = StateStore()
    items = [
        ("p01", "7월 한 달 동안 코딩테스트와 자소서를 병행한다", "ground"),
        ("p02", "평일 오전은 알고리즘 이론에 배정한다", None),
        ("p03", "예산 30만원은 강의와 교재에 나눈다", "ground"),
        ("p04", "사용자는 유료 강의를 선호한다", "reject"),
        ("p05", "교재는 중고로 구한다", None),
    ]
    if shuffle:
        items = list(reversed(items))
    for pid, text, act in items:
        s.add(Premise(id=pid, text=text,
                      derived_from=["p03"] if pid == "p05" else []))
        if act == "ground":
            s.ground(pid, Evidence(type="user_statement", turn=1, pointer=f"c_{pid}"))
        elif act == "reject":
            s.reject(pid, Evidence(type="ui_reject", turn=5, pointer=f"op_{pid}"))
    s.correct("p03", "예산 15만원은 무료 자료 위주로 쓴다",
              Evidence(type="ui_correction", turn=5, pointer="op_c"), new_id="p03b")
    s.flag_derived("p03")
    return s


def main() -> None:
    print("=" * 78)
    print("M4 컨텍스트 구성기 검증")
    print("=" * 78)

    store = make_store()

    # ── 1. 결정성 (문서 완료 기준) ───────────────────────────────
    section("1. 같은 상태면 바이트 단위로 같은 프롬프트")
    fps = set()
    for _ in range(5):
        sysp, msgs = build_context(store, HISTORY, "다시 짜줘", mode="H-mark")
        fps.add(context_fingerprint(sysp, msgs))
    check(len(fps) == 1, "5회 호출이 동일한 지문", f"지문 {len(fps)}종")

    # 삽입 순서만 다른 동일 상태
    store2 = make_store(shuffle=True)
    s1, m1 = build_context(store, HISTORY, "다시 짜줘", mode="H-mark")
    s2, m2 = build_context(store2, HISTORY, "다시 짜줘", mode="H-mark")
    check(context_fingerprint(s1, m1) == context_fingerprint(s2, m2),
          "삽입 순서가 달라도 동일한 프롬프트",
          "dict 순회 순서에 기대고 있다")

    # ── 2. 상태 블록 구조 ────────────────────────────────────────
    section("2. 상태 블록이 네 구역을 모두 가지는가")
    block = build_state_block(store)
    for key in ["확인된 전제", "미확인 전제", "정정된 전제", "거부된 전제"]:
        check(key in block, f"'{key}' 구역 존재")
    check(block.index("확인된 전제") < block.index("미확인 전제") < block.index("정정된 전제"),
          "구역 순서 고정")

    # ── 3. 금지 항목이 명시되는가 ────────────────────────────────
    section("3. 사용 금지가 프롬프트에 드러나는가")
    check("사용자는 유료 강의를 선호한다" in block, "거부된 전제가 블록에 나타남")
    check("사용 금지" in block, "금지 표현 존재")
    check("예산 30만원은 강의와 교재에 나눈다" in block,
          "옛 값이 '사용 금지'로 명시됨", "옛 값을 빼면 모델이 그걸 다시 쓸 수 있다")
    check("예산 15만원은 무료 자료 위주로 쓴다" in block, "새 값이 제시됨")

    # ── 4. 재검토 표시 ───────────────────────────────────────────
    section("4. 파생 전제 재검토 표시")
    check("재검토 대상" in block, "재검토 구역 존재")
    check("교재는 중고로 구한다" in block.split("재검토 대상")[1],
          "파생 전제가 재검토로 표시됨")
    p05 = store.get("p05")
    check(p05 is not None and p05.flagged_review, "flag_derived가 상태에 반영됨")
    check(p05.status == "proposed",
          "재검토 표시가 status를 바꾸지 않음", "자동으로 고치면 안 된다")

    # ── 5. 대화 기록 변형 ────────────────────────────────────────
    section("5. 대화 기록 세 변형")
    # M5의 "영향 찾기"가 산출하는 것: (더 이상 유효하지 않은 세그먼트 원문, 대체 내용)
    # 전제 문구를 과거 응답에서 문자열로 찾는 방식은 안 된다.
    # 전제는 "예산 30만원은 강의와 교재에 나눈다"인데 응답은
    # "예산 30만원 기준으로"라 쓰여 거의 안 걸린다.
    SUPERSEDED = [("예산 30만원 기준으로 계획을 짜겠습니다",
                   "예산 15만원은 무료 자료 위주로 쓴다")]
    full = build_history_block(HISTORY, store, "H-full")
    mark = build_history_block(HISTORY, store, "H-mark", SUPERSEDED)
    state = build_history_block(HISTORY, store, "H-state")
    check(len(full) == len(HISTORY), "H-full은 원문 그대로", f"{len(full)}/{len(HISTORY)}")
    check(all(a["content"] == b["content"] for a, b in zip(full, HISTORY)),
          "H-full은 내용 변경 없음")
    check(len(state) == 1, "H-state는 최근 1턴만", f"{len(state)}턴")
    marked = [m for m in mark if "[정정됨" in m["content"]]
    check(len(marked) == 1, "H-mark는 옛 전제에 표시", f"{len(marked)}건")
    check("예산 15만원은 무료 자료 위주로 쓴다" in marked[0]["content"],
          "표시에 새 값이 들어감")
    check(all(m["role"] == "assistant" for m in marked),
          "표시는 assistant 발화에만", "사용자 발화를 고치면 기록 위조다")

    for mode in HISTORY_MODES:
        fps2 = {context_fingerprint(*build_context(store, HISTORY, "x", mode=mode,
                                                   superseded=SUPERSEDED))
                for _ in range(3)}
        check(len(fps2) == 1, f"{mode} 결정성")

    # 전달 순서가 달라도 같은 표시가 나와야 한다
    sup2 = [("없는 부분", "무언가"), *SUPERSEDED]
    a = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-mark",
                                           superseded=SUPERSEDED))
    b = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-mark",
                                           superseded=list(reversed(sup2))))
    check(a == b, "없는 세그먼트·전달 순서가 지문을 바꾸지 않음")

    # ── 6. 변형 간 차이 ──────────────────────────────────────────
    section("6. 변형이 실제로 다른 프롬프트를 만드는가")
    fp = {}
    for mode in HISTORY_MODES:
        fp[mode] = context_fingerprint(*build_context(store, HISTORY, "x", mode=mode,
                                                      superseded=SUPERSEDED))
    check(len(set(fp.values())) == 3, "세 변형이 서로 다름", f"{len(set(fp.values()))}종")

    # 정정된 세그먼트가 없으면 H-mark는 H-full과 같다. 이게 정상 동작이다.
    # 표시할 게 없는데 표시를 만들어내면 기록 위조다.
    bare_mark = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-mark"))
    bare_full = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-full"))
    check(bare_mark == bare_full,
          "정정 세그먼트가 없으면 H-mark = H-full",
          "표시할 게 없는데 기록을 건드렸다")

    # ── 7. 상태 변화가 프롬프트에 반영되는가 ────────────────────
    section("7. 상태가 바뀌면 프롬프트도 바뀌는가")
    before = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-mark"))
    store.add(Premise(id="p06", text="새로 추가된 전제"))
    after = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-mark"))
    check(before != after, "전제 추가가 프롬프트를 바꿈")

    store.ground("p06", Evidence(type="ui_ack", turn=9, pointer="op_x"))
    after2 = context_fingerprint(*build_context(store, HISTORY, "x", mode="H-mark"))
    check(after != after2, "상태 전이가 프롬프트를 바꿈")
    blk = build_state_block(store)
    gpos = blk.index("확인된 전제")
    ppos = blk.index("미확인 전제")
    check(gpos < blk.index("새로 추가된 전제") < ppos,
          "승격된 전제가 확인 구역으로 이동")

    # ── 8. 빈 상태 ───────────────────────────────────────────────
    section("8. 빈 상태에서도 깨지지 않는가")
    empty = StateStore()
    try:
        s, m = build_context(empty, [], "처음 질문입니다", mode="H-state")
        check("(없음)" in s, "빈 구역이 '(없음)'으로 표시됨")
        check(len(m) == 1 and m[0]["content"] == "처음 질문입니다", "발화만 남음")
    except Exception as e:
        check(False, "빈 상태 처리", repr(e))

    print("\n" + "=" * 78)
    print(f"총계: {_passed} PASS / {_failed} FAIL")
    print("=" * 78)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
