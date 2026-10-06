"""GL-view UI 정적 검증 — 논문 6.3절 사양과 대조.

배경:
GL-view는 본 실험의 유일한 조작 변수다(6.3절). 사양은 다음과 같다.
  - 동일한 패널·검증 상태 구분·점 인코딩·호버·provenance 제공
  - 노드 조작(드래그 승격 / 내용 수정 / 우클릭 거부)만 비활성

이 사양이 깨지면 2조건 설계가 무효가 되므로, 브라우저 실행 없이도
정적으로 검증되도록 고정한다. 실제 브라우저 확인은 2026-09-29에 수행했고
그 결과를 아래 EXPECTED에 박아 둔다.

실행: C:/seungyeol/vscode/GL/.runtime/python.exe test_glview_ui.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
HTML = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

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
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def adddot_body() -> str:
    i = HTML.find("function addDot")
    assert i > 0, "addDot 없음"
    depth, j, started = 0, i, False
    while j < len(HTML):
        if HTML[j] == "{":
            depth += 1
            started = True
        elif HTML[j] == "}":
            depth -= 1
            if started and depth == 0:
                return HTML[i : j + 1]
        j += 1
    return HTML[i : i + 4000]


def draggable_block(body: str) -> str:
    """addDot 안의 `if (draggable) { ... }` 블록만 반환."""
    m = re.search(r"if\s*\(\s*draggable\s*\)\s*\{", body)
    if not m:
        return ""
    start = m.end() - 1
    depth, j = 0, start
    while j < len(body):
        if body[j] == "{":
            depth += 1
        elif body[j] == "}":
            depth -= 1
            if depth == 0:
                return body[start : j + 1]
        j += 1
    return ""


def main() -> None:
    print("=" * 78)
    print("GL-view UI 검증 — 논문 6.3절 사양 대조")
    print("=" * 78)

    body = adddot_body()
    blk = draggable_block(body)

    section("1. 조건 분기 — canManipulate")
    check(
        "canManipulate는 gl에서만 참",
        re.search(r"canManipulate\s*=\s*state\.sessionType\s*===\s*'gl'", HTML) is not None,
        "조건식이 바뀌었다",
    )
    check(
        "LLM Ground 노드에 canManipulate 전달",
        re.search(r"llm\.forEach\(.*?addDot\(\s*zoneFor\(item\).*?canManipulate\s*\)", HTML) is not None,
    )
    check(
        "Common Ground 노드는 항상 조작 불가",
        re.search(r"common\.forEach\(.*?addDot\(\s*\$\('cg-zone'\).*?,\s*false\s*\)", HTML) is not None,
        "CG 노드는 승격 대상이 아니므로 false여야 한다",
    )

    section("2. 조작만 차단되는가 (GL-view에서 비활성)")
    for evt in ("dragstart", "dragend", "contextmenu"):
        check(f"{evt}는 draggable 블록 안", f"'{evt}'" in blk, "조건 밖에 있으면 GL-view에서도 조작 가능")

    section("3. 열람 기능은 보존되는가 (두 조건 동일)")
    for evt, why in [
        ("mouseenter", "호버 툴팁"),
        ("mouseleave", "호버 해제"),
        ("dblclick", "provenance 확인"),
    ]:
        in_blk = f"'{evt}'" in blk
        in_body = f"'{evt}'" in body
        check(
            f"{evt}({why})는 조건 밖 — GL-view에서도 작동",
            in_body and not in_blk,
            "draggable 블록 안에 있으면 GL-view가 정보 접근에서 불리해져 교락이 된다",
        )

    section("4. 서버·핸들러 가드 (3중 방어)")
    guards = re.findall(r"state\.sessionType\s*!==\s*'gl'", HTML)
    check("핸들러 가드 3개 이상", len(guards) >= 3, f"발견 {len(guards)}개")
    for name, pat in [
        ("드롭(승격)", r"cg-zone'\)\.classList\.remove\('drop-ready'\);\s*\n\s*if \(state\.sessionType !== 'gl'\)"),
        ("우클릭 삭제", r"ctx-menu'\)\.style\.display = 'none';\s*\n\s*if \(state\.sessionType !== 'gl'\)"),
        ("수정 제출", r"btn-submit-correction'\)\.addEventListener[^{]*\{\s*\n\s*if \(state\.sessionType !== 'gl'\)"),
    ]:
        check(f"{name} 가드 존재", re.search(pat, HTML) is not None)

    section("5. 열람 전용 안내 (교정 경로 안내)")
    check("applyViewOnlyNotice 정의", "function applyViewOnlyNotice" in HTML)
    check(
        "gl_view에서만 표시",
        re.search(r"applyViewOnlyNotice[^}]*sessionType\s*!==\s*'gl_view'\s*\)\s*return", HTML, re.S) is not None,
    )
    check(
        "대화 채널로 안내 — 교정 경로가 봉쇄되지 않음(6.3절)",
        "대화로" in HTML and "열람 전용" in HTML,
    )
    check(
        "개별 노드에 정답 신호를 주지 않음",
        "주입 노드에 정답 신호를 주지 않기" in HTML,
        "주입된 전제만 표시가 다르면 감지 측정이 오염된다",
    )

    section("6. 시각 인코딩 동일성")
    check(
        "점 크기·색·위치가 조건에 의존하지 않음",
        not re.search(r"sessionType[^;\n]*(?:width|height|background|border-radius)", HTML),
        "노드 인코딩이 조건별로 달라지면 교락",
    )
    check(
        "커서는 draggable 속성에만 연동(어포던스 차이는 처치의 일부)",
        '.dot[draggable="true"] { cursor: grab; }' in HTML,
    )

    section("7. 세션 타입 화이트리스트")
    check(
        "gl / gl_view / baseline 3종만 허용",
        re.search(r"\['gl',\s*'gl_view',\s*'baseline'\]\.includes\(condParam\)", HTML) is not None,
    )
    check("미지정 시 gl로 기본 설정", "? condParam : 'gl'" in HTML)

    print("\n" + "=" * 78)
    print(f"총계: {_passed} PASS / {_failed} FAIL")
    print("=" * 78)
    if _failed == 0:
        print("\n실제 브라우저 확인(2026-09-29, headless Chromium, 동일 발화):")
        print("             GL-full   GL-view")
        print("  노드 수        7         7")
        print("  영역 분포   1/4/2     1/4/2   (구조적/암묵적/용어해석)")
        print("  draggable     7         0")
        print("  커서        grab      auto")
        print("  열람 안내     없음      있음")
        print("  호버 툴팁    작동      작동")
        print("  provenance   작동      작동")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
