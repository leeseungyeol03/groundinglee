"""계획 산출물 파서 — 5섹션 마크다운을 주소 가능한 필드 집합으로 변환.

배경(논문 5.3절 재설계):
지금까지 시스템 상태는 전제의 평평한 리스트였다. 평평한 리스트는 "전제 A가 계획의
어느 부분을 만들었는가"를 표현하지 못하므로, A가 교정되어도 A가 만든 계획 내용은
그대로 남는다. 즉 누적을 되돌릴 수 없다.

이 모듈은 그 전제조건을 만든다. 파트너 LLM의 응답(config.yaml의 5섹션 고정 템플릿)을
**안정적 주소를 가진 필드 집합**으로 분해하여, 이후 전제→필드 귀속(provenance)을
붙일 수 있게 한다.

주소 체계:
    goals[0]                    목표 1번
    routine.월.morning          월요일 오전 블록
    milestones[2]               3주차
    budget.오픽 응시료           예산 항목
    exceptions[1]               예외 규칙 2번

주소는 재생성 후에도 같은 자리를 가리킨다. 내용이 바뀌었는지는 content_hash로 판별한다.

파서는 결정적이다 — LLM 호출이 없다. 실제 세션 로그 11건으로 검증했다
(test_plan_parser.py).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

# 5섹션 제목. config.yaml의 고정 템플릿과 대응한다.
SECTION_PATTERNS = [
    ("goals",      r"목표"),
    ("routine",    r"주차별\s*일정|주간\s*루틴"),
    ("milestones", r"주차별\s*마일스톤|마일스톤"),
    ("budget",     r"예산\s*개요|예산"),
    ("exceptions", r"예외\s*규칙|예외"),
]

# 주소 체계는 영문 정규 키를 쓴다(feedback.md §4.1).
# 결정적 채점·교차 도메인 확장·필드 잠금에 모두 이 키가 쓰이므로,
# 표시 문자열(한글)과 주소 키(영문)를 분리한다.
TIME_SLOTS = [("am", "오전"), ("pm", "오후"), ("eve", "저녁")]

# 표시용 한글 요일 → 주소 키
DAY_MAP = {
    "월": "mon", "화": "tue", "수": "wed", "목": "thu",
    "금": "fri", "토": "sat", "일": "sun",
}
WEEKDAYS = list(DAY_MAP)

# 기본 계획 주차 수. 방학 계획 기준 4주이며 config로 조정한다.
# 주차를 명시하지 않은 루틴 표는 전 주차에 동일 적용된 것으로 보고 w1로 둔다.
DEFAULT_PLAN_WEEKS = 4

# "1주차", "Week 2", "2주" 등에서 주차 번호를 뽑는다.
_WEEK_PAT = re.compile(r"(?:week\s*|w)?\s*(\d+)\s*주(?:차)?|week\s*(\d+)", re.I)


def _week_no(text: str) -> int | None:
    """소제목·셀에서 주차 번호를 추출한다. 없으면 None."""
    m = _WEEK_PAT.search(text or "")
    if not m:
        return None
    raw = m.group(1) or m.group(2)
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 52 else None

# 값이 비어 있음을 뜻하는 표기 — 필드로 만들지 않는다.
_EMPTY_MARKS = {"", "-", "–", "—", "/", "x", "X"}


def content_hash(text: str) -> str:
    """내용 동일성 판별용 짧은 해시. 강조 표기·공백 차이는 무시한다."""
    normalized = re.sub(r"[*_`\s]+", "", text or "")
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]


def _strip_md(text: str) -> str:
    """마크다운 강조 표기 제거."""
    return re.sub(r"[*_`]+", "", text or "").strip()


def _split_row(line: str) -> list[str]:
    """마크다운 표 행을 셀 목록으로. 앞뒤 빈 셀은 버린다."""
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return [_strip_md(c) for c in cells]


def _is_separator(line: str) -> bool:
    """표 구분선(|---|---|) 여부."""
    return bool(re.fullmatch(r"[\s|:\-–—]+", line.strip())) and "-" in line


def _table_rows(block: str) -> list[list[str]]:
    """블록에서 표 본문 행만 추출 (헤더·구분선 제외)."""
    lines = [l for l in block.split("\n") if l.strip().startswith("|")]
    rows = [_split_row(l) for l in lines if not _is_separator(l)]
    return rows[1:] if rows else []   # 첫 행은 헤더


def _table_header(block: str) -> list[str]:
    lines = [l for l in block.split("\n") if l.strip().startswith("|")]
    return _split_row(lines[0]) if lines else []


def _trim_trailing_chatter(body: str) -> str:
    """마지막 섹션 뒤에 붙는 맺말·후속 질문을 잘라낸다.

    마지막 섹션은 다음 헤딩이 없어 경계가 없으므로, 파트너 LLM이 덧붙이는
    "몇 가지 여쮤볼게요!" 같은 질문 불릿이 계획 필드로 잡힐 수 있다.
    계획이 아닌 내용이 provenance 귀속 대상이 되면 안 되므로 수평선에서 끊는다.
    """
    m = re.search(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$", body, re.M)
    if not m:
        return body
    head = body[: m.start()]
    # 수평선이 섹션 머리에 있어 본문이 통째로 날아가는 경우는 원본 유지
    return head if head.strip() else body


def split_sections(text: str) -> dict[str, str]:
    """응답에서 5섹션 본문을 잘라낸다. 섹션 뒤의 잡담·질문은 자동으로 제외된다."""
    out: dict[str, str] = {}
    # "## 1. 목표" 형태의 헤딩 위치를 모두 찾는다
    heads = list(re.finditer(r"^#{1,4}\s*(\d)\s*[.)]\s*(.+?)\s*$", text or "", re.M))
    for i, m in enumerate(heads):
        title = m.group(2)
        body_start = m.end()
        is_last = i + 1 >= len(heads)
        body_end = len(text) if is_last else heads[i + 1].start()
        body = text[body_start:body_end]
        if is_last:
            body = _trim_trailing_chatter(body)
        for key, pat in SECTION_PATTERNS:
            if key in out:
                continue
            if re.search(pat, title):
                out[key] = body
                break
    return out


def _parse_goals(block: str) -> list[dict[str, Any]]:
    """번호 목록 또는 불릿 목록."""
    fields = []
    for line in block.split("\n"):
        s = line.strip()
        m = re.match(r"^(?:\d+[.)]|[-*•])\s+(.*)$", s)
        if not m:
            continue
        content = _strip_md(m.group(1))
        if content in _EMPTY_MARKS:
            continue
        fields.append({"address": f"priority.{len(fields) + 1}", "section": "priority", "content": content})
    return fields


def _parse_one_week_table(block: str, week: int) -> list[dict[str, Any]]:
    """요일×시간블록 표 하나를 주소 가능한 필드로 바꾼다.

    헤더에서 시간 블록 열 위치를 찾아 매핑한다. 헤더에 시간 범위가 붙어도
    ("오전 (9~12시)") 접두어로 식별한다.
    """
    header = _table_header(block)
    slot_col: dict[str, int] = {}
    for idx, cell in enumerate(header):
        for slot_key, label in TIME_SLOTS:
            if cell.startswith(label) and slot_key not in slot_col:
                slot_col[slot_key] = idx
    if not slot_col:
        return []

    fields = []
    for row in _table_rows(block):
        if not row:
            continue
        day = row[0].strip()
        day_ko = next((d for d in WEEKDAYS if day.startswith(d)), None)
        if not day_ko:
            continue
        day_key = DAY_MAP[day_ko]
        for slot_key, _ in TIME_SLOTS:
            col = slot_col.get(slot_key)
            if col is None or col >= len(row):
                continue
            content = row[col].strip()
            if content in _EMPTY_MARKS:
                continue
            fields.append({
                "address": f"schedule.w{week}.{day_key}.{slot_key}",
                "section": "schedule",
                "content": content,
                "week": week,
                "day": day_key,
                "slot": slot_key,
            })
    return fields


def _split_week_blocks(block: str) -> list[tuple[int | None, str]]:
    """일정 섹션을 주차별 덩어리로 나눈다.

    "### 1주차" 같은 소제목이 있으면 그 단위로 자르고, 없으면 (None, 전체)
    하나만 반환한다. None은 "주차 구분 없는 반복 루틴"을 뜻한다.
    """
    heads = list(re.finditer(r"^\s{0,3}#{2,5}\s*(.+)$", block, re.M))
    marked = [(m, _week_no(m.group(1))) for m in heads]
    marked = [(m, w) for m, w in marked if w is not None]
    if not marked:
        return [(None, block)]

    out: list[tuple[int | None, str]] = []
    for idx, (m, week) in enumerate(marked):
        start = m.end()
        end = marked[idx + 1][0].start() if idx + 1 < len(marked) else len(block)
        out.append((week, block[start:end]))
    return out


def _parse_routine(block: str, plan_weeks: int = DEFAULT_PLAN_WEEKS) -> list[dict[str, Any]]:
    """주차별 일정 → schedule.w{N}.{day}.{slot}

    세 가지 입력 형태를 받는다.
      1) 주차 소제목 + 주차마다 표      → 소제목의 주차 번호를 쓴다
      2) 표 첫 열이 주차인 단일 표       → 행마다 주차를 읽는다
      3) 주차 구분 없는 단일 루틴 표      → 매주 반복으로 보고 w1..w{plan_weeks}에 전개

    3번을 전개하는 이유는 의미가 그렇기 때문이다. "주간 루틴"은 그 주만이 아니라
    기간 전체에 적용되는 반복 일정이므로, 특정 주차를 건드리는 요구사항 변경이
    들어왔을 때 어느 주차가 영향받는지 주소로 구분할 수 있어야 한다.
    """
    weeks = max(1, int(plan_weeks or DEFAULT_PLAN_WEEKS))

    # (2) 표 첫 열이 주차인 경우
    header = _table_header(block)
    slot_cols = [c for c in header if any(c.startswith(l) for _, l in TIME_SLOTS)]
    if header and len(header) >= 3 and slot_cols and _week_no(header[0]) is None \
            and (header[0].startswith("주차") or header[0].startswith("주")):
        fields = []
        for row in _table_rows(block):
            if len(row) < 3:
                continue
            w = _week_no(row[0])
            if w is None:
                continue
            # 주차 열(0번)만 떼고 나머지는 그대로 넣는다.
            # 요일 열을 다시 붙이면 슬롯 열이 한 칸씩 밀린다.
            sub_head = "| " + " | ".join(header[1:]) + " |"
            sub_sep = "|" + "---|" * len(header[1:])
            sub_row = "| " + " | ".join(row[1:]) + " |"
            fields.extend(_parse_one_week_table("\n".join([sub_head, sub_sep, sub_row]), w))
        if fields:
            return fields

    # (1) 주차 소제목이 있는 경우 / (3) 없는 경우
    blocks = _split_week_blocks(block)
    fields = []
    for week, chunk in blocks:
        if week is not None:
            fields.extend(_parse_one_week_table(chunk, week))
            continue
        base = _parse_one_week_table(chunk, 1)
        if not base:
            continue
        fields.extend(base)
        for w in range(2, weeks + 1):
            for f in base:
                g = dict(f)
                g["address"] = f"schedule.w{w}.{f['day']}.{f['slot']}"
                g["week"] = w
                fields.append(g)
    return fields



def _parse_milestones(block: str) -> list[dict[str, Any]]:
    """주차별 표(2열 또는 3열) 또는 불릿 목록.

    열 수가 다르므로 위치를 고정하지 않는다. 첫 열을 주차 라벨, 마지막 열을 내용으로 본다.
    """
    fields = []
    rows = _table_rows(block)
    if rows:
        for row in rows:
            if len(row) < 2:
                continue
            label, content = row[0].strip(), row[-1].strip()
            if content in _EMPTY_MARKS:
                continue
            fields.append({
                "address": f"milestone.w{len(fields) + 1}",
                "section": "milestone",
                "content": content,
                "label": label,
            })
        return fields

    for line in block.split("\n"):
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*)$", line)
        if not m:
            continue
        content = _strip_md(m.group(1))
        if content in _EMPTY_MARKS:
            continue
        fields.append({
            "address": f"milestone.w{len(fields) + 1}",
            "section": "milestone",
            "content": content,
        })
    return fields


def _parse_budget(block: str) -> list[dict[str, Any]]:
    """항목/내용 표 또는 불릿. 항목명을 주소로 쓴다 (순서가 바뀌어도 추적 가능)."""
    fields = []
    seen: set[str] = set()

    def _add(item: str, value: str) -> None:
        item = item.strip()
        value = value.strip()
        if not item or value in _EMPTY_MARKS:
            return
        key = item
        n = 2
        while key in seen:               # 항목명 중복 시 접미사
            key = f"{item}#{n}"
            n += 1
        seen.add(key)
        fields.append({
            "address": f"budget.{key}",
            "section": "budget",
            "content": value,
            "label": item,
        })

    rows = _table_rows(block)
    if rows:
        # 표가 3열 이상이면 마지막 열이 "비고"·"산출 근거"일 수 있다.
        # 금액 열을 헤더로 찾고, 못 찾으면 마지막 열로 돌아간다.
        header = _table_header(block)
        amount_col = next(
            (i for i, c in enumerate(header)
             if i > 0 and any(k in c for k in ("금액", "비용", "예산", "원"))),
            None,
        )
        for row in rows:
            if len(row) < 2:
                continue
            col = amount_col if (amount_col is not None and amount_col < len(row)) else -1
            _add(row[0], row[col])
        return fields

    for line in block.split("\n"):
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*)$", line)
        if not m:
            continue
        body = _strip_md(m.group(1))
        parts = re.split(r"\s*[:：]\s*", body, maxsplit=1)
        if len(parts) == 2:
            _add(parts[0], parts[1])
        elif body not in _EMPTY_MARKS:
            _add(body, body)
    return fields


def _parse_exceptions(block: str) -> list[dict[str, Any]]:
    """불릿 목록. 표로 오는 경우도 받는다."""
    fields = []
    rows = _table_rows(block)
    if rows:
        for row in rows:
            content = " → ".join(c for c in row if c and c not in _EMPTY_MARKS)
            if content:
                fields.append({
                    "address": f"constraint.{len(fields) + 1}",
                    "section": "constraint",
                    "content": content,
                })
        return fields

    for line in block.split("\n"):
        m = re.match(r"^\s*(?:\d+[.)]|[-*•])\s+(.*)$", line)
        if not m:
            continue
        content = _strip_md(m.group(1))
        if content in _EMPTY_MARKS:
            continue
        fields.append({
            "address": f"constraint.{len(fields) + 1}",
            "section": "constraint",
            "content": content,
        })
    return fields


_PARSERS = {
    "goals":      _parse_goals,
    "routine":    _parse_routine,
    "milestones": _parse_milestones,
    "budget":     _parse_budget,
    "exceptions": _parse_exceptions,
}


def has_plan(text: str) -> bool:
    """응답이 계획 산출물인지 판별. 5섹션 중 3개 이상이면 계획으로 본다.

    S1(목표 탐색) 단계 응답이나 단순 질문 응답을 계획으로 오인하지 않기 위한 기준이다.
    """
    return len(split_sections(text)) >= 3


def parse_plan(text: str, turn: int = 0,
               plan_weeks: int = DEFAULT_PLAN_WEEKS) -> list[dict[str, Any]]:
    """5섹션 응답 → 주소 가능한 필드 목록.

    각 필드:
        address       안정적 주소 (재생성 후에도 같은 자리를 가리킴)
        section       goals | routine | milestones | budget | exceptions
        content       필드 내용 (마크다운 표기 제거됨)
        content_hash  내용 동일성 판별용
        set_at_turn   이 내용이 기록된 턴
        label         (선택) 표의 첫 열 라벨 — 마일스톤 주차, 예산 항목명
    """
    sections = split_sections(text)
    fields: list[dict[str, Any]] = []
    for key, parser in _PARSERS.items():
        block = sections.get(key)
        if not block:
            continue
        produced = parser(block, plan_weeks) if key == "routine" else parser(block)
        for field in produced:
            field["content_hash"] = content_hash(field["content"])
            field["set_at_turn"] = turn
            fields.append(field)
    fields.extend(_derive_meta_fields(fields, turn))
    return fields


def _derive_meta_fields(fields: list[dict[str, Any]], turn: int) -> list[dict[str, Any]]:
    """계획에서 결정적으로 계산되는 파생 필드.

    LLM이 쓰는 것이 아니라 산출물에서 계산한다. 결정적이므로 검증기와
    채점에 그대로 쓰이며, 요구사항 변경의 영향을 집계 수준에서 드러낸다
    (예: 예산을 반으로 줄였는데 budget.total이 그대로면 개정이 안 된 것).
    """
    sched = [f for f in fields if f.get("section") == "schedule"]
    budget = [f for f in fields if f.get("section") == "budget"]

    meta: list[dict[str, Any]] = []

    def _add(key: str, value: str) -> None:
        meta.append({
            "address": f"meta.{key}",
            "section": "meta",
            "content": value,
            "content_hash": content_hash(value),
            "set_at_turn": turn,
            "derived": True,
        })

    if sched:
        weeks = {f.get("week") for f in sched if f.get("week")}
        total_slots = max(len(weeks), 1) * len(WEEKDAYS) * len(TIME_SLOTS)
        filled = len(sched)
        _add("density", f"{filled}/{total_slots}")
        rest = sum(1 for f in sched if _is_rest(f["content"]))
        _add("rest_ratio", f"{rest}/{filled}")

    if budget:
        total = sum(_parse_amount(f["content"]) for f in budget)
        if total > 0:
            _add("budget_total", f"{int(total)}")

    return meta


_REST_WORDS = ("휴식", "자유", "예비", "공백", "없음", "휴일", "재충전")


def _is_rest(content: str) -> bool:
    c = (content or "").strip()
    return any(w in c for w in _REST_WORDS)


def merge_plan_fields(
    parsed: list[dict[str, Any]],
    prior: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """새로 파싱한 필드를 이전 귀속과 병합한다.

    반환: (unchanged, to_attribute)
      unchanged    — (주소, content_hash)가 같아 이전 귀속을 그대로 쓰는 필드
      to_attribute — 내용이 바뀌었거나 신규라 귀속을 다시 받아야 하는 필드

    계획은 매 턴 통째로 재생성되지만 실측상 대부분의 필드는 내용까지 그대로다.
    주소만 같고 내용이 바뀐 경우 이전 귀속을 재사용하면 오염이므로 반드시 재귀속한다.
    """
    index = {(f.get("address"), f.get("content_hash")): f for f in prior}
    unchanged: list[dict[str, Any]] = []
    to_attribute: list[dict[str, Any]] = []

    for field in parsed:
        previous = index.get((field["address"], field["content_hash"]))
        if previous is None:
            to_attribute.append(dict(field))
            continue
        merged = dict(field)
        merged["premise_ids"] = list(previous.get("premise_ids", []))
        merged["set_at_turn"] = previous.get("set_at_turn", field["set_at_turn"])
        merged["status"] = previous.get("status", "active")
        unchanged.append(merged)

    return unchanged, to_attribute


def to_plan_dict(fields: list[dict[str, Any]]) -> dict[str, Any]:
    """파싱된 필드 → scoring.py가 기대하는 plan dict 형태.

    scoring.build_plan_json()은 이 구조를 입력으로 받도록 작성되어 있으나
    지금까지 이를 만드는 코드가 없었다. 파서가 그 자리를 채운다.
    """
    plan: dict[str, Any] = {
        "goals": [],
        "weekly_routine": {},
        "milestones": [],
        "budget": {"expenses": []},
        "exception_rules": [],
    }
    for f in fields:
        sec, content = f["section"], f["content"]
        if sec == "priority":
            plan["goals"].append({"content": content, "address": f["address"]})
        elif sec == "schedule":
            _, wk, day, slot = f["address"].split(".", 3)
            plan["weekly_routine"].setdefault(wk, {}).setdefault(day, {})[slot] = content
        elif sec == "milestone":
            plan["milestones"].append({
                "week": f.get("label", ""), "content": content, "address": f["address"],
            })
        elif sec == "budget":
            amount = _parse_amount(content)
            plan["budget"]["expenses"].append({
                "item": f.get("label", ""), "raw": content,
                "amount": amount, "address": f["address"],
            })
        elif sec == "constraint":
            plan["exception_rules"].append({"content": content, "address": f["address"]})
    return plan


def _parse_amount(text: str) -> float:
    """예산 문자열에서 금액 추출. '[확인 필요]' 등 미확정 표기는 0."""
    if "확인 필요" in text:
        return 0.0
    m = re.search(r"([\d,]+)\s*(만원|만|원)", text)
    if not m:
        return 0.0
    value = float(m.group(1).replace(",", ""))
    return value * 10000 if m.group(2) in {"만원", "만"} else value
