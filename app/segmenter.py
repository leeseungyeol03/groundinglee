"""M0. 분절기와 버전 정렬 — 도메인 스키마를 대신하는 유일한 구조.

왜 필요한가:
P2(사용 추적)와 P3(수정 전파)는 "어느 문장이 어느 전제에 기댔나", "어느 문장이
바뀌었나"를 따진다. 그러려면 응답을 일관된 단위로 쪼개고, 재생성 후에도 같은
문장을 같은 ID로 추적해야 한다.

이전 설계는 도메인별 주소 체계(schedule.w1.mon.am 같은 것)로 이 문제를 풀었다.
그 방식은 결정적이라는 장점이 있었으나 도메인마다 스키마를 새로 짜야 했다.
여기서는 응답이 원래 갖는 구조(문단·목록·표)만 써서 도메인 독립으로 만든다.

설계 원칙:
  1. 분절은 **완전히 결정적**이다. 같은 입력이면 같은 출력이다. 모델을 쓰지 않는다.
  2. 정렬은 유사도 함수를 갈아끼울 수 있게 둔다. 임베딩이 없어도 어휘 유사도로
     동작하고, 임베딩이 준비되면 그걸 주입한다.
  3. 분절 단위는 LLM이 원래 내는 단위를 따른다. 문장을 인위적으로 더 쪼개지 않는다.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# --------------------------------------------------------------------------- #
# 분절
# --------------------------------------------------------------------------- #

# 마크다운 구조 표지
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*)$")
_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_HR = re.compile(r"^\s*(?:-{3,}|={3,}|\*{3,})\s*$")

# 한국어 문장 종결. 종결어미 + 문장부호를 함께 본다.
# kss 같은 전용 분리기가 있으면 더 낫지만, 의존성 없이 결정적으로 동작해야 하므로
# 규칙으로 처리한다. 과분할보다 미분할이 안전하다(세그먼트가 너무 짧으면
# 로그확률 추정이 불안정해진다).
_SENT_END = re.compile(
    r"(?<=[.!?。])\s+"                       # 문장부호 뒤 공백
    r"|(?<=다)\.\s+"                          # ~다.
    r"|(?<=요)\.\s+"                          # ~요.
    r"|(?<=[다요])\s*\n"                      # 줄바꿈으로 끝나는 종결
)

_MD_STRIP = re.compile(r"[*_`~]+")


def _clean(text: str) -> str:
    return _MD_STRIP.sub("", text or "").strip()


def _split_sentences(text: str) -> list[str]:
    """한 덩어리 안을 문장으로 나눈다. 짧은 조각은 앞 문장에 붙인다."""
    raw = [s.strip() for s in _SENT_END.split(text) if s and s.strip()]
    if not raw:
        return []
    out: list[str] = []
    for s in raw:
        # 너무 짧은 조각은 독립 세그먼트로 두지 않는다.
        # 로그확률을 토큰 수로 나누므로 분모가 작으면 분산이 커진다.
        if out and len(s) < 8:
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


@dataclass
class Segment:
    """응답 안의 추적 가능한 최소 단위."""
    id: str
    index: int
    text: str
    kind: str                     # heading | sentence | list_item | table_row
    path: str                     # 속한 구조 경로 (예: "2. 주차별 일정 > 1주차")
    turn: int = 0
    aligned_prev: str | None = None
    state: str = "active"         # active | superseded | flagged_review
    depends_on: list[str] = field(default_factory=list)
    # 원문 문자 구간. 로그확률 채점에서 토큰 범위로 변환하려면 필요하다.
    # 단일 순전파로 전체 응답을 채점한 뒤 세그먼트별로 잘라야 비용이 맞는다.
    start: int = -1
    end: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "index": self.index, "text": self.text,
            "kind": self.kind, "path": self.path, "turn": self.turn,
            "aligned_prev": self.aligned_prev, "state": self.state,
            "depends_on": list(self.depends_on),
            "start": self.start, "end": self.end,
        }


def _seg_id(turn: int, index: int, text: str) -> str:
    """턴·위치·내용으로 만드는 안정 ID.

    내용이 같아도 위치가 다르면 다른 세그먼트다. 정렬은 align()이 따로 한다.
    """
    h = hashlib.sha1(f"{turn}|{index}|{_clean(text)}".encode("utf-8")).hexdigest()[:8]
    return f"seg_{h}"


def segment(text: str, turn: int = 0) -> list[Segment]:
    """응답 텍스트를 세그먼트 목록으로. 완전히 결정적이다.

    구조 표지(제목·목록·표)를 먼저 인식하고, 일반 문단만 문장으로 나눈다.
    표 구분선과 수평선은 내용이 없으므로 버린다.

    각 세그먼트는 원문에서의 문자 구간(start, end)을 함께 기록한다.
    로그확률 채점이 단일 순전파로 전체 응답을 훑은 뒤 세그먼트별로 잘라야 하므로,
    토큰 오프셋으로 변환할 기준이 필요하다.
    """
    src = text or ""
    segments: list[Segment] = []
    path_stack: list[str] = []
    # 버퍼는 (줄 내용, 원문 시작 위치) 쌍을 모은다
    buffer: list[tuple[str, int]] = []

    def _add(content: str, kind: str, start: int, end: int) -> None:
        idx = len(segments)
        segments.append(Segment(
            id=_seg_id(turn, idx, content),
            index=idx,
            text=content,
            kind=kind,
            path=" > ".join(path_stack),
            turn=turn,
            start=start,
            end=end,
        ))

    def flush_paragraph() -> None:
        """버퍼의 줄들을 이어 붙여 문장으로 나누고, 각 문장의 원문 위치를 복원한다."""
        if not buffer:
            return
        # 이어 붙인 텍스트와, 그 안의 위치 → 원문 위치 매핑을 함께 만든다
        joined_parts: list[str] = []
        pos_map: list[int] = []          # joined 문자 i → 원문 위치
        for k, (line, line_start) in enumerate(buffer):
            if k > 0:
                joined_parts.append(" ")
                pos_map.append(line_start)   # 삽입한 공백은 다음 줄 시작으로 본다
            joined_parts.append(line)
            pos_map.extend(range(line_start, line_start + len(line)))
        joined = "".join(joined_parts)
        buffer.clear()

        cursor = 0
        for sent in _split_sentences(joined):
            found = joined.find(sent, cursor)
            if found < 0:
                found = cursor
            cursor = found + len(sent)
            c = _clean(sent)
            if not c:
                continue
            s_off = pos_map[found] if found < len(pos_map) else -1
            e_idx = min(cursor, len(pos_map)) - 1
            e_off = (pos_map[e_idx] + 1) if 0 <= e_idx < len(pos_map) else -1
            _add(c, "sentence", s_off, e_off)

    offset = 0
    for line in src.split("\n"):
        line_start = offset
        offset += len(line) + 1          # +1 은 제거된 개행

        if not line.strip():
            flush_paragraph()
            continue
        if _HR.match(line) or _TABLE_SEP.match(line):
            flush_paragraph()
            continue

        m = _HEADING.match(line)
        if m:
            flush_paragraph()
            level = len(m.group(1))
            title = _clean(m.group(2))
            del path_stack[level - 1:]
            path_stack.append(title)
            if title:
                _add(title, "heading",
                     line_start + m.start(2), line_start + m.end(2))
            continue

        m = _TABLE_ROW.match(line)
        if m:
            flush_paragraph()
            cells = [_clean(c) for c in m.group(1).split("|")]
            cells = [c for c in cells if c]
            if cells:
                _add(" | ".join(cells), "table_row",
                     line_start + m.start(1), line_start + m.end(1))
            continue

        m = _LIST.match(line)
        if m:
            flush_paragraph()
            c = _clean(m.group(1))
            if c:
                _add(c, "list_item",
                     line_start + m.start(1), line_start + m.end(1))
            continue

        buffer.append((line.strip(), line_start + (len(line) - len(line.lstrip()))))

    flush_paragraph()
    return segments

# --------------------------------------------------------------------------- #
# 버전 정렬
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+")


def _ngrams(text: str, n: int = 2) -> set[str]:
    s = re.sub(r"\s+", "", text or "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def lexical_similarity(a: str, b: str) -> float:
    """어휘 유사도. 임베딩 없이 결정적으로 계산한다.

    문자 바이그램과 토큰을 함께 본다. 바이그램만 쓰면 어순 바뀜에 둔감하고,
    토큰만 쓰면 짧은 문장에서 표본이 부족하다.
    """
    if a == b:
        return 1.0
    ga, gb = _ngrams(a), _ngrams(b)
    g = len(ga & gb) / len(ga | gb) if (ga | gb) else 0.0
    ta, tb = set(_TOKEN.findall(a)), set(_TOKEN.findall(b))
    t = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    return 0.5 * g + 0.5 * t


SimilarityFn = Callable[[str, str], float]


def align(
    prev: list[Segment],
    curr: list[Segment],
    sim_fn: SimilarityFn | None = None,
    threshold: float = 0.45,
    structure_bonus: float = 0.1,
) -> dict[str, Any]:
    """두 버전의 세그먼트를 일대일 대응시킨다.

    헝가리안 매칭으로 전역 최적 대응을 찾고, 임계값 아래 쌍은 끊어서
    신규(appeared)와 삭제(vanished)로 분류한다.

    탐욕 매칭을 쓰지 않는 이유는, 비슷한 문장이 여러 개일 때(주차별 일정처럼
    형식이 반복되는 경우) 앞에서 잘못 집으면 뒤가 연쇄로 어긋나기 때문이다.

    structure_bonus: 같은 구조 경로(path)와 같은 종류(kind)면 유사도를 올린다.
    "월 | 공부 | 알바" 같은 표 행은 본문 유사도만으로는 구분이 어렵다.
    """
    sim = sim_fn or lexical_similarity
    if not prev or not curr:
        return {
            "pairs": [],
            "appeared": [s.id for s in curr],
            "vanished": [s.id for s in prev],
            "mean_sim": 0.0,
        }

    n, m = len(prev), len(curr)
    score = [[0.0] * m for _ in range(n)]
    for i, p in enumerate(prev):
        for j, c in enumerate(curr):
            s = sim(p.text, c.text)
            if p.path == c.path:
                s = min(1.0, s + structure_bonus)
            if p.kind == c.kind:
                s = min(1.0, s + structure_bonus * 0.5)
            score[i][j] = s

    rows, cols = _hungarian(score)

    pairs = []
    matched_prev, matched_curr = set(), set()
    for i, j in zip(rows, cols):
        if score[i][j] >= threshold:
            pairs.append({
                "prev": prev[i].id, "curr": curr[j].id,
                "sim": round(score[i][j], 4),
                "changed": prev[i].text != curr[j].text,
            })
            matched_prev.add(i)
            matched_curr.add(j)

    for pair in pairs:
        for c in curr:
            if c.id == pair["curr"]:
                c.aligned_prev = pair["prev"]

    sims = [p["sim"] for p in pairs]
    return {
        "pairs": pairs,
        "appeared": [c.id for j, c in enumerate(curr) if j not in matched_curr],
        "vanished": [p.id for i, p in enumerate(prev) if i not in matched_prev],
        "mean_sim": (sum(sims) / len(sims)) if sims else 0.0,
    }


def _hungarian(score: list[list[float]]) -> tuple[list[int], list[int]]:
    """최대 가중 이분 매칭. scipy가 있으면 쓰고, 없으면 탐욕으로 내려간다."""
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = -np.asarray(score, dtype=float)
        r, c = linear_sum_assignment(cost)
        return list(r), list(c)
    except Exception:
        # 폴백: 점수 내림차순 탐욕. 결정적이되 최적은 아니다.
        n, m = len(score), len(score[0])
        cand = sorted(
            ((score[i][j], i, j) for i in range(n) for j in range(m)),
            key=lambda x: (-x[0], x[1], x[2]),
        )
        used_r, used_c, rows, cols = set(), set(), [], []
        for _, i, j in cand:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            rows.append(i)
            cols.append(j)
        return rows, cols


def changed_segments(alignment: dict[str, Any]) -> list[str]:
    """정렬 결과에서 내용이 바뀐 세그먼트의 현재 ID 목록."""
    return [p["curr"] for p in alignment["pairs"] if p["changed"]]
