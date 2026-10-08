"""전제 상태 저장소 — M4·M5가 공유하는 데이터 모델 (feedback2.md 4.3).

왜 분리하는가:
M4(컨텍스트 구성)와 M5(수정 전파)가 같은 상태를 읽고 쓴다. 상태가 모듈마다
흩어지면 "같은 상태면 같은 프롬프트"라는 M4의 완료 기준을 보장할 수 없다.

상태 전이는 Traum의 계산적 그라운딩 모델에서 착안했다(feedback2.md 4.3).
    proposed → grounded    사용자 증거가 있을 때만
    proposed/grounded → corrected    수정되었을 때. 새 전제가 생기고 superseded_by로 연결
    proposed → rejected    사용자가 거부
    grounded → retracted   사용자가 철회

**행동 검사 결과는 상태 전이의 근거가 아니다**(원칙 2). operative/inert는
`dependence`에만 기록하고 `status`는 건드리지 않는다. "LLM이 쓰고 있다"는
"사용자와 확인되었다"가 아니기 때문이다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

# 사용자 증거 유형. 대화 채널과 인터페이스 채널을 구분해 기록한다.
EVIDENCE_TYPES = {
    "user_statement",    # 발화에서 직접 진술
    "user_ack",          # 발화로 승인
    "user_correction",   # 발화로 수정
    "ui_ack",            # 인터페이스에서 승인
    "ui_correction",     # 인터페이스에서 수정
    "ui_reject",         # 인터페이스에서 거부
}

STATUSES = {"proposed", "grounded", "corrected", "rejected", "retracted"}

# 행동 검사 판정. status와 독립이다.
VERDICTS = {"operative", "inert", "inconclusive", "untested"}


@dataclass
class Evidence:
    type: str
    turn: int
    pointer: str = ""          # 원문 위치 또는 조작 로그 ID
    attribution: str = "designated"   # designated(인터페이스) | estimated(대화)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Premise:
    id: str
    text: str
    origin: str = "session"            # memory | session
    category: str = "implicit"         # implicit | structural | term
    status: str = "proposed"
    evidence: list[Evidence] = field(default_factory=list)
    candidate_sources: list[str] = field(default_factory=list)
    derived_from: list[str] = field(default_factory=list)
    superseded_by: str | None = None
    # 행동 검사 결과 — status와 분리해 보관한다
    verdict: str = "untested"
    tested_turn: int | None = None
    depends_segments: list[str] = field(default_factory=list)
    # 수정 전파 중 재검토 표시
    flagged_review: bool = False

    @property
    def is_grounded(self) -> bool:
        return self.status in ("grounded", "corrected")

    @property
    def is_active(self) -> bool:
        """현재 생성에 쓰일 수 있는 전제인가."""
        return self.status in ("proposed", "grounded", "corrected")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [e.to_dict() if isinstance(e, Evidence) else e
                         for e in self.evidence]
        return d


class StateStore:
    """전제와 세그먼트 상태를 모아 두고 버전을 남긴다.

    모든 변경은 이력으로 남긴다. 실험 재현과 논문 그림에 쓰이고,
    "고친 게 반영됐는가"를 따지려면 이전 상태를 되짚을 수 있어야 한다.
    """

    def __init__(self) -> None:
        self.premises: dict[str, Premise] = {}
        self.history: list[dict[str, Any]] = []
        self._version = 0

    # ---------------------------------------------------------------- #
    def add(self, premise: Premise) -> Premise:
        self.premises[premise.id] = premise
        self._log("add", premise.id, {"text": premise.text, "status": premise.status})
        return premise

    def get(self, pid: str) -> Premise | None:
        return self.premises.get(pid)

    def ground(self, pid: str, evidence: Evidence) -> None:
        """사용자 증거로 확인 상태로 올린다. 증거 없이는 호출할 수 없다."""
        if evidence.type not in EVIDENCE_TYPES:
            raise ValueError(f"알 수 없는 증거 유형: {evidence.type}")
        p = self.premises[pid]
        p.evidence.append(evidence)
        p.status = "grounded"
        self._log("ground", pid, {"evidence": evidence.to_dict()})

    def correct(self, pid: str, new_text: str, evidence: Evidence,
                new_id: str | None = None) -> Premise:
        """전제를 수정한다. 옛 전제는 corrected로 남기고 새 전제를 만든다.

        덮어쓰지 않는 이유는, 옛 값이 응답에 남아 있는지 검사하려면
        옛 값을 알고 있어야 하기 때문이다(M5 반영 재검사).
        """
        old = self.premises[pid]
        nid = new_id or f"{pid}_c{self._version + 1}"
        new = Premise(
            id=nid, text=new_text, origin=old.origin, category=old.category,
            status="corrected", evidence=[evidence],
            derived_from=list(old.derived_from),
        )
        old.status = "corrected"
        old.superseded_by = nid
        self.premises[nid] = new
        self._log("correct", pid, {"new_id": nid, "old_text": old.text,
                                   "new_text": new_text})
        return new

    def reject(self, pid: str, evidence: Evidence) -> None:
        p = self.premises[pid]
        p.evidence.append(evidence)
        p.status = "rejected"
        self._log("reject", pid, {})

    def set_verdict(self, pid: str, verdict: str, turn: int,
                    segments: Iterable[str] = ()) -> None:
        """행동 검사 결과를 기록한다. status는 건드리지 않는다(원칙 2)."""
        if verdict not in VERDICTS:
            raise ValueError(f"알 수 없는 판정: {verdict}")
        p = self.premises[pid]
        p.verdict = verdict
        p.tested_turn = turn
        p.depends_segments = list(segments)
        self._log("verdict", pid, {"verdict": verdict, "n_segments": len(p.depends_segments)})

    def flag_derived(self, pid: str, depth: int = 2) -> list[str]:
        """전제 p에서 파생된 LG 전제에 재검토 표시를 단다. 깊이 제한이 있다.

        자동으로 고치지 않는다. 사용자가 판단한다(feedback2.md M5-2).
        """
        flagged: list[str] = []
        frontier = [pid]
        for _ in range(max(depth, 0)):
            nxt = []
            for parent in frontier:
                for q in self.premises.values():
                    if parent in q.derived_from and not q.flagged_review:
                        q.flagged_review = True
                        flagged.append(q.id)
                        nxt.append(q.id)
            frontier = nxt
            if not frontier:
                break
        if flagged:
            self._log("flag_derived", pid, {"flagged": flagged})
        return flagged

    # ---------------------------------------------------------------- #
    def by_status(self, *statuses: str) -> list[Premise]:
        """상태로 거른다. 출력 순서는 id 기준으로 고정한다.

        M4가 바이트 단위로 같은 프롬프트를 만들려면 정렬이 결정적이어야 한다.
        dict 순회 순서에 기대면 실행마다 달라질 수 있다.
        """
        return sorted((p for p in self.premises.values() if p.status in statuses),
                      key=lambda p: p.id)

    def superseded_pairs(self) -> list[tuple[Premise, Premise]]:
        """(옛 전제, 새 전제) 쌍. 정정 표시와 반영 검사에 쓴다."""
        out = []
        for p in self.by_status("corrected"):
            if p.superseded_by and p.superseded_by in self.premises:
                out.append((p, self.premises[p.superseded_by]))
        return out

    def _log(self, op: str, pid: str, payload: dict[str, Any]) -> None:
        self._version += 1
        self.history.append({"v": self._version, "op": op, "id": pid, **payload})

    def snapshot(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "premises": {k: v.to_dict() for k, v in sorted(self.premises.items())},
        }

    def to_json(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True, indent=1)
