"""API 사용량 계측 — E2 규모 산정의 근거.

왜 필요한가:
E2 본 실험은 문서 초안 기준 약 3,800 에피소드다. 에피소드당 API 호출이 몇 번이고
토큰이 얼마나 드는지 모르면 예산도 일정도 세울 수 없다. 응답 객체에 usage가
이미 들어 있으므로 기록하는 데 추가 비용이 없다.

로컬 모델(파트너)은 여기 들어오지 않는다. 비용이 전기와 시간뿐이라 따로 센다.
"""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 2026-10 기준 공개 단가 (USD per 1M tokens). 청구서로 재확인할 것.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-5-20250929": (3.00, 15.00),
}


@dataclass
class Meter:
    """호출 지점별 토큰 누적."""
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_in: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_out: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    models: dict[str, str] = field(default_factory=dict)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def record(self, site: str, model: str, response: Any) -> None:
        u = getattr(response, "usage", None)
        if u is None:
            return
        with self._lock:
            self.calls[site] += 1
            self.tokens_in[site] += int(getattr(u, "input_tokens", 0) or 0)
            self.tokens_out[site] += int(getattr(u, "output_tokens", 0) or 0)
            self.models[site] = model

    def cost(self) -> float:
        total = 0.0
        for site in self.calls:
            pin, pout = PRICING.get(self.models.get(site, ""), (0.0, 0.0))
            total += self.tokens_in[site] / 1e6 * pin
            total += self.tokens_out[site] / 1e6 * pout
        return total

    def summary(self, n_episodes: int = 1) -> dict[str, Any]:
        sites = sorted(self.calls, key=lambda s: -self.calls[s])
        rows = []
        for s in sites:
            pin, pout = PRICING.get(self.models.get(s, ""), (0.0, 0.0))
            c = (self.tokens_in[s] / 1e6 * pin) + (self.tokens_out[s] / 1e6 * pout)
            rows.append({
                "site": s, "model": self.models.get(s, "?"),
                "calls": self.calls[s],
                "calls_per_episode": self.calls[s] / max(n_episodes, 1),
                "in": self.tokens_in[s], "out": self.tokens_out[s],
                "cost_usd": c,
            })
        tot = self.cost()
        return {
            "n_episodes": n_episodes,
            "total_calls": sum(self.calls.values()),
            "calls_per_episode": sum(self.calls.values()) / max(n_episodes, 1),
            "total_in": sum(self.tokens_in.values()),
            "total_out": sum(self.tokens_out.values()),
            "total_cost_usd": tot,
            "cost_per_episode_usd": tot / max(n_episodes, 1),
            "by_site": rows,
        }

    def report(self, n_episodes: int = 1) -> str:
        s = self.summary(n_episodes)
        lines = [
            f"  API 호출 {s['total_calls']}회 ({s['calls_per_episode']:.1f}회/에피소드)",
            f"  토큰 입력 {s['total_in']:,} / 출력 {s['total_out']:,}",
            f"  비용 ${s['total_cost_usd']:.3f} (${s['cost_per_episode_usd']:.4f}/에피소드)",
            "",
            f"  {'지점':22} {'호출':>5} {'회/에피':>7} {'입력':>9} {'출력':>8} {'비용$':>8}",
        ]
        for r in s["by_site"]:
            lines.append(
                f"  {r['site']:22} {r['calls']:>5} {r['calls_per_episode']:>7.1f} "
                f"{r['in']:>9,} {r['out']:>8,} {r['cost_usd']:>8.4f}")
        return "\n".join(lines)

    def project(self, n_episodes: int, target: int) -> str:
        """본 실험 규모로 외삽한다."""
        s = self.summary(n_episodes)
        c = s["cost_per_episode_usd"] * target
        calls = s["calls_per_episode"] * target
        return (f"  {target:,} 에피소드 외삽: API {calls:,.0f}회, 약 ${c:,.0f}")

    def save(self, path: Path, n_episodes: int = 1) -> None:
        path.write_text(json.dumps(self.summary(n_episodes), ensure_ascii=False,
                                   indent=1), encoding="utf-8")


# 전역 계측기. 실험 스크립트가 공유한다.
METER = Meter()
