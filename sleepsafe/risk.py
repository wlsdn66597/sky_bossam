"""시간에 따른 위험 판정 (아키텍처 명세서 3.3절).

- 엎드림, 얼굴 가림 조건마다 '연속으로 이어진 시간'을 따로 재요.
- 어느 하나가 risk_threshold_s(기본 5초) 이상 이어지면 위험 이벤트를 시작해요.
- 위험 중에는 정상 판정이 clear_threshold_s(기본 3초) 이어져야 끝나요.
  보호자가 알림을 확인했다고 끝나지는 않아요.
- 분석 불가(아기 안 보임·어두움)가 gap_tolerance_s보다 길어지면 조건 타이머를 초기화해요.
  0으로 두면 명세서처럼 공백이 생기는 즉시 초기화해요.
- 조건이 miss_tolerance_s 이하로 잠깐 안 잡히는 건 탐지가 깜빡인 것으로 보고 타이머를 유지해요.
  그보다 오래 안 잡혀야 '해소'로 봐요. 0으로 두면 한 프레임만 안 잡혀도 초기화해요.
- 2차 알림(1차 후 15초 미확인)은 보호자 확인 여부를 아는 서버가 맡아요.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

from .decision import FrameDecision

CONDITIONS = ("prone", "face_covered")  # 앞에 있을수록 우선


@dataclass
class RiskConfig:
    risk_threshold_s: float = 5.0
    clear_threshold_s: float = 3.0
    gap_tolerance_s: float = 1.0
    miss_tolerance_s: float = 1.0


@dataclass
class RiskEvent:
    kind: str                             # risk_started / risk_ended / status_changed
    ts: float                             # 이벤트가 생긴 시각
    event_id: Optional[str] = None
    risk_type: Optional[str] = None       # prone / face_covered
    started_at: Optional[float] = None    # 조건이 처음 관찰된 시각
    confirmed_at: Optional[float] = None  # 위험으로 확정된 시각
    ended_at: Optional[float] = None
    status: Optional[str] = None          # status_changed일 때 새 상태
    reason: str = ""


class RiskMonitor:
    """프레임 판단을 시간 순서대로 넣으면 상태와 이벤트를 돌려줘요.

    상태: normal(정상) / caution(주의) / risk(위험) / unavailable(분석 불가)
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.config = config or RiskConfig()
        self.status = "unavailable"
        self.reason = "starting"
        self.active: Optional[RiskEvent] = None  # 진행 중인 위험 이벤트
        self._since: Dict[str, Optional[float]] = {c: None for c in CONDITIONS}    # 조건이 시작된 시각
        self._last_on: Dict[str, Optional[float]] = {c: None for c in CONDITIONS}  # 조건이 마지막으로 잡힌 시각
        self._last_valid: Optional[float] = None
        self._normal_since: Optional[float] = None

    def elapsed(self, ts: float) -> Dict[str, float]:
        """조건별로 지금까지 이어진 시간(초)."""
        return {c: (ts - s if s is not None else 0.0) for c, s in self._since.items()}

    def update(self, ts: float, d: FrameDecision) -> List[RiskEvent]:
        out: List[RiskEvent] = []
        cfg = self.config

        if not d.valid:
            if self._last_valid is None or ts - self._last_valid > cfg.gap_tolerance_s:
                # 연속성이 끊김 → 타이머 초기화
                self._since = {c: None for c in CONDITIONS}
                self._last_on = {c: None for c in CONDITIONS}
                self._normal_since = None
                if self.active is None:
                    self._set_status("unavailable", d.reason, ts, out)
                else:
                    self.reason = d.reason  # 위험은 정상 3초가 확인될 때까지 유지
            return out

        self._last_valid = ts
        on = {"prone": d.is_prone, "face_covered": d.is_face_covered}
        for c in CONDITIONS:
            if on[c]:
                if self._since[c] is None:
                    self._since[c] = ts
                self._last_on[c] = ts
            elif self._since[c] is not None and ts - self._last_on[c] > cfg.miss_tolerance_s:
                self._since[c] = None  # 잠깐이 아니라 계속 안 잡힘 → 해소

        if self.active is None:
            for c in CONDITIONS:
                start = self._since[c]
                # 확정은 조건이 실제로 잡힌 프레임에서 (스냅샷이 위험 장면이 되도록)
                if on[c] and start is not None and ts - start >= cfg.risk_threshold_s:
                    self.active = RiskEvent(kind="risk_started", ts=ts, event_id=uuid.uuid4().hex,
                                            risk_type=c, started_at=start, confirmed_at=ts)
                    self._normal_since = None
                    out.append(self.active)
                    self._set_status("risk", c, ts, out)
                    return out
            running = [c for c in CONDITIONS if self._since[c] is not None]
            self._set_status("caution" if running else "normal", running[0] if running else "", ts, out)
            return out

        # 위험 진행 중
        if any(on.values()):
            self._normal_since = None
            self.reason = self.active.risk_type or ""
            return out
        if self._normal_since is None:
            self._normal_since = ts
        if ts - self._normal_since >= cfg.clear_threshold_s:
            out.append(replace(self.active, kind="risk_ended", ts=ts, ended_at=ts))
            self.active = None
            self._normal_since = None
            self._set_status("normal", "", ts, out)
        return out

    def _set_status(self, status: str, reason: str, ts: float, out: List[RiskEvent]) -> None:
        changed = status != self.status
        self.status, self.reason = status, reason
        if changed:
            out.append(RiskEvent(kind="status_changed", ts=ts, status=status, reason=reason))
