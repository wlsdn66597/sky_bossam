"""프레임 한 장의 탐지 결과 → 자세·얼굴 상태·위험 유형 판단.

평가(scripts/evaluate.py)와 실시간 실행(edge/run_edge.py)이 같은 함수를 써요.
그래서 PC에서 잰 정확도와 라즈베리파이에서의 동작이 같은 규칙을 따라요.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from .classes import AIRWAY_CLASSES, BODY_CLASSES


@dataclass(frozen=True)
class Detection:
    cls: str                                   # 클래스 이름
    conf: float                                # 신뢰도
    xyxy: Tuple[float, float, float, float]    # (x1, y1, x2, y2)


@dataclass(frozen=True)
class FrameDecision:
    valid: bool                       # False면 분석 불가
    reason: str = ""                  # 분석 불가 이유: no_baby, dark
    posture: Optional[str] = None     # supine / side / prone
    airway: Optional[str] = None      # airway_visible / airway_covered / None(얼굴 박스 없음)
    body: Optional[Detection] = None
    face: Optional[Detection] = None

    @property
    def is_prone(self) -> bool:
        return self.valid and self.posture == "prone"

    @property
    def is_face_covered(self) -> bool:
        return self.valid and self.airway == "airway_covered"

    @property
    def risk_type(self) -> Optional[str]:
        """위험 유형. 엎드림이면 얼굴 상태와 상관없이 엎드림으로 봐요."""
        if self.is_prone:
            return "prone"
        if self.is_face_covered:
            return "face_covered"
        return None


def center_inside(inner: Detection, outer: Detection, margin: float = 0.1) -> bool:
    """inner 박스의 중심이 outer 박스(가장자리 여유 margin 포함) 안에 있는지."""
    x1, y1, x2, y2 = outer.xyxy
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    cx = (inner.xyxy[0] + inner.xyxy[2]) / 2
    cy = (inner.xyxy[1] + inner.xyxy[3]) / 2
    return x1 - mx <= cx <= x2 + mx and y1 - my <= cy <= y2 + my


def decide(dets: Sequence[Detection], conf_body: float = 0.5, conf_airway: float = 0.4,
           covered_without_body: bool = False) -> FrameDecision:
    """아기 1명을 가정하고 판단해요.

    - 몸 박스: 신뢰도가 가장 높은 것 1개 → 자세
    - 얼굴 박스: 그 몸 박스 안에 있는 것 중 신뢰도가 가장 높은 것 1개 → 코·입 상태
    - 몸 박스가 없으면 분석 불가(no_baby).
      단 covered_without_body=True면, 코·입 가림 박스만 잡혀도 얼굴 가림으로 봐요
      (이불에 몸이 다 덮이면 몸 박스가 안 잡히는 경우 대비)
    - 얼굴 박스가 없으면 airway=None (얼굴 가림 근거가 없으므로 가림으로 보지 않음)
    """
    bodies = [d for d in dets if d.cls in BODY_CLASSES and d.conf >= conf_body]
    if not bodies:
        if covered_without_body:
            covered = [d for d in dets if d.cls == "airway_covered" and d.conf >= conf_airway]
            if covered:
                face = max(covered, key=lambda d: d.conf)
                return FrameDecision(valid=True, reason="body_not_found", airway=face.cls, face=face)
        return FrameDecision(valid=False, reason="no_baby")
    body = max(bodies, key=lambda d: d.conf)
    faces = [d for d in dets
             if d.cls in AIRWAY_CLASSES and d.conf >= conf_airway and center_inside(d, body)]
    face = max(faces, key=lambda d: d.conf) if faces else None
    return FrameDecision(valid=True, posture=body.cls, airway=face.cls if face else None,
                         body=body, face=face)
