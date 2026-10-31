"""이미지 읽기·밝기 측정·스냅샷 인코딩·화면 표시."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np

from .decision import Detection, FrameDecision

COLORS = {  # BGR
    "supine": (80, 200, 80), "side": (200, 160, 60), "prone": (40, 40, 230),
    "airway_visible": (200, 200, 200), "airway_covered": (0, 140, 255),
}
STATUS_COLORS = {"normal": (80, 200, 80), "caution": (0, 200, 255),
                 "risk": (40, 40, 230), "unavailable": (160, 160, 160)}


def imread(path: Union[str, Path]) -> np.ndarray:
    """한글 경로에서도 읽히도록 numpy로 읽어서 디코딩해요 (윈도우 cv2.imread 한글 경로 문제 회피)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없어요: {path}")
    return img


def brightness(img_bgr: np.ndarray) -> float:
    """평균 밝기 (0~255)."""
    small = cv2.resize(img_bgr, (64, 48), interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


def encode_jpeg_b64(img_bgr: np.ndarray, quality: int = 80, max_width: int = 640) -> str:
    h, w = img_bgr.shape[:2]
    if w > max_width:
        img_bgr = cv2.resize(img_bgr, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG 인코딩 실패")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def draw(img_bgr: np.ndarray, dets: Sequence[Detection], decision: Optional[FrameDecision],
         status: str, info: str = "") -> np.ndarray:
    """탐지 박스와 상태를 그려요. (OpenCV는 한글을 못 그려서 영어로 표시)"""
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = COLORS.get(d.cls, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{d.cls} {d.conf:.2f}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    label = status.upper()
    if decision is not None and not decision.valid:
        label += f" ({decision.reason})"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, f"{label}  {info}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                STATUS_COLORS.get(status, (255, 255, 255)), 2, cv2.LINE_AA)
    return out
