"""영상 입력. read()가 (프레임 BGR, 시각[초])을 돌려주고, 끝나면 None을 돌려줘요.

- 카메라(실시간): 시각 = time.monotonic()
- 영상 파일: 시각 = 영상 안에서의 재생 시각 (실제보다 빨리 처리해도 5초 판정이 정확해요)
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

Frame = Tuple[np.ndarray, float]


class OpenCVSource:
    def __init__(self, src, live: bool, size: Optional[Tuple[int, int]] = None):
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"입력을 열 수 없어요: {src}")
        if live and size:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.live = live
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._index = 0

    def read(self) -> Optional[Frame]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        ts = time.monotonic() if self.live else self._index / self.fps
        self._index += 1
        return frame, ts

    def close(self) -> None:
        self.cap.release()


class Picamera2Source:
    """라즈베리파이 카메라 (sudo apt install -y python3-picamera2)."""

    live = True

    def __init__(self, size: Tuple[int, int] = (640, 480)):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(main={"size": tuple(size), "format": "RGB888"})
        self.cam.configure(config)
        self.cam.start()

    def read(self) -> Optional[Frame]:
        frame = self.cam.capture_array()  # RGB888 형식은 메모리상 BGR 순서라 OpenCV에 바로 써요
        return frame, time.monotonic()

    def close(self) -> None:
        self.cam.stop()


def open_source(spec: str, size: Tuple[int, int] = (640, 480)):
    """'picamera2' → 파이 카메라, 숫자 → 웹캠 번호, 그 외 → 영상 파일 경로."""
    if spec == "picamera2":
        return Picamera2Source(size)
    if spec.isdigit():
        return OpenCVSource(int(spec), live=True, size=size)
    return OpenCVSource(spec, live=False)
