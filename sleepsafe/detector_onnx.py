"""ONNX Runtime으로 YOLO 탐지 모델 실행.

PyTorch 없이 numpy로 전처리·후처리를 해서 라즈베리파이 4에서도 가볍게 돌아가요.
YOLOv8/YOLO11 출력(1, 4+클래스수, N)과 YOLO26 같은 NMS 없는 출력(1, 300, 6)을 모두 처리해요.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .decision import Detection


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """비율을 유지해 size×size로 맞추고 남는 곳은 회색(114)으로 채워요 (Ultralytics와 같은 방식)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - nw) / 2, (size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """겹치는 박스 중 점수가 가장 높은 것만 남겨요. 남긴 인덱스를 점수 순으로 돌려줘요."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        w = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        h = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


def postprocess(output: np.ndarray, conf: float, iou: float, ratio: float, pad: Tuple[int, int],
                orig_shape: Tuple[int, int], max_det: int = 300
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모델 출력 → 원본 이미지 좌표의 (xyxy[N,4], 신뢰도[N], 클래스[N])."""
    out = output[0] if output.ndim == 3 else output
    if out.shape[-1] == 6 and out.shape[0] >= out.shape[-1]:
        # NMS가 모델 안에 들어간 출력: [x1, y1, x2, y2, score, cls]
        m = out[:, 4] >= conf
        xyxy, confs, cls = out[m, :4].astype(np.float32), out[m, 4], out[m, 5].astype(np.int64)
        order = confs.argsort()[::-1][:max_det]
        xyxy, confs, cls = xyxy[order], confs[order], cls[order]
    else:
        # YOLOv8/YOLO11 출력: (4+클래스수, N) → 박스 (cx, cy, w, h) + 클래스별 점수
        pred = out.T
        scores = pred[:, 4:]
        cls = scores.argmax(1)
        confs = scores[np.arange(len(cls)), cls]
        m = confs >= conf
        b, confs, cls = pred[m, :4], confs[m], cls[m]
        xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                         b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=1)
        keep = nms(xyxy + cls[:, None] * 7680.0, confs, iou)[:max_det]  # 클래스별로 따로 NMS
        xyxy, confs, cls = xyxy[keep], confs[keep], cls[keep]

    xyxy = xyxy.astype(np.float32).copy()
    xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad[0]) / ratio
    xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad[1]) / ratio
    h, w = orig_shape
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return xyxy, confs.astype(np.float32), cls.astype(np.int64)


class OnnxDetector:
    """detector(이미지 BGR) → [Detection, ...]"""

    def __init__(self, model_path: Union[str, Path], imgsz: Optional[int] = None, conf: float = 0.25,
                 iou: float = 0.7, threads: Optional[int] = None,
                 names: Optional[Union[Dict[int, str], Sequence[str]]] = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(str(model_path), sess_options=so,
                                            providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        fixed = inp.shape[-1] if isinstance(inp.shape[-1], int) else None
        meta_imgsz = ast.literal_eval(meta["imgsz"])[0] if "imgsz" in meta else None
        if fixed and imgsz and int(imgsz) != fixed:
            print(f"[알림] 모델 입력 크기가 {fixed}로 고정돼 있어 imgsz={imgsz} 대신 {fixed}를 써요.")
        self.imgsz = int(fixed or imgsz or meta_imgsz or 640)
        if names is None and "names" in meta:
            names = ast.literal_eval(meta["names"])
        if isinstance(names, (list, tuple)):
            names = dict(enumerate(names))
        if not names:
            raise ValueError("모델에 클래스 이름 정보가 없어요. names를 직접 넘겨 주세요.")
        self.names = {int(k): str(v) for k, v in names.items()}
        self.conf, self.iou = conf, iou

    def __call__(self, img_bgr: np.ndarray) -> List[Detection]:
        x, r, pad = letterbox(img_bgr, self.imgsz)
        x = x[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
        x = np.ascontiguousarray(x, dtype=np.float32)[None] / np.float32(255.0)
        out = self.session.run(None, {self.input_name: x})[0]
        xyxy, confs, cls = postprocess(out, self.conf, self.iou, r, pad, img_bgr.shape[:2])
        return [Detection(self.names.get(int(c), str(int(c))), float(s), tuple(float(v) for v in b))
                for b, s, c in zip(xyxy, confs, cls)]
