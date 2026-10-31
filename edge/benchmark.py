"""ONNX 모델 추론 속도 재기 (카메라 없이). 라즈베리파이 4에서 모델·입력 크기별로 비교할 때 써요.

python edge/benchmark.py --model models/A_original_320_aug.onnx
python edge/benchmark.py --model models/A_original_320_aug.onnx models/A_original_416_aug.onnx
python edge/benchmark.py --model models/x.onnx --image 사진.jpg --runs 50 --threads 4

시간은 실제 실행과 같은 범위(크기 맞추기 + 추론 + 박스 정리)를 재요. 카메라 읽기는 빠져 있어요.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import load_yaml, repo_path  # noqa: E402


def main(argv=None) -> None:
    import cv2
    import numpy as np
    import onnxruntime as ort

    from sleepsafe.detector_onnx import OnnxDetector
    from sleepsafe.imaging import imread

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", required=True, help="ONNX 모델 (여러 개면 차례로 비교)")
    ap.add_argument("--image", help="시험할 사진 (없으면 640×480 가짜 화면)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, help="ONNX Runtime 스레드 수 (기본: configs/edge.yaml)")
    args = ap.parse_args(argv)

    cfg = load_yaml("configs/edge.yaml")
    threads = args.threads or cfg.get("threads")
    if args.image:
        frame = imread(args.image)
    else:
        frame = np.full((480, 640, 3), 200, np.uint8)
        cv2.rectangle(frame, (220, 140), (420, 380), (90, 120, 200), -1)
    print(f"기기 {platform.machine()} / {platform.system()}, onnxruntime {ort.__version__}, "
          f"스레드 {threads or '자동'}, 화면 {frame.shape[1]}×{frame.shape[0]}, 측정 {args.runs}회")

    conf = min(cfg.get("conf_body", 0.25), cfg.get("conf_airway", 0.25))
    for model in args.model:
        detector = OnnxDetector(repo_path(model), conf=conf, threads=threads)
        for _ in range(args.warmup):
            detector(frame)
        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            detector(frame)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        mean = statistics.mean(times)
        p90 = times[min(len(times) - 1, int(len(times) * 0.9))]
        print(f"{Path(model).name} (입력 {detector.imgsz}): 평균 {mean:.0f}ms, 중앙값 {statistics.median(times):.0f}ms, "
              f"상위10% {p90:.0f}ms → 최대 약 {1000 / mean:.1f}fps")


if __name__ == "__main__":
    main()
