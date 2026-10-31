"""학습한 모델을 라즈베리파이 4용으로 내보내요 (기본 ONNX, 선택 NCNN).

python scripts/export_model.py --weights runs/train/A_original_416/weights/best.pt --imgsz 416
python scripts/export_model.py --weights ... --imgsz 320 --formats onnx ncnn
결과: models/<실험이름>.onnx  → 파이로 복사해서 configs/edge.yaml의 model에 지정
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import ROOT, repo_path  # noqa: E402

MODELS = ROOT / "models"


def export(weights, imgsz: int, formats: Sequence[str] = ("onnx",), name: Optional[str] = None,
           opset: Optional[int] = None) -> List[Path]:
    from ultralytics import YOLO

    weights = repo_path(weights)
    name = name or (weights.parent.parent.name if weights.parent.name == "weights" else weights.stem)
    MODELS.mkdir(exist_ok=True)
    outs = []
    for fmt in formats:
        kwargs = {"format": fmt, "imgsz": int(imgsz)}
        if fmt == "onnx" and opset:
            kwargs["opset"] = int(opset)
        produced = Path(YOLO(str(weights)).export(**kwargs))
        dst = MODELS / (f"{name}.onnx" if fmt == "onnx" else f"{name}_{fmt}_model")
        if produced.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(produced, dst)
        else:
            shutil.copy2(produced, dst)
        outs.append(dst)
        print(f"내보내기 완료: {dst}")
    return outs


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True, help="파이에서 쓸 입력 크기 (학습 때와 같게)")
    ap.add_argument("--formats", nargs="+", default=["onnx"], choices=["onnx", "ncnn"])
    ap.add_argument("--name", help="파일 이름 (기본: 실험 이름)")
    ap.add_argument("--opset", type=int, help="파이의 onnxruntime이 오래돼 열리지 않으면 12처럼 낮춰 보세요")
    args = ap.parse_args(argv)
    export(args.weights, args.imgsz, args.formats, args.name, args.opset)


if __name__ == "__main__":
    main()
