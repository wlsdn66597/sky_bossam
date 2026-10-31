"""YOLO 학습 한 번 실행. 기본값은 configs/train.yaml.

python scripts/train.py --data datasets/sleepsafe/data_original.yaml --imgsz 416 --name A_original_416
python scripts/train.py --data datasets/sleepsafe/data_original.yaml --set epochs=50 batch=32
결과: runs/train/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import ROOT, load_yaml, parse_overrides, repo_path  # noqa: E402

PROJECT = ROOT / "runs" / "train"


def train(data: Path, name: str, imgsz: Optional[int] = None, overrides: Optional[Dict[str, Any]] = None,
          config: str = "configs/train.yaml") -> Path:
    from ultralytics import YOLO

    cfg = load_yaml(config)
    cfg.update(overrides or {})
    if imgsz:
        cfg["imgsz"] = int(imgsz)
    model = YOLO(cfg.pop("model"))
    model.train(data=str(repo_path(data)), project=str(PROJECT), name=name, exist_ok=True, **cfg)
    best = PROJECT / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"학습 결과가 없어요: {best}")
    return best


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="prepare_dataset.py가 만든 data_*.yaml")
    ap.add_argument("--name", required=True, help="실험 이름 (결과 폴더 이름)")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="설정 덮어쓰기")
    args = ap.parse_args(argv)
    best = train(Path(args.data), args.name, args.imgsz, parse_overrides(args.set), args.config)
    print(f"학습 완료: {best}")


if __name__ == "__main__":
    main()
