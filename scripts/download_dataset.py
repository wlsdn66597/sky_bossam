"""Roboflow에서 Baby-Position 데이터셋(YOLOv8 형식)을 받아요.

1) https://app.roboflow.com 에 가입하고 Settings → API Keys에서 Private API Key를 복사
2) 환경변수로 넣고 실행 (키를 코드나 깃에 넣지 마세요)
   Windows PowerShell:  $env:ROBOFLOW_API_KEY="키"; python scripts/download_dataset.py
   Linux/macOS:         ROBOFLOW_API_KEY=키 python scripts/download_dataset.py

웹사이트에서 직접 받아도 돼요: 데이터셋 페이지 → Download Dataset → YOLOv8 → zip을
configs/dataset.yaml의 raw_dir(기본 datasets/raw/baby-position-v1)에 풀면 됩니다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import load_yaml, repo_path  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("환경변수 ROBOFLOW_API_KEY가 없어요. 파일 맨 위 설명을 참고하세요.")

    from roboflow import Roboflow

    r = cfg["roboflow"]
    out = repo_path(cfg["raw_dir"])
    out.parent.mkdir(parents=True, exist_ok=True)
    project = Roboflow(api_key=key).workspace(r["workspace"]).project(r["project"])
    project.version(r["version"]).download(r["format"], location=str(out), overwrite=True)
    print(f"받기 완료: {out}")
    print("다음 단계: python scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
