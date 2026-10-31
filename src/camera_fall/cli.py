"""Small entry points keep heavyweight training imports away from Pi runtime."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .common import ROOT, require


def main(command):
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=f"camera-fall-research: {command}")
    if command == "register_dataset":
        from .data import register as fn
        p.add_argument("--dataset",default="baby_crib_v1")
        p.add_argument("--input",required=True)
        p.add_argument("--registry",default="data/registry/datasets.json")
        p.add_argument("--version",default="1")
        p.add_argument("--source-url",default="https://universe.roboflow.com/darios-workspace-w6cci/baby-crib-observability/dataset/1")
        p.add_argument("--terms-file")
        p.add_argument("--approve-for-training",action="store_true")
    elif command in ("prepare_baby_position","prepare_baby_crib"):
        from .data import prepare as fn
        p.add_argument("--dataset",default="baby_position_v1" if command == "prepare_baby_position" else "baby_crib_v1")
        p.add_argument("--registry",default="data/registry/datasets.json")
        p.add_argument("--groups",help="CSV keyed by image_path relative to original data.yaml")
    elif command == "make_splits":
        from .data import split_data as fn
        p.add_argument("--seed",type=int,default=42)
        p.add_argument("--mode",choices=["grouped","original"],default="grouped")
        p.add_argument("--group-aware",action="store_true",help="Explicit grouped-mode assertion")
        p.add_argument("--out",default="data/splits")
        p.add_argument("--duplicate-review")
        p.add_argument("--hash-distance",type=int,default=4)
    elif command == "prepare_own_videos":
        from .data import prepare_videos as fn
        p.add_argument("--videos",default="data/manifests/videos.jsonl")
        p.add_argument("--events",default="data/manifests/events.jsonl")
        p.add_argument("--registry",default="data/registry/datasets.json")
        p.add_argument("--extract",action="store_true")
        p.add_argument("--split",choices=["train","val","test"])
        p.add_argument("--sample-hz",type=float,default=1.)
    elif command == "train_detector":
        from .detector import train as fn
        p.add_argument("--config",default="configs/detector_n416.yaml")
        p.add_argument("--smoke",action="store_true")
        p.add_argument("--expected-gpu",default="L4")
        p.add_argument("--allow-reproduction-split",action="store_true")
    elif command == "eval_detector":
        from .detector import evaluate as fn
        p.add_argument("--weights",required=True)
        p.add_argument("--split",choices=["val","test"],default="val")
        p.add_argument("--imgsz",type=int,default=416)
        p.add_argument("--device",default="0")
        p.add_argument("--out",default="reports/detector_val")
        p.add_argument("--allow-reproduction-split",action="store_true")
    elif command == "extract_features":
        from .video import extract as fn
        p.add_argument("--weights",required=True)
        p.add_argument("--split",choices=["train","val","test"],required=True)
        p.add_argument("--policy",default="configs/risk_r0.yaml")
        p.add_argument("--sample-hz",type=float,default=5.)
        p.add_argument("--imgsz",type=int,default=416)
        p.add_argument("--device",default="cpu")
        p.add_argument("--confidence",type=float,default=.25)
    elif command == "tune_r0":
        from .evaluation import tune as fn
        p.add_argument("--features",required=True)
        p.add_argument("--events",default="data/manifests/events.jsonl")
        p.add_argument("--split",choices=["val"],default="val")
        p.add_argument("--config",default="configs/risk_r0.yaml")
        p.add_argument("--out",default="reports/selected_r0.yaml")
    elif command == "eval_events":
        from .evaluation import evaluate as fn
        p.add_argument("--features",required=True)
        p.add_argument("--events",default="data/manifests/events.jsonl")
        p.add_argument("--split",choices=["val","test"],default="test")
        p.add_argument("--policy",choices=["r0"],default="r0")
        p.add_argument("--frozen-config",required=True)
        p.add_argument("--out",default="reports/event_metrics.json")
    elif command == "train_temporal":
        from .temporal import train as fn
        p.add_argument("--features",required=True)
        p.add_argument("--events",default="data/manifests/events.jsonl")
        p.add_argument("--config",default="configs/temporal_gru.yaml")
        p.add_argument("--model",choices=["r1","r2"],default="r2")
        p.add_argument("--out",default="runs/temporal")
        p.add_argument("--expected-gpu",default="L4")
    elif command == "export_onnx":
        from .detector import export as fn
        p.add_argument("--weights",required=True)
        p.add_argument("--imgsz",type=int,default=416)
        p.add_argument("--out",default="exports/body_n416")
        p.add_argument("--samples",type=int,default=32)
        p.add_argument("--policy",default="configs/risk_r0.yaml")
    elif command in ("replay_video","benchmark_runtime","run_camera"):
        from .video import replay,live
        fn = live if command == "run_camera" else replay
        p.add_argument("--weights",required=True)
        p.add_argument("--source",required=True)
        p.add_argument("--geometry",required=True)
        p.add_argument("--policy",default="configs/risk_r0.yaml")
        p.add_argument("--imgsz",type=int,default=416)
        p.add_argument("--confidence",type=float,default=.25)
        p.add_argument("--out",default=f"reports/{command}")
        if command == "run_camera":
            p.add_argument("--seconds",type=float,default=60.)
        else:
            p.add_argument("--video-id",default="replay")
            p.add_argument("--device",default="cpu")
            p.add_argument("--sample-hz",type=float,default=5.)
            p.add_argument("--timestamps-file")
            p.add_argument("--overlay",action="store_true")
    elif command == "preflight":
        from .utilities import preflight as fn
        p.add_argument("--expected-gpu",default="L4")
        p.add_argument("--out",default="reports/preflight")
    elif command == "download_roboflow":
        from .utilities import download_roboflow as fn
        p.add_argument("--out",default="data/raw/baby_crib")
    elif command == "mini_rgbd_to_mp4":
        from .utilities import mini_rgbd_to_mp4 as fn
        p.add_argument("--root",required=True,help="Extracted MINI-RGBD_web directory")
        p.add_argument("--out",default="data/raw/mini_videos")
        p.add_argument("--sequence",default="all",help="01 through 12, or all")
        p.add_argument("--ffmpeg",default="ffmpeg")
        p.add_argument("--ffprobe",default="ffprobe")
    else:
        raise ValueError(f"Unknown command: {command}")
    args = p.parse_args()
    try:
        if command == "make_splits":
            require(not args.group_aware or args.mode == "grouped","--group-aware conflicts with original mode")
        if hasattr(args,"samples"):
            require(args.samples > 0,"samples must be positive")
        if hasattr(args,"sample_hz"):
            require(args.sample_hz > 0,"sample-hz must be positive")
        if hasattr(args,"confidence"):
            require(0 <= args.confidence <= 1,"confidence must lie in [0,1]")
        if hasattr(args,"seconds"):
            require(args.seconds > 0,"seconds must be positive")
        fn(args)
    except (ValueError,FileNotFoundError,ImportError) as exc:
        p.exit(2,f"ERROR: {exc}\n")
