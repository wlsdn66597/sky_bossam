from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .common import dump,lock_environment,metadata,require


def preflight(args):
    from .detector import gpu_check
    result = dict(metadata(),**gpu_check(args.expected_gpu))
    dump(Path(args.out)/"environment.json",result)
    lock_environment(args.out)
    print(json.dumps({k:result[k] for k in ("gpu","vram_gib","cuda")},ensure_ascii=False))


def download_roboflow(args):
    from getpass import getpass
    from roboflow import Roboflow
    require(not Path(args.out).exists(),"Download destination already exists")
    key = os.environ.get("ROBOFLOW_API_KEY") or getpass("Roboflow API key (hidden): ")
    require(key.strip(),"No API key provided")
    project = Roboflow(api_key=key).workspace("darios-workspace-w6cci").project("baby-crib-observability")
    project.version(1).download("yolov8",location=str(Path(args.out).resolve()))
    print("Downloaded v1. Register this folder to inventory images, labels and checksums.")


def mini_rgbd_to_mp4(args):
    root = Path(args.root)
    require(root.is_dir(),"Extract MINI-RGBD_web.zip first")
    sequences = [f"{i:02d}" for i in range(1,13)] if args.sequence == "all" else [args.sequence]
    require(all(s in {f"{i:02d}" for i in range(1,13)} for s in sequences),"Sequence must be 01..12")
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    for seq in sequences:
        rgb = root/seq/"rgb"
        expected = {f"syn_{i:05d}.png" for i in range(1000)}
        actual = {p.name for p in rgb.glob("syn_*.png")}
        require(actual == expected,f"{seq}: expected RGB frames syn_00000.png .. syn_00999.png; depth images are not RGB")
        dest = out/f"mini_rgbd_{seq}.mp4"
        require(not dest.exists(),f"Output exists: {dest}")
        subprocess.run([args.ffmpeg,"-hide_banner","-loglevel","error","-n","-framerate","30","-start_number","0","-i",str(rgb/"syn_%05d.png"),"-frames:v","1000","-c:v","libx264","-preset","fast","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(dest)],check=True)
        result = json.loads(subprocess.check_output([args.ffprobe,"-v","error","-select_streams","v:0","-count_frames","-show_entries","stream=width,height,r_frame_rate,nb_read_frames,duration","-of","json",str(dest)],text=True))
        stream = result["streams"][0]
        require(int(stream["nb_read_frames"]) == 1000 and stream["r_frame_rate"] == "30/1","Encoded frame count/FPS mismatch")
        dump(dest.with_suffix(".json"),dict(dataset="MINI-RGBD",subject_kind="synthetic",sequence=seq,stream=stream,audio=False,fall_labels=False,use="pipeline/domain-shift evaluation only",license="CC BY-NC-SA 4.0",source="https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html"))
        print(dest)
