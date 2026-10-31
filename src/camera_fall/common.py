from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    if text.lstrip().startswith(("{", "[")):
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows(path, optional=False):
    p = Path(path)
    if optional and not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def metadata(seed=None):
    packages = {}
    for name in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless", "PyYAML", "onnx", "onnxruntime", "scikit-learn", "pandas", "pyarrow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git = None
    return dict(schema_version=1, python=sys.version, platform=platform.platform(), packages=packages, git_sha=git, seed=seed)


def lock_environment(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    content = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (directory / "requirements.lock.txt").write_text(content, encoding="utf-8")


def finite_number(value, name, minimum=None):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value), f"Invalid {name}: {value}")
    if minimum is not None:
        require(value >= minimum, f"{name} must be >= {minimum}")


def approved(registry, dataset_id):
    matches = [x for x in registry if x["dataset_id"] == dataset_id]
    require(len(matches) == 1, f"Dataset not uniquely registered: {dataset_id}")
    entry = matches[0]
    require(entry.get("approved_for_training") and entry.get("license_verified"), f"Dataset terms not confirmed: {dataset_id}. Register with your terms evidence first.")
    evidence = Path(entry["license_evidence_file"])
    require(evidence.is_file() and sha256(evidence) == entry["license_evidence_sha256"], f"Terms evidence missing/changed: {dataset_id}")
    return entry


def load_split(split, directory="data/splits"):
    require(split in ("train", "val", "test"), "split must be train/val/test")
    data = load(Path(directory) / f"{split}.json")
    for path, checksum in data["manifest_hashes"].items():
        require(Path(path).is_file() and sha256(path) == checksum, f"Manifest changed after split: {path}; rebuild split explicitly")
    return data


def selected_videos(split, directory="data/splits"):
    s = load_split(split, directory)
    all_rows = rows("data/manifests/videos.jsonl")
    found = {v["video_id"]: v for v in all_rows}
    require(set(s["video_ids"]) <= set(found), "Missing video IDs")
    return [found[i] for i in s["video_ids"]]
