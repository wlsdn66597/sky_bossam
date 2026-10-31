"""원본 데이터셋 → 우리 5개 클래스로 정리하고, 실험용 분할 목록을 만들어요.

- 클래스 정리: configs/dataset.yaml의 class_map대로 바꾸고 null인 클래스(보호자 손, 장난감)는 버려요.
- 분할 A (original): Roboflow에 원래 나뉜 train/valid/test 그대로
- 분할 B (holdout_<출처>): 한 출처의 사진을 통째로 테스트로 빼요 → 처음 보는 촬영 환경에서의 성능

결과 (out_dir, 기본 datasets/sleepsafe):
  images/, labels/              모든 사진과 바뀐 라벨
  splits/<분할>/{train,val,test}.txt
  data_<분할>.yaml               학습 때 --data로 넘기는 파일
  records.json, stats.json      사진별 출처·클래스, 출처·분할별 박스 수

python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --holdout doll_blanket real_baby
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import load_yaml, repo_path  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RAW_SPLITS = {"train": "train", "valid": "val", "test": "test"}  # Roboflow 폴더 → 우리 이름


def load_names(raw: Path) -> List[str]:
    names = yaml.safe_load((raw / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    return list(names)


def source_of(filename: str, groups: List[dict], default: str) -> str:
    for g in groups:
        if re.match(g["pattern"], filename):
            return g["name"]
    return default


def remap_lines(text: str, src_names: List[str], class_map: Dict[str, Optional[str]],
                out_index: Dict[str, int]) -> List[str]:
    lines = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        src = src_names[int(parts[0])]
        if src not in class_map:
            raise KeyError(f"configs/dataset.yaml의 class_map에 '{src}'를 추가해 주세요.")
        dst = class_map[src]
        if dst is not None:
            lines.append(" ".join([str(out_index[dst])] + parts[1:]))
    return lines


def write_split(out: Path, name: str, files: Dict[str, List[str]], classes: List[str]) -> Path:
    sdir = out / "splits" / name
    sdir.mkdir(parents=True, exist_ok=True)
    data = {"path": str(out.resolve()), "names": list(classes)}
    for split in ("train", "val", "test"):
        p = sdir / f"{split}.txt"
        p.write_text("".join(f"{(out / 'images' / f).resolve()}\n" for f in files.get(split, [])),
                     encoding="utf-8")
        data[split] = str(p.resolve())
    y = out / f"data_{name}.yaml"
    y.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return y


def count_boxes(records: List[dict], classes: List[str]) -> Dict[str, int]:
    c = Counter(cls for r in records for cls in r["classes"])
    return {k: c.get(k, 0) for k in classes}


def print_table(title: str, rows: Dict[str, Dict[str, int]], classes: List[str]) -> None:
    print(f"\n{title}")
    head = f"{'':16}" + "".join(f"{c:>15}" for c in classes) + f"{'사진 수':>10}"
    print(head)
    for name, row in rows.items():
        print(f"{name:16}" + "".join(f"{row.get(c, 0):>15}" for c in classes) + f"{row.get('_images', 0):>10}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dataset.yaml")
    ap.add_argument("--raw", help="원본 폴더 (기본: config의 raw_dir)")
    ap.add_argument("--out", help="결과 폴더 (기본: config의 out_dir)")
    ap.add_argument("--holdout", nargs="*", help="테스트로 뺄 출처 이름들 (기본: config의 holdout_groups)")
    args = ap.parse_args(argv)

    cfg = load_yaml(args.config)
    raw = repo_path(args.raw or cfg["raw_dir"])
    out = repo_path(args.out or cfg["out_dir"])
    if not (raw / "data.yaml").exists():
        sys.exit(f"{raw / 'data.yaml'} 이 없어요. 먼저 scripts/download_dataset.py로 받아 주세요.")

    classes: List[str] = list(cfg["classes"])
    out_index = {c: i for i, c in enumerate(classes)}
    class_map = cfg["class_map"]
    groups = cfg.get("source_groups", [])
    default_group = cfg.get("default_group", "other")
    src_names = load_names(raw)

    for sub in ("images", "labels", "splits"):  # 다시 실행해도 깨끗하게
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)

    records: List[dict] = []
    used = set()
    for raw_split, split in RAW_SPLITS.items():
        img_dir = raw / raw_split / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = raw / raw_split / "labels" / f"{img.stem}.txt"
            lines = remap_lines(lbl.read_text(encoding="utf-8") if lbl.exists() else "",
                                src_names, class_map, out_index)
            name = img.name if img.name not in used else f"{split}_{img.name}"
            used.add(name)
            shutil.copy2(img, out / "images" / name)
            (out / "labels" / f"{Path(name).stem}.txt").write_text(
                "".join(f"{line}\n" for line in lines), encoding="utf-8")
            records.append({"image": name, "split": split,
                            "source": source_of(img.name, groups, default_group),
                            "classes": [classes[int(line.split()[0])] for line in lines]})
    if not records:
        sys.exit(f"{raw} 아래에서 사진을 찾지 못했어요 (train/images, valid/images, test/images).")

    # 분할 A: 원래 분할
    original = defaultdict(list)
    for r in records:
        original[r["split"]].append(r["image"])
    made = [write_split(out, "original", original, classes)]

    # 분할 B: 출처 하나를 통째로 테스트로
    rng = random.Random(cfg.get("seed", 0))
    holdouts = args.holdout if args.holdout is not None else cfg.get("holdout_groups", [])
    sources = sorted({r["source"] for r in records})
    for g in holdouts:
        if g not in sources:
            print(f"[경고] 출처 '{g}'의 사진이 없어 holdout_{g} 분할을 건너뛰어요. 있는 출처: {sources}")
            continue
        test = [r["image"] for r in records if r["source"] == g]
        rest = [r["image"] for r in records if r["source"] != g]
        rng.shuffle(rest)
        n_val = max(1, int(len(rest) * cfg.get("val_fraction", 0.15)))
        made.append(write_split(out, f"holdout_{g}", {"train": rest[n_val:], "val": rest[:n_val], "test": test},
                                classes))

    # 통계
    by_source = {}
    for s in sources:
        recs = [r for r in records if r["source"] == s]
        by_source[s] = dict(count_boxes(recs, classes), _images=len(recs))
    by_split = {}
    for s in ("train", "val", "test"):
        recs = [r for r in records if r["split"] == s]
        by_split[s] = dict(count_boxes(recs, classes), _images=len(recs))
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "stats.json").write_text(json.dumps({"by_source": by_source, "by_original_split": by_split},
                                               ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"사진 {len(records)}장 정리 완료 → {out}")
    print_table("[출처별 박스 수]", by_source, classes)
    print_table("[원래 분할별 박스 수]", by_split, classes)
    for g in holdouts:
        if g in by_source:
            missing = [c for c in classes if by_source[g].get(c, 0) == 0]
            if missing:
                print(f"[주의] holdout_{g} 테스트에는 {missing} 박스가 없어 그 클래스는 평가되지 않아요.")
    print("\n만든 데이터 파일:")
    for y in made:
        print(" ", y)


if __name__ == "__main__":
    main()
