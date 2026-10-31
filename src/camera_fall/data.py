"""Auditable detection conversion, conservative grouped splits and video validation."""
from __future__ import annotations

import csv
import math
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .common import approved, dump, load, object_sha, rel, require, rows, sha256, write_rows

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BODY = {"baby", "babybody", "babysupine", "babysideways", "babyprone", "supine", "side", "sideways", "prone"}
IGNORE = {"airwayvisible", "airwaycovered", "activeparent", "toybeside"}
GROUP_KEYS = ("source_group", "subject_id", "bed_id", "session_id", "video_id", "event_group_id", "duplicate_group_id", "sha256")


def normalize_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def convert_labels(text, names):
    """Never merge face-only boxes into body boxes, or assume empty == negative."""
    normalized = {int(i): normalize_name(n) for i, n in names.items()}
    unknown = set(normalized.values()) - BODY - IGNORE
    require(not unknown, f"Unknown classes: {sorted(unknown)}. Supply an explicit mapping after inspection.")
    kept, original = [], []
    for line in text.splitlines():
        if not line.strip():
            continue
        v = line.split()
        require(len(v) == 5, "Expected YOLO detection box: class cx cy w h; segmentation is unsupported")
        require(v[0].isdigit() and int(v[0]) in normalized, f"Invalid class: {v[0]}")
        c = int(v[0])
        x, y, w, h = map(float, v[1:])
        require(all(math.isfinite(a) for a in (x, y, w, h)), "Non-finite box")
        require(0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1, "Invalid normalized box")
        require(x - w/2 >= -1e-5 and x + w/2 <= 1.00001 and y - h/2 >= -1e-5 and y + h/2 <= 1.00001, "Box extends outside image")
        if normalized[c] in BODY:
            kept.append("0 " + " ".join(f"{a:.8f}" for a in (x, y, w, h)))
            original.append(names[c])
    return kept, original


def register(args):
    source = Path(args.input).resolve()
    require(source.exists(), f"Missing input: {source}")
    archive_sha = None
    if source.is_file():
        require(zipfile.is_zipfile(source), "Input must be a folder or ZIP")
        archive_sha = sha256(source)
        dest = Path("data/raw") / args.dataset
        require(not dest.exists(), f"Extraction destination exists: {dest}")
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                target = (dest / info.filename).resolve()
                require(target.is_relative_to(dest.resolve()), "Unsafe archive path")
                require((info.external_attr >> 16) & 0o170000 != 0o120000, "Archive symlink rejected")
            z.extractall(dest)
        source = dest.resolve()
    inventory = [{"path": p.relative_to(source).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)} for p in sorted(source.rglob("*")) if p.is_file()]
    require(inventory, "Empty dataset")
    evidence = Path(args.terms_file).resolve() if args.terms_file else None
    require(not args.approve_for_training or evidence is not None and evidence.is_file(), "--approve-for-training requires --terms-file with your source/terms record")
    entry = dict(dataset_id=args.dataset, version=args.version, source_url=args.source_url, root=rel(source), status="registered", license_verified=bool(evidence), approved_for_training=args.approve_for_training, license_evidence_file=rel(evidence) if evidence else None, license_evidence_sha256=sha256(evidence) if evidence else None, archive_sha256=archive_sha, inventory_sha256=object_sha(inventory), actual_image_count=sum(Path(i["path"]).suffix.lower() in IMAGE_EXT for i in inventory))
    registry = load(args.registry) if Path(args.registry).exists() else []
    require(not any(x["dataset_id"] == args.dataset for x in registry), "Dataset already registered; use a new version ID instead of replacing its history")
    dump(Path("data/registry") / f"{args.dataset}.inventory.json", inventory)
    dump(args.registry, registry + [entry])
    print(f"Registered {args.dataset}: {entry['actual_image_count']} images")


def dhash(path):
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8))
        px = list(im.getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            result = (result << 1) | (px[y*9+x] > px[y*9+x+1])
    return f"{result:016x}"


def prepare(args):
    entry = approved(load(args.registry), args.dataset)
    root = Path(entry["root"])
    yamls = list(root.rglob("data.yaml"))
    require(len(yamls) == 1, "Expected exactly one data.yaml in raw dataset")
    base = yamls[0].parent
    names = load(yamls[0])["names"]
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    convert_labels("", names)  # Reject unsupported names even when unused.
    inventory = load(Path("data/registry") / f"{args.dataset}.inventory.json")
    checksums = {x["path"]: x["sha256"] for x in inventory}
    attrs = {}
    if args.groups:
        with open(args.groups, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                require(row["image_path"] not in attrs, "Duplicate group CSV key")
                attrs[row["image_path"]] = row
    records, held = [], []
    used_attrs = set()
    for split, dirname in (("train", "train"), ("val", "valid"), ("test", "test")):
        folder = base / dirname / "images"
        if split == "val" and not folder.exists():
            folder = base / "val/images"
        require(folder.is_dir(), f"Missing original split directory: {folder}")
        for img in sorted(folder.rglob("*")):
            if img.suffix.lower() not in IMAGE_EXT:
                continue
            key = img.relative_to(base).as_posix()
            label = folder.parent / "labels" / img.relative_to(folder).with_suffix(".txt")
            for p in (img, label):
                require(p.is_file(), f"Missing image/label: {p}; do not assume absent labels are negatives")
                require(checksums.get(p.relative_to(root).as_posix()) == sha256(p), f"Raw dataset changed since registration: {p}")
            boxes, posture = convert_labels(label.read_text(encoding="utf-8-sig"), names)
            a = attrs.get(key, {})
            if a:
                used_attrs.add(key)
            if a.get("subject_kind") == "adult":
                held.append(dict(image_path=key, reason="adult_not_in_p0"))
                continue
            if not boxes and a.get("negative_verified", "").lower() != "true":
                held.append(dict(image_path=key, reason="no_body_label_not_verified_negative"))
                continue
            image_id = args.dataset + "_" + object_sha(key)[:16]
            newlabel = Path("data/prepared") / args.dataset / "labels" / f"{image_id}.txt"
            newlabel.parent.mkdir(parents=True, exist_ok=True)
            newlabel.write_text("\n".join(boxes) + ("\n" if boxes else ""), encoding="utf-8")
            record = dict(image_id=image_id, dataset_id=args.dataset, image_path=rel(img), label_path=rel(newlabel), sha256=sha256(img), label_sha256=sha256(newlabel), dhash=dhash(img), original_split=split, original_classes=posture, negative_verified=not boxes, frame_ts_s=None, split=None)
            for k in GROUP_KEYS:
                if k != "sha256":
                    record[k] = a.get(k) or None
            record["source_group"] = a.get("source_group") or f"{args.dataset}:unknown_source"
            record["subject_kind"] = a.get("subject_kind") or "unknown"
            for k in ("posture", "visibility", "modality"):
                record[k] = a.get(k) or "unknown"
            records.append(record)
    require(set(attrs) == used_attrs | {x["image_path"] for x in held if x["image_path"] in attrs}, "Group CSV has unmatched paths")
    require(records, "No usable body images after conversion")
    out = Path("data/manifests/images.jsonl")
    previous = rows(out, optional=True)
    require(not any(x["dataset_id"] == args.dataset for x in previous), "Dataset already prepared; preserve manifest and explicitly start a new experiment")
    write_rows(out, previous + records)
    dump(f"reports/data_audit_{args.dataset}.json", dict(class_names=names, used=len(records), held=held, original_splits=dict(Counter(x["original_split"] for x in records)), evidence=entry))
    print(f"Prepared {len(records)} images; held {len(held)}. Groups unknown by default; strict splits need reviewed metadata.")


def components(records, extra_pairs=()):
    parent = list(range(len(records)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        parent[find(a)] = find(b)
    seen = {}
    for i, row in enumerate(records):
        for key in GROUP_KEYS:
            value = row.get(key)
            if value:
                token = (key, str(value))
                if token in seen:
                    union(i, seen[token])
                seen[token] = i
    by_id = {r.get("image_id", r.get("video_id")): i for i, r in enumerate(records)}
    for a, b in extra_pairs:
        require(a in by_id and b in by_id, "Unknown ID in duplicate review")
        union(by_id[a], by_id[b])
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(i)
    return list(groups.values())


def split_data(args):
    images = rows("data/manifests/images.jsonl", optional=True)
    videos = rows("data/manifests/videos.jsonl", optional=True)
    require(images or videos, "No manifests prepared")
    ids = [r["image_id"] for r in images] + [r["video_id"] for r in videos]
    require(len(ids) == len(set(ids)), "Duplicate IDs in manifests")
    review = {}
    if args.duplicate_review:
        with open(args.duplicate_review, encoding="utf-8-sig", newline="") as f:
            review = {(x["image_a"], x["image_b"]): x["decision"] for x in csv.DictReader(f)}
    candidates, duplicate_pairs = [], []
    for i, a in enumerate(images):
        for b in images[i+1:]:
            if a["sha256"] == b["sha256"]:
                duplicate_pairs.append((a["image_id"], b["image_id"]))
                continue
            if a.get("dhash") is None or b.get("dhash") is None:
                continue
            distance = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            if distance <= args.hash_distance:
                key = (a["image_id"], b["image_id"])
                decision = review.get(key, "unreviewed")
                require(decision in ("unreviewed", "duplicate", "distinct"), "Duplicate review decision must be duplicate/distinct")
                candidates.append(dict(image_a=key[0], image_b=key[1], hamming=distance, decision=decision))
                if decision == "duplicate":
                    duplicate_pairs.append(key)
    Path("reports").mkdir(exist_ok=True)
    with open("reports/duplicate_candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_a", "image_b", "hamming", "decision"])
        w.writeheader(); w.writerows(candidates)
    records = images + videos
    grouping = components(records, duplicate_pairs)
    allocation = {}
    if args.mode == "original":
        require(not videos, "Original reproduction mode is only for public still images; use grouped splits for videos")
        allocation = {i: r["original_split"] for i, r in enumerate(records)}
    else:
        require(not any(c["decision"] == "unreviewed" for c in candidates), "Review reports/duplicate_candidates.csv, then rerun with --duplicate-review. No automatic near-duplicate identity inference.")
        require(len(grouping) >= 3, f"Only {len(grouping)} independent connected groups. Add source/subject/session metadata; no random frame split fallback.")
        rng = random.Random(args.seed)
        rng.shuffle(grouping)
        grouping.sort(key=len, reverse=True)
        counts = Counter()
        goals = dict(train=.70 * len(records), val=.15 * len(records), test=.15 * len(records))
        for j, group in enumerate(grouping):
            remaining = len(grouping) - j
            empty = [s for s in goals if counts[s] == 0]
            s = empty[0] if remaining == len(empty) else max(goals, key=lambda x: goals[x] - counts[x])
            for i in group:
                allocation[i] = s
            counts[s] += len(group)
    hashes = {p: sha256(p) for p in ("data/manifests/images.jsonl", "data/manifests/videos.jsonl") if Path(p).exists()}
    exact_cross = []
    for a, b in duplicate_pairs:
        if allocation[ids.index(a)] != allocation[ids.index(b)]:
            exact_cross.append([a, b])
    group_cross = [g for g in grouping if len({allocation[i] for i in g}) > 1]
    require(args.mode == "original" or not exact_cross and not group_cross, "Internal split leakage")
    destination = Path(args.out)
    require(not any((destination / f"{s}.json").exists() for s in ("train", "val", "test")), "Split already frozen; use a new project copy for a different split")
    for split in ("train", "val", "test"):
        ix = [i for i in range(len(records)) if allocation[i] == split]
        require(ix, f"Empty split: {split}")
        dump(destination / f"{split}.json", dict(schema_version=1, seed=args.seed, mode=args.mode, manifest_hashes=hashes, group_constraints=list(GROUP_KEYS), image_ids=[records[i]["image_id"] for i in ix if "image_id" in records[i]], video_ids=[records[i]["video_id"] for i in ix if "image_id" not in records[i]], groups=[{k: records[i].get(k) for k in GROUP_KEYS} for i in ix]))
        for i in ix:
            r = records[i]
            if "image_id" not in r:
                continue
            for kind, source in (("images", r["image_path"]), ("labels", r["label_path"])):
                dest = Path("data/detection") / split / kind / (r["image_id"] + Path(source).suffix)
                dest.parent.mkdir(parents=True, exist_ok=True)
                require(not dest.exists(), f"Would overwrite prepared data: {dest}")
                shutil.copy2(source, dest)
    dump("configs/datasets.yaml", dict(path=str(Path("data/detection").resolve()), train="train/images", val="val/images", test="test/images", names={0: "baby_body"}))
    dump("reports/leakage_report.json", dict(mode=args.mode, independent_groups=len(grouping), cross_split_duplicate_pairs=exact_cross, cross_split_group_count=len(group_cross), unreviewed_near_duplicates=sum(x["decision"] == "unreviewed" for x in candidates), generalization_claim_allowed=args.mode == "grouped", caveat="Original split is reproduction only; unknown identity/session independence is not asserted."))
    print(f"Fixed {args.mode} splits saved. Review reports/leakage_report.json before interpreting metrics.")


def validate_events(videos, events):
    by_id = {v["video_id"]: v for v in videos}
    require(len(by_id) == len(videos), "Duplicate video IDs")
    ids = set()
    allowed = {"edge_approach", "rail_climb_attempt", "unassisted_bed_exit", "fall_observed", "caregiver_pickup", "occlusion", "camera_unavailable"}
    for e in events:
        require(e["event_id"] not in ids, "Duplicate event ID")
        ids.add(e["event_id"])
        require(e["video_id"] in by_id, "Unknown event video")
        v = by_id[e["video_id"]]
        require(e["event_type"] in allowed, "Unknown event type")
        require(e.get("event_group_id") == v.get("event_group_id"), "Event group mismatch")
        require(isinstance(e.get("surrogate"), bool), "Event needs surrogate boolean")
        require(e["surrogate"] == (v["subject_kind"] in ("doll", "synthetic")), "Surrogate/subject mismatch")
        times = [e.get(k) for k in ("action_start_s", "critical_ts_s", "fall_onset_s", "impact_s", "end_s")]
        known = [x for x in times if x is not None]
        require(known and all(math.isfinite(x) and 0 <= x <= v["duration_s"] for x in known), "Event time outside video")
        require(known == sorted(known), "Event timestamps not ordered")
        require(e.get("action_start_s") is not None and e.get("end_s") is not None, "Event start/end required")
        if e["event_type"] == "unassisted_bed_exit":
            require(e.get("critical_ts_s") is not None, "Exit needs critical_ts_s")


def prepare_videos(args):
    from .geometry import validate_geometry
    import cv2
    videos, events = rows(args.videos), rows(args.events)
    require(videos, "Empty videos manifest")
    registry = load(args.registry)
    validate_events(videos, events)
    for v in videos:
        approved(registry, v["dataset_id"])
        require(v["subject_kind"] in ("doll", "real_infant", "synthetic"), "P0 does not use adult subjects")
        require(all(v.get(k) for k in ("source_group", "session_id", "event_group_id", "camera_setup_id")), "Video missing source/session/group/camera IDs")
        require(v["width"] > 0 and v["height"] > 0 and v["duration_s"] > 0, "Invalid video dimensions/duration")
        g = load(Path("data/geometry") / f"{v['camera_setup_id']}.json")
        validate_geometry(g)
        cap = cv2.VideoCapture(str(v["path"]))
        try:
            require(cap.isOpened(), f"Cannot decode {v['path']}")
            require(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == v["width"] and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == v["height"], "Video dimensions differ from manifest")
        finally:
            cap.release()
        for a, b in v.get("annotation_coverage", []):
            require(0 <= a < b <= v["duration_s"], "Invalid annotation_coverage")
    dump("reports/video_audit.json", dict(videos=len(videos), events=len(events), video_manifest_sha256=sha256(args.videos), events_sha256=sha256(args.events)))
    if args.extract:
        from .video import frames
        from .common import load_split
        require(args.split is not None, "--extract requires --split, fixed before extracting frames")
        allowed_ids = set(load_split(args.split)["video_ids"])
        require(allowed_ids, "No videos in chosen split")
        for v in videos:
            if v["video_id"] not in allowed_ids:
                continue
            folder = Path("data/unlabeled_frames") / v["video_id"]
            require(not folder.exists(), f"Frame output exists: {folder}")
            folder.mkdir(parents=True)
            manifest = []
            for index, ts, frame in frames(v["path"], sample_hz=args.sample_hz, timestamps_file=v.get("timestamps_file")):
                p = folder / f"frame_{index:08d}.jpg"
                require(cv2.imwrite(str(p), frame), "Frame write failed")
                manifest.append(dict(image_path=rel(p), video_id=v["video_id"], frame_ts_s=ts, split=args.split, label_status="needs_manual_body_boxes", **{k: v.get(k) for k in GROUP_KEYS if k != "video_id"}))
            write_rows(folder / "frames.jsonl", manifest)
        print("Frames extracted without labels; not included in training until manually annotated.")
