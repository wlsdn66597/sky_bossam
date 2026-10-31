from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .common import approved, dump, load, load_split, lock_environment, metadata, rel, require, rows, sha256


def letterbox(frame, size):
    import cv2
    h,w = frame.shape[:2]
    scale = min(size/w, size/h)
    nw,nh = round(w*scale), round(h*scale)
    left,top = round((size-nw)/2-.1), round((size-nh)/2-.1)
    image = cv2.resize(frame, (nw,nh), interpolation=cv2.INTER_LINEAR)
    image = cv2.copyMakeBorder(image, top, size-nh-top, left, size-nw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
    tensor = np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)[None], dtype=np.float32) / 255.
    return tensor, (scale, left, top, w, h)


def nms(boxes, scores, threshold=.45):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if not len(rest):
            break
        lo = np.maximum(boxes[i,:2], boxes[rest,:2])
        hi = np.minimum(boxes[i,2:], boxes[rest,2:])
        wh = np.maximum(0., hi-lo)
        inter = wh[:,0]*wh[:,1]
        a = np.prod(np.maximum(0., boxes[i,2:]-boxes[i,:2]))
        b = np.prod(np.maximum(0., boxes[rest,2:]-boxes[rest,:2]), axis=1)
        overlap = inter/np.maximum(a+b-inter, 1e-12)
        order = rest[overlap <= threshold]
    return keep


def decode(raw, transform, confidence=.25, iou_threshold=.45):
    raw = np.asarray(raw)
    require(raw.ndim == 3 and raw.shape[0] == 1 and raw.shape[1] == 5, f"Expected YOLOv8 single-class [1,5,N], got {raw.shape}; cannot use legacy 5-class model")
    require(np.isfinite(raw).all(), "Non-finite detector output")
    pred = raw[0].T
    pred = pred[pred[:,4] >= confidence]
    boxes = np.concatenate((pred[:,:2]-pred[:,2:4]/2, pred[:,:2]+pred[:,2:4]/2), axis=1)
    selected = nms(boxes, pred[:,4], iou_threshold)[:100]
    scale, left, top, w, h = transform
    result = []
    for i in selected:
        b = boxes[i].copy()
        b[[0,2]] = np.clip((b[[0,2]]-left)/scale/w, 0., 1.)
        b[[1,3]] = np.clip((b[[1,3]]-top)/scale/h, 0., 1.)
        if b[2] > b[0] and b[3] > b[1]:
            result.append(dict(bbox=b.tolist(), confidence=float(pred[i,4]), class_id=0))
    return result


class Detector:
    def __init__(self, weights, imgsz=416, device="cpu", confidence=.25, iou_threshold=.45, allow_unvalidated=False):
        self.path = Path(weights)
        require(self.path.is_file(), f"Missing trained model: {weights}")
        self.size, self.confidence, self.iou_threshold = imgsz, confidence, iou_threshold
        self.device = f"cuda:{device}" if str(device).isdigit() else device
        if self.path.suffix == ".onnx":
            import onnxruntime as ort
            meta = load(self.path.with_name("model_meta.json"))
            require(meta["classes"] == ["baby_body"] and meta["imgsz"] == imgsz, "Model class/input size mismatch")
            require(meta["sha256"] == sha256(weights), "ONNX differs from metadata checksum")
            require(meta.get("parity_passed") or allow_unvalidated, "ONNX export parity was not verified")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.session = ort.InferenceSession(str(weights), sess_options=options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            require(self.session.get_inputs()[0].shape == [1,3,imgsz,imgsz], "Expected static batch-1 model")
            self.model = None
        else:
            import torch
            from ultralytics import YOLO
            yolo = YOLO(str(weights))
            require(list(yolo.names.values()) == ["baby_body"], f"Expected baby_body, found {yolo.names}")
            self.model = yolo.model.to(self.device).float().eval()
            self.session = None

    def raw(self, tensor):
        if self.session is not None:
            return self.session.run(None, {self.input_name:tensor})[0]
        import torch
        with torch.inference_mode():
            output = self.model(torch.from_numpy(tensor).to(self.device))
        return (output[0] if isinstance(output, (tuple,list)) else output).detach().cpu().numpy()

    def predict(self, frame):
        tensor, transform = letterbox(frame, self.size)
        return decode(self.raw(tensor), transform, self.confidence, self.iou_threshold)


def gpu_check(expected="L4"):
    import torch
    require(torch.cuda.is_available(), "CUDA GPU unavailable; training will not silently fall back to CPU")
    name = torch.cuda.get_device_name(0)
    require(expected.lower() in name.lower(), f"Expected {expected}, got {name}. Set --expected-gpu explicitly for a different experiment.")
    x = torch.randn(256,256,device="cuda:0")
    _ = x @ x
    torch.cuda.synchronize()
    return dict(gpu=name, vram_gib=torch.cuda.get_device_properties(0).total_memory/2**30, cuda=torch.version.cuda)


def audit_training_data(allow_reproduction=False):
    registry = load("data/registry/datasets.json")
    all_images = {r["image_id"]:r for r in rows("data/manifests/images.jsonl")}
    counts, split_hashes = {}, {}
    for s in ("train", "val", "test"):
        spec = load_split(s)
        require(spec["mode"] == "grouped" or allow_reproduction, "Original split needs --allow-reproduction-split; it is not a generalization benchmark")
        require(spec["image_ids"], f"No detection images in {s}")
        counts[s] = len(spec["image_ids"])
        split_hashes[s] = sha256(Path("data/splits") / f"{s}.json")
        expected_images = set()
        expected_labels = set()
        for image_id in spec["image_ids"]:
            r = all_images[image_id]
            approved(registry, r["dataset_id"])
            for kind, src, checksum in (("images",r["image_path"],r["sha256"]),("labels",r["label_path"],r["label_sha256"])):
                p = Path("data/detection") / s / kind / (image_id+Path(src).suffix)
                require(p.is_file() and sha256(p) == checksum, f"Materialized data missing/modified: {p}")
                (expected_images if kind == "images" else expected_labels).add(p.name)
        for kind, expected in (("images",expected_images),("labels",expected_labels)):
            actual = {p.name for p in (Path("data/detection")/s/kind).iterdir() if p.is_file()}
            require(actual == expected, f"Unexpected files in detection/{s}/{kind}")
    return dict(image_counts=counts, split_hashes=split_hashes, manifest_sha256=sha256("data/manifests/images.jsonl"), mode=load_split("train")["mode"])


def train(args):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    import torch
    config = load(args.config)
    require(config["model"] in ("yolov8n.pt", "yolov8s.pt"), "Use explicit YOLOv8n/s baseline")
    require(config["device"] == 0, "This baseline requires single GPU device 0")
    evidence = audit_training_data(args.allow_reproduction_split)
    env = dict(metadata(config["seed"]), **gpu_check(args.expected_gpu), **evidence)
    if args.smoke:
        config.update(imgsz=320, batch=4, epochs=2, patience=2, workers=2, name=config["name"]+"_smoke")
        smoke_root = Path("runs/smoke_inputs")
        smoke_root.mkdir(parents=True, exist_ok=True)
        ds = load(config["data"])
        for s, limit in (("train",16),("val",8)):
            files = sorted((Path(ds["path"])/s/"images").iterdir())[:limit]
            (smoke_root/f"{s}.txt").write_text("\n".join(str(p.resolve()) for p in files)+"\n", encoding="utf-8")
            ds[s] = str((smoke_root/f"{s}.txt").resolve())
        dump(smoke_root/"data.yaml",ds)
        config["data"] = str(smoke_root/"data.yaml")
    get_cfg(overrides=config)  # Reject unsupported flags before training.
    run = Path(config["project"])/config["name"]
    require(not run.exists(), f"Run already exists: {run}. Choose a new name in config.")
    model = YOLO(config["model"])
    epoch_times = []
    started = [None]
    def on_start(trainer):
        started[0] = time.perf_counter()
    def on_end(trainer):
        epoch_times.append(time.perf_counter()-started[0])
    model.add_callback("on_train_epoch_start", on_start)
    model.add_callback("on_train_epoch_end", on_end)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    completed = False
    try:
        model.train(**config)
        completed = True
    finally:
        actual = Path(model.trainer.save_dir) if model.trainer else run
        env.update(elapsed_s=time.perf_counter()-start, gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30, epoch_train_seconds=epoch_times, completed=completed, requested_config=config)
        if model.trainer:
            env["resolved_config"] = vars(model.trainer.args)
            env["best_epoch"] = getattr(model.trainer,"best_epoch",None)
            n = len(model.trainer.train_loader.dataset)
            env["actual_train_images"] = n
            env["training_images_per_s"] = n*len(epoch_times)/sum(epoch_times) if epoch_times else None
        dump(actual/"run_metadata.json",env)
        lock_environment(actual)


def evaluate(args):
    from ultralytics import YOLO
    import yaml
    audit = audit_training_data(args.allow_reproduction_split)
    require(Path(args.weights).is_file(), "Missing model")
    model = YOLO(args.weights)
    require(list(model.names.values()) == ["baby_body"], "Not a single-class baby_body detector")
    spec = load_split(args.split)
    index = {r["image_id"]:r for r in rows("data/manifests/images.jsonl")}
    selected = [index[i] for i in spec["image_ids"]]
    out = Path(args.out)
    require(not out.exists(), "Evaluation output already exists; choose --out")
    out.mkdir(parents=True)
    groups = {"all":selected}
    for field in ("subject_kind", "posture", "visibility", "modality"):
        for value in sorted({r.get(field,"unknown") for r in selected}):
            groups[f"{field}_{value}"] = [r for r in selected if r.get(field,"unknown") == value]
    results = {}
    for i,(label,group) in enumerate(groups.items()):
        filelist = out/f"subset_{i}.txt"
        filelist.write_text("\n".join(str((Path("data/detection")/args.split/"images"/(r["image_id"]+Path(r["image_path"]).suffix)).resolve()) for r in group)+"\n",encoding="utf-8")
        ds = load("configs/datasets.yaml")
        ds[args.split] = str(filelist.resolve())
        config = out/f"subset_{i}.yaml"
        config.write_text(yaml.safe_dump(ds),encoding="utf-8")
        m = model.val(data=str(config),split=args.split,imgsz=args.imgsz,device=args.device,project=str(out),name=f"subset_{i}",plots=False)
        results[label] = dict(image_count=len(group),metrics={k:float(v) for k,v in m.results_dict.items()})
    dump(out/"detector_metrics.json",dict(weights_sha256=sha256(args.weights),split=args.split,audit=audit,groups=results))


def export(args):
    import cv2
    import onnx
    from ultralytics import YOLO
    require(Path(args.weights).is_file(), "Missing trained model")
    output = Path(args.out)
    require(not output.exists(), "Export destination exists; choose a new --out")
    selected = load_split("val")["image_ids"]
    images = {x["image_id"]:x for x in rows("data/manifests/images.jsonl")}
    require(selected, "Validation images required for export parity")
    model = YOLO(args.weights)
    require(list(model.names.values()) == ["baby_body"], "Only baby_body supported")
    path = Path(model.export(format="onnx",imgsz=args.imgsz,batch=1,dynamic=False,half=False,opset=17,nms=False,simplify=False,device="cpu"))
    onnx.checker.check_model(str(path))
    output.mkdir(parents=True)
    dest = output/"baby_body.onnx"
    shutil.copy2(path,dest)
    meta = dict(schema_version=1,architecture="YOLOv8 detect",classes=["baby_body"],imgsz=args.imgsz,input_layout="NCHW RGB float32 /255",output_layout="[1,5,N]: cx cy w h confidence",letterbox=dict(pad=114,scaleup=True,auto=False),nms=dict(confidence=.25,iou=.45),sha256=sha256(dest),source_weights_sha256=sha256(args.weights),parity_passed=False,environment=metadata())
    dump(output/"model_meta.json",meta)
    pt = Detector(args.weights,args.imgsz)
    ot = Detector(dest,args.imgsz,allow_unvalidated=True)
    checks = []
    for image_id in selected[:args.samples]:
        frame = cv2.imread(images[image_id]["image_path"])
        require(frame is not None, "Unreadable validation image")
        tensor, transform = letterbox(frame,args.imgsz)
        a,b = pt.raw(tensor),ot.raw(tensor)
        require(a.shape == b.shape, "ONNX raw output shape mismatch")
        raw_ok = bool(np.allclose(a,b,rtol=1e-3,atol=2e-3))
        da,db = decode(a,transform),decode(b,transform)
        final_ok = len(da) == len(db) and all(np.allclose(x["bbox"],y["bbox"],atol=1e-3) and abs(x["confidence"]-y["confidence"]) < 1e-3 for x,y in zip(da,db))
        checks.append(dict(image_id=image_id,max_abs_diff=float(np.max(abs(a-b))),raw_passed=raw_ok,boxes_passed=final_ok))
    dump(output/"parity_report.json",dict(images=checks,video_event_parity="not_run; compare replay events on your validation video before deployment"))
    require(all(x["raw_passed"] and x["boxes_passed"] for x in checks), "ONNX parity failed; see parity_report.json")
    meta["parity_passed"] = True
    dump(output/"model_meta.json",meta)
    from .risk import FEATURE_ORDER
    dump(output/"feature_order.json",dict(schema_version=1,features=FEATURE_ORDER))
    dump(output/"risk_policy.json",load(args.policy))
    lock_environment(output)
    print(f"Exported and checked {len(checks)} validation images: {dest}")
