"""학습한 모델 평가.

1) 박스 지표 (Ultralytics val): 클래스별 정밀도·재현율·mAP50
2) 사진 단위 판정: 라즈베리파이에서 쓰는 판정 규칙(sleepsafe.decision)으로 사진마다
   자세(바로/옆/엎드림), 코·입 상태, 위험 여부(엎드림 또는 얼굴 가림)를 맞혔는지

python scripts/evaluate.py --weights runs/train/A_original_416/weights/best.pt \
    --data datasets/sleepsafe/data_original.yaml --imgsz 416
python scripts/evaluate.py --weights models/A_original_416.onnx --backend onnx \
    --data datasets/sleepsafe/data_original.yaml --imgsz 416     # 파이와 같은 추론 코드로 평가
python scripts/evaluate.py --weights runs/train/A_original_320_aug/weights/best.pt \
    --data datasets/<데이터셋>/data_original.yaml --imgsz 320 --split all --name <결과이름>
                                                               # 다른 데이터셋 전체로 평가 (박스 지표 생략)
결과: results/eval_<이름>_<백엔드>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.classes import AIRWAY_CLASSES, BODY_CLASSES  # noqa: E402
from sleepsafe.config import ROOT, load_yaml, repo_path  # noqa: E402
from sleepsafe.decision import Detection, decide  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
POSTURE_LABELS = list(BODY_CLASSES) + ["none"]
AIRWAY_LABELS = list(AIRWAY_CLASSES) + ["none"]
RESULTS = ROOT / "results"


def data_names(data_yaml: Path) -> List[str]:
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def split_images(data_yaml: Path, split: str) -> List[Path]:
    """분할 하나의 사진 목록. split='all'이면 train·val·test를 모두 합쳐요 (외부 데이터셋 평가용)."""
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if split == "all":
        return [p for s in ("train", "val", "test") if data.get(s) for p in split_images(data_yaml, s)]
    entry = data.get(split)
    if not entry:
        raise ValueError(f"{data_yaml}에 '{split}' 분할이 없어요.")
    p = Path(entry)
    if not p.is_absolute():
        p = Path(data.get("path") or data_yaml.parent) / p
    if p.suffix == ".txt":
        return [Path(s.strip()) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
    return sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


def ground_truth(img: Path, names: List[str]) -> List[Detection]:
    """정답 라벨 → Detection. 신뢰도 자리에 박스 넓이를 넣어 큰 박스가 대표가 되게 해요."""
    dets = []
    lp = label_path(img)
    if lp.exists():
        for line in lp.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, w, h = map(float, p[1:5])
            dets.append(Detection(names[int(p[0])], w * h, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return dets


def predict_ultralytics(weights: Path, images: List[Path], imgsz: int, conf: float) -> Iterator[List[Detection]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    for i in range(0, len(images), 32):
        chunk = [str(p) for p in images[i:i + 32]]
        for r in model.predict(chunk, imgsz=imgsz, conf=conf, verbose=False, stream=True):
            b = r.boxes
            yield [Detection(r.names[int(c)], float(s), tuple(float(v) for v in xy))
                   for xy, s, c in zip(b.xyxy.tolist(), b.conf.tolist(), b.cls.tolist())]


def predict_onnx(model_path: Path, images: List[Path], imgsz: int, conf: float) -> Iterator[List[Detection]]:
    from sleepsafe.detector_onnx import OnnxDetector
    from sleepsafe.imaging import imread

    detector = OnnxDetector(model_path, imgsz=imgsz, conf=conf)
    for p in images:
        yield detector(imread(p))


def _ratio(a: int, b: int) -> Optional[float]:
    return a / b if b else None


def collect(weights, data, split: str = "test", imgsz: int = 640, conf: float = 0.05,
            backend: str = "ultralytics") -> Dict:
    """테스트 사진마다 정답 박스와 모델 박스를 모아요. 판정 기준을 여러 번 바꿔 볼 때 한 번만 실행하면 돼요.

    conf보다 높은 기준으로 나중에 거르면, 그 기준으로 바로 추론한 것과 결과가 같아요.
    """
    weights, data = repo_path(weights), repo_path(data)
    names = data_names(data)
    images = split_images(data, split)
    predict = predict_onnx if backend == "onnx" else predict_ultralytics
    records_file = Path(yaml.safe_load(data.read_text(encoding="utf-8")).get("path", data.parent)) / "records.json"
    sources = ({r["image"]: r["source"] for r in json.loads(records_file.read_text(encoding="utf-8"))}
               if records_file.exists() else {})
    return {"weights": str(weights), "data": str(data), "split": split, "imgsz": imgsz, "backend": backend,
            "images": [p.name for p in images],
            "sources": [sources.get(p.name, "unknown") for p in images],
            "gts": [ground_truth(p, names) for p in images],
            "preds": list(predict(weights, images, imgsz, conf))}


def score(col: Dict, conf_body: float = 0.5, conf_airway: float = 0.4,
          covered_without_body: bool = False) -> Dict:
    """모아 둔 박스로 사진마다 정답 판정과 모델 판정을 비교해요.

    정답 판정은 몸 박스가 없어도 코·입 가림 박스가 있으면 얼굴 가림으로 봐요 (사진에 실제로 있는 위험).
    """
    posture_cm = {g: Counter() for g in POSTURE_LABELS}
    airway_cm = {g: Counter() for g in AIRWAY_LABELS}
    risk = Counter()
    by_type = {"prone": Counter(), "face_covered": Counter()}
    by_source = defaultdict(Counter)
    for source, gt_dets, dets in zip(col["sources"], col["gts"], col["preds"]):
        gt = decide(gt_dets, conf_body=0.0, conf_airway=0.0, covered_without_body=True)
        pr = decide(dets, conf_body, conf_airway, covered_without_body)
        posture_cm[gt.posture or "none"][pr.posture or "none"] += 1
        airway_cm[gt.airway or "none"][pr.airway or "none"] += 1
        g_risk, p_risk = gt.risk_type is not None, pr.risk_type is not None
        risk[("tp" if p_risk else "fn") if g_risk else ("fp" if p_risk else "tn")] += 1
        if gt.risk_type:
            t = by_type[gt.risk_type]
            t["total"] += 1
            t["exact"] += int(pr.risk_type == gt.risk_type)
            t["any"] += int(p_risk)
        s = by_source[source]
        s["images"] += 1
        s["risk_correct"] += int(g_risk == p_risk)

    tp, fp, fn, tn = (risk[k] for k in ("tp", "fp", "fn", "tn"))
    n_body = sum(sum(posture_cm[g].values()) for g in BODY_CLASSES)
    return {
        "weights": col["weights"], "data": col["data"], "split": col["split"], "imgsz": col["imgsz"],
        "backend": col["backend"], "conf_body": conf_body, "conf_airway": conf_airway,
        "covered_without_body": covered_without_body, "images": tp + fp + fn + tn,
        "posture_accuracy": _ratio(sum(posture_cm[g][g] for g in BODY_CLASSES), n_body),
        "body_missed_rate": _ratio(sum(posture_cm[g]["none"] for g in BODY_CLASSES), n_body),
        "risk": {"accuracy": _ratio(tp + tn, tp + fp + fn + tn), "recall": _ratio(tp, tp + fn),
                 "precision": _ratio(tp, tp + fp), "specificity": _ratio(tn, tn + fp),
                 "tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "risk_by_type": {k: {"total": v["total"], "recall_exact": _ratio(v["exact"], v["total"]),
                             "recall_any_risk": _ratio(v["any"], v["total"])} for k, v in by_type.items()},
        "posture_confusion": {g: {p: posture_cm[g][p] for p in POSTURE_LABELS} for g in POSTURE_LABELS},
        "airway_confusion": {g: {p: airway_cm[g][p] for p in AIRWAY_LABELS} for g in AIRWAY_LABELS},
        "by_source": {k: {"images": v["images"], "risk_accuracy": _ratio(v["risk_correct"], v["images"])}
                      for k, v in sorted(by_source.items())},
    }


def image_level(weights, data, split: str = "test", imgsz: int = 640, conf_body: float = 0.5,
                conf_airway: float = 0.4, backend: str = "ultralytics",
                covered_without_body: bool = False) -> Dict:
    """사진마다 정답 판정과 모델 판정을 비교해요 (collect + score)."""
    col = collect(weights, data, split, imgsz, min(conf_body, conf_airway), backend)
    return score(col, conf_body, conf_airway, covered_without_body)


def box_metrics(weights, data, imgsz: int, split: str = "test", name: str = "eval") -> Dict:
    """Ultralytics val로 클래스별 정밀도·재현율·mAP를 구해요."""
    from ultralytics import YOLO

    r = YOLO(str(repo_path(weights))).val(data=str(repo_path(data)), split=split, imgsz=imgsz, plots=True,
                                          project=str(ROOT / "runs" / "val"), name=name, exist_ok=True)
    per_class = {}
    for i, c in enumerate(r.ap_class_index):
        p, rec, ap50, ap = r.box.class_result(i)
        per_class[r.names[int(c)]] = {"precision": float(p), "recall": float(rec),
                                      "map50": float(ap50), "map50_95": float(ap)}
    return {"map50": float(r.box.map50), "map50_95": float(r.box.map), "precision": float(r.box.mp),
            "recall": float(r.box.mr), "per_class": per_class,
            "speed_ms": {k: float(v) for k, v in r.speed.items()}, "save_dir": str(r.save_dir)}


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def print_report(res: Dict) -> None:
    print(f"\n[사진 단위 판정] {res['images']}장, 백엔드={res['backend']}, imgsz={res['imgsz']}")
    print(f"  자세 정확도: {_pct(res['posture_accuracy'])}   아기를 못 찾은 비율: {_pct(res['body_missed_rate'])}")
    r = res["risk"]
    print(f"  위험 판정: 정확도 {_pct(r['accuracy'])}, 재현율 {_pct(r['recall'])}, "
          f"정밀도 {_pct(r['precision'])}, 특이도 {_pct(r['specificity'])} "
          f"(TP {r['tp']} / FP {r['fp']} / FN {r['fn']} / TN {r['tn']})")
    for k, v in res["risk_by_type"].items():
        print(f"  {k}: 정답 {v['total']}장 중 같은 유형으로 맞힘 {_pct(v['recall_exact'])}, "
              f"위험으로는 잡음 {_pct(v['recall_any_risk'])}")
    for title, key, labels in (("자세", "posture_confusion", POSTURE_LABELS),
                               ("코·입", "airway_confusion", AIRWAY_LABELS)):
        print(f"\n  {title} 혼동행렬 (행=정답, 열=예측)")
        print(" " * 18 + "".join(f"{p:>16}" for p in labels))
        for g in labels:
            print(f"  {g:16}" + "".join(f"{res[key][g][p]:>16}" for p in labels))
    if res["by_source"]:
        print("\n  출처별 위험 판정 정확도")
        for k, v in res["by_source"].items():
            print(f"  {k:16} {v['images']:>6}장  {_pct(v['risk_accuracy'])}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="best.pt 또는 .onnx")
    ap.add_argument("--data", required=True, help="data_*.yaml")
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--split", default="test", help="train / val / test / all(전부 합침, 외부 데이터셋용)")
    ap.add_argument("--backend", choices=["ultralytics", "onnx"], default="ultralytics",
                    help="onnx: 파이에서 쓰는 sleepsafe.detector_onnx로 추론")
    ap.add_argument("--no-box", action="store_true", help="박스 지표(Ultralytics val)는 건너뜀")
    ap.add_argument("--name", help="결과 파일 이름 (기본: 가중치 폴더 이름)")
    args = ap.parse_args(argv)

    edge = load_yaml("configs/edge.yaml")
    weights = repo_path(args.weights)
    name = args.name or (weights.parent.parent.name if weights.parent.name == "weights" else weights.stem)
    result = {"name": name}
    if args.split == "all" and not args.no_box:
        print("split=all은 사진 단위 판정만 평가해요 (박스 지표 생략).")
    elif not args.no_box:
        result["box"] = box_metrics(weights, args.data, args.imgsz, args.split, name)
    result["image_level"] = image_level(weights, args.data, args.split, args.imgsz,
                                        edge["conf_body"], edge["conf_airway"], args.backend,
                                        edge.get("covered_without_body", False))
    print_report(result["image_level"])
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"eval_{name}_{args.backend}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()
