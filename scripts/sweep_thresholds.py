"""판정 기준을 바꿔 가며 사진 단위 위험 재현율·정밀도를 비교해요 (재학습 없이).

예측은 낮은 기준으로 한 번만 돌리고, 아래 값들을 조합해 다시 판정해요.
- conf_body: 몸(자세) 박스 최소 신뢰도
- conf_airway: 코·입 박스 최소 신뢰도
- covered_without_body: 몸 박스 없이 코·입 가림 박스만 잡혀도 얼굴 가림으로 볼지

python scripts/sweep_thresholds.py --exp A_original_416
python scripts/sweep_thresholds.py --exp A_original_416 A_original_320 --target-recall 0.9
python scripts/sweep_thresholds.py --exp A_original_416 --backend onnx   # 파이와 같은 추론 코드로
python scripts/sweep_thresholds.py --weights runs/train/A_original_320_aug/weights/best.pt \
    --data datasets/<데이터셋>/data_original.yaml --imgsz 320 --split all   # 다른 데이터셋 전체로
결과: results/sweep_<실험>_<백엔드>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import ROOT, load_yaml  # noqa: E402
from evaluate import collect, score  # noqa: E402

BODY_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
AIRWAY_GRID = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
RESULTS = ROOT / "results"
COLUMNS = ["covered_without_body", "conf_body", "conf_airway", "risk_recall", "risk_precision", "risk_acc",
           "prone_recall", "face_recall", "posture_acc", "fp", "fn"]


def flatten(m: Dict) -> Dict:
    r, t = m["risk"], m["risk_by_type"]
    return {"covered_without_body": m["covered_without_body"], "conf_body": m["conf_body"],
            "conf_airway": m["conf_airway"], "risk_recall": r["recall"], "risk_precision": r["precision"],
            "risk_acc": r["accuracy"], "prone_recall": t["prone"]["recall_any_risk"],
            "face_recall": t["face_covered"]["recall_any_risk"], "posture_acc": m["posture_accuracy"],
            "fp": r["fp"], "fn": r["fn"]}


def pick(rows: List[Dict], target: float) -> Dict:
    """목표 재현율을 넘는 것 중 정밀도가 가장 높은 것. 없으면 재현율이 가장 높은 것."""
    ok = [r for r in rows if (r["risk_recall"] or 0) >= target]
    if ok:
        return max(ok, key=lambda r: (r["risk_precision"] or 0, r["risk_recall"] or 0))
    return max(rows, key=lambda r: (r["risk_recall"] or 0, r["risk_precision"] or 0))


HEADER = ["몸없이가림", "몸기준", "얼굴기준", "위험재현율", "위험정밀도", "위험정확도",
          "엎드림재현율", "얼굴가림재현율", "자세정확도", "FP", "FN"]


def _p(v) -> str:
    return "-" if v is None else f"{v * 100:.1f}"


def print_rows(title: str, rows: List[Dict]) -> None:
    print(f"\n{title}")
    print("  " + " | ".join(HEADER))
    for r in rows:
        cells = ["예" if r["covered_without_body"] else "아니오", f"{r['conf_body']:.2f}", f"{r['conf_airway']:.2f}",
                 _p(r["risk_recall"]), _p(r["risk_precision"]), _p(r["risk_acc"]), _p(r["prone_recall"]),
                 _p(r["face_recall"]), _p(r["posture_acc"]), str(r["fp"]), str(r["fn"])]
        print("  " + " | ".join(cells))


def sweep(name: str, weights: str, data: str, imgsz: int, split: str, backend: str,
          target: float, edge: Dict) -> Dict:
    print(f"\n===== {name} ({backend}, imgsz {imgsz}) =====")
    col = collect(weights, data, split, imgsz, conf=0.05, backend=backend)
    rows = [flatten(score(col, b, a, cw)) for cw in (False, True) for b in BODY_GRID for a in AIRWAY_GRID]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"sweep_{name}_{backend}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    raw = score(col, edge["conf_body"], edge["conf_airway"], edge.get("covered_without_body", False))
    current = flatten(raw)
    best = pick(rows, target)
    n_risk = raw["risk"]["tp"] + raw["risk"]["fn"]
    print(f"테스트 사진 {len(col['images'])}장 (위험 사진 {n_risk}장)")
    print_rows("[지금 configs/edge.yaml 설정]", [current])
    print_rows(f"[추천: 위험 재현율 {target * 100:.0f}% 이상 중 정밀도 최고]" if (best["risk_recall"] or 0) >= target
               else f"[재현율 {target * 100:.0f}%를 넘는 조합이 없어요 → 재현율 최고]", [best])
    top = sorted(rows, key=lambda r: (-(r["risk_recall"] or 0), -(r["risk_precision"] or 0)))[:10]
    print_rows("[재현율 상위 10개]", top)
    print("\nconfigs/edge.yaml에 넣을 값 (추천):")
    print(f"  conf_body: {best['conf_body']}\n  conf_airway: {best['conf_airway']}\n"
          f"  covered_without_body: {'true' if best['covered_without_body'] else 'false'}")
    print(f"전체 조합: {out}")
    return best


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="*", default=[], help="run_experiments.py로 돌린 실험 이름")
    ap.add_argument("--weights", help="실험 이름 대신 모델을 직접 지정")
    ap.add_argument("--data", help="--weights와 함께: data_*.yaml")
    ap.add_argument("--imgsz", type=int, help="--weights와 함께: 입력 크기")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", choices=["ultralytics", "onnx"], default="ultralytics")
    ap.add_argument("--target-recall", type=float, default=0.9)
    args = ap.parse_args(argv)

    edge = load_yaml("configs/edge.yaml")
    jobs = []
    for name in args.exp:
        info_file = RESULTS / f"exp_{name}.json"
        if not info_file.exists():
            sys.exit(f"{info_file} 가 없어요. 먼저 run_experiments.py로 '{name}'을 실행하세요.")
        info = json.loads(info_file.read_text(encoding="utf-8"))
        if "image_level" not in info:
            print(f"[건너뜀] {name}: 평가 결과가 없어요 ({info.get('error', '')})")
            continue
        weights = info.get("onnx") if args.backend == "onnx" else info["weights"]
        if not weights:
            sys.exit(f"{name}: ONNX 모델이 없어요. --backend ultralytics로 실행하거나 먼저 내보내세요.")
        jobs.append((name, weights, info["image_level"]["data"], info["imgsz"]))
    if args.weights:
        if not (args.data and args.imgsz):
            sys.exit("--weights를 쓰면 --data와 --imgsz도 필요해요.")
        w = Path(args.weights)
        run = w.parent.parent.name if w.parent.name == "weights" else w.stem
        jobs.append((f"{run}_{Path(args.data).parent.name}", args.weights, args.data, args.imgsz))
    if not jobs:
        sys.exit("--exp 또는 --weights를 지정하세요.")

    for name, weights, data, imgsz in jobs:
        sweep(name, weights, data, imgsz, args.split, args.backend, args.target_recall, edge)


if __name__ == "__main__":
    main()
