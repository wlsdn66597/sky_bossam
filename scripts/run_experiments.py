"""configs/experiments.yaml의 실험을 차례로 실행해요.
학습 → 박스 지표 → 사진 단위 판정 → ONNX 내보내기 → results/summary.csv

python scripts/run_experiments.py                         # 전부
python scripts/run_experiments.py --only A_original_416 B_doll_blanket_416
python scripts/run_experiments.py --skip-train            # 이미 학습된 실험은 평가만 다시
python scripts/run_experiments.py --only A_original_416 --set epochs=3   # 빠른 동작 확인
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.config import ROOT, load_yaml, parse_overrides, repo_path  # noqa: E402
from evaluate import box_metrics, image_level  # noqa: E402
from export_model import export  # noqa: E402
from train import PROJECT, train  # noqa: E402

RESULTS = ROOT / "results"
COLUMNS = ["name", "data", "imgsz", "mAP50", "prone_P", "prone_R", "covered_P", "covered_R",
           "posture_acc", "risk_acc", "risk_recall", "risk_precision", "infer_ms", "onnx", "error"]
RATIO_COLUMNS = {"mAP50", "prone_P", "prone_R", "covered_P", "covered_R",
                 "posture_acc", "risk_acc", "risk_recall", "risk_precision"}


def summary_row(r: dict) -> dict:
    box = r.get("box", {})
    pc = box.get("per_class", {})
    img = r.get("image_level", {})
    risk = img.get("risk", {})
    return {
        "name": r["name"], "data": r["data"], "imgsz": r["imgsz"], "mAP50": box.get("map50"),
        "prone_P": pc.get("prone", {}).get("precision"), "prone_R": pc.get("prone", {}).get("recall"),
        "covered_P": pc.get("airway_covered", {}).get("precision"),
        "covered_R": pc.get("airway_covered", {}).get("recall"),
        "posture_acc": img.get("posture_accuracy"), "risk_acc": risk.get("accuracy"),
        "risk_recall": risk.get("recall"), "risk_precision": risk.get("precision"),
        "infer_ms": box.get("speed_ms", {}).get("inference"), "onnx": r.get("onnx", ""),
        "error": r.get("error", ""),
    }


def write_summary() -> list:
    rows = [summary_row(json.loads(p.read_text(encoding="utf-8"))) for p in sorted(RESULTS.glob("exp_*.json"))]
    with open(RESULTS / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글 깨짐 방지
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return rows


def _fmt(column: str, v) -> str:
    if v in (None, ""):
        return "-"
    if column in RATIO_COLUMNS:
        return f"{float(v) * 100:.1f}"
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--only", nargs="*", help="이 이름의 실험만 실행")
    ap.add_argument("--skip-train", action="store_true", help="best.pt가 있으면 학습 생략")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="모든 실험의 학습 설정 덮어쓰기")
    args = ap.parse_args(argv)

    experiments = load_yaml(args.config)["experiments"]
    out_dir = repo_path(load_yaml("configs/dataset.yaml")["out_dir"])
    edge = load_yaml("configs/edge.yaml")
    cli = parse_overrides(args.set)
    RESULTS.mkdir(exist_ok=True)

    for exp in experiments:
        name = exp["name"]
        if args.only and name not in args.only:
            continue
        print(f"\n===== {name} =====")
        result = {"name": name, "data": exp["data"], "imgsz": exp["imgsz"],
                  "overrides": {**(exp.get("overrides") or {}), **cli}}
        try:
            data = out_dir / f"data_{exp['data']}.yaml"
            if not data.exists():
                raise FileNotFoundError(f"{data} 가 없어요. 먼저 python scripts/prepare_dataset.py")
            best = PROJECT / name / "weights" / "best.pt"
            if not (args.skip_train and best.exists()):
                best = train(data, name, exp["imgsz"], result["overrides"])
            result["weights"] = str(best)
            result["box"] = box_metrics(best, data, exp["imgsz"], name=name)
            result["image_level"] = image_level(best, data, "test", exp["imgsz"],
                                                edge["conf_body"], edge["conf_airway"],
                                                covered_without_body=edge.get("covered_without_body", False))
            if not args.no_export:
                result["onnx"] = str(export(best, exp["imgsz"], name=name)[0])
        except Exception as e:  # 한 실험이 실패해도 나머지는 계속
            traceback.print_exc()
            result["error"] = f"{type(e).__name__}: {e}"
        (RESULTS / f"exp_{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")

    rows = write_summary()
    print("\n===== 요약 (비율은 %, infer_ms는 학습 PC 기준) → results/summary.csv =====")
    shown = [c for c in COLUMNS if c != "onnx"]
    print(" | ".join(shown))
    for row in rows:
        print(" | ".join(_fmt(c, row[c]) for c in shown))


if __name__ == "__main__":
    main()
