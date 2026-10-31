"""실시간 실행: 카메라 → 탐지 → 위험 판정(5초) → 알림.

노트북 웹캠:   python edge/run_edge.py --source 0 --show
영상 파일:     python edge/run_edge.py --source test.mp4 --show
라즈베리파이 4: python edge/run_edge.py --source picamera2 --server http://<서버IP>:8000
설정 기본값은 configs/edge.yaml (명령줄 옵션이 우선)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleepsafe.classes import KOREAN  # noqa: E402
from sleepsafe.config import load_yaml, repo_path  # noqa: E402
from sleepsafe.decision import FrameDecision, decide  # noqa: E402
from sleepsafe.risk import RiskConfig, RiskMonitor  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/edge.yaml")
    ap.add_argument("--source", default="0", help="picamera2 / 웹캠 번호 / 영상 파일")
    ap.add_argument("--model", help="ONNX 모델 경로")
    ap.add_argument("--imgsz", type=int)
    ap.add_argument("--infer-fps", type=float)
    ap.add_argument("--server", help="웹 알림 서버 주소 (예: http://192.168.0.10:8000)")
    ap.add_argument("--device-id")
    ap.add_argument("--show", action="store_true", help="화면에 결과 표시 (q로 종료)")
    ap.add_argument("--max-seconds", type=float, help="이 시간(입력 기준)만큼만 실행")
    ap.add_argument("--stats-every", type=float, default=10.0,
                    help="이 간격(초)마다 평균 추론 시간·처리 속도 출력 (0이면 끔)")
    return ap.parse_args(argv)


def event_payload(ev, to_wall, decision: FrameDecision, model_name: str, settings_version: int) -> dict:
    p = {"event_id": ev.event_id, "kind": ev.kind, "risk_type": ev.risk_type,
         "model": model_name, "settings_version": settings_version}
    for key in ("started_at", "confirmed_at", "ended_at"):
        value = getattr(ev, key)
        if value is not None:
            p[key] = round(to_wall(value), 3)
    if ev.kind == "risk_started":
        p.update(posture=decision.posture, airway=decision.airway,
                 confidence=round(decision.body.conf, 3) if decision.body else None)
    return p


def main(argv=None) -> None:
    import cv2

    from sleepsafe.detector_onnx import OnnxDetector
    from sleepsafe.imaging import brightness, draw, encode_jpeg_b64
    from sleepsafe.notify import ConsoleNotifier, HttpNotifier
    from sleepsafe.sources import open_source

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_yaml(args.config)
    model_path = repo_path(args.model or cfg["model"])
    infer_fps = args.infer_fps or cfg.get("infer_fps", 3)
    server = args.server or cfg["server"].get("url")
    device_id = args.device_id or cfg["server"]["device_id"]
    snap = cfg.get("snapshot", {})

    detector = OnnxDetector(model_path, imgsz=args.imgsz or cfg.get("imgsz"),
                            conf=min(cfg["conf_body"], cfg["conf_airway"]), threads=cfg.get("threads"))
    monitor = RiskMonitor(RiskConfig(**cfg["risk"]))
    notifier = (HttpNotifier(server, device_id, heartbeat_s=cfg["server"].get("heartbeat_s", 2.0))
                if server else ConsoleNotifier())
    source = open_source(str(args.source), tuple(cfg.get("camera_size", (640, 480))))
    print(f"모델 {model_path.name} (입력 {detector.imgsz}), 입력 {args.source}, "
          f"알림 {'서버 ' + server if server else '콘솔'}")

    settings_version = 0
    origin = None          # 입력 시각 → 실제 시각 변환용
    first_ts = None
    last_infer = None
    infer_ms = 0.0
    stats_start, stats_ms = None, []  # 속도 출력용
    try:
        while True:
            item = source.read()
            if item is None:
                break
            frame, ts = item
            if origin is None:
                origin, first_ts = time.time() - ts, ts
            if args.max_seconds and ts - first_ts > args.max_seconds:
                break
            if last_infer is not None and ts - last_infer < 1.0 / infer_fps:
                continue
            fps = 1.0 / (ts - last_infer) if last_infer is not None and ts > last_infer else 0.0
            last_infer = ts

            t0 = time.perf_counter()
            if brightness(frame) < cfg.get("min_brightness", 0):
                dets, decision = [], FrameDecision(valid=False, reason="dark")
            else:
                dets = detector(frame)
                decision = decide(dets, cfg["conf_body"], cfg["conf_airway"],
                                  cfg.get("covered_without_body", False))
            infer_ms = (time.perf_counter() - t0) * 1000
            if args.stats_every > 0:
                stats_start = ts if stats_start is None else stats_start
                stats_ms.append(infer_ms)
                if ts - stats_start >= args.stats_every:
                    print(f"[{ts - first_ts:7.1f}s] 속도: 추론 평균 {sum(stats_ms) / len(stats_ms):.0f}ms "
                          f"(최대 {max(stats_ms):.0f}ms), 처리 {len(stats_ms) / (ts - stats_start):.1f}fps, "
                          f"상태 {KOREAN[monitor.status]}")
                    stats_start, stats_ms = ts, []

            for ev in monitor.update(ts, decision):
                if ev.kind == "status_changed":
                    why = KOREAN.get(ev.reason, ev.reason)
                    print(f"[{ts - first_ts:7.1f}s] 상태: {KOREAN[ev.status]}" + (f" ({why})" if why else ""))
                    continue
                payload = event_payload(ev, lambda t: origin + t, decision, model_path.name, settings_version)
                if ev.kind == "risk_started" and snap.get("enabled", True):
                    payload["snapshot_jpeg_b64"] = encode_jpeg_b64(frame, snap.get("jpeg_quality", 80),
                                                                   snap.get("max_width", 640))
                label = "위험 시작" if ev.kind == "risk_started" else "위험 종료"
                print(f"[{ts - first_ts:7.1f}s] {label}: {KOREAN.get(ev.risk_type, ev.risk_type)} ({ev.event_id[:8]})")
                notifier.send_event(payload)

            notifier.update_status({
                "ts": round(origin + ts, 3), "status": monitor.status, "reason": monitor.reason,
                "posture": decision.posture, "airway": decision.airway,
                "elapsed": {k: round(v, 1) for k, v in monitor.elapsed(ts).items()},
                "active_event_id": monitor.active.event_id if monitor.active else None,
                "fps": round(fps, 2), "infer_ms": round(infer_ms, 1),
                "model": model_path.name, "settings_version": settings_version,
            })
            s = notifier.settings  # 서버에서 기준 시간을 바꾸면 여기로 들어와요
            if s and s.get("version", 0) != settings_version:
                monitor.config.risk_threshold_s = float(s.get("risk_threshold_s", monitor.config.risk_threshold_s))
                settings_version = s.get("version", 0)
                print(f"설정 적용 v{settings_version}: 기준 시간 {monitor.config.risk_threshold_s}초")

            if args.show:
                cv2.imshow("sky_bossam", draw(frame, dets, decision, monitor.status,
                                           f"{fps:.1f}fps {infer_ms:.0f}ms"))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()
        notifier.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
