from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

from .common import approved, dump, load, load_split, metadata, object_sha, require, rows, selected_videos, sha256, write_rows
from .detector import Detector
from .risk import FeatureExtractor, R0, FEATURE_ORDER


def frames(path, sample_hz=None, timestamps_file=None):
    """File time from decoder timestamps; never use wall inference time or guess CFR."""
    import cv2
    require(sample_hz is None or sample_hz > 0, "sample_hz must be positive")
    times = rows(timestamps_file) if timestamps_file else None
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), f"Cannot open video: {path}")
    previous, emitted = None, None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if times is not None:
                require(index < len(times) and times[index]["frame_index"] == index, "Timestamp sidecar does not cover decoded frames in order")
                ts = times[index]["timestamp_s"]
            else:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            require(ts >= 0 and (previous is None or ts > previous), "Decoder timestamps invalid. Supply a per-frame --timestamps-file; do not silently replace VFR timestamps with index/fps.")
            previous = ts
            if emitted is None or sample_hz is None or ts-emitted >= 1/sample_hz-1e-7:
                yield index,ts,frame
                emitted = ts
            index += 1
        require(index > 0, "No decoded frames")
        if times is not None:
            require(index == len(times), "Timestamp sidecar length mismatch")
    finally:
        cap.release()


def frame_features(extractor, detector, frame, ts):
    import cv2
    w,h = extractor.g["reference_width"], extractor.g["reference_height"]
    calibrated = frame.shape[1] == w and frame.shape[0] == h
    detections = detector.predict(frame)
    return extractor.step(ts, detections, float(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY).mean()/255.), calibrated)


def annotate(frame, row, result, geometry):
    import cv2
    import numpy as np
    h,w = frame.shape[:2]
    points = np.array([[int(x*w),int(y*h)] for x,y in geometry["bed_polygon"]],dtype=np.int32)
    cv2.polylines(frame,[points],True,(255,180,0),2)
    if row["bbox"]:
        x1,y1,x2,y2 = row["bbox"]
        cv2.rectangle(frame,(int(x1*w),int(y1*h)),(int(x2*w),int(y2*h)),(0,220,255),2)
    color = (0,0,255) if result["active_warning"] else (0,220,220)
    cv2.putText(frame,f"{row['video_ts_s']:.2f}s {result['state']}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
    return frame


def replay(args):
    import cv2
    import numpy as np
    out = Path(args.out)
    require(not out.exists(), "Replay output exists; choose another --out")
    out.mkdir(parents=True)
    geometry,config = load(args.geometry),load(args.policy)
    detector = Detector(args.weights,args.imgsz,args.device,args.confidence)
    extractor,policy = FeatureExtractor(geometry,config["max_observation_gap_s"]),R0(config)
    frame_log, events, latencies = [],[],[]
    writer = None
    start = time.perf_counter()
    last_index = -1
    try:
        for index,ts,frame in frames(args.source,args.sample_hz,args.timestamps_file):
            tick = time.perf_counter()
            row = frame_features(extractor,detector,frame,ts)
            result = policy.step(row)
            latency = time.perf_counter()-tick
            last_index = index
            latencies.append(latency)
            for e in result["events"]:
                e.update(video_id=args.video_id,camera_setup_id=geometry["camera_setup_id"])
                events.append(e)
            frame_log.append(dict(row,**{k:v for k,v in result.items() if k != "events"},process_ms=latency*1000))
            if args.overlay:
                if writer is None:
                    writer = cv2.VideoWriter(str(out/"overlay.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),args.sample_hz,(frame.shape[1],frame.shape[0]))
                    require(writer.isOpened(), "Cannot open overlay video writer")
                writer.write(annotate(frame,row,result,geometry))
    finally:
        if writer is not None:
            writer.release()
        write_rows(out/"frames.jsonl",frame_log)
        write_rows(out/"events.jsonl",events)
    require(frame_log, "No frames processed")
    elapsed = time.perf_counter()-start
    dump(out/"runtime_benchmark.json",dict(environment=metadata(),source_sha256=sha256(args.source),weights_sha256=sha256(args.weights),policy_sha256=sha256(args.policy),geometry_sha256=sha256(args.geometry),processed_frames=len(frame_log),decoded_through_frame=last_index,wall_elapsed_s=elapsed,effective_processing_fps=len(frame_log)/elapsed,p50_processing_ms=float(np.percentile(latencies,50)*1000),p95_processing_ms=float(np.percentile(latencies,95)*1000),sample_hz=args.sample_hz,timeline="decoder PTS",overlay_timeline="CFR sampled preview; use JSONL timestamps for event metrics",cloud_or_display_latency_ms=None,live_drop_rate=None))
    print(str(out))


def extract(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    config = load(args.policy)
    videos = selected_videos(args.split)
    require(videos, "No videos in fixed split")
    detector = Detector(args.weights,args.imgsz,args.device,args.confidence)
    model_sha = sha256(args.weights)
    cache = Path("data/features")/model_sha
    cache.mkdir(parents=True,exist_ok=True)
    schema = dict(schema_version=1,feature_order=FEATURE_ORDER,sample_hz=args.sample_hz,imgsz=args.imgsz,confidence=args.confidence,max_gap_s=config["max_observation_gap_s"],weights_sha256=model_sha,manifest_sha256=sha256("data/manifests/videos.jsonl"))
    if (cache/"feature_order.json").exists():
        require(load(cache/"feature_order.json") == schema, "Feature cache settings changed; use another experiment directory")
    else:
        dump(cache/"feature_order.json",schema)
    for v in videos:
        approved(load("data/registry/datasets.json"),v["dataset_id"])
        geometry_path = Path("data/geometry")/f"{v['camera_setup_id']}.json"
        geometry = load(geometry_path)
        require(geometry["calibration_valid"], f"Uncalibrated geometry: {geometry_path}")
        out = cache/f"{v['video_id']}.parquet"
        require(not out.exists(), f"Cache already exists: {out}; do not silently overwrite")
        extractor = FeatureExtractor(geometry,config["max_observation_gap_s"])
        feature_rows = []
        for index,ts,frame in frames(v["path"],args.sample_hz,v.get("timestamps_file")):
            feature_rows.append(dict(frame_features(extractor,detector,frame,ts),frame_index=index,video_id=v["video_id"]))
        require(feature_rows, "Empty feature sequence")
        # Detect an early decoder failure instead of silently evaluating a prefix.
        require(v["duration_s"]-feature_rows[-1]["video_ts_s"] <= max(1.,2/args.sample_hz), "Decoded duration materially shorter than manifest")
        pq.write_table(pa.Table.from_pylist(feature_rows),out)
        dump(out.with_suffix(".meta.json"),dict(video_sha256=sha256(v["path"]),geometry_sha256=sha256(geometry_path),cache_sha256=sha256(out),split=args.split,split_sha256=sha256(f"data/splits/{args.split}.json"),schema_sha256=object_sha(schema)))
    print(f"Features: {cache}")


def cached_videos(feature_dir, split):
    import pyarrow.parquet as pq
    schema = load(Path(feature_dir)/"feature_order.json")
    require(schema["feature_order"] == FEATURE_ORDER, "Feature order mismatch")
    require(schema["manifest_sha256"] == sha256("data/manifests/videos.jsonl"), "Video manifest changed since extraction")
    result = []
    for video in selected_videos(split):
        approved(load("data/registry/datasets.json"),video["dataset_id"])
        p = Path(feature_dir)/f"{video['video_id']}.parquet"
        require(p.is_file(), f"Missing features: {p}")
        m = load(p.with_suffix(".meta.json"))
        require(m["split"] == split and m["split_sha256"] == sha256(f"data/splits/{split}.json"), "Feature split provenance mismatch")
        require(m["cache_sha256"] == sha256(p) and m["schema_sha256"] == object_sha(schema), "Feature file/schema modified")
        require(m["geometry_sha256"] == sha256(Path("data/geometry")/f"{video['camera_setup_id']}.json"), "Geometry changed after extraction")
        require(m["video_sha256"] == sha256(video["path"]), "Video changed after extraction")
        result.append((video,pq.read_table(p).to_pylist()))
    require(result,"No feature videos in split")
    return result


def live(args):
    """USB/RTSP OpenCV live source. One-slot queue discards old frames."""
    import cv2
    geometry,config = load(args.geometry),load(args.policy)
    detector = Detector(args.weights,args.imgsz,"cpu",args.confidence)
    extractor,policy = FeatureExtractor(geometry,config["max_observation_gap_s"]),R0(config)
    q = queue.Queue(maxsize=1)
    stop = threading.Event()
    stats = dict(captured=0,dropped=0,processed=0)
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    require(cap.isOpened(),"Live camera could not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,geometry["reference_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,geometry["reference_height"])
    began = time.monotonic()
    def capture():
        try:
            while not stop.is_set():
                ok,frame = cap.read()
                ts = time.monotonic()-began
                if not ok:
                    break
                stats["captured"] += 1
                if q.full():
                    try:
                        q.get_nowait(); stats["dropped"] += 1
                    except queue.Empty:
                        pass
                q.put((ts,frame))
        finally:
            stop.set()
    threading.Thread(target=capture,daemon=True).start()
    out = Path(args.out)
    require(not out.exists(),"Live output already exists")
    out.mkdir(parents=True)
    age_ms,local_ms = [],[]
    try:
        with (out/"live.jsonl").open("w",encoding="utf-8") as f:
            while time.monotonic()-began < args.seconds:
                try:
                    ts,frame = q.get(timeout=.2)
                except queue.Empty:
                    ts = time.monotonic()-began
                    row = extractor.step(ts,[],None)
                    result = policy.step(row)
                    f.write(json.dumps(dict(row,**result))+"\n");f.flush()
                    if stop.is_set():
                        break
                    continue
                # Queued captures can predate the last timeout observation.
                if extractor.last_ts is not None and ts <= extractor.last_ts:
                    stats["dropped"] += 1
                    continue
                now = time.monotonic()-began
                if now-ts > config["max_observation_gap_s"]:
                    stats["dropped"] += 1
                    continue
                age_ms.append((now-ts)*1000)
                row = frame_features(extractor,detector,frame,ts)
                result = policy.step(row)
                latency = (time.monotonic()-began-ts)*1000
                local_ms.append(latency)
                stats["processed"] += 1
                f.write(json.dumps(dict(row,**result,local_latency_ms=latency))+"\n");f.flush()
                if result["events"]:
                    print(json.dumps(result["events"],ensure_ascii=False),flush=True)
    finally:
        stop.set()
        cap.release()
        import numpy as np
        dump(out/"runtime_benchmark.json",dict(environment=metadata(),**stats,elapsed_s=time.monotonic()-began,p95_capture_age_ms=float(np.percentile(age_ms,95)) if age_ms else None,p95_capture_to_local_result_ms=float(np.percentile(local_ms,95)) if local_ms else None,capture_timestamp_semantics="host monotonic immediately after cap.read; camera hardware/network buffering latency is not measured",server_latency_ms=None,guardian_display_latency_ms=None))
