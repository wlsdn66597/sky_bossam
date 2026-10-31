from __future__ import annotations

import itertools
import random
from pathlib import Path

import numpy as np

from .common import dump, load, object_sha, require, rows, sha256
from .risk import FEATURE_ORDER, R0
from .video import cached_videos


def alarm_episodes(events, end_s, merge_gap):
    episodes, active = [],None
    for e in sorted(events,key=lambda x:x["video_ts_s"]):
        ts = e["video_ts_s"]
        if e["event_type"] == "risk_started" and active is None:
            active = [ts,None]
        elif e["event_type"] in ("risk_ended","risk_interrupted") and active is not None:
            active[1] = ts
            episodes.append(active); active = None
    if active is not None:
        active[1] = end_s
        episodes.append(active)
    merged = []
    for start,end in episodes:
        if merged and start-merged[-1][1] <= merge_gap:
            merged[-1][1] = max(end,merged[-1][1])
        else:
            merged.append([start,end])
    return merged


def match_events(targets, alarm_times):
    """Match early warnings first, one-to-one; late detection never increases early TP."""
    used,early,late = set(),[],[]
    targets = sorted(targets,key=lambda e:e["critical_ts_s"])
    matched = set()
    for e in targets:
        candidates = [i for i,t in enumerate(alarm_times) if i not in used and e["action_start_s"] <= t <= e["critical_ts_s"]]
        if candidates:
            i = min(candidates,key=lambda i:alarm_times[i])
            used.add(i);matched.add(e["event_id"])
            early.append(dict(event_id=e["event_id"],alarm_index=i,lead_time_s=e["critical_ts_s"]-alarm_times[i]))
    for e in targets:
        if e["event_id"] in matched:
            continue
        candidates = [i for i,t in enumerate(alarm_times) if i not in used and e["critical_ts_s"] < t <= e["end_s"]]
        if candidates:
            i = min(candidates,key=lambda i:alarm_times[i])
            used.add(i)
            late.append(dict(event_id=e["event_id"],alarm_index=i,delay_s=alarm_times[i]-e["critical_ts_s"]))
    return dict(targets=len(targets),alarms=len(alarm_times),early=early,late=late,false_alarms=len(alarm_times)-len(used),unmatched_alarm_indices=[i for i in range(len(alarm_times)) if i not in used])


def aggregate(items):
    total = sum(x["targets"] for x in items)
    alarms = sum(x["alarms"] for x in items)
    early = sum(len(x["early"]) for x in items)
    late = sum(len(x["late"]) for x in items)
    false = sum(x["false_alarms"] for x in items)
    wall = sum(x["duration_s"] for x in items)
    valid = sum(x["valid_s"] for x in items)
    leads = [e["lead_time_s"] for x in items for e in x["early"]]
    return dict(target_events=total,alarm_episodes=alarms,early_matches=early,late_matches=late,unmatched_alarms=false,early_event_recall=early/total if total else None,early_event_precision=early/alarms if alarms else None,late_only_event_recall=late/total if total else None,early_or_late_recall=(early+late)/total if total else None,false_per_wall_hour=false/wall*3600 if wall else None,false_per_8_wall_hours=false/wall*28800 if wall else None,false_per_valid_hour=false/valid*3600 if valid else None,availability=valid/wall if wall else None,wall_seconds=wall,valid_seconds=valid,lead_median_s=float(np.median(leads)) if leads else None,lead_p10_s=float(np.percentile(leads,10)) if leads else None)


def score_sequences(sequences, events, config, predictor=None):
    results = []
    for v,sequence in sequences:
        policy = R0(config)
        alarm_events,outputs = [],[]
        for row in sequence:
            result = policy.step(row) if predictor is None else predictor.step(v["video_id"],row)
            alarm_events.extend(result["events"])
            outputs.append(result)
        episodes = alarm_episodes(alarm_events,v["duration_s"],config["alarm_merge_gap_s"])
        target = [e for e in events if e["video_id"] == v["video_id"] and e["event_type"] == "unassisted_bed_exit"]
        valid = 0.
        for i,row in enumerate(sequence):
            next_ts = sequence[i+1]["video_ts_s"] if i+1 < len(sequence) else v["duration_s"]
            if outputs[i]["observation_valid"]:
                valid += max(0.,min(next_ts-row["video_ts_s"],config["max_observation_gap_s"]))
        match = match_events(target,[a[0] for a in episodes])
        results.append(dict(match,video_id=v["video_id"],session_id=v["session_id"],subject_kind=v["subject_kind"],modality=v.get("modality","unknown"),duration_s=v["duration_s"],valid_s=min(valid,v["duration_s"]),episodes=episodes))
    return results


def tune(args):
    require(args.split == "val", "Policy tuning is only allowed on val")
    sequence = cached_videos(args.features,"val")
    events = rows(args.events)
    require(any(e["event_type"] == "unassisted_bed_exit" and e["video_id"] in {v["video_id"] for v,_ in sequence} for e in events), "Validation needs target events; cannot select warning thresholds without positives")
    base = load(args.config)
    candidates = []
    for distance,hold in itertools.product((.06,.10,.14),(.4,.6,1.0)):
        c = dict(base,near_distance_ratio=distance,warning_hold_s=hold)
        metrics = aggregate(score_sequences(sequence,events,c))
        # Explicit validation objective. No claim that this is a certified target.
        score = metrics["early_event_recall"] - .05*(metrics["false_per_wall_hour"] or 0)
        candidates.append(dict(config=c,metrics=metrics,selection_score=score))
    winner = max(candidates,key=lambda x:x["selection_score"])
    frozen = dict(winner["config"],selection_split="val",selection_objective="early_recall - 0.05 * unmatched_alarms_per_wall_hour",validation_split_sha256=sha256("data/splits/val.json"),events_sha256=sha256(args.events),feature_schema_sha256=sha256(Path(args.features)/"feature_order.json"))
    dump(args.out,frozen)
    dump(Path(args.out).with_suffix(".search.json"),candidates)
    print(args.out)


def evaluate(args):
    require(args.policy == "r0", "This executable evaluates R0 events. R1/R2 produce horizon predictions; see README for their separate validation scope.")
    config = load(args.frozen_config)
    require(config.get("selection_split") == "val", "Use a frozen policy produced by tune_r0 on val")
    require(config["feature_schema_sha256"] == sha256(Path(args.features)/"feature_order.json"), "Feature settings differ from threshold selection")
    require(config["events_sha256"] == sha256(args.events), "Event annotations changed since selection")
    sequences = cached_videos(args.features,args.split)
    events = rows(args.events)
    results = score_sequences(sequences,events,config)
    summary = aggregate(results)
    groups = {}
    for key in ("subject_kind","modality"):
        for value in sorted({x[key] for x in results}):
            groups[f"{key}:{value}"] = aggregate([x for x in results if x[key] == value])
    sessions = sorted({x["session_id"] for x in results})
    ci = None
    if len(sessions) >= 2:
        rng = random.Random(42)
        boot = []
        for _ in range(1000):
            sample = rng.choices(sessions,k=len(sessions))
            boot.append(aggregate([x for s in sample for x in results if x["session_id"] == s]))
        ci = {}
        for key in ("early_event_recall","false_per_wall_hour","availability"):
            nums = [b[key] for b in boot if b[key] is not None]
            ci[key] = np.percentile(nums,[2.5,97.5]).tolist() if nums else None
    dump(args.out,dict(schema_version=1,split=args.split,policy_sha256=sha256(args.frozen_config),summary=summary,groups=groups,session_bootstrap_95pct=ci,videos=results,scope="bed-exit warning; not fall confirmation; late-only recall uses annotated critical-to-end interval"))
    import csv
    with Path(args.out).with_suffix(".errors.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=["video_id","kind","id_or_time"])
        writer.writeheader()
        for r in results:
            for index in r["unmatched_alarm_indices"]:
                writer.writerow(dict(video_id=r["video_id"],kind="unmatched_alarm",id_or_time=r["episodes"][index][0]))
            matched = {e["event_id"] for e in r["early"]}
            for e in events:
                if e["video_id"] == r["video_id"] and e["event_type"] == "unassisted_bed_exit" and e["event_id"] not in matched:
                    writer.writerow(dict(video_id=r["video_id"],kind="no_early_warning",id_or_time=e["event_id"]))
    print(args.out)
