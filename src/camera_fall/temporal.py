"""Optional causal horizon experiments. Requires reviewed event/coverage annotations."""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np

from .common import dump, load, lock_environment, metadata, require, rows, sha256
from .risk import BINARY_INDICES, FEATURE_ORDER
from .video import cached_videos


def targets_at(t, horizons, video, events):
    y,m = [],[]
    target = [e for e in events if e["video_id"] == video["video_id"] and e["event_type"] == "unassisted_bed_exit"]
    blockers = [e for e in events if e["video_id"] == video["video_id"] and (e["event_type"] in ("occlusion","camera_unavailable") or e.get("visibility") in ("occluded","out_of_view","unobservable"))]
    after_exit = any(e["critical_ts_s"] <= t for e in target)
    for horizon in horizons:
        full = any(a <= t and t+horizon <= b for a,b in video.get("annotation_coverage",[]))
        blocked = any(e["action_start_s"] < t+horizon and e["end_s"] > t for e in blockers)
        known = bool(video.get("annotations_reviewed") and full and not blocked and not after_exit and t+horizon <= video["duration_s"])
        y.append(float(any(t < e["critical_ts_s"] <= t+horizon for e in target)))
        m.append(float(known))
    return y,m


def fit_preprocess(sequences):
    data = np.array([[np.nan if x is None else x for x in r["values"]] for _,seq in sequences for r in seq],dtype=np.float64)
    require(data.ndim == 2 and data.shape[1] == 20,"Feature shape mismatch")
    mask = np.isfinite(data)
    count = mask.sum(axis=0)
    mean = np.where(mask,data,0).sum(axis=0)/np.maximum(1,count)
    std = np.sqrt(np.where(mask,(data-mean)**2,0).sum(axis=0)/np.maximum(1,count))
    std[std < 1e-6] = 1.
    mean[BINARY_INDICES] = 0.;std[BINARY_INDICES] = 1.
    return dict(schema_version=1,feature_order=FEATURE_ORDER,mean=mean.tolist(),std=std.tolist(),observed_train_counts=count.tolist(),binary_indices=BINARY_INDICES,fill_after_standardization=0.)


def transform(sequence, preprocessing):
    x = np.array([[np.nan if x is None else x for x in r["values"]] for r in sequence],dtype=np.float32)
    m = np.array([r["masks"] for r in sequence],dtype=np.float32)
    require(np.array_equal(np.isfinite(x),m.astype(bool)),"Feature values/masks disagree")
    z = (x-np.array(preprocessing["mean"]))/np.array(preprocessing["std"])
    z[~m.astype(bool)] = 0.
    return np.concatenate([z,m],axis=1).astype(np.float32)


def windows(sequences, events, config, preprocessing, training=False):
    xx,yy,mm,keys = [],[],[],[]
    rng = random.Random(config["seed"])
    length = config["history_steps"]
    for video,seq in sequences:
        require(video.get("annotations_reviewed") is True and video.get("annotation_coverage"), f"Reviewed annotation_coverage missing: {video['video_id']}")
        values = transform(seq,preprocessing)
        starts = []
        for end in range(length-1,len(seq)):
            chunk = seq[end-length+1:end+1]
            times = [r["video_ts_s"] for r in chunk]
            # No interpolation from future samples; long gaps and track changes reset history.
            if any(b-a > config["max_gap_s"] for a,b in zip(times,times[1:])) or len({r["track_id"] for r in chunk}) != 1:
                continue
            if chunk[-1]["values"][15] != 1 or chunk[-1]["values"][16] != 1:
                continue
            y,m = targets_at(times[-1],config["horizons_s"],video,events)
            if any(m):
                starts.append((end,y,m))
        if training and len(starts) > config["max_train_windows_per_video"]:
            starts = rng.sample(starts,config["max_train_windows_per_video"])
        for end,y,m in starts:
            xx.append(values[end-length+1:end+1]);yy.append(y);mm.append(m)
            keys.append(dict(video_id=video["video_id"],video_ts_s=seq[end]["video_ts_s"]))
    require(xx,"No fully labeled causal windows. Check history, annotation coverage and event labels.")
    return np.array(xx),np.array(yy,dtype=np.float32),np.array(mm,dtype=np.float32),keys


def summaries(x):
    return np.concatenate([x[:,-1],x.mean(axis=1),x[:,-1]-x[:,0],x.std(axis=1)],axis=1)


def make_gru(config):
    import torch
    class CausalGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(40,config["hidden_size"],config["num_layers"],batch_first=True,dropout=config["dropout"] if config["num_layers"] > 1 else 0.,bidirectional=False)
            self.head = torch.nn.Linear(config["hidden_size"],len(config["horizons_s"]))
        def forward(self,x):
            output,_ = self.gru(x)
            return self.head(output[:,-1])
    return CausalGRU()


def train(args):
    from sklearn.metrics import average_precision_score,log_loss
    config = load(args.config)
    require(config["input_dim"] == 40,"This schema requires 20 values + 20 masks")
    output = Path(args.out)
    require(not output.exists(),"Temporal output exists; choose new --out")
    feature_schema = load(Path(args.features)/"feature_order.json")
    require(feature_schema["sample_hz"] == config["sample_hz"],"Temporal/cache sample rate mismatch")
    train_seq = cached_videos(args.features,"train")
    val_seq = cached_videos(args.features,"val")
    events = rows(args.events)
    preprocessing = fit_preprocess(train_seq)
    x,y,m,keys = windows(train_seq,events,config,preprocessing,training=True)
    vx,vy,vm,vkeys = windows(val_seq,events,config,preprocessing)
    for h in range(len(config["horizons_s"])):
        require(len(np.unique(y[m[:,h] > 0,h])) == 2,f"Train horizon {config['horizons_s'][h]}s needs positive and negative events")
        require(len(np.unique(vy[vm[:,h] > 0,h])) == 2,f"Validation horizon {config['horizons_s'][h]}s needs positive and negative events")
    output.mkdir(parents=True)
    dump(output/"preprocess.json",preprocessing)
    info = dict(metadata(config["seed"]),config=config,mode=args.model,train_windows=len(x),val_windows=len(vx),train_videos=len(train_seq),val_videos=len(val_seq),events_sha256=sha256(args.events),feature_schema_sha256=sha256(Path(args.features)/"feature_order.json"),scope="horizon prediction experiment; alarm policy/event-level comparison and Pi deployment are not certified by these metrics")
    if args.model == "r1":
        from sklearn.linear_model import LogisticRegression
        sx,sv = summaries(x),summaries(vx)
        models,probabilities = [],[]
        for h in range(y.shape[1]):
            select = m[:,h] > 0
            reg = LogisticRegression(max_iter=2000,random_state=config["seed"],class_weight="balanced")
            reg.fit(sx[select],y[select,h])
            models.append(dict(coef=reg.coef_[0].tolist(),intercept=float(reg.intercept_[0])))
            probabilities.append(reg.predict_proba(sv)[:,1])
        probabilities = np.stack(probabilities,axis=1)
        dump(output/"r1_coefficients.json",dict(preprocessing=preprocessing,history_steps=config["history_steps"],horizons_s=config["horizons_s"],summary_order=["last","mean","delta","std"],models=models,probability_calibrated=False))
    else:
        import torch
        from .detector import gpu_check
        info.update(gpu_check(args.expected_gpu))
        torch.manual_seed(config["seed"]);np.random.seed(config["seed"])
        net = make_gru(config).cuda()
        optimizer = torch.optim.AdamW(net.parameters(),lr=config["lr"],weight_decay=config["weight_decay"])
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(m))
        loader = torch.utils.data.DataLoader(dataset,batch_size=config["batch_size"],shuffle=True)
        val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy),torch.from_numpy(vm)),batch_size=config["batch_size"])
        best,stale,best_state = float("inf"),0,None
        history = []
        for epoch in range(config["epochs"]):
            net.train()
            for bx,by,bm in loader:
                bx,by,bm = bx.cuda(),by.cuda(),bm.cuda()
                optimizer.zero_grad()
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(bx),by,reduction="none")*bm).sum()/bm.sum().clamp_min(1)
                loss.backward();optimizer.step()
            net.eval()
            numerator,denominator = 0.,0.
            with torch.inference_mode():
                for bx,by,bm in val_loader:
                    logits = net(bx.cuda()).cpu()
                    numerator += float((torch.nn.functional.binary_cross_entropy_with_logits(logits,by,reduction="none")*bm).sum())
                    denominator += float(bm.sum())
            value = numerator/denominator
            history.append(dict(epoch=epoch,val_masked_bce=value))
            if value < best:
                best,stale = value,0
                best_state = {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            else:
                stale += 1
            if stale >= config["patience"]:
                break
        require(best_state is not None,"No valid GRU checkpoint")
        net.load_state_dict(best_state)
        net.cpu().eval()
        torch.save(dict(state_dict=best_state,config=config,preprocessing=preprocessing),output/"temporal_gru.pt")
        with torch.inference_mode():
            probabilities = torch.cat([torch.sigmoid(net(torch.from_numpy(vx[i:i+64]))) for i in range(0,len(vx),64)]).numpy()
        torch.onnx.export(net,torch.zeros(1,config["history_steps"],40),str(output/"temporal_gru.onnx"),opset_version=17,input_names=["history"],output_names=["logits"],dynamic_axes=None,dynamo=False)
        import onnxruntime as ort
        session = ort.InferenceSession(str(output/"temporal_gru.onnx"),providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            for sample in vx[:min(32,len(vx))]:
                a = net(torch.from_numpy(sample[None])).numpy()
                b = session.run(None,{"history":sample[None]})[0]
                require(np.allclose(a,b,atol=1e-5,rtol=1e-4),"GRU ONNX logit parity failed")
        info.update(history=history,onnx_logit_parity_passed=True,onnx_event_parity="not_run")
    report = {}
    for h,horizon in enumerate(config["horizons_s"]):
        select = vm[:,h] > 0
        truth,score = vy[select,h],probabilities[select,h]
        report[str(horizon)] = dict(valid_windows=int(select.sum()),positive_windows=int(truth.sum()),auprc=float(average_precision_score(truth,score)),log_loss=float(log_loss(truth,np.clip(score,1e-7,1-1e-7))),note="Overlapping windows are not independent events. No test-set threshold tuning.")
    dump(output/"validation_horizons.json",report)
    from .common import write_rows
    write_rows(output/"validation_predictions.jsonl",[dict(k,probabilities=p.tolist(),target_mask=mask.tolist()) for k,p,mask in zip(vkeys,probabilities,vm)])
    dump(output/"temporal_model_meta.json",dict(info,horizons_s=config["horizons_s"],input_shape=[1,config["history_steps"],40],output="logits; sigmoid applied once for probabilities",thresholds=None,event_policy_selected=False))
    lock_environment(output)
