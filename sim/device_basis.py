"""3단계: 기기와 같은 방식(1초 창)으로 YAMNet 임베딩을 계산하고, 모듈이 쓰는 기준 파일 두 개를 만든다.

    python sim/device_basis.py embed   # 창마다 임베딩·울음 점수 → CRY_DATA/sim/device_yamnet_{enes,public}.npz
    python sim/device_basis.py basis   # 공개 울음 창(Enes 제외)으로 표준화 + PCA 16차원 → artifacts/cry_personal_basis.npz
    python sim/device_basis.py ctx     # Enes 울음으로 맥락 로지스틱 회귀(수유·기저귀·안아주기) → artifacts/cry_personal_ctx.json
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
import cry_personal as cp  # noqa: E402

EMB_ENES = os.path.join(c.OUT, "device_yamnet_enes.npz")
EMB_PUB = os.path.join(c.OUT, "device_yamnet_public.npz")
MAP = {"F": "feed", "I": "diaper", "S": "hold"}               # Enes 라벨 → 행동


def device_windows(x):
    """기기처럼 1초마다 최근 0.975초를 본다."""
    if len(x) < c.WIN:
        x = np.pad(x, (0, c.WIN - len(x)))
    return [x[s:s + c.WIN] for s in range(0, len(x) - c.WIN + 1, c.HOP)]


def embed_files(paths, yam, tag):
    E, C, owner = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        x, _ = sf.read(p, dtype="float32")
        for w in device_windows(x):
            e, s = yam.window(w)
            E.append(e.astype(np.float16))
            C.append(max(s[j] for j in c.CRY_IDX))
            owner.append(i)
        if (i + 1) % 300 == 0:
            print(f"  {tag} {i + 1}/{len(paths)} {time.time() - t0:.0f}s", flush=True)
    return np.stack(E), np.array(C, np.float32), np.array(owner)


def cmd_embed(_):
    yam = c.Yam()
    m = pd.read_csv(os.path.join(c.ENES, "meta.csv"), dtype={"date": str, "time": str})
    E, C, ow = embed_files([os.path.join(c.ENES, "wav", i + ".wav") for i in m.id], yam, "enes")
    os.makedirs(c.OUT, exist_ok=True)
    np.savez(EMB_ENES, ids=m.id.values, X=E, cry=C, ep=ow)
    pub = sorted(glob.glob(os.path.join(c.JOINT, "wav", "dac", "*.wav")) +
                 glob.glob(os.path.join(c.JOINT, "wav", "baidu", "*.wav")))
    E2, C2, ow2 = embed_files(pub, yam, "public")
    np.savez(EMB_PUB, X=E2, cry=C2, ep=ow2)
    print(f"Enes 창 {len(E)}개(울음 판정 {float((C >= c.CRY_THR).mean()):.0%}), 공개 데이터 창 {len(E2)}개")


def cmd_basis(_):
    z = np.load(EMB_PUB)
    X = z["X"].astype(np.float64)[z["cry"] >= c.CRY_THR]
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    _, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
    os.makedirs(c.ART, exist_ok=True)
    np.savez(c.BASIS_NPZ, mean=mu, std=sd, pca=Vt[:16].T)
    print(f"저장 {c.BASIS_NPZ}: 공개 울음 창 {len(X)}개, PCA 16차원 설명 분산 {float((S[:16] ** 2).sum() / (S ** 2).sum()):.0%}")


def bouts():
    """Enes 울음 단위 표: 아기, 정답 행동, 시작·끝 시각. 아기·시각 순으로 정렬."""
    m = pd.read_csv(os.path.join(c.ENES, "meta.csv"), dtype={"date": str, "time": str})
    dt = (m.date.str.zfill(8) + m.time.str.zfill(4)).where(m.date.str.len() <= 8, m.date + m.time)
    m["t"] = pd.to_datetime(dt.str[:12], format="%d%m%Y%H%M", errors="coerce")
    rows = []
    for b, g in m.groupby("bout"):
        g = g.sort_values("ep")
        if g.stop.iloc[0] not in MAP:
            continue
        t0 = time.mktime(pd.Timestamp(g.t.iloc[0]).timetuple())
        span = float((g.t0 + g.dur).max() - g.t0.min())
        rows.append({"bout": b, "baby": g.baby.iloc[0], "y": "FIS".index(g.stop.iloc[0]), "t0": t0, "t1": t0 + span,
                     "key": g.key.iloc[0]})
    rows.sort(key=lambda r: (r["baby"], r["t0"], r["key"]))
    return rows


def ctx_train(rows, exclude=None):
    """맥락 로지스틱 회귀(3클래스). 학습용 특징은 부모 답이 모두 있다고 보고 만든다."""
    from sklearn.linear_model import LogisticRegression
    F, Y = [], []
    by = collections.defaultdict(list)
    for r in rows:
        by[r["baby"]].append(r)
    for bb, rs in by.items():
        if bb == exclude:
            continue
        last_cry, last_feed = None, None
        for r in rs:
            F.append(cp.ctx_features(r["t0"], last_cry, last_feed))
            Y.append(r["y"])
            last_cry = r["t1"]
            if r["y"] == 0:
                last_feed = r["t1"]
    F, Y = np.array(F), np.array(Y)
    mu, sd = F.mean(0), F.std(0) + 1e-6
    lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit((F - mu) / sd, Y)
    return {"coef": lr.coef_.tolist(), "intercept": lr.intercept_.tolist(), "mean": mu.tolist(), "std": sd.tolist(),
            "features": ["hour_sin", "hour_cos", "h_since_prev_cry", "h_since_feed"], "classes": cp.ACTIONS3}


def cmd_ctx(_):
    os.makedirs(c.ART, exist_ok=True)
    json.dump(ctx_train(bouts()), open(c.CTX_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"저장 {c.CTX_JSON}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("embed", "basis", "ctx"):
        sp.add_parser(name)
    a = ap.parse_args()
    {"embed": cmd_embed, "basis": cmd_basis, "ctx": cmd_ctx}[a.cmd](a)


if __name__ == "__main__":
    main()
