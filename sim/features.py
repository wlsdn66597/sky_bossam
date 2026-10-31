"""2단계: 울음마다 정답 세계와 소리 블록에 쓸 특징을 계산한다.

    python sim/features.py egemaps --workers 8   # eGeMAPS 88 + MFCC 40(librosa)          → CRY_DATA/sim/feats_egemaps.npz
    python sim/features.py blocks --workers 8    # 모듈과 같은 소리 통계 82 + 스펙트럼 모양 22 + Enes 리듬 5 → feats_blocks.npz

blocks는 기기처럼 1초 창마다 YAMNet 울음 점수로 거른 창(0.3 이상, 없으면 전부)에서만 소리 통계를 낸다.
"""
import argparse
import os
import sys
import time

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common
    common.limit_threads()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import common as c  # noqa: E402
import cry_personal as cp  # noqa: E402

_W = {}


# --- eGeMAPS + MFCC ----------------------------------------------------------------
def _init_eg():
    import opensmile
    _W["smile"] = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                  feature_level=opensmile.FeatureLevel.Functionals)


def mfcc_stats(x):
    """소리 나는 프레임(최대 -30dB 이내)의 MFCC 20개 평균·표준편차."""
    import librosa
    mf = librosa.feature.mfcc(y=x, sr=c.SR, n_mfcc=20, n_fft=1024, hop_length=256)
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0][: mf.shape[1]]
    db = 20 * np.log10(r + 1e-9)
    act = db > db.max() - 30
    mf = mf[:, act] if act.sum() >= 3 else mf
    return np.r_[mf.mean(1), mf.std(1)]


def _eg_one(job):
    k, paths = job
    x = c.load_clip(paths)
    if len(x) < c.SR:
        x = np.pad(x, (0, c.SR - len(x)))
    return k, _W["smile"].process_signal(x, c.SR).values[0].astype(np.float32), mfcc_stats(x).astype(np.float32)


def cmd_egemaps(a):
    from multiprocessing import Pool
    m = c.items()
    G = np.zeros((len(m), 88), np.float32)
    M = np.zeros((len(m), 40), np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init_eg) as pool:
        for n, (k, g, mf) in enumerate(pool.imap_unordered(_eg_one, list(enumerate(m.paths)), chunksize=4)):
            G[k], M[k] = g, mf
            if (n + 1) % 500 == 0:
                print(f"  {n + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(c.OUT, exist_ok=True)
    np.savez(c.FEATS_EG, egemaps=np.nan_to_num(G), mfcc=np.nan_to_num(M))
    print(f"저장 {c.FEATS_EG} {time.time() - t0:.0f}s", flush=True)


# --- 모듈과 같은 소리 통계 + 스펙트럼 모양 + 리듬 ------------------------------------------
def _init_blk():
    import librosa
    _W["yam"], _W["fb"] = c.Yam(), librosa.filters.mel(sr=c.SR, n_fft=cp.N_FFT)


def spectral_shape(x):
    """스펙트럼 모양 22개: 대비(7대역) 평균·표준편차, 평탄도·롤오프·대역폭·영교차 평균·표준편차(소리 나는 프레임)."""
    import librosa
    kw = dict(n_fft=1024, hop_length=256)
    con = librosa.feature.spectral_contrast(y=x, sr=c.SR, **kw)
    flat = librosa.feature.spectral_flatness(y=x, **kw)
    roll = np.log(librosa.feature.spectral_rolloff(y=x, sr=c.SR, **kw) + 1)
    bw = np.log(librosa.feature.spectral_bandwidth(y=x, sr=c.SR, **kw) + 1)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=1024, hop_length=256)
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    F = np.vstack([con, np.log(flat + 1e-6), roll, bw, zcr])[:, : len(r)]
    db = 20 * np.log10(r[: F.shape[1]] + 1e-9)
    act = db > db.max() - 30
    F = F[:, act] if act.sum() >= 3 else F
    return np.r_[F.mean(1), F.std(1)]


def _blk_one(job):
    k, paths = job
    x = c.load_clip(paths)
    xw = np.pad(x, (0, max(0, c.WIN - len(x))))
    wins = [xw[s:s + c.WIN] for s in range(0, len(xw) - c.WIN + 1, c.HOP)]
    cry, embs = [], []
    for w in wins:
        e, sc = _W["yam"].window(w)
        embs.append(e)
        cry.append(max(sc[i] for i in c.CRY_IDX) >= c.CRY_THR)
    use = [w for w, ok in zip(wins, cry) if ok] or wins
    emb = np.mean([e for e, ok in zip(embs, cry) if ok] or embs, 0)
    fr = [cp.frame_features(w, _W["fb"]) for w in use]
    ms = cp.audio_stats(*(np.concatenate([f[j] for f in fr]) for j in range(5)))   # MFCC 40 + 리듬 8 + 목소리 14 + 스펙트럼 20
    xs = np.pad(x, (0, max(0, c.SR - len(x))))
    return k, ms.astype(np.float32), spectral_shape(xs).astype(np.float32), emb.astype(np.float32)


def enes_rhythm(m):
    """Enes 울음마다 리듬 5개(메타 정보): 음절 속도, 음절 사이 쉼, 유성 비율, 울음 길이(로그), 조각 수. 공개 데이터는 NaN."""
    e = pd.read_csv(os.path.join(c.ENES, "meta.csv"), dtype={"date": str, "time": str})
    R = np.full((len(m), 5), np.nan)
    by = {b: g for b, g in e.groupby("bout")}
    for i in np.where(m.source.values == "enes")[0]:
        g = by[m.bout.values[i]]
        d = g.dur.sum()
        R[i] = [g.n_syl.sum() / d, np.average(g.gap_med.fillna(g.gap_med.median()), weights=g.dur), g.voiced.sum() / d,
                np.log(d), len(g)]
    return R


def cmd_blocks(a):
    from multiprocessing import Pool
    m = c.items()
    MS = np.zeros((len(m), 82), np.float32)
    SP = np.zeros((len(m), 22), np.float32)
    EM = np.zeros((len(m), 1024), np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init_blk) as pool:
        for n, (k, ms, sp, em) in enumerate(pool.imap_unordered(_blk_one, list(enumerate(m.paths)), chunksize=4)):
            MS[k], SP[k], EM[k] = ms, sp, em
            if (n + 1) % 500 == 0:
                print(f"  {n + 1}/{len(m)} {time.time() - t0:.0f}s", flush=True)
    os.makedirs(c.OUT, exist_ok=True)
    np.savez(c.FEATS_BLK, audio_dev=np.nan_to_num(MS), spec=np.nan_to_num(SP), rhythm=enes_rhythm(m), emb=EM)
    print(f"저장 {c.FEATS_BLK} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("egemaps", "blocks"):
        p = sp.add_parser(name)
        p.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"egemaps": cmd_egemaps, "blocks": cmd_blocks}[a.cmd](a)


if __name__ == "__main__":
    main()
