"""4단계: 모듈이 쓰는 음향 변환 파일을 만든다(artifacts/cry_personal_acoustic.npz).

    python sim/artifacts.py

담는 것
  - YAMNet 임베딩(1024) → 음향 특징 134개(요약 6 + eGeMAPS 88 + MFCC 40) 릿지 회귀. 공개 데이터만 쓰고 원인 라벨은 쓰지 않는다.
    규제 세기는 공개 데이터 그룹 20%를 떼어 고른다.
  - 소리 통계 블록(MFCC · 리듬 · 목소리 · 스펙트럼)의 공개 데이터 기준 평균·표준편차, MFCC 멜 필터.
  - 블록마다 'Enes 아기 안 퍼짐'(라벨 안 씀). 모듈이 블록을 이 값으로 나눠 같은 비중으로 붙인다.
필요: prep.npz(1단계), feats_egemaps.npz · feats_blocks.npz(2단계), cry_personal_basis.npz(3단계).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
import cry_personal as cp  # noqa: E402


def acoustic_targets(A, f):
    import opensmile
    gm = list(opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                              feature_level=opensmile.FeatureLevel.Functionals).feature_names)
    names = ["f0_med", "f0_iqr", "loud_db", "loud_cv", "active", "centroid"] + gm + \
        [f"mfcc{i}_mean" for i in range(20)] + [f"mfcc{i}_std" for i in range(20)]
    return np.c_[A, f["egemaps"], f["mfcc"]].astype(np.float64), names


def ridge_map(m, X, T):
    """공개 데이터로 YAMNet(X: [N, 1024]) → 목표 특징(T: [N, d]) 릿지 회귀. 반환: 처음 보는 그룹 R², 변환 계수."""
    from sklearn.linear_model import Ridge
    pub = (m.source != "enes").values
    xm, xs = X[pub].mean(0), X[pub].std(0) + 1e-6
    tm, ts = T[pub].mean(0), T[pub].std(0) + 1e-9
    Xs, Ts = (X - xm) / xs, (T - tm) / ts
    ug = np.random.default_rng(0).permutation(np.unique(m.group.values[pub]))
    hold = pub & m.group.isin(ug[: len(ug) // 5]).values
    fit = pub & ~hold

    def r2(g):
        return (1 - ((g.predict(Xs[hold]) - Ts[hold]) ** 2).sum(0) / ((Ts[hold] - Ts[hold].mean(0)) ** 2).sum(0)).mean()
    best = max(np.logspace(0, 5, 11), key=lambda al: r2(Ridge(alpha=al).fit(Xs[fit], Ts[fit])))
    g = Ridge(alpha=best).fit(Xs[pub], Ts[pub])
    return r2(Ridge(alpha=best).fit(Xs[fit], Ts[fit])), {"x_mean": xm, "x_std": xs, "coef": g.coef_,
                                                           "intercept": g.intercept_, "alpha": best}


def main():
    import librosa
    m, A, E = c.load_prep()
    T, names = acoustic_targets(A, np.load(c.FEATS_EG))
    r2, p = ridge_map(m, E, T)
    z = {"x_mean": p["x_mean"].astype(np.float32), "x_std": p["x_std"].astype(np.float32),
         "coef": p["coef"].astype(np.float32), "intercept": p["intercept"].astype(np.float32), "names": np.array(names)}
    print(f"음향 변환: 1024 → {len(names)}, 규제 {p['alpha']:.0f}, 처음 보는 그룹 R² {r2:.2f}")

    pub = (m.source != "enes").values
    MS = np.load(c.FEATS_BLK)["audio_dev"].astype(float)
    for k, (lo, hi) in cp.AUDIO_SLICES.items():
        z[f"{k}_mean"] = MS[pub, lo:hi].mean(0).astype(np.float32)
        z[f"{k}_std"] = (MS[pub, lo:hi].std(0) + 1e-6).astype(np.float32)
    z["mel_fb"] = librosa.filters.mel(sr=c.SR, n_fft=cp.N_FFT).astype(np.float32)

    b = np.load(c.BASIS_NPZ)
    dev = cp.CryPersonal({k: b[k] for k in ("mean", "std", "pca")}, None, acoustic_path=z,
                         blocks=["yamnet16", "acoustic", *cp.AUDIO_SLICES])
    order = c.enes_order(m, min_n=2)
    for blk in ("yamnet16", "acoustic", *cp.AUDIO_SLICES):
        dev.blocks, dev.w[blk] = [blk], 1.0
        Z = np.array([dev._project(E[i], MS[i]) for i in range(len(m))])
        d = np.concatenate([Z[ix] - Z[ix].mean(0) for ix in order.values()])
        z[f"spread_{blk}"] = np.float32(np.sqrt((d ** 2).sum(1).mean()))
    os.makedirs(c.ART, exist_ok=True)
    np.savez(c.ACOUSTIC_NPZ, **z)
    print(f"저장 {c.ACOUSTIC_NPZ}", {k: round(float(z[k]), 3) for k in z if k.startswith("spread_")})


if __name__ == "__main__":
    main()
