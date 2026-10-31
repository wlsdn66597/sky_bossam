"""울음 개인화 힌트 (파이에서 실행, numpy만 사용).

부모가 울음이 끝난 뒤 "무엇으로 그쳤나요?"에 답한 기록으로 그 아기에게 맞춰 배운다.

흐름
  1) 울음이 감지되는 동안 1초마다 YAMNet 임베딩(과 그 1초 소리)을 모은다(add_window).
  2) 울음이 시작되고 몇 초 뒤 힌트를 낸다(hint) — 부모가 무엇을 해 볼지 정할 때 쓰는 참고. 1·2순위를 같이 낸다.
  3) 울음이 끝나면(end) 부모에게 묻고, 답을 그 울음의 소리 특징과 함께 저장한다(feedback).
힌트 = 맥락(마지막 수유 후 시간·직전 울음 후 시간·시간대) × 이 아기의 이전 답 빈도 × 이전 울음 소리와의 유사도

소리 비교 공간 — blocks로 고른다
  acoustic  YAMNet 임베딩 → 음향 특징 134개(음높이·크기·eGeMAPS·MFCC) 선형 변환(acoustic_path의 coef)
  mfcc      울음 소리에서 직접 계산한 MFCC 20개의 평균·표준편차(40개). librosa.feature.mfcc와 같은 계산을 numpy로 한다
  yamnet16  예전 YAMNet 16차원(basis)
  rhythm    프레임 세기로 계산한 울음 리듬 8개(덩어리·쉼 길이와 변동, 세기 기울기 등). 리듬으로 만든 가상 세계에서
            +0.06~0.09, 다른 세계에서는 -0.03~0. 원인이 어디에 드러날지 모르니 기본에 넣는다
  블록마다 '아기 안 퍼짐'으로 나눠 같은 비중으로 붙인다(acoustic_path의 spread_*). YAMNet은 '무슨 소리인가'를 가르도록
  학습돼 같은 울음끼리의 음높이·세기·음색 차이에 둔하다. 정답 규칙을 여러 특징으로 만든 가상 세계에서 YAMNet 16차원이
  가장 약했다.
자가 검증(gate)
  - 울음이 끝날 때마다, 그 울음의 답을 받기 전에 '소리 포함'과 '소리 없이(맥락 × 이력)' 예측을 저장해 둔다.
  - 답 5개부터 소리 예측을 기록하고, 최근 20개 중 비교가 8개 이상일 때 소리 포함 예측이 2개 이상 더 맞으면 소리를 섞는다.
  - 실제 Enes 재생에서 소리를 항상 섞으면 이력만보다 나빴다(0.433 → 0.350). 아기마다 다를 수 있어 기기가 스스로 재고 켠다.
  - 확정이 아닌 '쓸수록 맞춰지는 실험적 힌트'로만 보여주고, 확신이 낮거나 이전 답이 적으면 숨긴다(show / show2=False).
행동(actions): 기본 3개(수유 / 기저귀 / 안아주기)이고 1·2순위를 함께 보여준다(top2, show2). ACTIONS6으로 늘릴 수 있다
  (행동이 늘면 1순위는 내려가고 무작위 대비 이득은 비슷하다).

필요 파일(artifacts/, sim/ 스크립트로 다시 만들 수 있다)
  yamnet_emb.tflite          sim/export_yamnet_tflite.py — 점수 + 임베딩 출력
  cry_personal_acoustic.npz  sim/artifacts.py       — 음향 변환, MFCC 멜 필터·표준화, 블록 퍼짐
  cry_personal_basis.npz     sim/device_basis.py basis — YAMNet 16차원 표준화·PCA
  cry_personal_ctx.json      sim/device_basis.py ctx   — 맥락 로지스틱 회귀 계수(수유 / 기저귀 / 안아주기 3개로 학습)
"""
import base64
import json
import math
import os
import time

import numpy as np

ACTIONS3 = ["feed", "diaper", "hold"]                                   # Enes 라벨과 같은 3개(예전 기본)
ACTIONS6 = ["feed", "diaper", "sleep", "hold", "burp", "temp"]           # 실사용 6개
ACTIONS = ACTIONS3                                                      # 기본 3개 + 1·2순위 표시
EXTRA = ["self", "unknown", "other", "sick"]                            # 기록만 하는 답
LABEL_KO = {"feed": "수유", "diaper": "기저귀", "sleep": "재우기", "hold": "안아주기·자세 바꾸기", "burp": "트림·가스 빼기",
            "temp": "온도·환경 조절", "self": "저절로 그침", "unknown": "모름", "other": "기타", "sick": "아파 보임"}
CAP_H = 12.0
N_FFT, HOP_MFCC, N_MFCC = 1024, 256, 20


def ctx_features(t, last_cry_t, last_feed_t):
    """맥락 특징: 하루 중 시각(sin·cos), 직전 울음 후 시간, 마지막 수유 후 시간(모르면 상한)."""
    lt = time.localtime(t)
    h = lt.tm_hour + lt.tm_min / 60
    since = lambda p: CAP_H if p is None else float(min(max((t - p) / 3600, 0), CAP_H))
    return np.array([math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                     since(last_cry_t), since(last_feed_t)], float)


# --- MFCC (librosa.feature.mfcc(n_mfcc=20, n_fft=1024, hop_length=256)과 같은 값, numpy만) ---------------
def dct_matrix(n_out=N_MFCC, n_in=128):
    k, n = np.arange(n_out)[:, None], np.arange(n_in)[None, :]
    D = np.sqrt(2 / n_in) * np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    D[0] /= np.sqrt(2)
    return D


SR = 16000
F0_MIN, F0_MAX, YIN_TH = 150.0, 2000.0, 0.15
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8001)]


def _frames(x):
    x = np.asarray(x, float)
    xp = np.pad(x, N_FFT // 2)
    n = 1 + len(x) // HOP_MFCC
    return xp[np.arange(N_FFT)[None, :] + HOP_MFCC * np.arange(n)[:, None]]


def yin_frames(fr, sr=SR, fmin=F0_MIN, fmax=F0_MAX, th=YIN_TH):
    """프레임마다 YIN 음높이(Hz, 무성음이면 nan)와 비주기성(0 = 맑은 음, 1 = 잡음). numpy만."""
    tmax, tmin = int(sr / fmin), max(2, int(sr / fmax))
    W = fr.shape[1] - tmax
    a, b = fr[:, :W], fr[:, :W + tmax]
    nf = 1 << int(np.ceil(np.log2(W + tmax + 1)))
    r = np.fft.irfft(np.conj(np.fft.rfft(a, nf)) * np.fft.rfft(b, nf), nf)[:, : tmax + 1]
    c = np.cumsum(np.c_[np.zeros(len(fr)), b ** 2], axis=1)
    e0 = c[:, W][:, None]
    et = c[:, W:W + tmax + 1] - c[:, : tmax + 1]
    d = np.maximum(e0 + et - 2 * r, 0)
    cm = np.cumsum(d[:, 1:], axis=1) / np.arange(1, tmax + 1)
    dn = np.c_[np.ones(len(fr)), d[:, 1:] / (cm + 1e-12)]
    f0, ap = np.full(len(fr), np.nan), np.ones(len(fr))
    for i in range(len(fr)):
        row = dn[i]
        below = np.where(row[tmin:] < th)[0]
        if len(below):
            t = tmin + below[0]
            while t + 1 <= tmax and row[t + 1] < row[t]:
                t += 1
        else:
            t = tmin + int(np.argmin(row[tmin:]))
        ap[i] = row[t]
        if row[t] < th and 0 < t < tmax:                       # 포물선 보간
            y0, y1, y2 = row[t - 1], row[t], row[t + 1]
            den = y0 - 2 * y1 + y2
            tt = t + (0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0)
            f0[i] = sr / tt
    return f0, ap


def spectral_frame_feats(S, sr=SR):
    """파워 스펙트럼 프레임마다 10개: 무게중심·대역폭·롤오프(85%)(로그), 평탄도(로그), 엔트로피, 변화량, 대역 에너지 비율 4개."""
    f = np.linspace(0, sr / 2, S.shape[1])
    P = S + 1e-12
    tot = P.sum(1, keepdims=True)
    p = P / tot
    cen = (p * f).sum(1)
    bw = np.sqrt((p * (f - cen[:, None]) ** 2).sum(1))
    roll = f[np.minimum((np.cumsum(p, 1) < 0.85).sum(1), len(f) - 1)]
    flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    ent = -(p * np.log(p)).sum(1) / np.log(S.shape[1])
    flux = np.r_[0.0, np.sqrt((np.maximum(np.diff(p, axis=0), 0) ** 2).sum(1))]
    bands = np.stack([P[:, (f >= lo) & (f < hi)].sum(1) for lo, hi in BANDS], 1) / tot
    return np.c_[np.log(cen + 1), np.log(bw + 1), np.log(roll + 1), np.log(flat + 1e-12), ent, flux,
                 10 * np.log10(bands + 1e-12)]


def frame_features(x, mel_fb, D=None):
    """한 구간(파이는 1초 창)의 프레임 특징. 반환: MFCC [n, 20], 세기 dB [n], 음높이 Hz [n], 비주기성 [n], 스펙트럼 [n, 10]."""
    D = dct_matrix(N_MFCC, mel_fb.shape[0]) if D is None else D
    fr = _frames(x)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    S = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    mel = 10 * np.log10(np.maximum(S @ mel_fb.T, 1e-10))
    mel = np.maximum(mel, mel.max() - 80)
    f0, ap = yin_frames(fr)
    return mel @ D.T, 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9), f0, ap, spectral_frame_feats(S)


def mfcc_frames(x, mel_fb, D=None):
    """한 구간(파이는 1초 창)의 MFCC [프레임, 20]과 프레임 크기(RMS dB). mel_fb = librosa.filters.mel(16000, 1024)."""
    D = dct_matrix(N_MFCC, mel_fb.shape[0]) if D is None else D
    fr = _frames(x)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    S = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
    db = 10 * np.log10(np.maximum(S @ mel_fb.T, 1e-10))
    db = np.maximum(db, db.max() - 80)
    return db @ D.T, 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)


def mfcc_stats(mf, db):
    """소리 나는 프레임(울음 전체 최대 -30dB 이내)의 MFCC 평균·표준편차 40개."""
    act = db > db.max() - 30
    mf = mf[act] if act.sum() >= 3 else mf
    return np.r_[mf.mean(0), mf.std(0)]


def rhythm_stats(db, hop_s=HOP_MFCC / 16000):
    """울음 리듬 8개(프레임 세기에서): 소리 나는 비율, 초당 울음 덩어리 수, 덩어리·쉼 평균 길이(초), 덩어리·쉼 길이
    변동(표준편차/평균), 세기 기울기(dB/초, 커지는지 잦아드는지), 세기 폭(dB, 상위 10% - 하위 10%)."""
    db = np.asarray(db, float)
    act = db > db.max() - 30
    d = np.diff(np.r_[0, act.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (en - st) >= 3                                     # 48ms 미만 덩어리는 버린다
    st, en = st[keep], en[keep]
    burst = (en - st) * hop_s
    pause = (st[1:] - en[:-1]) * hop_s if len(st) > 1 else np.zeros(0)
    dur = max(len(db) * hop_s, 1e-6)
    cv = lambda v: float(v.std() / (v.mean() + 1e-9)) if len(v) > 1 else 0.0  # noqa: E731
    ia = np.where(act)[0]
    slope = float(np.polyfit(ia * hop_s, db[ia], 1)[0]) if len(ia) >= 3 else 0.0
    rng = float(np.subtract(*np.percentile(db[ia], [90, 10]))) if len(ia) >= 3 else 0.0
    return np.array([act.mean(), len(burst) / dur, burst.mean() if len(burst) else 0.0,
                     pause.mean() if len(pause) else 0.0, cv(burst), cv(pause), slope, rng])


def _bursts(act, min_len=3):
    d = np.diff(np.r_[0, act.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    keep = (en - st) >= min_len
    return st[keep], en[keep]


def voice_stats(db, f0, ap, hop_s=HOP_MFCC / SR):
    """목소리 질·음높이 윤곽 14개(소리 나는 프레임): 유성 비율, 음높이 중앙값·사분위 범위·상위 10%(반음),
    과발성 비율(1kHz 초과), jitter(이웃 프레임 음높이 변화, 반음), shimmer(이웃 프레임 세기 변화, dB), HNR(dB, YIN 비주기성),
    울음 덩어리 안 음높이 기울기(반음/초)·곡률(반음/초²), 오르는·내리는 덩어리 비율, 덩어리 안 음높이 폭, 시작 음높이 튐."""
    db, f0, ap = (np.asarray(v, float) for v in (db, f0, ap))
    act = db > db.max() - 30
    v = act & ~np.isnan(f0)
    st = np.full(len(f0), np.nan)
    st[v] = 12 * np.log2(f0[v] / 100)
    out = np.zeros(14)
    out[0] = v.sum() / max(act.sum(), 1)
    if v.sum() < 3:
        return out
    sv = st[v]
    out[1], out[2], out[3] = np.median(sv), np.subtract(*np.percentile(sv, [75, 25])), np.percentile(sv, 90)
    out[4] = np.mean(f0[v] > 1000)
    pair = v[1:] & v[:-1]
    if pair.any():
        out[5] = np.median(np.abs(np.diff(st))[pair])
        out[6] = np.median(np.abs(np.diff(db))[pair])
    a = np.clip(ap[v], 1e-3, 1 - 1e-3)
    out[7] = np.median(10 * np.log10((1 - a) / a))
    slopes, curvs, ranges, jumps = [], [], [], []
    for s0, e0 in zip(*_bursts(act)):
        idx = np.arange(s0, e0)[v[s0:e0]]
        if len(idx) < 4:
            continue
        tt = (idx - idx[0]) * hop_s
        c2, c1, _ = np.polyfit(tt, st[idx], 2) if len(idx) >= 5 else (0.0, *np.polyfit(tt, st[idx], 1))
        slopes.append(np.polyfit(tt, st[idx], 1)[0])
        curvs.append(c2)
        ranges.append(np.ptp(st[idx]))
        jumps.append(st[idx[:3]].mean() - np.median(st[idx]))
    if slopes:
        sl = np.array(slopes)
        out[8], out[9] = np.mean(sl), np.mean(curvs)
        out[10], out[11] = np.mean(sl > 2), np.mean(sl < -2)
        out[12], out[13] = np.mean(ranges), np.mean(jumps)
    return np.nan_to_num(out)


def spectral_stats(sp, db):
    """스펙트럼 프레임 특징 10개의 평균·표준편차(소리 나는 프레임) 20개."""
    act = db > db.max() - 30
    sp = np.asarray(sp, float)
    sp = sp[act] if act.sum() >= 3 else sp
    return np.r_[sp.mean(0), sp.std(0)]


AUDIO_SLICES = {"mfcc": (0, 40), "rhythm": (40, 48), "voice": (48, 62), "spectral": (62, 82)}


def audio_stats(mf, db, f0=None, ap=None, sp=None):
    """울음마다 저장하는 소리 통계 = MFCC 40 + 리듬 8 (+ 목소리 14 + 스펙트럼 20 = 82개)."""
    base = np.r_[mfcc_stats(mf, db), rhythm_stats(db)]
    if f0 is None:
        return base
    return np.r_[base, voice_stats(db, f0, ap), spectral_stats(sp, db)]


def _pack(v, dtype=np.float16):
    return base64.b64encode(np.asarray(v, dtype).tobytes()).decode()


def _unpack(s, dtype=np.float16):
    return np.frombuffer(base64.b64decode(s), dtype).astype(float)


class CryPersonal:
    def __init__(self, basis_path, ctx_path=None, state_path=None, k_min=10, temp=3.0, show_min=0.5,
                 hint_seconds=10, half_life=8, w_sound=1.0, proto_from="all", sound_method="lda", k_nn=3, show2_min=0.0,
                 acoustic_path=None, blocks=None, actions=None, gate=True, gate_n=8, gate_window=20, gate_margin=2,
                 track_from=5, use_not=True, semi=False, w_semi=0.5, lda_max_dim=12, ctx_mode="global", ctx_lambda=3.0,
                 ctx_min=5):
        """basis_path / acoustic_path: npz 경로 또는 딕셔너리. ctx_path: json 경로 또는 딕셔너리.
        blocks: 소리 비교 공간 묶음("acoustic", "mfcc", "rhythm", "yamnet16"). 없으면 쓸 수 있는 것 중
          acoustic + mfcc + rhythm.
        actions: 학습하는 답 목록(ACTIONS3 / ACTIONS6 또는 직접)."""
        self.actions = list(actions or ACTIONS)
        b = basis_path if isinstance(basis_path, dict) else np.load(basis_path)
        self.mu, self.sd, self.P = b["mean"], b["std"], b["pca"]
        a = None
        if acoustic_path is not None and (isinstance(acoustic_path, dict) or os.path.exists(acoustic_path)):
            a = acoustic_path if isinstance(acoustic_path, dict) else dict(np.load(acoustic_path))
        a = a or {}
        self.ac = tuple(np.asarray(a[k], float) for k in ("x_mean", "x_std", "coef", "intercept")) if "coef" in a else None
        self.mel_fb = np.asarray(a["mel_fb"], float) if "mel_fb" in a else None
        self.mfcc_norm = (np.asarray(a["mfcc_mean"], float), np.asarray(a["mfcc_std"], float)) if "mfcc_mean" in a else None
        self.dct = dct_matrix(N_MFCC, self.mel_fb.shape[0]) if self.mel_fb is not None else None
        # 소리 통계 블록(MFCC 40 · 리듬 8 · 목소리 14 · 스펙트럼 20)의 공개 데이터 기준 표준화
        self.norm = {k: (np.asarray(a[f"{k}_mean"], float), np.asarray(a[f"{k}_std"], float))
                     for k in AUDIO_SLICES if f"{k}_mean" in a}
        self.rhythm_norm = self.norm.get("rhythm")
        ok = {"yamnet16": True, "acoustic": self.ac is not None, **{k: k in self.norm for k in AUDIO_SLICES}}
        # 기본 = 쓸 수 있는 블록 전부(변환 파일이 없으면 YAMNet 16만). 정답을 8가지 특징으로 만든 가상 세계 평균에서
        #   전부 + 아기별 판별 분석이 가장 높았다(0.612 / 0.877, 예전 음향+MFCC+리듬 0.579 / 0.862)
        full = ("yamnet16", "acoustic", "mfcc", "rhythm", "voice", "spectral")
        want = blocks or ([k for k in full if ok[k]] if self.ac is not None else ["yamnet16"])
        self.blocks = [k for k in want if ok[k]] or ["yamnet16"]
        self.w = {k: 1.0 / float(a.get(f"spread_{k}", 1.0)) for k in ("yamnet16", "acoustic", *AUDIO_SLICES)}
        self.space = "+".join(self.blocks)
        self.ctx = None
        c = ctx_path if isinstance(ctx_path, dict) else (
            json.load(open(ctx_path, encoding="utf-8")) if ctx_path and os.path.exists(ctx_path) else None)
        if c:
            self.ctx = (np.array(c["coef"]), np.array(c["intercept"]), np.array(c["mean"]), np.array(c["std"]),
                        list(c.get("classes", ACTIONS3)))
        self.k_min, self.temp, self.show_min, self.hint_seconds = k_min, temp, show_min, hint_seconds
        # show2_min: 1·2순위를 함께 보여줄 기준(두 확률의 합). 행동 3개면 합이 늘 2/3 이상이라 기준을 두지 않고
        #   답이 k_min개 쌓이면 보여준다. 행동을 늘릴 때만 0.6 정도로 올린다
        self.show2_min = show2_min
        # half_life: 이력(이전 답 빈도)에서 최근 답일수록 크게 반영(답 8개마다 절반). 아기가 크면서 우는 이유
        #   비율이 바뀌어서다(재생 시험: 이력만 0.389 → 0.434). 소리 비교에는 쓰지 않는다
        # w_sound: 소리 유사도를 힌트에 얼마나 반영할지(확률의 거듭제곱). 0이면 계산·기록만 한다
        # proto_from: 이전 울음을 '울음 전체' 평균(all)으로 비교. 처음 몇 초만 쓰면 더 나빴다(0.428 → 0.381)
        self.half_life, self.w_sound, self.proto_from = half_life, w_sound, proto_from
        # sound_method: proto(행동별 평균과 코사인) / knn(가장 비슷한 이전 울음 k개의 가중 투표).
        #   유형 크기를 맞춘 가상 세계에서 proto가 knn보다 일관되게 나았다
        #   lda: 이 아기의 답으로 행동을 가르는 방향을 배우는 판별 분석(답이 적을수록 proto에 가깝게 축소)
        self.sound_method, self.k_nn, self.lda_max_dim = sound_method, k_nn, lda_max_dim
        # use_not: '해 봤는데 안 그쳤어요'(feedback_not)를 부분 정보로 쓴다. semi: 답 없는 울음에 현재 모델의
        #   추정을 w_semi 무게로 붙여 같이 배운다
        self.use_not, self.semi, self.w_semi = use_not, semi, w_semi
        # ctx_mode: global(모든 아기 공통 맥락 모델) / personal(이 아기의 답으로 다시 배운 맥락 모델, 답 ctx_min개부터).
        #   personal은 공통 모델 계수 쪽으로 ctx_lambda만큼 당겨(L2) 답이 적을 때 크게 흔들리지 않게 한다
        self.ctx_mode, self.ctx_lambda, self.ctx_min = ctx_mode, ctx_lambda, ctx_min
        self._ctx_cache = (None, None)
        # gate: 최근 gate_window개 답에서 소리 포함 예측이 gate_margin개 이상 더 맞아야 켠다(비교 gate_n개 이상).
        #   track_from: 답이 이만큼 쌓이면 소리 예측을 기록하기 시작한다(힌트에는 k_min부터)
        self.gate, self.gate_n, self.gate_window, self.gate_margin = gate, gate_n, gate_window, gate_margin
        self.track_from = track_from
        self.state_path = state_path
        self.episodes = []   # {"id","t0","t1","e","eh","m","mh","z","zh","action","pred_sound","pred_base"}
        self.last_feed = None
        self.cur = None
        if state_path and os.path.exists(state_path):
            self.load()

    # --- 울음 구간 ---------------------------------------------------------------
    def start(self, t):
        self.cur = {"t0": t, "embs": [], "mf": [], "db": [], "f0": [], "ap": [], "sp": []}

    def add_window(self, emb, audio=None):
        """울음으로 판정된 1초 창의 YAMNet 임베딩(1024)과 그 창의 소리(16kHz, MFCC 블록을 쓸 때)."""
        if self.cur is None:
            return
        self.cur["embs"].append(np.asarray(emb, np.float32))
        if audio is not None and self.mel_fb is not None:
            if any(k in self.blocks for k in ("voice", "spectral")):
                mf, db, f0, ap, sp = frame_features(audio, self.mel_fb, self.dct)
                for key, v in (("f0", f0), ("ap", ap), ("sp", sp)):
                    self.cur[key].append(v.astype(np.float32))
            else:
                mf, db = mfcc_frames(audio, self.mel_fb, self.dct)
            self.cur["mf"].append(mf.astype(np.float32))
            self.cur["db"].append(db.astype(np.float32))

    def _cur_mfcc(self, n_windows=None):
        if not self.cur or not self.cur["mf"]:
            return None
        k = slice(None) if n_windows is None else slice(0, n_windows)
        cat = lambda key: np.concatenate(self.cur[key][k])  # noqa: E731
        if self.cur["f0"]:
            return audio_stats(cat("mf"), cat("db"), cat("f0"), cat("ap"), cat("sp"))
        return audio_stats(cat("mf"), cat("db"))

    def end(self, t, mfcc=None):
        """울음이 끝났을 때 호출. 저장한 울음 번호를 돌려준다(부모 답을 받을 때 쓴다).
        mfcc: (울음 전체, 처음 hint_seconds초) MFCC 통계를 밖에서 줄 때(시험용). 답을 받기 전에 '소리 포함'·'소리 없이'
        예측을 남겨 자가 검증에 쓴다."""
        if self.cur is None or not self.cur["embs"]:
            self.cur = None
            return None
        e = np.mean(self.cur["embs"], 0)
        eh = np.mean(self.cur["embs"][: self.hint_seconds], 0)
        m, mh = mfcc if mfcc is not None else (self._cur_mfcc(), self._cur_mfcc(self.hint_seconds))
        ep = {"id": len(self.episodes), "t0": self.cur["t0"], "t1": t, "e": _pack(e), "eh": _pack(eh),
              "m": None if m is None else _pack(m, np.float32), "mh": None if mh is None else _pack(mh, np.float32),
              "z": np.round(self._project(e, m), 4).tolist(), "zh": np.round(self._project(eh, mh), 4).tolist(), "action": None,
              "pred_sound": None, "pred_base": None}
        labeled = self._labeled()
        last_cry = self.episodes[-1]["t1"] if self.episodes else None
        ep["c"] = ctx_features(self.cur["t0"], last_cry, self.last_feed).round(4).tolist()   # 아기별 맥락 학습용
        base = self._context(self.cur["t0"], last_cry) * self._prior(labeled)
        ep["pred_base"] = self.actions[int(np.argmax(base))]
        if len(labeled) >= self.track_from:
            snd = self._sound_model()(ep["zh"])[0] ** max(self.w_sound, 1.0)
            ep["pred_sound"] = self.actions[int(np.argmax(base * snd))]
        self.episodes.append(ep)
        self.cur = None
        self.save()
        return ep["id"]

    def feedback(self, ep_id, action):
        """부모 답 기록. actions에 없는 답(저절로·모름 등)은 기록만 한다. 수유는 마지막 수유 시각으로도 쓴다."""
        ep = self.episodes[ep_id]
        ep["action"] = action
        if action == "feed":
            self.last_feed = max(self.last_feed or 0, ep["t1"])
        self.save()

    def feedback_not(self, ep_id, action):
        """'해 봤는데 안 그쳤어요' 기록(그 행동은 아니었다). 최종 답이 없는 울음에서 부분 정보로 쓴다."""
        ep = self.episodes[ep_id]
        ep.setdefault("not", [])
        if action not in ep["not"]:
            ep["not"].append(action)
        self.save()

    def log_feed(self, t):
        """앱에서 수유를 따로 기록했을 때."""
        self.last_feed = max(self.last_feed or 0, t)
        self.save()

    # --- 힌트 --------------------------------------------------------------------
    def hint(self, t=None, embs=None, mfcc=None):
        """지금 울음의 힌트. embs·mfcc를 안 주면 지금까지 모은 창 중 처음 hint_seconds개를 쓴다."""
        t = t if t is not None else (self.cur["t0"] if self.cur else time.time())
        if embs is None:
            embs = self.cur["embs"][: self.hint_seconds] if self.cur else []
            mfcc = self._cur_mfcc(self.hint_seconds)
        last_cry = self.episodes[-1]["t1"] if self.episodes else None
        p_ctx = self._context(t, last_cry)
        labeled = self._labeled()
        p_prior = self._prior(labeled)
        parts = {"context": p_ctx, "history": p_prior}
        p = p_ctx * p_prior
        check = self.check()
        sound_on = False
        if len(labeled) >= self.k_min and len(embs):
            p_sound = self._sound_model()(self._project(np.mean(embs, 0), mfcc))[0]
            parts["sound"] = p_sound
            sound_on = bool(self.w_sound) and (not self.gate or check["sound_better"])
            if sound_on:
                p = p * p_sound ** self.w_sound
        p = p / p.sum()
        order = np.argsort(-p)
        top, top2 = int(order[0]), [self.actions[i] for i in order[:2]]   # 먼저 해 볼 1·2순위
        since_feed = None if self.last_feed is None else round((t - self.last_feed) / 3600, 1)
        return {"probs": {a: round(float(v), 3) for a, v in zip(self.actions, p)}, "top": self.actions[top],
                "top_ko": LABEL_KO.get(self.actions[top], self.actions[top]), "confidence": round(float(p[top]), 3),
                "top2": top2, "top2_ko": [LABEL_KO.get(a, a) for a in top2], "top2_conf": round(float(p[order[:2]].sum()), 3),
                "show": bool(len(labeled) >= self.k_min and p[top] >= self.show_min),
                "show2": bool(len(labeled) >= self.k_min and p[order[:2]].sum() >= self.show2_min),
                "n_feedback": len(labeled), "hours_since_feed": since_feed, "sound_on": sound_on,
                "recent": check, "parts": {k: [round(float(x), 3) for x in v] for k, v in parts.items()}}

    def check(self):
        """자가 검증: 최근 답 gate_window개에서 미리 낸 예측이 맞은 수(소리 포함 / 소리 없이)."""
        ev = [e for e in self.episodes if e["action"] in self.actions and e.get("pred_sound") is not None]
        ev = ev[-self.gate_window:]
        hit_s = sum(e["pred_sound"] == e["action"] for e in ev)
        hit_b = sum(e["pred_base"] == e["action"] for e in ev)
        return {"n": len(ev), "hit_sound": int(hit_s), "hit_base": int(hit_b),
                "sound_better": bool(len(ev) >= self.gate_n and hit_s >= hit_b + self.gate_margin)}

    # --- 내부 --------------------------------------------------------------------
    def _labeled(self):
        return [e for e in self.episodes if e["action"] in self.actions]

    def _project(self, emb, m=None):
        emb = np.asarray(emb, float)
        out = []
        for k in self.blocks:
            if k == "acoustic":
                xm, xs, W, b = self.ac
                v = ((emb - xm) / xs) @ W.T + b
            elif k in AUDIO_SLICES:                               # 소리 통계에서 그 블록 구간(없는 예전 기록은 0)
                lo, hi = AUDIO_SLICES[k]
                mu_, sd_ = self.norm[k]
                v = np.zeros(hi - lo) if m is None or len(m) < hi else (np.asarray(m, float)[lo:hi] - mu_) / sd_
            else:
                v = ((emb - self.mu) / self.sd) @ self.P
            out.append(v * self.w[k])
        return np.concatenate(out)

    def _weights(self, n):
        if not self.half_life or n == 0:
            return np.ones(n)
        return 0.5 ** ((n - 1 - np.arange(n)) / self.half_life)

    def _prior(self, labeled):
        w = self._weights(len(labeled))
        K = len(self.actions)
        cnt = np.bincount([self.actions.index(e["action"]) for e in labeled], weights=w, minlength=K).astype(float)
        return (cnt + 1) / (cnt.sum() + K)

    def _sound(self, z, labeled=None):
        return self._sound_model()(z)[0]

    def _soft_labels(self):
        """울음마다 행동 무게 [울음, 행동]. 최종 답 = 1, '아니었어요'만 있으면 남은 행동에 나눠 합 1, 그 밖은 0."""
        W = np.zeros((len(self.episodes), len(self.actions)))
        for i, e in enumerate(self.episodes):
            if e["action"] in self.actions:
                W[i, self.actions.index(e["action"])] = 1.0
            elif self.use_not and e.get("not"):
                ok = [k for k, a in enumerate(self.actions) if a not in e["not"]]
                if ok:
                    W[i, ok] = 1.0 / len(ok)
        return W

    def _sound_model(self):
        """지금까지의 기록으로 소리 모델을 만든다. 반환: 울음 특징들 → [n, 행동] 확률 함수.
        답이 없는 울음도 '이 아기 평균'(기준점)과 판별 분석의 주성분에는 쓴다."""
        key = "zh" if self.proto_from == "first" else "z"
        if self.sound_method == "knn":
            labeled = self._labeled()
            return lambda q: np.array([self._knn(z, labeled) for z in np.atleast_2d(np.asarray(q, float))])
        Z = np.array([e.get(key, e["z"]) for e in self.episodes], float)
        mu = Z.mean(0)
        X = Z - mu
        W = self._soft_labels()
        fit = self._fit_lda if self.sound_method == "lda" else self._fit_proto
        if self.semi:
            un = W.sum(1) == 0
            if un.any() and W.sum() > 0:
                W[un] = fit(X, W)(X[un]) * self.w_semi
        f = fit(X, W)
        return lambda q: f(np.atleast_2d(np.asarray(q, float)) - mu)

    def _fit_proto(self, X, W):
        """행동별 (무게) 평균과 코사인 유사도 → softmax(temp)."""
        m = W.sum(0)
        C = (W.T @ X) / np.maximum(m, 1e-9)[:, None]
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        has = m > 0

        def pred(Q):
            Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
            s = np.where(has[None], Qn @ Cn.T, -1.0)
            e = np.exp(self.temp * (s - s.max(1, keepdims=True)))
            return e / e.sum(1, keepdims=True)
        return pred

    def _fit_lda(self, X, W):
        """아기별 판별 분석. 이 아기 울음의 주성분 r개 위에서 행동별 평균과 행동 안 퍼짐(Sw)을 재고, 퍼짐을 답 수에
        따라 축소한 뒤(답이 적을수록 단위 행렬 쪽 = 거리 비교) 행동을 가르는 방향에 무게를 둔다."""
        m, n_eff = W.sum(0), W.sum()
        r = max(1, min(int(n_eff // 3), self.lda_max_dim, X.shape[0] - 1, X.shape[1]))
        r = max(r, min(2, X.shape[0] - 1, X.shape[1]))
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:r].T
        Xr = X @ V
        M = (W.T @ Xr) / np.maximum(m, 1e-9)[:, None]
        R = Xr[:, None, :] - M[None]
        Sw = np.einsum("nk,nki,nkj->ij", W, R, R) / max(n_eff, 1e-9)
        a = r / (r + n_eff)
        S = (1 - a) * Sw + a * (np.trace(Sw) / r + 1e-9) * np.eye(r)
        Si = np.linalg.inv(S)
        b = -0.5 * np.einsum("ki,ij,kj->k", M, Si, M)
        has = m > 0

        def pred(Q):
            s = (Q @ V) @ Si @ M.T + b
            s = np.where(has[None], s, -np.inf)
            e = np.exp(np.maximum(s - s.max(1, keepdims=True), -30)) * has[None]
            return e / e.sum(1, keepdims=True)
        return pred

    def _knn(self, z, labeled):
        key = "zh" if self.proto_from == "first" else "z"
        Z = np.array([e.get(key, e["z"]) for e in self.episodes])
        mu = Z.mean(0)
        P = np.array([e.get(key, e["z"]) for e in labeled]) - mu
        q = z - mu
        sims = P @ q / (np.linalg.norm(P, axis=1) * np.linalg.norm(q) + 1e-9)
        top = np.argsort(-sims)[: self.k_nn]
        y = np.array([self.actions.index(labeled[i]["action"]) for i in top])
        p = np.bincount(y, weights=np.exp(self.temp * sims[top]), minlength=len(self.actions)) + 0.05
        return p / p.sum()

    def _context(self, t, last_cry):
        """맥락 모델은 수유 / 기저귀 / 안아주기 3개로 학습했다. 그 밖의 행동은 세 확률의 평균(중립)을 준다."""
        K = len(self.actions)
        if self.ctx is None:
            return np.ones(K) / K
        W, b, m, s, classes = self.ctx
        if self.ctx_mode == "personal":
            W, b = self._ctx_personal(W, b, m, s, classes)
        x = (ctx_features(t, last_cry, self.last_feed) - m) / s
        z = W @ x + b
        pc = np.exp(z - z.max())
        pc = pc / pc.sum()
        p = np.array([pc[classes.index(a)] if a in classes else pc.mean() for a in self.actions])
        return p / p.sum()

    def _ctx_personal(self, W0, b0, m, s, classes):
        """이 아기의 답(울음 시작 때 맥락 특징 → 행동)으로 다항 로지스틱 회귀를 다시 맞춘다.
        목적 = 음의 로그우도 + ctx_lambda·‖(W, b) − 공통 모델‖². 답이 ctx_min개 미만이면 공통 모델을 그대로 쓴다."""
        ev = [e for e in self.episodes if e["action"] in classes and e.get("c") is not None]
        if len(ev) < self.ctx_min:
            return W0, b0
        if self._ctx_cache[0] == len(ev):
            return self._ctx_cache[1]
        X = (np.array([e["c"] for e in ev]) - m) / s
        Y = np.eye(len(classes))[[classes.index(e["action"]) for e in ev]]
        W, b = W0.copy(), b0.copy()
        for _ in range(200):                                    # 경사 하강(파라미터 15개라 충분)
            z = X @ W.T + b
            P = np.exp(z - z.max(1, keepdims=True))
            P /= P.sum(1, keepdims=True)
            G = P - Y
            W -= 0.5 * (G.T @ X + 2 * self.ctx_lambda * (W - W0)) / len(ev)
            b -= 0.5 * (G.sum(0) + 2 * self.ctx_lambda * (b - b0)) / len(ev)
        self._ctx_cache = (len(ev), (W, b))
        return W, b

    # --- 저장 --------------------------------------------------------------------
    def save(self):
        if not self.state_path:
            return
        tmp = self.state_path + ".tmp"
        json.dump({"episodes": self.episodes, "last_feed": self.last_feed, "space": self.space,
                   "actions": self.actions}, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, self.state_path)

    def load(self):
        d = json.load(open(self.state_path, encoding="utf-8"))
        self.episodes, self.last_feed = d["episodes"], d["last_feed"]
        if d.get("space", "yamnet16") != self.space:     # 비교 공간이 바뀌면 저장한 원 임베딩·MFCC로 다시 바꾼다
            dim = len(self._project(np.zeros(1024)))
            for e in self.episodes:
                if "e" in e:
                    m = _unpack(e["m"], np.float32) if e.get("m") else None
                    mh = _unpack(e["mh"], np.float32) if e.get("mh") else None
                    e["z"] = self._project(_unpack(e["e"]), m).tolist()
                    e["zh"] = self._project(_unpack(e["eh"]), mh).tolist()
            self.episodes = [e for e in self.episodes if len(e["z"]) == dim]
        for e in self.episodes:                          # 예전 상태 파일: 예측을 번호로 저장했다
            for k in ("pred_sound", "pred_base"):
                if isinstance(e.get(k), int):
                    e[k] = ACTIONS3[e[k]]
