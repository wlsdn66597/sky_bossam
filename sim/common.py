"""시뮬레이션 공통: 경로, 울음 목록, 소리 읽기, 음향 요약, YAMNet(TFLite).

데이터 위치는 환경변수 CRY_DATA로 준다(기본 ./data). 필요한 입력은 README의 '데이터 준비'를 본다.
"""
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)                                       # 저장소 루트의 cry_personal

DATA = os.path.expanduser(os.environ.get("CRY_DATA", os.path.join(ROOT, "data")))
ENES = os.path.join(DATA, "enes3")                             # Enes 울음 조각과 meta.csv
JOINT = os.path.join(DATA, "joint")                            # 공개 울음 wav와 meta.csv · meta_unlab.csv
ART = os.path.join(ROOT, "artifacts")                          # 모듈이 쓰는 변환 파일
OUT = os.path.join(DATA, "sim")                                # 중간 결과(특징)
RUNS = os.path.join(DATA, "runs")                              # 보고서

SR, WIN, HOP, CRY_THR = 16000, 15600, 16000, 0.3               # 16kHz, 0.975초 창을 1초마다, 울음 점수 기준
CLIP_S = 20                                                    # 울음마다 앞 20초만 쓴다
CRY_IDX = (19, 20)                                             # YAMNet 'Crying, sobbing' · 'Baby cry, infant cry'

PREP = os.path.join(OUT, "prep.npz")                           # 음향 요약 6 + YAMNet 임베딩
FEATS_EG = os.path.join(OUT, "feats_egemaps.npz")              # eGeMAPS 88 + MFCC 40
FEATS_BLK = os.path.join(OUT, "feats_blocks.npz")              # 모듈과 같은 소리 통계 82 + 스펙트럼 모양 + 리듬
ITEMS = os.path.join(OUT, "items.csv")
ACOUSTIC_NPZ = os.path.join(ART, "cry_personal_acoustic.npz")
BASIS_NPZ = os.path.join(ART, "cry_personal_basis.npz")
CTX_JSON = os.path.join(ART, "cry_personal_ctx.json")
YAMNET = os.path.join(ART, "yamnet_emb.tflite")


def items():
    """울음 목록: Enes(조각을 울음 단위로 묶음, 시각 포함) + 공개 데이터(Enes 제외)."""
    e = pd.read_csv(os.path.join(ENES, "meta.csv"), dtype={"date": str, "time": str})
    # 파일명 하나가 날짜·시각 경계가 한 칸 밀려 있다(101020192_058 = 10102019_2058). 붙여서 다시 자른다.
    dt = (e.date.str.zfill(8) + e.time.str.zfill(4)).where(e.date.str.len() <= 8, e.date + e.time)
    ts = pd.to_datetime(dt.str[:12], format="%d%m%Y%H%M", errors="coerce")
    e["ts"] = (ts - pd.Timestamp(0)).dt.total_seconds() + e.t0
    rows = []
    for b, g in e.groupby("bout"):
        g = g.sort_values("ep")
        rows.append({"source": "enes", "group": f"enes:{g.baby.iloc[0]}", "baby": g.baby.iloc[0], "t": g.ts.min(),
                     "stop": g.stop.iloc[0], "bout": b,
                     "paths": "|".join(os.path.join(ENES, "wav", i + ".wav") for i in g.id)})
    j = pd.read_csv(os.path.join(JOINT, "meta.csv"))
    u = pd.read_csv(os.path.join(JOINT, "meta_unlab.csv"))
    for r in pd.concat([j[j.dataset != "enes"], u[u.dataset != "enes_other"]]).itertuples():
        rows.append({"source": r.dataset, "group": r.group, "baby": r.group if r.dataset == "corvin" else "",
                     "t": np.nan, "stop": "", "bout": -1, "paths": os.path.join(JOINT, r.path)})
    return pd.DataFrame(rows)


def load_items():
    m = pd.read_csv(ITEMS)
    m["baby"], m["stop"] = m.baby.fillna("").astype(str), m.stop.fillna("").astype(str)
    return m


def load_prep():
    """울음 목록, 음향 요약 6 [N, 6], YAMNet 평균 임베딩 [N, 1024]."""
    z = np.load(PREP)
    return load_items(), z["A"].astype(np.float64), z["E"].astype(np.float64)


def enes_order(m, min_n=15):
    """Enes 아기마다 울음 번호를 시간순으로. 울음이 min_n개 이상인 아기만."""
    en = (m.source == "enes").values
    order = {}
    for b in sorted(bb for bb, n in m.baby[en].value_counts().items() if n >= min_n):
        ix = np.where(m.baby.values == b)[0]
        order[b] = ix[np.lexsort((m.bout.values[ix], m.t.values[ix]))]
    return order


def load_clip(paths):
    """'|'로 이은 조각들을 이어 모노 16kHz, 앞 CLIP_S초."""
    xs, n = [], 0
    for p in paths.split("|"):
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            import librosa
            x = librosa.resample(x, orig_sr=sr, target_sr=SR)
        xs.append(x)
        n += len(x)
        if n >= CLIP_S * SR:
            break
    return np.concatenate(xs)[: CLIP_S * SR]


def acoustic(x):
    """음향 요약 6개: 음높이(반음) 중앙값·사분위 범위, 크기(dB)·변동, 소리 나는 비율, 음색(스펙트럼 중심)."""
    import librosa
    if len(x) < SR:
        x = np.pad(x, (0, SR - len(x)))
    r = librosa.feature.rms(y=x, frame_length=1024, hop_length=256)[0]
    f = librosa.yin(x, fmin=150, fmax=1000, sr=SR, frame_length=1024, hop_length=256)
    c = librosa.feature.spectral_centroid(y=x, sr=SR, n_fft=1024, hop_length=256)[0]
    k = min(len(r), len(f), len(c))
    r, f, c = r[:k], f[:k], c[:k]
    db = 20 * np.log10(r + 1e-9)
    act = db > db.max() - 30
    a = act if act.any() else np.ones(k, bool)
    semi = 12 * np.log2(f[a] / 100)
    return np.nan_to_num([np.median(semi), np.subtract(*np.percentile(semi, [75, 25])), np.mean(db[a]),
                          np.std(r[a]) / (np.mean(r[a]) + 1e-9), act.mean(), np.log(np.median(c[a]) + 1)])


def interpreter(path=YAMNET):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_content=open(path, "rb").read(), num_threads=1)   # 한글 경로 대비 model_content


class Yam:
    """기기와 같은 YAMNet(0.975초 창, 1초 간격). 출력: 임베딩 1024 · 점수 521."""

    def __init__(self):
        it = interpreter()
        it.allocate_tensors()
        self.it, self.inp = it, it.get_input_details()[0]["index"]
        self.out = {int(d["shape"][-1]): d["index"] for d in it.get_output_details()}

    def window(self, w):
        self.it.set_tensor(self.inp, w.astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.out[1024]).reshape(-1).copy(), self.it.get_tensor(self.out[521]).reshape(-1)

    def clip(self, x):
        """울음 판정 창(울음 점수 0.3 이상, 없으면 전부)의 평균 임베딩과 울음 창 비율."""
        if len(x) < WIN:
            x = np.pad(x, (0, WIN - len(x)))
        embs, cry = [], []
        for s in range(0, len(x) - WIN + 1, HOP):
            e, sc = self.window(x[s:s + WIN])
            embs.append(e)
            cry.append(max(sc[i] for i in CRY_IDX))
        embs, g = np.array(embs), np.array(cry) >= CRY_THR
        return (embs[g] if g.any() else embs).mean(0), float(g.mean())


def bal_np(yt, yp):
    """균형 정확도: 정답에 나온 행동마다 맞힌 비율의 평균."""
    yt, yp = np.asarray(yt), np.asarray(yp)
    return float(np.mean([np.mean(yp[yt == c] == c) for c in range(3) if (yt == c).any()]))


def limit_threads():
    """작업자 하나에 코어 하나."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, "1")
