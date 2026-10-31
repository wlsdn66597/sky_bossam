"""5단계: 가상 아기 시뮬레이션과 실제 Enes 확인. 모듈(cry_personal.py)을 그대로 시간순으로 재생한다.

    python sim/simulate.py long --workers 8            # 가상 세계, 쌓인 답 수별 정확도 → CRY_DATA/runs/LONG_REPORT_g11.md
    python sim/simulate.py long --seeds 2 --half       # 시드 수 조절, 부모가 절반만 답하는 경우도 함께
    python sim/simulate.py real --workers 8            # 실제 Enes 라벨과 귀무(아기 안 라벨 섞기) → REAL_REPORT.md

가상 아기
  울음 15개 이상인 Enes 아기를 시드마다 섞어 group명씩 묶는다(11명이면 울음 약 275개). 아기마다 평균을 빼 목소리 차이를
  지우고, 각자의 시간 순서를 지켜 번갈아 넣는다.
정답 만들기(baby_rule)
  가상 아기 안에서 정답 세계의 특징을 표준화(10개 넘으면 PCA 10) → k-평균 3유형 → 유형 크기를 같게 맞춰 배정 →
  유형마다 행동(수유·기저귀·안아주기)을 아기마다 무작위로 짝짓는다. 모든 아기 공통 규칙은 없다.
정답 세계 8가지: 요약 6 / eGeMAPS / MFCC / YAMNet / 스펙트럼 모양 / 리듬 / 목소리 / 혼합.
  정답을 만든 특징과 모듈의 비교 블록이 같으면 점수가 부풀려진다(순환). 혼합 세계는 모든 특징 묶음의 주성분을
  무작위 비율로 섞어 어느 블록과도 그대로 겹치지 않는다.
재생: 울음마다 그때까지의 답만으로 힌트를 먼저 내고, 그다음 답을 받는다. 모든 예측이 처음 보는 울음에 대한 것이다.
지표: 1순위 균형 정확도 / 정답이 1·2순위 안에 든 비율(행동별 평균). 무작위 0.33 / 0.67.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as c  # noqa: E402
import cry_personal as cp  # noqa: E402

CFG = {"이력만": dict(w_sound=0.0),
       "음향+MFCC+리듬, 자가 검증": dict(blocks=["acoustic", "mfcc", "rhythm"]),
       "블록 전부 + LDA, 자가 검증 (모듈 기본)": dict(sound_method="lda"),
       "블록 전부 + LDA, 소리 항상 켬": dict(sound_method="lda", gate=False)}
ANSWERS = {"답 100%": 1.0, "답 50%": 0.5}
WORLDS = ("요약 6", "eGeMAPS", "MFCC", "YAMNet", "스펙트럼 모양", "리듬", "목소리", "혼합")
BUCKETS = [(10, 19), (20, 39), (40, 79), (80, 119), (120, 159), (160, 199), (200, 999)]
_W = {}


# --- 준비 ----------------------------------------------------------------------------
def _family_pca(worlds, enes):
    """혼합 세계용: 특징 묶음마다 Enes 울음으로 표준화 → 주성분 5개."""
    out = {}
    for k, F in worlds.items():
        X = np.nan_to_num(F[enes].astype(float))
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
        P = np.zeros((len(F), 5))
        P[enes] = Z @ Vt[:5].T
        out[k] = P
    return out


def _init():
    m, A, E = c.load_prep()
    f, f2 = np.load(c.FEATS_EG), np.load(c.FEATS_BLK)
    b = np.load(c.BASIS_NPZ)
    MS = f2["audio_dev"].astype(float)
    _W.update(m=m, E0=E, MS=MS, basis={k: b[k] for k in ("mean", "std", "pca")}, ac=dict(np.load(c.ACOUSTIC_NPZ)),
              order=c.enes_order(m),
              worlds={"요약 6": A, "eGeMAPS": f["egemaps"].astype(float), "MFCC": MS[:, :40], "YAMNet": E,
                      "스펙트럼 모양": f2["spec"].astype(float), "리듬": f2["rhythm"], "목소리": MS[:, 48:62]})
    _W["fam5"] = _family_pca(_W["worlds"], (m.source == "enes").values)


def mixed_world(seed):
    """혼합 세계: 모든 특징 묶음의 주성분 5개를 묶음마다 무작위 비율·무작위 방향으로 섞은 10차원."""
    rng = np.random.default_rng(1000 + seed)
    fams = _W["fam5"]
    w = rng.dirichlet(np.ones(len(fams)))
    return sum(w[i] * (F @ rng.standard_normal((5, 10))) for i, F in enumerate(fams.values()))


def virtual_babies(world, seed, gsize, F=None):
    """Enes 아기 gsize명씩 묶은 가상 아기. 반환: {가상 아기: 울음 순서}, 임베딩, 소리 통계, 정답 세계 특징."""
    E0, MS, order = _W["E0"], _W["MS"], _W["order"]
    rng = np.random.default_rng(seed)
    babies = list(order)
    rng.shuffle(babies)
    groups = [babies[i:i + gsize] for i in range(0, len(babies) - len(babies) % gsize, gsize)]
    En, Mn, Fn = E0.copy(), MS.copy(), (_W["worlds"][world] if F is None else F).copy()
    allix = np.concatenate(list(order.values()))
    gE, gM = E0[allix].mean(0), MS[allix].mean(0)
    for ix in order.values():                                 # 아기마다 평균을 빼고 전체 평균을 더한다
        En[ix] += gE - E0[ix].mean(0)
        Mn[ix] += gM - MS[ix].mean(0)
        Fn[ix] = (Fn[ix] - Fn[ix].mean(0)) / (Fn[ix].std(0) + 1e-9)
    vorder = {}
    for gi, grp in enumerate(groups):
        pos = np.concatenate([np.arange(len(order[b])) / len(order[b]) for b in grp])
        ix = np.concatenate([order[b] for b in grp])
        vorder[f"v{gi}"] = ix[np.argsort(pos + rng.random(len(pos)) * 1e-6)]
    return vorder, En, Mn, Fn


def baby_rule(m, F, order, K, seed):
    """가상 아기 안에서 표준화(차원이 많으면 PCA 10) → k-평균 K유형 → 유형 크기를 같게 배정 → 유형마다 행동을 무작위로."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    fr = np.ones(K) / K
    y = np.zeros(len(m), int)
    for ix in order.values():
        Z = (F[ix] - F[ix].mean(0)) / (F[ix].std(0) + 1e-9)
        if Z.shape[1] > 10:
            Z = PCA(10, random_state=seed).fit_transform(Z)
        C = KMeans(K, n_init=10, random_state=seed).fit(Z).cluster_centers_
        slot = rng.permutation(K)                               # 유형 j → 행동 slot[j]
        raw = fr * len(ix)
        cap = np.floor(raw).astype(int)
        cap[np.argsort(-(raw - cap))[: len(ix) - cap.sum()]] += 1
        cols = np.concatenate([[j] * cap[slot[j]] for j in range(K)])
        D = ((Z[:, None] - C[None]) ** 2).sum(-1)[:, cols]
        r, cc = linear_sum_assignment(D)
        y[ix[r]] = slot[cols[cc]]
    return y


# --- 재생 ----------------------------------------------------------------------------
def replay(y, actions, vorder, En, Mn, seed, answers):
    """울음마다 힌트를 먼저 내고 답을 받는다. 답 여부는 울음마다 미리 뽑아 설정 간에 같다.
    반환: (설정, 답 방식, 정답, 1순위, 2순위 적중, 몇 번째 울음, 소리 켜짐)."""
    rec = []
    for ix in vorder.values():
        draw = np.random.default_rng([seed, len(ix), int(ix[0])]).random(len(ix))
        for aname in answers:
            for cname, kw in CFG.items():
                dev = cp.CryPersonal(_W["basis"], None, acoustic_path=_W["ac"], actions=actions, hint_seconds=10,
                                     use_not=False, semi=False, **{"sound_method": "proto", **kw})
                for pos, i in enumerate(ix):
                    t = 1.7e9 + pos * 3600.0
                    h = dev.hint(t, embs=[En[i]], mfcc=Mn[i])
                    rec.append((cname, aname, y[i], actions.index(h["top"]), actions[y[i]] in h["top2"], pos, h["sound_on"]))
                    dev.start(t)
                    dev.add_window(En[i])
                    eid = dev.end(t + 60, mfcc=(Mn[i], Mn[i]))
                    if draw[pos] < ANSWERS[aname]:
                        dev.feedback(eid, actions[y[i]])
    return rec


def _score(x):
    """(world, seed)마다 1순위 균형 정확도와 2순위 안 적중률(행동별 평균)을 내서 평균."""
    t1, t2 = [], []
    for _, g in x.groupby(["world", "seed"]):
        t1.append(c.bal_np(g.y, g.pred))
        t2.append(np.mean([g.hit2[g.y == k].mean() for k in g.y.unique()]))
    return np.mean(t1), np.mean(t2)


def _pool(task, jobs, workers):
    import warnings
    from multiprocessing import Pool
    warnings.filterwarnings("ignore")
    rows, t0 = [], time.time()
    with Pool(workers, initializer=_init) as pool:
        for n, r in enumerate(pool.imap_unordered(task, jobs)):
            rows += r
            if (n + 1) % 8 == 0 or n + 1 == len(jobs):
                print(f"  {n + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    cols = ["kind", "world", "seed", "cfg", "ans", "y", "pred", "hit2", "pos", "on"]
    return pd.DataFrame(rows, columns=cols)


def _write(name, txt):
    os.makedirs(c.RUNS, exist_ok=True)
    open(os.path.join(c.RUNS, name), "w", encoding="utf-8").write(txt)
    print(txt)


# --- 가상 세계, 긴 사용 기간 ---------------------------------------------------------------
def _long_task(job):
    world, seed, gsize, answers = job
    F = mixed_world(seed) if world == "혼합" else None
    vorder, En, Mn, Fn = virtual_babies(world, seed, gsize, F)
    y = baby_rule(_W["m"], Fn, vorder, 3, seed)
    return [("sim", world, seed) + r for r in replay(y, cp.ACTIONS3, vorder, En, Mn, seed, answers)]


def cmd_long(a):
    answers = ["답 100%"] + (["답 50%"] if a.half else [])
    D = _pool(_long_task, [(w, s, a.group, answers) for w in WORLDS for s in range(a.seeds)], a.workers)
    os.makedirs(c.RUNS, exist_ok=True)
    D.to_csv(os.path.join(c.RUNS, f"long_g{a.group}.csv"), index=False)
    _write(f"LONG_REPORT_g{a.group}.md", report_long(D, a.group, a.seeds))


def report_long(D, group, seeds):
    worlds = [w for w in WORLDS if w in set(D.world)]
    lab = lambda lo, hi: f"{lo}~{hi}" if hi < 999 else f"{lo}+"  # noqa: E731
    L = [f"# 쌓인 답 수별 정확도 (행동 3개, 아기 {group}명을 이어 붙인 가상 아기, 울음 약 {25 * group}개)", "",
         f"- 가상 아기 {22 // group}명 × 시드 {seeds}개 × 정답 세계 {len(worlds)}가지.",
         "- 칸 = 그 구간 울음의 1순위 균형 정확도 / 정답이 1·2순위 안에 든 비율. 무작위 0.33 / 0.67.",
         "- 구간 = 몇 번째 울음인지(답 100%면 쌓인 답 수와 같다). 힌트는 답 10개부터 보인다.",
         "- 블록 전부는 혼합 외 7개 세계와 특징이 겹친다(순환). 혼합 세계 줄을 함께 본다.", ""]
    for aname in dict.fromkeys(D.ans):
        A = D[D.ans == aname]
        for title, ws in ((f"{len(worlds)}개 세계 평균 ({aname})", worlds), (f"혼합 세계만 ({aname})", ["혼합"])):
            if not set(ws) & set(A.world):
                continue
            L += [f"## {title}", "", "| 설정 | " + " | ".join(f"울음 {lab(lo, hi)}번째" for lo, hi in BUCKETS) + " | 소리 켜짐(20번째~) |",
                  "|---|" + "---:|" * (len(BUCKETS) + 1)]
            for cname in CFG:
                g = A[(A.cfg == cname) & A.world.isin(ws)]
                if g.empty:
                    continue
                cells = []
                for lo, hi in BUCKETS:
                    x = g[(g.pos >= lo) & (g.pos <= hi)]
                    if len(x) < 50 * len(ws):
                        cells.append("-")
                        continue
                    v = np.mean([_score(x[x.world == w]) for w in ws if (x.world == w).any()], 0)
                    cells.append(f"{v[0]:.2f} / {v[1]:.2f}")
                on = f"{g[g.pos >= 20].on.mean():.0%}" if "자가 검증" in cname else "-"
                L.append(f"| {cname} | " + " | ".join(cells) + f" | {on} |")
            n = [int(((A.cfg == "이력만") & A.world.isin(ws) & (A.pos >= lo) & (A.pos <= hi)).sum()) for lo, hi in BUCKETS]
            L += ["| (울음 수) | " + " | ".join(str(v) for v in n) + " | |", ""]
    return "\n".join(L) + "\n"


# --- 실제 Enes ----------------------------------------------------------------------
def _real_task(seed):
    """실제 Enes 라벨(울음을 그치게 한 행동, 답 100%). seed > 0이면 아기 안에서 라벨을 섞은 귀무."""
    m = _W["m"]
    order = c.enes_order(m, min_n=2)
    y = np.array(["FIS".index(s) if s in ("F", "I", "S") else 0 for s in m.stop.astype(str)])
    if seed:
        rng = np.random.default_rng(seed)
        for ix in order.values():
            y[ix] = rng.permutation(y[ix])
    return [("real", "실제 Enes", seed) + r for r in replay(y, cp.ACTIONS3, order, _W["E0"], _W["MS"], seed, ["답 100%"])]


def cmd_real(a):
    D = _pool(_real_task, list(range(a.perms + 1)), a.workers)
    os.makedirs(c.RUNS, exist_ok=True)
    D.to_csv(os.path.join(c.RUNS, "real.csv"), index=False)
    L = ["# 실제 Enes (답 100%, 답 10개 이상 = 11번째 울음부터)", "",
         f"| 설정 | 실제 | 귀무 95% (아기 안 라벨 섞기 {a.perms}번) | 소리 켜짐 |", "|---|---:|---:|---:|"]
    for cname in CFG:
        x = D[(D.cfg == cname) & (D.pos >= 10)]
        r1, r2 = _score(x[x.seed == 0])
        nul = [_score(g) for _, g in x[x.seed > 0].groupby("seed")]
        n1, n2 = (np.quantile([v[0] for v in nul], .95), np.quantile([v[1] for v in nul], .95)) if nul else (np.nan, np.nan)
        on = f"{x[x.seed == 0].on.mean():.0%}" if "자가 검증" in cname else "-"
        L.append(f"| {cname} | {r1:.3f} / {r2:.3f} | {n1:.3f} / {n2:.3f} | {on} |")
    _write("REAL_REPORT.md", "\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("long", help="가상 세계에서 쌓인 답 수별 정확도")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--group", type=int, default=11, help="가상 아기 하나에 묶을 Enes 아기 수(11이면 울음 약 275개)")
    p.add_argument("--half", action="store_true", help="부모가 절반만 답하는 경우(답 50%%)도 함께 잰다")
    p = sp.add_parser("real", help="실제 Enes 라벨과 귀무")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--perms", type=int, default=20, help="귀무(아기 안 라벨 섞기) 횟수")
    a = ap.parse_args()
    {"long": cmd_long, "real": cmd_real}[a.cmd](a)


if __name__ == "__main__":
    main()
