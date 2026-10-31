"""1단계: 울음 목록을 만들고, 울음마다 음향 요약 6개와 YAMNet 평균 임베딩을 계산한다.

    python sim/prep.py --workers 8          # → CRY_DATA/sim/items.csv, prep.npz
    python sim/prep.py --limit 20           # 20개만 계산해 보고 저장하지 않는다
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

import common as c  # noqa: E402

_W = {}


def _init():
    _W["yam"] = c.Yam()


def _one(job):
    k, paths = job
    x = c.load_clip(paths)
    emb, g = _W["yam"].clip(x)
    return k, np.asarray(c.acoustic(x), np.float32), emb.astype(np.float32), np.float32(g)


def main():
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만 계산해 보고 저장하지 않음")
    a = ap.parse_args()
    m = c.items()
    if a.limit:
        m = m.head(a.limit)
    N = len(m)
    A, E, G = np.zeros((N, 6), np.float32), np.zeros((N, 1024), np.float32), np.zeros(N, np.float32)
    t0 = time.time()
    with Pool(a.workers, initializer=_init) as pool:
        for n, (k, Ak, Ek, Gk) in enumerate(pool.imap_unordered(_one, list(enumerate(m.paths)), chunksize=2)):
            A[k], E[k], G[k] = Ak, Ek, Gk
            if (n + 1) % 100 == 0:
                print(f"  {n + 1}/{N} {time.time() - t0:.0f}s", flush=True)
    if a.limit:
        print(f"시험 {N}개 {time.time() - t0:.1f}s (저장 안 함)\n음향 요약:\n{A.round(2)}\n울음 창 비율: {G.round(2).tolist()}")
        return
    os.makedirs(c.OUT, exist_ok=True)
    m.drop(columns="paths").to_csv(c.ITEMS, index=False)
    np.savez(c.PREP, A=A, E=E, G=G)
    print(f"저장 {c.PREP}: 울음 {N}개 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
