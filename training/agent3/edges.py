"""Bucket boundaries from the feature distribution of random deals.

preflop : K_PRE quantile bins of all-in equity over all 376,992 hands
flop/turn: E_POST EHS bins (finer near the top), and inside each EHS bin
           C_POST equal-mass CUR bins (separates draws from made hands)
river   : K_RIVER HS bins (finer near the top)
Features come from agent3.street_features, i.e. the exact estimator used at
play time.  Output: out/edges.npz
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C_POST, E_POST, K_PRE, K_RIVER, OUT, load_agent3, quantile_levels  # noqa: E402

N_BOARDS = 2400        # random boards per street
HANDS_PER_BOARD = 32
WORKERS = 7
GAMMA_PRE, GAMMA_POST, GAMMA_RIVER = 1.2, 1.5, 1.6


def _feat_job(args):
    street, seed, nboards = args
    a3 = load_agent3()
    rng = np.random.default_rng(seed)
    nb = (0, 3, 4, 5)[street]
    E, C = [], []
    for _ in range(nboards):
        perm = rng.permutation(36)
        board = perm[:nb]
        rest = perm[nb:]
        holes = np.array([rng.choice(rest, 5, replace=False) for _ in range(HANDS_PER_BOARD)])
        e, c = a3.street_features(holes, board, rng)
        E.append(e)
        C.append(c)
    return np.concatenate(E), np.concatenate(C)


def main():
    t0 = time.time()
    pre = np.load(os.path.join(OUT, "preflop.npz"))
    ehs_pre = pre["ehs"]
    pre_edges = np.unique(np.quantile(ehs_pre, quantile_levels(K_PRE, GAMMA_PRE)))
    pre_bucket = np.searchsorted(pre_edges, ehs_pre, side="right").astype(np.uint8)
    out = {"pre_edges": pre_edges, "pre_bucket": pre_bucket}
    jobs = [(s, 100 * s + w, N_BOARDS // WORKERS) for s in (1, 2, 3) for w in range(WORKERS)]
    with Pool(WORKERS) as pool:
        res = pool.map(_feat_job, jobs)
    for s in (1, 2, 3):
        parts = [r for j, r in zip(jobs, res) if j[0] == s]
        e = np.concatenate([p[0] for p in parts])
        c = np.concatenate([p[1] for p in parts])
        if s == 3:
            out["e3"] = np.unique(np.quantile(e, quantile_levels(K_RIVER, GAMMA_RIVER)))
            print("river edges", np.round(out["e3"], 3), "P(hs==1) %.3f" % np.mean(e >= 1.0))
            continue
        ee = np.quantile(e, quantile_levels(E_POST, GAMMA_POST))
        eb = np.searchsorted(ee, e, side="right")
        ce = np.zeros((E_POST, C_POST - 1))
        for b in range(E_POST):
            sel = c[eb == b]
            ce[b] = np.quantile(sel, np.arange(1, C_POST) / C_POST) if len(sel) else 0.5
        out["e%d" % s] = ee
        out["c%d" % s] = ce
        print("street", s, "ehs edges", np.round(ee, 3))
        print("   corr(ehs,cur) %.3f" % np.corrcoef(e, c)[0, 1])
    out["nbuckets"] = np.array([len(pre_edges) + 1, E_POST * C_POST, E_POST * C_POST,
                                len(out["e3"]) + 1], np.int64)
    print("buckets per street", out["nbuckets"])
    np.savez(os.path.join(OUT, "edges.npz"), **out)
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
