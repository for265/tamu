"""Build agent4_tab.npy (5-card score table) and agent4_pre.npy (preflop strength
percentile table, uint8) in the repo root.  Usage: python3 training/agent4/build_tables.py [samples] [procs]"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import REPO, load_agent4  # noqa: E402

A = None


def _init():
    global A
    A = load_agent4()


def _chunk(args):
    lo, hi, S, seed = args
    rng = np.random.default_rng(seed)
    A._RNG = rng
    from itertools import combinations
    combos = ALL[lo:hi]
    out = np.zeros(len(combos))
    for s in range(0, len(combos), 400):
        H = combos[s:s + 400]
        n = len(H)
        rows = np.repeat(np.arange(n), S)
        hb = np.zeros((n * S, 36), bool)
        hb[np.arange(n * S)[:, None], H[rows]] = True
        keys = rng.random((n * S, 36))
        keys[hb] = 2.0
        part = np.argpartition(keys, 9, axis=1)[:, :10]
        opp = np.sort(part[:, :5], axis=1)
        bd = np.sort(part[:, 5:], axis=1)
        tc = A._tcodes(bd)
        me = A._evalPair(A._pcodes(H)[rows], tc)
        th = A._evalPair(A._pcodes(opp), tc)
        eq = ((me > th) + 0.5 * (me == th)).reshape(n, S).mean(1)
        out[s:s + n] = eq
    return out


if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    A = load_agent4()
    t = time.time()
    tab = A._build_score_table()
    np.save(os.path.join(REPO, "agent4_tab.npy"), tab)
    print("score table", tab.shape, "built in %.1fs" % (time.time() - t))
    from itertools import combinations
    ALL = np.array(list(combinations(range(36), 5)), dtype=np.int64)
    N = len(ALL)
    chunks = [(i, min(i + 4000, N), S, 1000 + i) for i in range(0, N, 4000)]
    t = time.time()
    with Pool(procs, initializer=_init) as p:
        res = p.map(_chunk, chunks)
    eq = np.concatenate(res)
    order = np.argsort(eq, kind="stable")
    rank = np.empty(N)
    # mid-rank percentile (ties share rank)
    srt = np.sort(eq)
    q = (np.searchsorted(srt, eq, "left") + np.searchsorted(srt, eq, "right")) / (2.0 * N)
    pre = np.zeros(N, np.uint8)
    pre[A._hash5(ALL)] = np.round(q * 255).astype(np.uint8)
    np.save(os.path.join(REPO, "agent4_pre.npy"), pre)
    print("pre table done in %.1fs; eq range %.3f-%.3f" % (time.time() - t, eq.min(), eq.max()))
