"""Preflop table: all-in equity vs a random hand for every 5-card holding.

Hands are grouped into suit-isomorphism classes (the multiset of per-suit rank
masks); one representative per class is evaluated by Monte Carlo with
S_PRE samples (opponent hand + 5 board cards), then mapped back to every hand
by colex index.  Output: out/preflop.npz with
    ehs   float32 (376992,)  equity per colex index
    cls   int32   (376992,)  class id per colex index
"""
from __future__ import annotations

import os
import sys
import time
from itertools import combinations
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, load_agent3  # noqa: E402

S_PRE = 6000
WORKERS = 7


def all_hands():
    return np.array(list(combinations(range(36), 5)), np.int64)


def canonical_keys(h):
    masks = np.zeros((len(h), 4), np.int64)
    for k in range(5):
        np.add.at(masks, (np.arange(len(h)), h[:, k] & 3), 1 << (h[:, k] >> 2))
    masks = -np.sort(-masks, axis=1)
    return (masks[:, 0] << 27) | (masks[:, 1] << 18) | (masks[:, 2] << 9) | masks[:, 3]


def _equity_chunk(args):
    reps, seed = args
    a3 = load_agent3()
    rng = np.random.default_rng(seed)
    out = np.empty(len(reps))
    for i0 in range(0, len(reps), 16):
        holes = reps[i0:i0 + 16]
        n = len(holes)
        m = np.ones((n, 36), bool)
        m[np.arange(n)[:, None], holes] = False
        avail = np.nonzero(m)[1].reshape(n, 31)
        acc = np.zeros(n)
        per = 1000
        for _ in range(S_PRE // per):
            smp = a3.sample_cards(avail, per, 10, rng)
            tx = a3.triple_codes(smp[..., 5:])
            own = a3.combine(a3.pair_codes(holes)[:, None, :], tx)
            oth = a3.combine(a3.pair_codes(smp[..., :5]), tx)
            acc += ((own > oth).sum(1) + 0.5 * (own == oth).sum(1))
        out[i0:i0 + n] = acc / S_PRE
    return out


def main():
    t0 = time.time()
    a3 = load_agent3()
    h = all_hands()
    ci = a3.colex(h)
    assert len(np.unique(ci)) == len(h) and ci.max() == len(h) - 1
    keys = canonical_keys(h)
    uk, first, inv = np.unique(keys, return_index=True, return_inverse=True)
    reps = h[first]
    print("hands", len(h), "classes", len(uk), flush=True)
    chunks = np.array_split(np.arange(len(reps)), WORKERS * 8)
    with Pool(WORKERS) as pool:
        res = pool.map(_equity_chunk, [(reps[c], 1000 + i) for i, c in enumerate(chunks)])
    ehs_cls = np.concatenate(res)
    ehs = np.empty(len(h), np.float32)
    cls = np.empty(len(h), np.int32)
    ehs[ci] = ehs_cls[inv]
    cls[ci] = inv
    np.savez(os.path.join(OUT, "preflop.npz"), ehs=ehs, cls=cls)
    print("done in %.1fs; ehs range %.3f..%.3f mean %.4f" % (time.time() - t0, ehs.min(), ehs.max(), ehs.mean()))


if __name__ == "__main__":
    main()
