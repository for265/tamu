"""Public-chance-sampling Linear CFR on the bucketed abstract game.

Each iteration samples one full board and a pool of M random 5-card hands
(from the 31 non-board cards).  Both players' ranges are vectors over the
pool; pairs of hands that share a card are excluded via a validity matrix,
so card removal inside the sample is exact.  The whole betting tree
(41,490 nodes, built from game.Game) is traversed full-width, layer by layer
(all nodes at one depth in a single numpy op):

  forward : regret matching at every decision node for every hand's bucket,
            reach probabilities for both players
  leaves  : fold      v_p = payoff_p * (VALID @ reach_opp)
            showdown  v_p = contrib  * (W @ reach_opp),  W = VALID * sign(rank_i - rank_j)
            (two GEMMs over all terminals)
  backward: counterfactual values, regret and average-strategy increments,
            summed per (node, bucket) with bincount.

Info set = (betting node, bucket of the acting player's hand on the current
street) -- perfect recall of betting, imperfect recall of cards.  Linear CFR:
iteration t's regret and strategy increments are weighted by t.  Workers keep
local increments and add them to shared-memory tables every FLUSH boards
(under a lock); reads of the regret table are lock-free.

Usage:
  python3 cfr.py --minutes 50 --workers 7            # train
  python3 cfr.py --bench                             # time a few boards
Output: out/cfr_regret.npy, out/cfr_strat.npy, out/cfr_meta.npz (checkpoints)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"          # one core per worker (we use at most 7 workers)
from multiprocessing import Event, Lock, Process, Value, shared_memory

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, load_agent3  # noqa: E402
from tree import build_tree, payoffs  # noqa: E402

M_HANDS = 112
FLUSH = 4


class Layout:
    """Tree arrays plus per-depth layers and table row layout."""

    def __init__(self, nbuckets):
        t = build_tree()
        self.t = t
        self.nb = np.asarray(nbuckets, np.int64)
        kind, depth, street = t["kind"], t["depth"], t["street"].astype(np.int64)
        player = t["player"].astype(np.int64)
        n = len(kind)
        self.n = n
        ch = t["children"].astype(np.int64)
        ch_d = np.where(ch < 0, n, ch)               # illegal -> dummy zero node n
        dec = np.where(kind == 0)[0]
        self.rowbase = np.full(n, -1, np.int64)
        self.layers = []
        r = 0
        for d in range(int(depth.max()) + 1):
            for pl in (0, 1):
                idx = dec[(depth[dec] == d) & (player[dec] == pl)]
                if len(idx) == 0:
                    continue
                k = self.nb[street[idx]]
                lo = r
                self.rowbase[idx] = lo + np.cumsum(k) - k
                r = lo + int(k.sum())
                legal = ch[idx] >= 0
                L = dict(idx=idx, lo=lo, hi=r, p=pl, st=street[idx],
                         base=self.rowbase[idx], lbase=self.rowbase[idx] - lo,
                         legal=legal.astype(np.float32),
                         unif=(legal / legal.sum(1, keepdims=True)).astype(np.float32),
                         ch=ch_d[idx])
                L["fwd"] = []
                for a in range(4):
                    m = legal[:, a]
                    if m.any():
                        L["fwd"].append((a, np.where(m)[0], ch[idx[m], a]))
                self.layers.append(L)
        self.rows = r
        pay = payoffs(t)
        self.fold = np.where(kind == 1)[0]
        self.pay_f = pay[self.fold]                                  # (F, 2)
        self.sd = np.where(kind == 2)[0]
        self.c_sd = t["contrib0"][self.sd].astype(np.float32)        # (S,)


class Trainer:
    def __init__(self, lay: Layout, edges, pre_bucket, R, S, seed):
        self.lay = lay
        self.edges = edges
        self.pre_bucket = pre_bucket
        self.R, self.S = R, S
        self.a3 = load_agent3()
        self.rng = np.random.default_rng(seed)
        m = M_HANDS
        self.REACH = [np.zeros((lay.n + 1, m), np.float32) for _ in range(2)]
        self.VAL = [np.zeros((lay.n + 1, m), np.float32) for _ in range(2)]
        self.dR = np.zeros((lay.rows, 4), np.float32)
        self.dS = np.zeros((lay.rows, 4), np.float32)
        self.timing = np.zeros(4)

    def sample(self):
        rng, m = self.rng, M_HANDS
        perm = rng.permutation(36)
        board, rest = perm[:5], perm[5:]
        keys = rng.random((m, 31))
        hands = np.sort(rest[np.argsort(keys, axis=1)[:, :5]], axis=1)
        a3 = self.a3
        bk = np.empty((4, m), np.int64)
        bk[0] = self.pre_bucket[a3.colex(hands)]
        for s, nb in ((1, 3), (2, 4), (3, 5)):
            e, c = a3.street_features(hands, board[:nb], rng)
            bk[s] = a3.bucketize(s, e, c, self.edges)
        rank = a3.best_rank(hands, board).astype(np.int32)
        masks = (np.int64(1) << hands).sum(1)
        valid = ((masks[:, None] & masks[None, :]) == 0).astype(np.float32)
        w = valid * np.sign(rank[:, None] - rank[None, :]).astype(np.float32)
        return bk, valid, w

    def iterate(self, weight):
        lay, R = self.lay, self.R
        t0 = time.perf_counter()
        bk, valid, w = self.sample()
        t1 = time.perf_counter()
        REACH, VAL = self.REACH, self.VAL
        REACH[0][0] = 1.0
        REACH[1][0] = 1.0
        sigs = []
        for L in lay.layers:
            rows = L["base"][:, None] + bk[L["st"]]                 # (k, M)
            reg = R[rows]                                           # (k, M, 4)
            np.maximum(reg, 0, out=reg)
            reg *= L["legal"][:, None, :]
            ssum = reg.sum(-1, keepdims=True)
            sig = np.where(ssum > 0, reg / np.where(ssum > 0, ssum, 1.0), L["unif"][:, None, :])
            sig = np.ascontiguousarray(sig.transpose(0, 2, 1))     # (k, 4, M)
            sigs.append(sig)
            rp, ro = REACH[L["p"]], REACH[1 - L["p"]]
            r_p = rp[L["idx"]]
            r_o = ro[L["idx"]]
            for a, sel, ch in L["fwd"]:
                rp[ch] = r_p[sel] * sig[sel, a]
                ro[ch] = r_o[sel]
        t2 = time.perf_counter()
        F, SD = lay.fold, lay.sd
        VAL[0][F] = lay.pay_f[:, 0:1] * (REACH[1][F] @ valid)
        VAL[1][F] = lay.pay_f[:, 1:2] * (REACH[0][F] @ valid)
        wt = np.ascontiguousarray(w.T)
        VAL[0][SD] = lay.c_sd[:, None] * (REACH[1][SD] @ wt)
        VAL[1][SD] = lay.c_sd[:, None] * (REACH[0][SD] @ wt)
        t3 = time.perf_counter()
        ar4 = np.arange(4)[None, :, None]
        for L, sig in zip(reversed(lay.layers), reversed(sigs)):
            idx, p = L["idx"], L["p"]
            vp_arr, vo_arr = VAL[p], VAL[1 - p]
            cvp = vp_arr[L["ch"]]                                    # (k, 4, M)
            vp = (sig * cvp).sum(1)                                  # (k, M)
            vp_arr[idx] = vp
            vo_arr[idx] = vo_arr[L["ch"]].sum(1)
            cvp -= vp[:, None, :]
            cvp *= L["legal"][:, :, None]
            sig *= REACH[p][idx][:, None, :]
            flat = ((L["lbase"][:, None] + bk[L["st"]])[:, None, :] * 4 + ar4).ravel()
            size = (L["hi"] - L["lo"]) * 4
            self.dR[L["lo"]:L["hi"]] += (weight * np.bincount(flat, weights=cvp.ravel(), minlength=size)
                                         ).reshape(-1, 4).astype(np.float32)
            self.dS[L["lo"]:L["hi"]] += (weight * np.bincount(flat, weights=sig.ravel(), minlength=size)
                                         ).reshape(-1, 4).astype(np.float32)
        t4 = time.perf_counter()
        self.timing += (t1 - t0, t2 - t1, t3 - t2, t4 - t3)

    def flush(self, lock):
        with lock:
            self.R += self.dR
            self.S += self.dS
        self.dR[:] = 0
        self.dS[:] = 0


def _load_inputs():
    a3 = load_agent3()  # noqa: F841  (ensures import works before forking)
    e = np.load(os.path.join(OUT, "edges.npz"))
    edges = {k: e[k] for k in ("e1", "c1", "e2", "c2", "e3")}
    return edges, e["pre_bucket"], e["nbuckets"]


def _worker(wid, shm_names, rows, counter, lock, stop, seed):
    edges, pre_bucket, nbuckets = _load_inputs()
    lay = Layout(nbuckets)
    shr = shared_memory.SharedMemory(name=shm_names[0])
    shs = shared_memory.SharedMemory(name=shm_names[1])
    R = np.ndarray((rows, 4), np.float32, buffer=shr.buf)
    S = np.ndarray((rows, 4), np.float32, buffer=shs.buf)
    tr = Trainer(lay, edges, pre_bucket, R, S, seed)
    k = 0
    while not stop.is_set():
        with counter.get_lock():
            counter.value += 1
            t = counter.value
        tr.iterate(float(t))
        k += 1
        if k % FLUSH == 0:
            tr.flush(lock)
    tr.flush(lock)
    del R, S
    shr.close()
    shs.close()


def save(R, S, nbuckets, lay, boards, tag=""):
    np.save(os.path.join(OUT, "cfr_regret%s.npy" % tag), R)
    np.save(os.path.join(OUT, "cfr_strat%s.npy" % tag), S)
    np.savez(os.path.join(OUT, "cfr_meta%s.npz" % tag), rowbase=lay.rowbase, nbuckets=nbuckets,
             boards=boards, m_hands=M_HANDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--checkpoint-min", type=float, default=10.0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    edges, pre_bucket, nbuckets = _load_inputs()
    lay = Layout(nbuckets)
    print("rows", lay.rows, "table entries", lay.rows * 4, "layers", len(lay.layers), flush=True)
    if args.bench:
        R = np.zeros((lay.rows, 4), np.float32)
        S = np.zeros((lay.rows, 4), np.float32)
        tr = Trainer(lay, edges, pre_bucket, R, S, 0)
        t0 = time.perf_counter()
        for i in range(6):
            tr.iterate(float(i + 1))
            tr.flush(Lock())
            v0, v1 = float(tr.VAL[0][0].sum()), float(tr.VAL[1][0].sum())
            assert abs(v0 + v1) <= 1e-3 * (abs(v0) + abs(v1)) + 1e-2, (v0, v1)   # zero-sum check
            print("  root value sums %.2f %.2f" % (v0, v1))
        dt = time.perf_counter() - t0
        print("per board %.3fs  (features %.3f forward %.3f leaves %.3f backward %.3f)" %
              ((dt / 6,) + tuple(tr.timing / 6)))
        return
    shr = shared_memory.SharedMemory(create=True, size=lay.rows * 16)
    shs = shared_memory.SharedMemory(create=True, size=lay.rows * 16)
    R = np.ndarray((lay.rows, 4), np.float32, buffer=shr.buf)
    S = np.ndarray((lay.rows, 4), np.float32, buffer=shs.buf)
    R[:] = 0
    S[:] = 0
    counter = Value("q", 0)
    if args.resume:
        R[:] = np.load(os.path.join(OUT, "cfr_regret.npy"))
        S[:] = np.load(os.path.join(OUT, "cfr_strat.npy"))
        counter.value = int(np.load(os.path.join(OUT, "cfr_meta.npz"))["boards"])
    lock = Lock()
    stop = Event()
    procs = [Process(target=_worker, args=(w, (shr.name, shs.name), lay.rows, counter, lock, stop,
                                           12345 + 7919 * w + counter.value))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    t0 = time.time()
    last_ck = t0
    try:
        while time.time() - t0 < args.minutes * 60:
            time.sleep(10)
            el = time.time() - t0
            print("t=%5.0fs boards=%d (%.1f/s)" % (el, counter.value, counter.value / max(el, 1)), flush=True)
            if time.time() - last_ck > args.checkpoint_min * 60:
                with lock:
                    save(R.copy(), S.copy(), nbuckets, lay, counter.value)
                last_ck = time.time()
                print("checkpoint saved", flush=True)
    finally:
        stop.set()
        for p in procs:
            p.join()
        save(R.copy(), S.copy(), nbuckets, lay, counter.value)
        print("final save: boards", counter.value, flush=True)
        del R, S
        shr.close()
        shr.unlink()
        shs.close()
        shs.unlink()


if __name__ == "__main__":
    main()
