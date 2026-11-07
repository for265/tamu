
"""
agent4.py - fast, range-aware exploiter for heads-up Short Deck PLO5.

Design (see the comments in the code for details):
  1. Vectorised numpy equity engine (perfect-hash score table, agent4_tab.npy).
  2. Opponent range inferred by weighting sampled hands with the likelihood of
     the opponent's observed actions, under a learned threshold-style model.
  3. EV of every legal action (with fold equity) and a softmax mix.

Run it:
    python agent4.py                 # listens on port 5008
    PORT=5010 python agent4.py       # or pick a port
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "bongo"
AGENT_NAME = "cloud4"

# ===== YOUR CODE HERE =====

import math
import random
import time
from itertools import combinations

import numpy as np

from evaluator import get_table

_DIR = os.path.dirname(os.path.abspath(__file__))
TIME_BANK = 8.0
TARGET = 3.7              # seconds of the bank we plan to use per half
HANDS_PER_HALF = 1000
_RNG = np.random.default_rng()
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
_NCARD = 36
_NHASH5 = 376992
_NTRIP = 7140

# ---------- 0. Perfect-hash score table ----------------------------------------
# A "pair" (2 hole cards a<b) has index C(b,2)+a in 0..629, a "triple" of board
# cards a<b<c has index C(c,3)+C(b,2)+a in 0..7139, and _TAB[pair, triple] is
# the dense rank of that five-card hand (0 if the cards overlap).

_B = np.array([[math.comb(n, k) for n in range(_NCARD)] for k in range(1, 6)], dtype=np.int64)


def _hash5(S):
    """Index in 0..376991 of sorted-ascending 5-card sets S[..., 5]."""
    return _B[0][S[..., 0]] + _B[1][S[..., 1]] + _B[2][S[..., 2]] + _B[3][S[..., 3]] + _B[4][S[..., 4]]


_CP = np.zeros((_NCARD, _NCARD), np.int32)
_CT = np.zeros((_NCARD, _NCARD, _NCARD), np.int32)
for _a in range(_NCARD):
    for _b in range(_a + 1, _NCARD):
        _CP[_a, _b] = math.comb(_b, 2) + _a
        for _c in range(_b + 1, _NCARD):
            _CT[_a, _b, _c] = math.comb(_c, 3) + math.comb(_b, 2) + _a
_PA, _PB = (np.array(x) for x in zip(*combinations(range(5), 2)))
_TRI = {n: tuple(np.array(x) for x in zip(*combinations(range(n), 3))) for n in (3, 4, 5)}


def _build_score_table():
    """Build the (630, 7140) uint16 table from evaluator.get_table()."""
    tab = get_table()
    combos = np.array(list(combinations(range(_NCARD), 5)), dtype=np.int64)
    masks = np.left_shift(np.int64(1), combos).sum(1)
    sc = np.fromiter((tab[m] for m in masks.tolist()), dtype=np.int64, count=len(masks))
    _, inv = np.unique(sc, return_inverse=True)
    sc5 = np.zeros(_NHASH5, np.uint16)
    sc5[_hash5(combos)] = inv + 1
    pairs = np.array(list(combinations(range(_NCARD), 2)), dtype=np.int8)
    trips = np.array(list(combinations(range(_NCARD), 3)), dtype=np.int8)
    pidx = _CP[pairs[:, 0], pairs[:, 1]]
    tidx = _CT[trips[:, 0], trips[:, 1], trips[:, 2]]
    out = np.zeros((630, _NTRIP), np.uint16)
    for s in range(0, 630, 30):
        pp = pairs[s:s + 30]
        m = np.concatenate([np.broadcast_to(pp[:, None, :], (len(pp), _NTRIP, 2)),
                            np.broadcast_to(trips[None, :, :], (len(pp), _NTRIP, 3))], axis=2)
        m = np.sort(m, axis=2)
        dup = (m[..., 1:] == m[..., :-1]).any(2)
        h = np.where(dup, 0, _hash5(m.astype(np.int64)))
        val = np.where(dup, 0, sc5[h])
        out[pidx[s:s + 30, None], tidx[None, :]] = val
    return out


def _load_tab():
    try:
        t = np.load(os.path.join(_DIR, "agent4_tab.npy"))
        if t.shape == (630, _NTRIP) and t.dtype == np.uint16:
            return t
    except Exception:
        pass
    return _build_score_table()


def _load_pre():
    try:
        t = np.load(os.path.join(_DIR, "agent4_pre.npy"))
        if t.shape == (_NHASH5,):
            return t
    except Exception:
        pass
    return np.full(_NHASH5, 128, np.uint8)       # degrade gracefully: no preflop info


_TAB = _load_tab()
_TABF = np.ascontiguousarray(_TAB).ravel()
_PRE = _load_pre()


def _pcodes(H):
    """Pair codes (N, 10) of sorted 5-card hands H (N, 5)."""
    return _CP[H[:, _PA], H[:, _PB]]


def _tcodes(Bd):
    """Triple codes (M, K) of sorted boards Bd (M, n), n in 3..5."""
    ia, ib, ic = _TRI[Bd.shape[1]]
    return _CT[Bd[:, ia], Bd[:, ib], Bd[:, ic]]


def _eval1(PC, TC):
    """Best score of each hand row against ONE board (TC shape (1, K)) -> (N,)."""
    idx = PC[:, :, None] * _NTRIP + TC[0][None, None, :]
    return _TABF.take(idx).reshape(len(PC), -1).max(1)


def _evalN(PC, TC):
    """Scores of N hands on R shared boards: PC (N,10), TC (R,K) -> (N,R)."""
    idx = PC[:, None, :, None] * _NTRIP + TC[None, :, None, :]
    return _TABF.take(idx).reshape(len(PC), len(TC), -1).max(2)


def _evalPair(PC, TC):
    """Row-wise: PC (M,10), TC (M,K) -> (M,)."""
    idx = PC[:, :, None] * _NTRIP + TC[:, None, :]
    return _TABF.take(idx).reshape(len(PC), -1).max(1)


def _rand_subsets(avail, n, k):
    """n random k-subsets (sorted rows) of the array avail (rejection sampling)."""
    m = len(avail)
    out = np.empty((0, k), np.int64)
    while len(out) < n:
        X = _RNG.integers(0, m, (int((n - len(out)) * 1.7) + 8, k))
        X.sort(axis=1)
        X = X[(X[:, 1:] != X[:, :-1]).all(1)]
        out = np.concatenate([out, avail[X]]) if len(out) else avail[X]
    return out[:n]


def _pct(s):
    """Mid-rank percentile of each value among the sample s."""
    srt = np.sort(s)
    return (np.searchsorted(srt, s, "left") + np.searchsorted(srt, s, "right")) / (2.0 * len(s))


# ---------- 1. Opponent model ---------------------------------------------------
# Context = (street, depth class, facing a bet?, size class of the bet faced).
# Action 0 = fold, 1 = check/call, 2 = raise.  Threshold model: with weight
# lam the opponent acts by hand-strength percentile q (weakest fold, middle
# call, top raise, widths = the marginal frequencies); with weight 1-lam it
# acts independently of its cards.  Marginals come from ALL observed actions
# (folds included), so showdown censoring only affects lam, which is shrunk.

_DEF = {
    (True, 0): (0.15, 0.55, 0.30),
    (True, 1): (0.30, 0.55, 0.15),
    (True, 2): (0.40, 0.50, 0.10),
    (True, 3): (0.50, 0.50, 0.00),
    (False, 0): (0.00, 0.65, 0.35),
}
for _d in (1, 2, 3):
    _DEF[(False, _d)] = _DEF[(False, 0)]
_K1, _K2 = 10.0, 16.0
_W = 0.10
_GRID = np.linspace(0.0, 0.9, 10)
_LPRIOR = np.log((_GRID + 0.15) * (1.15 - _GRID))


def _sc(to_call, pot):
    return "h" if to_call < 0.8 * max(pot - to_call, 1) else "p"


def _probs(mem, street, d, facing, sc):
    cnt = mem["cnt"]
    dflt = np.array(_DEF[(facing, d)])
    n = cnt.get((street, d, facing))
    par = dflt if n is None else (n + _K1 * dflt) / (n.sum() + _K1)
    if sc:
        nf = cnt.get((street, d, facing, sc))
        if nf is not None:
            return (nf + _K2 * par) / (nf.sum() + _K2)
    return par


def _count(mem, street, d, facing, sc, a):
    cnt = mem["cnt"]
    for k in ((street, d, facing), (street, d, facing, sc)) if sc else ((street, d, facing),):
        v = cnt.get(k)
        if v is None:
            v = cnt[k] = np.zeros(3)
        v[a] += 1.0


def _lam(mem, street):
    return mem["lam"][0 if street == 0 else 1]


def _react(q, p3, lam, can_raise):
    """Opp P(fold), P(call), P(raise) for strength percentiles q."""
    f, c, r = float(p3[0]), float(p3[1]), float(p3[2])
    if not can_raise:
        c += r
        r = 0.0
    fh = np.clip((f - q) / _W + 0.5, 0.0, 1.0) if f > 0 else 0.0
    rh = np.clip((q - (1.0 - r)) / _W + 0.5, 0.0, 1.0) if r > 0 else 0.0
    pf = lam * fh + (1.0 - lam) * f
    pr = lam * rh + (1.0 - lam) * r
    pc = np.maximum(1.0 - pf - pr, 0.01)
    tot = pf + pc + pr
    return pf / tot, pc / tot, pr / tot


def _refit_lam(mem):
    """Posterior-mean lam per group (preflop / postflop) from showdown records."""
    sd = mem["sd"]
    if not sd:
        return
    cache = {}
    rows = []
    for (grp, st, d, fa, sc, a, q, canr) in sd:
        key = (st, d, fa, sc)
        p3 = cache.get(key)
        if p3 is None:
            p3 = cache[key] = _probs(mem, st, d, fa, sc)
        f, c, r = p3
        if not canr:
            c, r = c + r, 0.0
        rows.append((grp, f, r, a, q))
    arr = np.array(rows, dtype=float)
    for g in (0, 1):
        m = arr[:, 0] == g
        if not m.any():
            continue
        f, r, a, q = arr[m, 1], arr[m, 2], arr[m, 3], arr[m, 4]
        ll = np.zeros(len(_GRID))
        for i, lam in enumerate(_GRID):
            fh = np.clip((f - q) / _W + 0.5, 0.0, 1.0) * (f > 0)
            rh = np.clip((q - (1.0 - r)) / _W + 0.5, 0.0, 1.0) * (r > 0)
            pf = lam * fh + (1 - lam) * f
            pr = lam * rh + (1 - lam) * r
            pc = np.maximum(1.0 - pf - pr, 0.01)
            p = np.where(a == 0, pf, np.where(a == 2, pr, pc)) / (pf + pr + pc)
            ll[i] = np.log(np.maximum(p, 1e-3)).sum()
        post = ll + _LPRIOR
        post = np.exp(post - post.max())
        mem["lam"][g] = float((post * _GRID).sum() / post.sum())


# ---------- 2. Replaying the hand history ----------------------------------------

def _aidx(act):
    return 0 if act == Action.FOLD else (1 if act == Action.CHECK_CALL else 2)


def _sync(obs, mem):
    """Replay obs.action_history; count + store opponent actions not seen yet."""
    if mem.get("hand") != obs.hand_index:
        mem["hand"] = obs.hand_index
        mem["nproc"] = 0
        mem["recs"] = []
        mem["hole"] = tuple(obs.hole_cards)
        mem["pending"] = None
        mem["hands_seen"] += 1
    hist = obs.action_history
    my_c, opp_c = (1, 2) if obs.is_button else (2, 1)
    my_b, opp_b = my_c, opp_c
    street = 0
    raises = 0
    n0 = mem["nproc"]
    for i, e in enumerate(hist):
        if int(e.street) != street:
            street = int(e.street)
            my_b = opp_b = 0
            raises = 0
        own, oth = (my_b, opp_b) if e.is_me else (opp_b, my_b)
        if not e.is_me and i >= n0:
            to_call = max(0, oth - own)
            fa = to_call > 0
            sc = _sc(to_call, my_c + opp_c) if fa else ""
            d = min(raises, 3)
            a = _aidx(e.action)
            _count(mem, street, d, fa, sc, a)
            mem["recs"].append((street, d, fa, sc, a, raises < 4))
        if e.is_me:
            my_b = e.street_total
            my_c += e.amount
        else:
            opp_b = e.street_total
            opp_c += e.amount
        if e.action in _RAISES:
            raises += 1
    mem["nproc"] = len(hist)


# ---------- 3. Range inference + equity + EV ---------------------------------------

_BASE = {0: (180, 6), 1: (170, 8), 2: (170, 8), 3: (0, 0)}      # resample size, runouts


def _decide(obs, mem, scale):
    legal = obs.legal_actions
    st = int(obs.street)
    board = tuple(obs.board)
    L = len(board)
    mine = np.sort(np.array(obs.hole_cards, dtype=np.int64))
    blocked = np.zeros(_NCARD, bool)
    blocked[list(obs.hole_cards)] = True
    if L:
        blocked[list(board)] = True
    avail = np.nonzero(~blocked)[0]
    n0 = max(90, int(520 * scale))
    H = _rand_subsets(avail, n0, 5)
    H.sort(axis=1)
    pc = _pcodes(H)
    qs = {}
    scores = {}

    def q_for(s):
        q = qs.get(s)
        if q is None:
            if s == 0:
                q = _PRE[_hash5(H)] / 255.0
            else:
                Bd = np.sort(np.array(board[:(2 + s)], dtype=np.int64))[None, :]
                sc_ = _eval1(pc, _tcodes(Bd))
                scores[s] = sc_
                q = _pct(sc_)
            qs[s] = q
        return q

    # --- likelihood of the opponent's observed actions this hand
    logw = np.zeros(n0)
    for (s, d, fa, sc, a, canr) in mem["recs"]:
        p3 = _probs(mem, s, d, fa, sc)
        pf, pc_, pr = _react(q_for(s), p3, _lam(mem, s), canr)
        p = (pf, pc_, pr)[a]
        logw += np.log(np.maximum(p, 0.02))
    qcur_pool = q_for(st)
    w_pool = np.exp(logw - logw.max())

    mypc = _pcodes(mine[None, :])
    if st == 3:
        Bd = np.sort(np.array(board, dtype=np.int64))[None, :]
        tcb = _tcodes(Bd)
        ms = int(_eval1(mypc, tcb)[0])
        if 3 not in scores:
            q_for(3)
        sc_ = scores[3]
        e = (ms > sc_) * 1.0 + (ms == sc_) * 0.5
        wt = w_pool
        qcur = qcur_pool
    else:
        n1, R = _BASE[st]
        n1 = max(40, int(n1 * scale))
        R = max(2, int(round(R * min(scale, 1.5))))
        if np.ptp(logw) < 1e-9:
            sel = np.arange(min(n1, n0))
        else:
            cum = np.cumsum(w_pool)
            pos = (np.arange(n1) + _RNG.random()) / n1 * cum[-1]
            sel = np.minimum(np.searchsorted(cum, pos), n0 - 1)
        Hs = H[sel]
        pcs = pc[sel]
        qcur = qcur_pool[sel]
        wt = np.ones(len(sel))
        n = len(sel)
        if st == 0:
            # per-hand independent runouts
            hm0 = np.zeros((n, _NCARD), bool)
            hm0[np.arange(n)[:, None], Hs] = True
            cand = _rand_subsets(avail, int(n * R * 2.7), 5)
            owner = np.tile(np.arange(n), int(n * R * 2.7) // n + 1)[:len(cand)]
            ok = ~hm0[owner[:, None], cand].any(1)
            cand, owner = cand[ok][:n * R], owner[ok][:n * R]
            tcr = _tcodes(cand)
            theirs = _evalPair(pcs[owner], tcr)
            ms_ = _evalPair(np.broadcast_to(mypc, (len(cand), 10)), tcr)
            O = (ms_ > theirs) * 1.0 + (ms_ == theirs) * 0.5
            cntv = np.bincount(owner, minlength=n)
            e = np.where(cntv > 0, np.bincount(owner, weights=O, minlength=n) / np.maximum(cntv, 1), 0.5)
            wt = (cntv > 0).astype(float)
        else:
            need = 5 - L
            run = _rand_subsets(avail, R, need)
            full = np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64), (R, L)), run], axis=1)
            full.sort(axis=1)
            tcr = _tcodes(full)
            ms_ = _evalN(mypc, tcr)[0]                       # (R,)
            theirs = _evalN(pcs, tcr)                        # (n, R)
            hm = np.zeros((n, _NCARD), bool)
            hm[np.arange(n)[:, None], Hs] = True
            valid = ~hm[:, run].any(2)                       # (n, R)
            O = (ms_[None, :] > theirs) * 1.0 + (ms_[None, :] == theirs) * 0.5
            vc = valid.sum(1)
            e = np.where(vc > 0, (O * valid).sum(1) / np.maximum(vc, 1), 0.5)
            wt = (vc > 0).astype(float)
    W = wt.sum()
    if W <= 0:
        wt = np.ones(len(e))
        W = wt.sum()
    eq = float((wt * e).sum() / W)

    # --- EV of each legal action (chips relative to folding now)
    P = obs.pot
    c = obs.to_call
    V = {}
    if Action.FOLD in legal:
        V[Action.FOLD] = 0.0
    V[Action.CHECK_CALL] = eq * (P + c) - c
    info = {"eq": eq}
    for A in _RAISES:
        if A not in legal:
            continue
        T = obs.raise_to[A]
        r = T - obs.my_bet
        c_opp = T - obs.opp_bet
        add = min(c_opp, obs.opp_stack)
        pot_call = P + r + add
        d_after = min(obs.raises_this_street + 1, 3)
        canr = (obs.raises_this_street + 1 < 4 and obs.my_stack - r > 0 and obs.opp_stack > c_opp)
        p3 = _probs(mem, st, d_after, True, _sc(c_opp, P + r))
        pf, pcl, pr = _react(qcur, p3, _lam(mem, st), canr)
        Wf = float((wt * pf).sum() / W)
        Wcall = wt * pcl
        v = Wf * P
        v += float((Wcall * (e * pot_call - r)).sum() / W)
        Wr = float((wt * pr).sum() / W)
        if Wr > 1e-6:
            err = float((wt * pr * e).sum() / (wt * pr).sum())
            ecall = min(0.75 * pot_call, obs.my_stack - r)
            pot2 = P + r + add + 2 * ecall
            v += Wr * max(-r, err * pot2 - r - ecall)
        V[A] = v
        info[A] = (Wf, v)
    mem["dbg"] = (V, info)
    return V


def _pick(V, tau):
    best = max(V.values())
    acts = list(V)
    ws = [math.exp(max((V[a] - best) / tau, -30.0)) for a in acts]
    x = random.random() * sum(ws)
    for a, w in zip(acts, ws):
        x -= w
        if x <= 0:
            return a
    return acts[-1]


# ---------- 4. Entry points ----------------------------------------------------------

def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["cnt"] = {}
    memory["lam"] = [0.45, 0.45]
    memory["sd"] = []
    memory["hand"] = None
    memory["nproc"] = 0
    memory["recs"] = []
    memory["pending"] = None
    memory["hands_seen"] = 0
    memory["decisions"] = 0
    memory["time_used"] = 0.0
    memory["cost"] = {0: 0.0035, 1: 0.0035, 2: 0.003, 3: 0.002}
    memory["hands_total"] = getattr(config, "num_hands", HANDS_PER_HALF) or HANDS_PER_HALF
    memory["bank"] = getattr(config, "time_bank_seconds", TIME_BANK) or TIME_BANK


def _cheap(obs):
    if obs.to_call == 0:
        return Action.CHECK_CALL
    return Action.CHECK_CALL if obs.to_call <= 0.3 * obs.pot else Action.FOLD


def _fallback(obs):
    return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    t0 = time.perf_counter()
    try:
        _sync(obs, memory)
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        bank = memory["bank"]
        if memory["time_used"] > 0.82 * bank:          # emergency: nearly out of time, play cheap
            return _cheap(obs)
        hands_left = max(memory["hands_total"] - obs.hand_index, 1)
        dph = (memory["decisions"] + 3.0) / (memory["hands_seen"] + 1.0)
        per = (bank * (TARGET / 8.0) - memory["time_used"]) / (hands_left * max(dph, 1.5))
        st = int(obs.street)
        scale = max(0.1, min(1.7, per / memory["cost"][st]))
        V = _decide(obs, memory, scale)
        act = _pick(V, TAU)
        _set_pending(obs, memory, act)
        memory["decisions"] += 1
        dt = time.perf_counter() - t0
        memory["cost"][st] = 0.9 * memory["cost"][st] + 0.1 * dt / scale
        return act
    except Exception:
        memory["exc"] = memory.get("exc", 0) + 1
        memory["exc_tb"] = traceback.format_exc()
        return _fallback(obs)
    finally:
        memory["time_used"] += time.perf_counter() - t0


TAU = 0.4              # softmax temperature over action EVs (chips)


def _set_pending(obs, mem, act):
    """Remember the opponent response to my action that no later observation will show."""
    st = int(obs.street)
    if act in _RAISES:
        T = obs.raise_to[act]
        r = T - obs.my_bet
        mem["pending"] = ("raise", st, min(obs.raises_this_street + 1, 3), _sc(T - obs.opp_bet, obs.pot + r),
                          obs.raises_this_street + 1 < 4)
    elif (act == Action.CHECK_CALL and obs.to_call == 0 and st == 3
          and not any(int(e.street) == 3 and not e.is_me for e in obs.action_history)):
        mem["pending"] = ("check", st, 0, "", True)
    else:
        mem["pending"] = None


def _showdown_q(mem, s, opp_hole, board_full):
    if s == 0:
        return float(_PRE[_hash5(np.sort(np.array(opp_hole, dtype=np.int64)))] / 255.0)
    L = 2 + s
    bd = list(board_full[:L])
    blocked = np.zeros(_NCARD, bool)
    blocked[bd] = True
    blocked[list(mem["hole"])] = True
    blocked[list(opp_hole)] = False
    avail = np.nonzero(~blocked)[0]
    avail = np.array([c for c in avail if c not in opp_hole], dtype=np.int64)
    H = _rand_subsets(avail, 120, 5)
    H.sort(axis=1)
    Bd = np.sort(np.array(bd, dtype=np.int64))[None, :]
    tc = _tcodes(Bd)
    ref = _eval1(_pcodes(H), tc)
    mine = _eval1(_pcodes(np.sort(np.array(opp_hole, dtype=np.int64))[None, :]), tc)[0]
    return float((np.sum(ref < mine) + 0.5 * np.sum(ref == mine)) / len(ref))


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Update the opponent model from the finished hand."""
    t0 = time.perf_counter()
    try:
        _hand_end(result, memory)
    except Exception:
        memory["exc"] = memory.get("exc", 0) + 1
        memory["exc_tb"] = traceback.format_exc()
    finally:
        memory["time_used"] += time.perf_counter() - t0


def _hand_end(result, mem):
    if mem.get("hand") != result.hand_index:
        # I never acted: the opponent (button) folded to the big blind.
        if not result.went_to_showdown and result.winner == "me":
            _count(mem, 0, 0, True, "h", 0)
        return
    pend = mem.get("pending")
    recs = list(mem["recs"])
    if pend is not None:
        kind, st, d, sc, canr = pend
        if kind == "raise":
            if not result.went_to_showdown and result.winner == "me":
                a = 0
            elif result.went_to_showdown:
                a = 1
            else:
                a = None
            if a is not None:
                _count(mem, st, d, True, sc, a)
                recs.append((st, d, True, sc, a, canr))
        elif kind == "check" and result.went_to_showdown:
            _count(mem, st, d, False, "", 1)
            recs.append((st, d, False, "", 1, True))
    mem["pending"] = None
    mem["hand"] = None
    if result.went_to_showdown and result.opp_hole and len(result.board) == 5:
        qc = {}
        for (s, d, fa, sc, a, canr) in recs:
            if a == 0:
                continue
            if s not in qc:
                qc[s] = _showdown_q(mem, s, result.opp_hole, result.board)
            mem["sd"].append((0 if s == 0 else 1, s, d, fa, sc, a, qc[s], canr))
        mem["nsd"] = mem.get("nsd", 0) + 1
        if mem["nsd"] % 8 == 0:
            _refit_lam(mem)

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
