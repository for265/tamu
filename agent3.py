
"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
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
AGENT_NAME = "cloud"


# ===== YOUR CODE HERE =====
#
# agent3: offline CFR blueprint for heads-up Short Deck PLO5.
#
# * Stacks reset to 80 chips every hand, so the betting tree is identical for
#   every hand (15,952 decision nodes). It is stored exactly in
#   agent3_blueprint.npz and found by replaying obs.action_history.
# * Cards are abstracted into buckets per street (same code at train time):
#     preflop    all-in equity vs a random hand (precomputed for every hand)
#     flop/turn  equity vs a random hand (EHS) x current made-hand strength (CUR)
#     river      hand strength vs a random hand
#   estimated by Monte Carlo with a vectorised 2+3 evaluator (~0.3-0.6 ms).
# * The abstract game was solved offline with public-chance-sampling Linear
#   CFR (code in training/agent3/). At runtime we sample an action from the
#   blueprint's average strategy, with a small capped opponent adjustment.
# * Any unseen node or error -> equity-vs-pot-odds fallback. Never illegal.

import math
import random
import time
from itertools import combinations, combinations_with_replacement

import numpy as np

from evaluator import score_five

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_FILE = os.path.join(_HERE, "agent3_blueprint.npz")

# ---------- 1. Vectorised 2+3 evaluator ------------------------------------------
# Every 5-card rank multiset (each rank at most 4 times) has a unique sum of
# these per-rank weights (max 24918). A hole pair gets the code
#   weight sum + 6*KM*pair_suit   (pair_suit = suit if suited else 4)
# and a board triple gets
#   weight sum + KM*triple_suit   (triple_suit = suit if monotone else 5)
# so code(pair) + code(triple) indexes one int16 table of dense hand ranks
# (flush ranks where pair_suit == triple_suit < 4). Bigger rank = better hand,
# equal = tie, exactly as evaluator.py orders scores.
_RANK_W = np.array([0, 1, 5, 22, 94, 312, 992, 2422, 5624], np.int32)
_KM = 24919
_CW = _RANK_W[np.arange(36) >> 2].astype(np.int32)
_CS = (np.arange(36) & 3).astype(np.int32)


def _build_rank_table():
    plain = np.zeros(_KM, np.int64)
    flush = np.zeros(_KM, np.int64)
    for m in combinations_with_replacement(range(9), 5):
        if max(m.count(r) for r in m) > 4:
            continue
        key = int(sum(int(_RANK_W[r]) for r in m))
        if len(set(m)) == 5:
            plain[key] = score_five([r * 4 + (i % 2) for i, r in enumerate(m)])   # not a flush
            flush[key] = score_five([r * 4 for r in m])                           # flush
        else:
            seen, cards = {}, []
            for r in m:
                k = seen.get(r, 0)
                seen[r] = k + 1
                cards.append(r * 4 + k)
            plain[key] = score_five(cards)
    uniq = np.unique(np.concatenate([plain, flush]))
    plain = np.searchsorted(uniq, plain).astype(np.int16)
    flush = np.searchsorted(uniq, flush).astype(np.int16)
    tab = np.empty((5, 6, _KM), np.int16)
    for ps in range(5):
        for ts in range(6):
            tab[ps, ts] = flush if ps == ts else plain
    return tab.ravel()


_TAB = _build_rank_table()
_PI = np.array([i for i, _ in combinations(range(5), 2)])
_PJ = np.array([j for _, j in combinations(range(5), 2)])
_TRI = {n: np.array(list(combinations(range(n), 3))) for n in (3, 4, 5)}
_PX = (_CW[:, None] + _CW[None, :]
       + 6 * _KM * np.where(_CS[:, None] == _CS[None, :], _CS[:, None], 4)).astype(np.int32)
_BINOM = np.array([[math.comb(n, k) for k in range(6)] for n in range(37)], np.int64)


def pair_codes(hole):
    """(..., 5) hole cards -> (..., 10) codes of the 10 hole pairs."""
    return _PX[hole[..., _PI], hole[..., _PJ]]


def triple_codes(board):
    """(..., nb) board cards -> (..., C(nb, 3)) codes of the board triples."""
    tri = _TRI[board.shape[-1]]
    b0 = board[..., tri[:, 0]]
    b1 = board[..., tri[:, 1]]
    b2 = board[..., tri[:, 2]]
    s0 = _CS[b0]
    return (_CW[b0] + _CW[b1] + _CW[b2]
            + _KM * np.where((s0 == _CS[b1]) & (s0 == _CS[b2]), s0, 5))


def combine(px, tx):
    """Best rank over all (pair, triple) combinations; leading dims broadcast."""
    return _TAB[px[..., :, None] + tx[..., None, :]].max(axis=(-2, -1))


def best_rank(hole, board):
    """Best dense rank using exactly 2 of 5 hole cards and 3 of the board."""
    return combine(pair_codes(hole), triple_codes(board))


def colex(holes):
    """Colex index (0..376991) of sorted 5-card hands, shape (N, 5) -> (N,)."""
    h = np.sort(np.asarray(holes, np.int64).reshape(-1, 5), axis=1)
    return (_BINOM[h[:, 0], 1] + _BINOM[h[:, 1], 2] + _BINOM[h[:, 2], 3]
            + _BINOM[h[:, 3], 4] + _BINOM[h[:, 4], 5])


# ---------- 2. Hand features (Monte Carlo vs a random hand) -----------------------
S_DRAW = 256    # samples on flop / turn  (opponent hand + runout)
S_RIVER = 384   # samples on the river    (opponent hand)


def sample_cards(avail, n_samples, k, rng):
    """k distinct cards per sample from each row of avail (N, R) -> (N, S, k)."""
    n, r = avail.shape
    a = np.repeat(avail[:, None, :], n_samples, axis=1)
    for d in range(k):
        j = (d + (rng.random((n, n_samples)) * (r - d)).astype(np.int64))[..., None]
        x = np.take_along_axis(a, j, -1)
        np.put_along_axis(a, j, a[..., d:d + 1].copy(), -1)
        a[..., d:d + 1] = x
    return a[..., :k]


def _win_rate(own, oth):
    return ((own > oth).sum(-1) + 0.5 * (own == oth).sum(-1)) / oth.shape[-1]


def street_features(holes, board, rng):
    """(EHS, CUR) for each hand (N, 5) on a 3/4/5-card board.

    EHS: P(win) + P(tie)/2 at showdown vs a uniformly random opponent hand,
         rest of the board run out at random.
    CUR: the same, but comparing the best hands made on the current board
         (equals EHS on the river)."""
    holes = np.asarray(holes, np.int64).reshape(-1, 5)
    board = np.asarray(board, np.int64)
    n, nb = len(holes), len(board)
    s = S_RIVER if nb == 5 else S_DRAW
    m = np.ones((n, 36), bool)
    m[np.arange(n)[:, None], holes] = False
    m[:, board] = False
    avail = np.nonzero(m)[1].reshape(n, 36 - 5 - nb)
    smp = sample_cards(avail, s, 10 - nb, rng)
    px_own = pair_codes(holes)                      # (N, 10)
    px_opp = pair_codes(smp[..., :5])               # (N, S, 10)
    tx_now = triple_codes(board)                    # (C(nb,3),)
    own_c = combine(px_own, tx_now)[:, None]
    oth_c = combine(px_opp, tx_now)
    cur = _win_rate(own_c, oth_c)
    if nb == 5:
        return cur, cur
    full = np.concatenate([np.broadcast_to(board, (n, s, nb)), smp[..., 5:]], axis=-1)
    tx = triple_codes(full)                         # (N, S, 10)
    ehs = _win_rate(combine(px_own[:, None, :], tx), combine(px_opp, tx))
    return ehs, cur


def bucketize(street, ehs, cur, edges):
    """Feature values -> bucket ids for a postflop street (arrays in, array out)."""
    if street == 3:
        return np.searchsorted(edges["e3"], ehs, side="right")
    e = np.searchsorted(edges["e%d" % street], ehs, side="right")
    ce = edges["c%d" % street][e]                       # (N, C-1) conditional CUR edges
    c = (np.asarray(cur)[:, None] >= ce).sum(1)
    return e * (ce.shape[1] + 1) + c


# ---------- 3. Blueprint data ----------------------------------------------------
def _load_blueprint():
    try:
        with np.load(_DATA_FILE) as z:
            bp = {k: z[k] for k in z.files}
        bp["children"] = bp["children"].astype(np.int64)
        if "node_freq" in bp:
            bp["node_freq"] = bp["node_freq"].astype(np.float32)
        bp["edges"] = {k: bp[k] for k in ("e1", "c1", "e2", "c2", "e3")}
        return bp
    except Exception:   # missing / broken file: play the fallback heuristic
        return None


_BP = _load_blueprint()
_ACTIONS = (Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)
RAISE_ACTIONS = (Action.RAISE_HALF, Action.RAISE_POT)
TIME_SOFT_LIMIT = 6.0     # seconds per half: beyond this, use cheaper features


def _find_node(obs, me):
    """Replay the hand's action history through the stored tree; -1 if off-tree."""
    bp = _BP
    ch = bp["children"]
    n = 0
    for a in obs.action_history:
        n = int(ch[n, int(a.action)])
        if n < 0:
            return -1
    if (bp["kind"][n] != 0 or bp["player"][n] != me or bp["street"][n] != int(obs.street)
            or bp["pot"][n] != obs.pot or bp["to_call"][n] != obs.to_call):
        return -1
    legal = {int(a) for a in obs.legal_actions}
    if legal != {i for i in range(4) if ch[n, i] >= 0}:
        return -1
    return n


def _features(obs, memory):
    """(ehs, bucket) for my hand on this street, cached per hand and street."""
    key = (obs.hand_index, int(obs.street))
    cache = memory.setdefault("feat", {})
    if key in cache:
        return cache[key]
    if len(cache) > 8:
        cache.clear()
    bp = _BP
    if obs.street == Street.PREFLOP:
        ci = int(colex([obs.hole_cards])[0])
        if bp is not None:
            res = (float(bp["pre_ehs"][ci]) / 255.0, int(bp["pre_bucket"][ci]))
        else:
            res = (0.5, -1)
    else:
        rng = memory.get("np_rng") or np.random.default_rng()
        hole = np.array(obs.hole_cards, np.int64)
        if memory.get("time_used", 0.0) > TIME_SOFT_LIMIT:   # emergency: tiny sample
            ehs, _ = _cheap_equity(hole, obs.board, rng)
            res = (ehs, -1)
        else:
            ehs, cur = street_features(hole[None, :], obs.board, rng)
            b = -1
            if bp is not None:
                b = int(bucketize(int(obs.street), ehs, cur, bp["edges"])[0])
            res = (float(ehs[0]), b)
    cache[key] = res
    return res


def _cheap_equity(hole, board, rng, n=32):
    board = np.asarray(board, np.int64)
    nb = len(board)
    m = np.ones(36, bool)
    m[hole] = False
    m[board] = False
    avail = np.nonzero(m)[0][None, :]
    smp = sample_cards(avail, n, 10 - nb, rng)[0]
    full = np.concatenate([np.broadcast_to(board, (n, nb)), smp[:, 5:]], axis=-1)
    own = best_rank(hole[None, :], full)
    oth = best_rank(smp[:, :5], full)
    v = float(((own > oth) + 0.5 * (own == oth)).mean())
    return v, v


def _heuristic(obs, ehs):
    """Cheap equity-vs-pot-odds fallback. Always returns a legal action."""
    legal = obs.legal_actions
    if obs.to_call > 0:
        odds = obs.to_call / float(obs.pot + obs.to_call)
        if ehs > 0.85 and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if ehs >= odds + 0.05 or ehs > 0.7:
            return Action.CHECK_CALL
        return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
    if ehs > 0.72 and Action.RAISE_HALF in legal:
        return Action.RAISE_HALF
    return Action.CHECK_CALL


def _safe_default(obs):
    return Action.CHECK_CALL if (obs.to_call == 0 or Action.FOLD not in obs.legal_actions) \
        else Action.FOLD


# ---------- 4. Opponent model: observed vs equilibrium frequencies ---------------
# Every opponent decision is located at its exact tree node, and the blueprint
# stores the equilibrium (self-play) action frequencies of every node
# (node_freq). Per street we sum observed and expected counts, and the binomial
# variance, of
#   aggression  bet/raise, at all opponent decisions
#   folding     fold, at opponent decisions facing our bet
# z = (observed - expected) / sqrt(variance). Only clear deviations (|z| > Z0)
# move our strategy, and by at most CAP of the probability mass involved, so
# against a balanced opponent we play the blueprint.
Z0 = 2.5
CAP = 0.30
ADAPT = os.environ.get("AGENT3_ADAPT", "1") != "0"     # switch for local A/B tests


def _opp(memory):
    st = memory.get("opp")
    if st is None:
        st = memory["opp"] = {k: [0.0, 0.0, 0.0, 0.0]
                              for k in ("ag_o", "ag_e", "ag_v", "fo_o", "fo_e", "fo_v")}
    return st


def _observe(st, node, action):
    """The opponent played `action` (int) at tree node `node`."""
    f = _BP["node_freq"][node]
    if f.sum() <= 0:
        return
    s = int(_BP["street"][node])
    pa = float(f[2] + f[3])
    st["ag_o"][s] += 1.0 if action >= 2 else 0.0
    st["ag_e"][s] += pa
    st["ag_v"][s] += pa * (1.0 - pa)
    if _BP["to_call"][node] > 0:
        pf = float(f[0])
        st["fo_o"][s] += 1.0 if action == 0 else 0.0
        st["fo_e"][s] += pf
        st["fo_v"][s] += pf * (1.0 - pf)


def _z(st, key, s):
    v = st[key + "_v"][s]
    if v < 3.0:                       # too little evidence
        return 0.0
    return (st[key + "_o"][s] - st[key + "_e"][s]) / math.sqrt(v)


def _adapt(probs, obs, ehs, memory):
    """Bounded shift of the blueprint toward exploiting measured deviations."""
    st = _opp(memory)
    s = int(obs.street)
    out = probs.copy()
    legal = obs.legal_actions
    if obs.to_call > 0:
        z = _z(st, "ag", s)
        if z > Z0 and out[0] > 0 and ehs > 0.25:
            d = min(CAP, 0.1 * (z - Z0)) * out[0]          # over-aggressive: call down lighter
            out[0] -= d
            out[1] += d
        elif z < -Z0 and out[1] > 0 and ehs < 0.6:
            d = min(CAP / 2, 0.05 * (-z - Z0)) * out[1]     # under-aggressive: respect bets
            out[1] -= d
            out[0] += d
    zf = _z(st, "fo", s)
    raise_mass = out[2] + out[3]
    if zf > Z0 and obs.to_call == 0 and ehs < 0.5 and out[1] > 0 and Action.RAISE_HALF in legal:
        d = min(CAP, 0.1 * (zf - Z0)) * out[1]              # over-folder: bluff more
        out[1] -= d
        out[2] += d
    elif zf < -Z0:
        w = min(CAP, 0.1 * (-zf - Z0))
        if ehs < 0.5 and raise_mass > 0:                    # under-folder: bluff less
            out[1 if obs.to_call == 0 else 0] += w * raise_mass
            out[2] *= 1.0 - w
            out[3] *= 1.0 - w
        elif ehs > 0.75 and out[1] > 0 and Action.RAISE_HALF in legal:
            d = w * out[1]                                  # ... and value-bet thinner
            out[1] -= d
            out[2] += d
    return out


# ---------- 5. Agent API -----------------------------------------------------------
def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["time_used"] = 0.0
    memory["np_rng"] = np.random.default_rng()
    memory["rng"] = random.Random()
    memory["feat"] = {}
    memory["adapt"] = ADAPT
    _opp(memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Blueprint lookup + sampling, with a safe fallback."""
    t0 = time.perf_counter()
    try:
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = _safe_default(obs)
    except Exception as exc:
        memory["n_exc"] = memory.get("n_exc", 0) + 1          # visible in local tests
        memory["last_exc"] = repr(exc)
        action = _safe_default(obs)
    memory["my_last"] = (memory.get("cur_node", -1), int(action))
    memory["time_used"] = memory.get("time_used", 0.0) + time.perf_counter() - t0
    return action


def _decide(obs, memory):
    me = 0 if obs.is_button else 1
    memory["cur_node"] = -1
    if _BP is not None:
        try:
            _track_opponent(obs, memory)
        except Exception:
            pass
    try:
        ehs, bucket = _features(obs, memory)
    except Exception:
        ehs, bucket = 0.5, -1
    probs = None
    if _BP is not None and bucket >= 0:
        node = _find_node(obs, me)
        memory["cur_node"] = node
        if node >= 0:
            row = int(_BP["rowbase"][node]) + bucket
            if _BP["conf"][row]:
                legal = np.zeros(4)
                for a in obs.legal_actions:
                    legal[int(a)] = 1.0
                probs = _BP["strategy"][row].astype(np.float64) * legal
                if probs.sum() <= 0:
                    probs = None
    if probs is None:
        memory["n_fallback"] = memory.get("n_fallback", 0) + 1
        return _heuristic(obs, ehs)
    probs = probs / probs.sum()
    if memory.get("adapt", True):
        try:
            probs = _adapt(probs, obs, ehs, memory)
        except Exception:
            pass
    rng = memory.get("rng") or random
    u = rng.random() * probs.sum()
    acc = 0.0
    for i in range(4):
        acc += probs[i]
        if u < acc and probs[i] > 0:
            return _ACTIONS[i]
    return _ACTIONS[int(np.argmax(probs))]


def _track_opponent(obs, memory):
    """Walk the tree along the history; record each opponent action once."""
    st = _opp(memory)
    hist = obs.action_history
    if memory.get("cur_hand") != obs.hand_index:
        memory["cur_hand"] = obs.hand_index
        memory["seen_n"] = 0
        memory["walk"] = 0
        memory["my_last"] = None
    n = memory.get("walk", -1)
    ch, kind = _BP["children"], _BP["kind"]
    for i in range(memory.get("seen_n", 0), len(hist)):
        if n < 0:
            break
        a = int(hist[i].action)
        if not hist[i].is_me and kind[n] == 0:
            _observe(st, n, a)
        n = int(ch[n, a])
    memory["walk"] = n
    memory["seen_n"] = len(hist)


def on_hand_end(result: HandResult, memory: dict) -> None:
    """The opponent's answer to our last action is never shown to us if it
    ended the hand: a fold (we win without showdown) or a call/check that
    led to showdown. Record it here."""
    t0 = time.perf_counter()
    try:
        last = memory.get("my_last")
        if (_BP is not None and last is not None and last[0] >= 0
                and memory.get("cur_hand") == result.hand_index):
            r = int(_BP["children"][last[0], last[1]])
            if r >= 0 and _BP["kind"][r] == 0:
                if not result.went_to_showdown and result.winner == "me":
                    _observe(_opp(memory), r, 0)
                elif result.went_to_showdown:
                    _observe(_opp(memory), r, 1)
        memory["my_last"] = None
    except Exception:
        pass
    memory["time_used"] = memory.get("time_used", 0.0) + time.perf_counter() - t0

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
