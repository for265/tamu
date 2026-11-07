
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

import random
import time
from itertools import combinations

from evaluator import (ONE_PAIR, STRAIGHT, FLUSH, category_name, category_of,
                       score_five)

_ALL_CARDS = tuple(range(36))
_PAIRS = tuple(combinations(range(5), 2))
_SAMPLES = {Street.PREFLOP: 60, Street.FLOP: 80, Street.TURN: 100, Street.RIVER: 120}
TIME_BANK = 8.0          # seconds of thinking per half (see game.TIME_BANK_SECONDS)
SAFETY = 0.6             # only plan to use this share of the bank
HANDS_PER_HALF = 1000
_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


# ---------- 1. Hand detection -------------------------------------------------

def best_made_hand(hole, board):
    """Best hand held right now using exactly 2 hole + 3 board cards.

    Works on a flop (3), turn (4) or river (5) board.
    Returns (score, category, name), or None before the flop.
    """
    if len(board) < 3:
        return None
    best = 0
    triples = list(combinations(board, 3))
    for i, j in _PAIRS:
        for combo in triples:
            s = score_five((hole[i], hole[j]) + combo)
            if s > best:
                best = s
    return best, category_of(best), category_name(best)


def detect_draws(hole, board):
    """What the hand can still become (flop/turn only).

    Returns flush_draw / straight_draw flags (2+3 rule) and `outs`,
    the number of unseen cards that improve the made-hand category.
    """
    info = {"flush_draw": False, "straight_draw": False, "outs": 0}
    if len(board) not in (3, 4):
        return info
    made = best_made_hand(hole, board)[1]
    seen = set(hole) | set(board)
    for c in _ALL_CARDS:
        if c in seen:
            continue
        cat = best_made_hand(hole, tuple(board) + (c,))[1]
        if cat > made:
            info["outs"] += 1
            if cat == FLUSH:
                info["flush_draw"] = True
            elif cat == STRAIGHT:
                info["straight_draw"] = True
    return info


# ---------- 2. Opponent model / bluff probability -------------------------------

def _opp_stats(memory):
    if not memory.get("opp"):
        memory["opp"] = {
            "agg_showdowns": 0,   # showdowns where opp bet/raised
            "agg_bluffs": 0,      # ...and showed a weak hand
            "folds_to_raise": 0, "raises_faced": 0,
            "opp_raises": 0, "opp_actions": 0,
        }
    return memory["opp"]


def bluff_probability(obs, memory):
    """P(opponent is bluffing | they are betting/raising now), 0..1.

    Showdown evidence of weak aggressive hands, shrunk toward a prior,
    then adjusted for overall aggression and bet size.
    """
    st = _opp_stats(memory)
    prior_bluff, prior_n = 0.25, 6.0
    p = (st["agg_bluffs"] + prior_bluff * prior_n) / (st["agg_showdowns"] + prior_n)
    if st["opp_actions"] >= 20:
        freq = st["opp_raises"] / st["opp_actions"]
        p *= 0.8 + 0.8 * min(freq, 0.5)           # frequent raisers bluff more
    size = obs.to_call / max(obs.pot - obs.to_call, 1)
    if size > 1.0:
        p *= 0.85                                   # overbets are usually value
    elif size < 0.4:
        p *= 1.15                                   # small bets are weaker
    return max(0.03, min(0.75, p))


# ---------- 3. Equity against the opponent's likely range -----------------------

def equity(obs, bluff_p=0.0, deadline=None):
    """Monte Carlo win probability vs the opponent's range.

    Opponent hands are sampled, ranked, and the strongest 60% count as
    their value range; the rest as bluffs. They are mixed by bluff_p.
    """
    seen = set(obs.hole_cards) | set(obs.board)
    deck = [c for c in _ALL_CARDS if c not in seen]
    need = 5 - len(obs.board)
    samples = []
    for k in range(_SAMPLES[obs.street]):
        if k >= 8 and deadline is not None and time.perf_counter() > deadline:
            break
        draw = random.sample(deck, 5 + need)
        full = tuple(obs.board) + tuple(draw[5:])
        mine = omaha_score(obs.hole_cards, full)
        theirs = omaha_score(tuple(draw[:5]), full)
        samples.append((theirs, 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0))
    if not samples:
        return 0.5
    if bluff_p <= 0 or len(samples) < 10:
        return sum(w for _, w in samples) / len(samples)
    samples.sort(key=lambda t: t[0], reverse=True)
    cut = int(len(samples) * 0.6)
    value, bluffs = samples[:cut], samples[cut:]
    pv = sum(w for _, w in value) / len(value)
    pb = sum(w for _, w in bluffs) / len(bluffs)
    return (1 - bluff_p) * pv + bluff_p * pb


# ---------- 4. Decision ------------------------------------------------------------

def _raise_action(obs, strong):
    """Pot raise when very strong, half-pot otherwise, if legal."""
    pref = (Action.RAISE_POT, Action.RAISE_HALF) if strong else _RAISES
    for a in pref:
        if a in obs.legal_actions:
            return a
    return None


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["hands_played"] = 0
    memory["last_history"] = ()
    memory["time_used"] = 0.0
    memory["hands_total"] = getattr(config, "num_hands", HANDS_PER_HALF) or HANDS_PER_HALF
    _opp_stats(memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Equity-based play with bluff-adjusted opponent range and sized raises."""
    t0 = time.perf_counter()
    # Spend the bank evenly: ~3 decisions per remaining hand, keep a reserve.
    hands_left = max(memory.get("hands_total", HANDS_PER_HALF) - obs.hand_index, 1)
    budget = (TIME_BANK * SAFETY - memory.get("time_used", 0.0)) / (hands_left * 3)
    budget = max(0.0, min(budget, 0.05))
    deadline = t0 + budget
    memory["last_history"] = obs.action_history
    try:
        return _decide(obs, memory, deadline)
    finally:
        memory["time_used"] = memory.get("time_used", 0.0) + time.perf_counter() - t0


def _decide(obs, memory, deadline):
    facing_bet = obs.to_call > 0
    bluff_p = bluff_probability(obs, memory) if facing_bet else 0.0
    eq = equity(obs, bluff_p, deadline)

    if len(obs.board) in (3, 4) and time.perf_counter() < deadline:   # small bonus for drawing potential
        outs = detect_draws(obs.hole_cards, obs.board)["outs"]
        eq = min(0.99, eq + 0.005 * min(outs, 8) * (1.0 if len(obs.board) == 3 else 0.5))

    if facing_bet:
        pot_odds = obs.to_call / (obs.pot + obs.to_call)
        if eq > 0.78 or (eq > 0.68 and bluff_p > 0.4):
            r = _raise_action(obs, strong=eq > 0.85)
            if r is not None:
                return r
        if eq >= pot_odds + 0.03:
            return Action.CHECK_CALL
        if bluff_p > 0.45 and eq >= pot_odds - 0.05:   # likely a bluff: call down
            return Action.CHECK_CALL
        return Action.FOLD

    if eq > 0.62:                                       # value bet
        r = _raise_action(obs, strong=eq > 0.8)
        if r is not None:
            return r
    st = _opp_stats(memory)
    fold_rate = st["folds_to_raise"] / st["raises_faced"] if st["raises_faced"] >= 10 else 0.3
    if 0.3 < eq < 0.5 and random.random() < fold_rate * 0.5:   # semi-bluff
        r = _raise_action(obs, strong=False)
        if r is not None:
            return r
    return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Update the opponent model from the finished hand."""
    memory["hands_played"] += 1
    st = _opp_stats(memory)
    hist = memory.get("last_history", ())
    opp_acts = [a for a in hist if not a.is_me]
    st["opp_actions"] += len(opp_acts)
    st["opp_raises"] += sum(a.action in _RAISES for a in opp_acts)
    if any(a.is_me and a.action in _RAISES for a in hist):
        st["raises_faced"] += 1
        if not result.went_to_showdown and result.winner == "me":
            st["folds_to_raise"] += 1
    if (result.went_to_showdown and result.opp_hole and len(result.board) == 5
            and any(a.action in _RAISES for a in opp_acts)):
        st["agg_showdowns"] += 1
        if category_of(omaha_score(result.opp_hole, result.board)) <= ONE_PAIR:
            st["agg_bluffs"] += 1

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
