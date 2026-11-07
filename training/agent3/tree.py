"""Build the complete betting tree of one hand directly from game.Game.

Stacks reset to 80 every hand, blinds are 1/2 and there are only 4 actions with
at most 4 raises per street, so the betting tree is identical for every hand
(about 16k decision nodes, 25.5k terminals).  We enumerate it with the real
engine (clone + apply), so legal actions, raise sizes, pots and payoffs are the
engine's by construction.

Player index p = seat in a Game built with button_seat=0, i.e.
    p = 0 -> button (posts SB, acts first preflop)
    p = 1 -> big blind (acts first postflop)

Node arrays (BFS order, root = 0; parents always precede children):
    kind      0 decision, 1 fold terminal, 2 showdown terminal
    player    acting player (decision nodes), -1 otherwise
    street    0..3 (for terminals: street where the hand ended)
    depth     number of actions from the root
    children  (N, 4) child per Action (FOLD, CHECK_CALL, RAISE_HALF, RAISE_POT), -1 if illegal
    pot, to_call, contrib0, contrib1   engine state on arrival at the node
    raise_half_to, raise_pot_to        raise_to amounts (decision nodes), 0 if illegal
    folder    seat that folded (fold terminals), -1 otherwise
"""
from __future__ import annotations

import os
import sys
from collections import deque

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import game as gm  # noqa: E402

_DUMMY_DEAL = gm.Deal(hole_cards=((0, 1, 2, 3, 4), (5, 6, 7, 8, 9)), board=(10, 11, 12, 13, 14))


class _NoShowdownGame(gm.Game):
    """Game whose showdown does not evaluate cards (we only need the tree)."""

    def _showdown(self) -> None:  # noqa: D401
        self._finish(winner_seat=None, showdown=True)


def build_tree() -> dict[str, np.ndarray]:
    root = _NoShowdownGame(_DUMMY_DEAL, button_seat=0)
    rows = []                      # per node dict
    queue = deque([(root, -1, -1, 0)])
    while queue:
        g, parent, pa, depth = queue.popleft()
        nid = len(rows)
        if parent >= 0:
            rows[parent]["children"][pa] = nid
        if g.is_terminal:
            res = g.result()
            folder = -1
            if not res.went_to_showdown:
                folder = g.history[-1].seat
            rows.append(dict(kind=1 if folder >= 0 else 2, player=-1, street=int(g.street), depth=depth,
                             children=[-1, -1, -1, -1], pot=g.pot, to_call=0,
                             contrib0=g.contributed[0], contrib1=g.contributed[1],
                             rh=0, rp=0, folder=folder))
            continue
        seat = g.to_act
        legal = g.legal_actions()
        rows.append(dict(kind=0, player=seat, street=int(g.street), depth=depth,
                         children=[-1, -1, -1, -1], pot=g.pot, to_call=g._to_call(seat),
                         contrib0=g.contributed[0], contrib1=g.contributed[1],
                         rh=g.raise_to(gm.Action.RAISE_HALF) if gm.Action.RAISE_HALF in legal else 0,
                         rp=g.raise_to(gm.Action.RAISE_POT) if gm.Action.RAISE_POT in legal else 0,
                         folder=-1))
        for a in legal:
            g2 = g.clone()
            g2.apply(a)
            queue.append((g2, nid, int(a), depth + 1))
    n = len(rows)
    out = {
        "kind": np.array([r["kind"] for r in rows], np.int8),
        "player": np.array([r["player"] for r in rows], np.int8),
        "street": np.array([r["street"] for r in rows], np.int8),
        "depth": np.array([r["depth"] for r in rows], np.int16),
        "children": np.array([r["children"] for r in rows], np.int32).reshape(n, 4),
        "pot": np.array([r["pot"] for r in rows], np.int16),
        "to_call": np.array([r["to_call"] for r in rows], np.int16),
        "contrib0": np.array([r["contrib0"] for r in rows], np.int16),
        "contrib1": np.array([r["contrib1"] for r in rows], np.int16),
        "raise_half_to": np.array([r["rh"] for r in rows], np.int16),
        "raise_pot_to": np.array([r["rp"] for r in rows], np.int16),
        "folder": np.array([r["folder"] for r in rows], np.int8),
    }
    # sanity: showdowns always have equal contributions (equal 80-chip stacks)
    sd = out["kind"] == 2
    assert np.all(out["contrib0"][sd] == out["contrib1"][sd])
    return out


def payoffs(tree: dict[str, np.ndarray]) -> np.ndarray:
    """(N, 2) chips won by each player at fold terminals (0 elsewhere).

    Showdown payoff for player p is contrib * sign(score_p - score_opp)."""
    n = len(tree["kind"])
    pay = np.zeros((n, 2), np.float32)
    f = np.where(tree["kind"] == 1)[0]
    folder = tree["folder"][f]
    lost = np.where(folder == 0, tree["contrib0"][f], tree["contrib1"][f]).astype(np.float32)
    pay[f, 0] = np.where(folder == 0, -lost, lost)
    pay[f, 1] = -pay[f, 0]
    return pay


if __name__ == "__main__":
    t = build_tree()
    k = t["kind"]
    print("nodes", len(k), "decision", int((k == 0).sum()), "fold", int((k == 1).sum()),
          "showdown", int((k == 2).sum()), "max depth", int(t["depth"].max()))
    for s in range(4):
        print("street", s, "decision nodes", int(((k == 0) & (t["street"] == s)).sum()))
