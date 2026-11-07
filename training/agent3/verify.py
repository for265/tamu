"""Check the stored tree / training payoffs against game.Game, and the agent.

1. Random action sequences on random deals in the real engine: at every
   decision the agent's tree walk must land on a node whose player, street,
   pot, to_call, legal actions and raise_to amounts match the engine; at the
   end the payoff used in training (fold: folder's contribution, showdown:
   contrib * sign) must equal game.result().nets.
2. agent3.choose_action returns a legal action at every decision.
"""
from __future__ import annotations

import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_agent3  # noqa: E402
from tree import build_tree, payoffs  # noqa: E402

import game as gm  # noqa: E402
from evaluator import omaha_score  # noqa: E402


def main(n_hands=3000):
    t = build_tree()
    pay = payoffs(t)
    a3 = load_agent3()
    rng = random.Random(7)
    ch = t["children"]
    checked = 0
    for h in range(n_hands):
        deal = gm.make_deal("verify", h)
        btn = h % 2
        g = gm.Game(deal, btn, hand_index=h)
        node = 0
        while not g.is_terminal:
            seat = g.to_act
            p = 0 if seat == btn else 1            # tree player index
            obs = g.observation(seat)
            # tree walk from the observation (what the agent does)
            n = a3._find_node(obs, p) if a3._BP is not None else node
            assert n == node, (h, n, node)
            assert t["kind"][node] == 0 and t["player"][node] == p
            assert t["street"][node] == int(obs.street) and t["pot"][node] == obs.pot
            assert t["to_call"][node] == obs.to_call
            legal = {int(a) for a in obs.legal_actions}
            assert legal == {i for i in range(4) if ch[node, i] >= 0}
            if gm.Action.RAISE_HALF in obs.legal_actions:
                assert obs.raise_to[gm.Action.RAISE_HALF] == t["raise_half_to"][node]
                assert obs.raise_to[gm.Action.RAISE_POT] == t["raise_pot_to"][node]
            act = a3.choose_action(obs, {}) if h % 3 == 0 else rng.choice(obs.legal_actions)
            assert act in obs.legal_actions
            g.apply(act)
            node = int(ch[node, int(act)])
            checked += 1
        res = g.result()
        k = t["kind"][node]
        assert k in (1, 2)
        nets_tree = [0.0, 0.0]
        if k == 1:
            nets_tree = [pay[node, 0], pay[node, 1]]
        else:
            c = t["contrib0"][node]
            s0 = omaha_score(deal.hole_cards[btn], deal.board)
            s1 = omaha_score(deal.hole_cards[1 - btn], deal.board)
            sg = (s0 > s1) - (s0 < s1)
            nets_tree = [c * sg, -c * sg]
        # tree player 0 = button seat
        assert nets_tree[0] == res.nets[btn] and nets_tree[1] == res.nets[1 - btn], (h, nets_tree, res.nets)
    print("verified %d hands, %d decisions: tree walk, legality, pot, to_call, raise sizes, payoffs OK"
          % (n_hands, checked))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("%.1fs" % (time.time() - t0))
