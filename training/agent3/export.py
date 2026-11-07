"""Export the trained blueprint to agent3_blueprint.npz at the repo root.

Contents (all small integer arrays, compressed):
  tree     children kind player street pot to_call      (betting tree)
  rowbase  first table row of each decision node (row = rowbase + bucket)
  strategy uint8 (rows, 4)  average strategy * 255 (purified: tiny probs -> 0)
  conf     bool  (rows,)    row has enough training mass to be trusted
  buckets  pre_bucket / pre_ehs (uint8 per colex hand index), e1 c1 e2 c2 e3
Usage: python3 export.py [--tag ""] [--purify 0.03] [--out agent3_blueprint.npz]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, REPO  # noqa: E402
from tree import build_tree  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--purify", type=float, default=0.03)
    ap.add_argument("--min-mass", type=float, default=1e-6,
                    help="rows whose strategy mass is below this fraction of the median are distrusted")
    ap.add_argument("--out", default=os.path.join(REPO, "agent3_blueprint.npz"))
    args = ap.parse_args()

    t = build_tree()
    meta = np.load(os.path.join(OUT, "cfr_meta%s.npz" % args.tag))
    R = np.load(os.path.join(OUT, "cfr_regret%s.npy" % args.tag)).astype(np.float64)
    S = np.load(os.path.join(OUT, "cfr_strat%s.npy" % args.tag)).astype(np.float64)
    e = np.load(os.path.join(OUT, "edges.npz"))
    pre = np.load(os.path.join(OUT, "preflop.npz"))
    rowbase = meta["rowbase"]
    nb = meta["nbuckets"]

    # legal mask per row
    kind, street = t["kind"], t["street"].astype(np.int64)
    legal_rows = np.zeros(S.shape, bool)
    dec = np.where(kind == 0)[0]
    for n in dec:
        lo = rowbase[n]
        legal_rows[lo:lo + nb[street[n]]] = t["children"][n] >= 0

    mass = S.sum(1)
    avg = np.where(mass[:, None] > 0, S / np.where(mass > 0, mass, 1)[:, None], 0.0)
    pos = np.maximum(R, 0) * legal_rows
    psum = pos.sum(1)
    cur = np.where(psum[:, None] > 0, pos / np.where(psum > 0, psum, 1)[:, None], 0.0)
    med = np.median(mass[mass > 0])
    use_avg = mass > args.min_mass * med
    strat = np.where(use_avg[:, None], avg, cur)
    conf = use_avg | (psum > 0)
    # purification: drop actions below the threshold, renormalise
    if args.purify > 0:
        keep = strat >= args.purify
        keep[np.arange(len(strat)), strat.argmax(1)] = True
        strat = strat * keep
        strat /= np.maximum(strat.sum(1, keepdims=True), 1e-12)
    strat = strat * legal_rows
    q = np.round(strat * 255).astype(np.int64)
    # make sure every trusted row keeps at least one legal action
    empty = q.sum(1) == 0
    q[empty, strat[empty].argmax(1)] = 255
    # equilibrium (self-play) action frequencies per node, for the opponent model
    node_freq = np.zeros((len(kind), 4), np.float32)
    for n in dec:
        lo = rowbase[n]
        tot = S[lo:lo + nb[street[n]]].sum(0)
        if tot.sum() > 0:
            node_freq[n] = tot / tot.sum()
    out = dict(node_freq=node_freq.astype(np.float16),
        children=t["children"].astype(np.int32), kind=t["kind"], player=t["player"],
        street=t["street"], pot=t["pot"], to_call=t["to_call"],
        rowbase=rowbase.astype(np.int32), strategy=q.astype(np.uint8), conf=conf,
        pre_bucket=e["pre_bucket"].astype(np.uint8),
        pre_ehs=np.round(pre["ehs"] * 255).astype(np.uint8),
        e1=e["e1"], c1=e["c1"], e2=e["e2"], c2=e["c2"], e3=e["e3"], nbuckets=nb,
        boards=meta["boards"],
    )
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, "%.2f MB" % (os.path.getsize(args.out) / 1e6),
          "boards", int(meta["boards"]), "rows", len(q),
          "avg-strategy rows %.1f%%" % (100 * use_avg.mean()), "trusted %.1f%%" % (100 * conf.mean()))


if __name__ == "__main__":
    main()
