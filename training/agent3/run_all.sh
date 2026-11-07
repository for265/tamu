#!/usr/bin/env bash
# Rebuild agent3_blueprint.npz from scratch (about 55 minutes on 6 cores).
#
#   PY=/path/to/python training/agent3/run_all.sh
#
# PY should be a python whose numpy uses an optimised BLAS (pip numpy wheels
# bundle OpenBLAS). The distro numpy here links the reference BLAS, which makes
# the GEMMs in cfr.py ~7x slower (whole iteration ~1.6x slower). The shipped
# blueprint was trained with numpy 2.5 / OpenBLAS; agent3.py itself runs on
# numpy 1.26 (the tournament image).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
$PY preflop.py                                     # preflop equity table (~25 s)
$PY edges.py                                       # bucket boundaries (~10 s)
$PY cfr.py --minutes "${MINUTES:-50}" --workers "${WORKERS:-6}"   # PCS Linear CFR
$PY export.py                                      # -> ../../agent3_blueprint.npz
python3 verify.py                                  # tree / payoffs / legality vs game.Game
