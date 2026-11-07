#!/usr/bin/env bash
# Duplicate-match evaluation of agent3 against the reference opponents.
# Usage (from anywhere):  training/agent3/evaluate.sh [matches] [workers] [seed] [opponents...]
# Uses the in-process harness /tmp/tamu_eval/duel.py (seats swapped, same deals).
set -euo pipefail
cd "$(dirname "$0")/../.."
MATCHES=${1:-8}
WORKERS=${2:-6}
SEED=${3:-duel}
shift 3 2>/dev/null || shift $#
OPPS=${*:-agent2.py random station half pot}
for o in $OPPS; do
  echo "=== agent3 vs $o"
  python3 /tmp/tamu_eval/duel.py agent3.py "$o" --matches "$MATCHES" --deals 1000 \
      --workers "$WORKERS" --seed "$SEED" | tail -n 1
done
