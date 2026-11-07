"""Shared helpers for the agent3 training pipeline.

The card-feature code (vectorised evaluator, Monte Carlo features, bucketing)
lives in agent3.py itself so that training and play use exactly the same code.
This module imports agent3.py with fastapi/uvicorn stubbed out.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Bucket counts per street and quantile shaping (see edges.py).
K_PRE = 100            # preflop buckets
E_POST = 16            # EHS bins on flop / turn
C_POST = 5             # CUR bins inside each EHS bin
K_RIVER = 64           # river buckets
N_BUCKETS = (K_PRE, E_POST * C_POST, E_POST * C_POST, K_RIVER)


def _stub_web() -> None:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        return
    except ImportError:
        pass

    class _App:
        def __init__(self, *a, **k):
            pass

        def get(self, *a, **k):
            return lambda f: f

        post = get

    fa = types.ModuleType("fastapi")
    fa.FastAPI = _App
    fa.Request = object
    fr = types.ModuleType("fastapi.responses")
    fr.JSONResponse = object
    uv = types.ModuleType("uvicorn")
    uv.run = lambda *a, **k: None
    sys.modules.update({"fastapi": fa, "fastapi.responses": fr, "uvicorn": uv})


def load_agent3():
    """Import agent3.py from the repo root as a module."""
    _stub_web()
    if "agent3_mod" in sys.modules:
        return sys.modules["agent3_mod"]
    spec = importlib.util.spec_from_file_location("agent3_mod", os.path.join(REPO, "agent3.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["agent3_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def quantile_levels(k: int, gamma: float):
    """k-1 cumulative quantile levels, finer near the top when gamma > 1."""
    import numpy as np
    x = np.arange(1, k) / k
    return 1.0 - (1.0 - x) ** gamma
