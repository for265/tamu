"""Helper: import agent4.py from the repo root even when fastapi/uvicorn are missing."""
import importlib.util
import os
import sys
import types

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def load_agent4():
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError:
        class _App:
            def __init__(self, *a, **k): pass
            def get(self, *a, **k): return lambda f: f
            post = get
        fa = types.ModuleType("fastapi"); fa.FastAPI = _App; fa.Request = object
        fr = types.ModuleType("fastapi.responses"); fr.JSONResponse = object
        uv = types.ModuleType("uvicorn"); uv.run = lambda *a, **k: None
        sys.modules.update({"fastapi": fa, "fastapi.responses": fr, "uvicorn": uv})
    spec = importlib.util.spec_from_file_location("agent4_mod", os.path.join(REPO, "agent4.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
