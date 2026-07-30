"""Run provenance: every output file gets a sidecar recording how to remake it."""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")


def ensure_dirs():
    for d in (RESULTS_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)


def git_commit():
    """Current commit hash, or ``"unknown"`` outside a repository.

    Recorded as the resolved SHA, not the string ``HEAD`` -- a sidecar that
    says ``HEAD`` pins nothing.
    """
    try:
        out = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    sha = out.stdout.strip()
    try:
        # Tracked modifications only: freshly written, not-yet-committed outputs
        # in results/ are not a reason to call the *code* state dirty.
        dirty = subprocess.run(
            ["git", "-C", PROJECT_ROOT, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            sha += "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass
    return sha


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_sidecar(output_name, parameters, seed, extra=None):
    """Write ``<output_name>.meta.json`` next to an output in ``results/``."""
    ensure_dirs()
    meta = {
        "output": output_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "seed": int(seed),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "parameters": _jsonable(parameters),
    }
    if extra:
        meta.update(_jsonable(extra))
    path = os.path.join(RESULTS_DIR, output_name + ".meta.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path


def save_json(output_name, payload, parameters, seed, extra=None):
    """Save a results payload plus its sidecar; returns the results path."""
    ensure_dirs()
    path = os.path.join(RESULTS_DIR, output_name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
    write_sidecar(output_name, parameters, seed, extra)
    return path


def load_json(output_name):
    with open(os.path.join(RESULTS_DIR, output_name), encoding="utf-8") as fh:
        return json.load(fh)
