from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np


def sha256_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return f"sha256:{h.hexdigest()}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def is_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        return len(out) > 0
    except Exception:
        return True


def env_info() -> Dict[str, Any]:
    import numpy as np
    import sys

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2))


def median_heuristic_lengthscale(x: np.ndarray) -> float:
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 points for median heuristic.")
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    tri = dists[np.triu_indices(n, k=1)]
    med = float(np.median(tri))
    if med <= 0.0:
        raise ValueError("Median heuristic lengthscale is non-positive.")
    return med
