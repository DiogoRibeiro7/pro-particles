from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def load_palmer_penguins(
    *,
    cache_path: Optional[Path] = None,
) -> ArrayF:
    """Load Palmer penguins bill length/depth data with optional caching.

    If cache_path exists, loads from CSV. Otherwise attempts to load via
    palmerpenguins and optionally writes to cache_path.
    """
    if cache_path is not None and cache_path.exists():
        data = np.loadtxt(cache_path, delimiter=",", dtype=np.float64)
        return data

    try:
        import pandas  # noqa: F401
        from palmerpenguins import load_penguins
    except ImportError as exc:
        raise ImportError(
            "palmerpenguins and pandas are required for this dataset."
        ) from exc

    df = load_penguins()
    cols = ["bill_length_mm", "bill_depth_mm"]
    data = df[cols].dropna().to_numpy(dtype=np.float64)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(cache_path, data, delimiter=",")
    return data
