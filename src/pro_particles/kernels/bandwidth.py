from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def median_heuristic_lengthscale(
    samples: ArrayF, max_points: int = 1024, rng: Optional[np.random.Generator] = None
) -> float:
    """Median heuristic for kernel lengthscale based on pairwise distances.

    Uses at most max_points samples (deterministic prefix unless rng is provided).
    """
    if samples.ndim != 2:
        raise ValueError("samples must be 2D.")
    n = samples.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 samples for median heuristic.")

    if n > max_points:
        if rng is None:
            x = samples[:max_points]
        else:
            idx = rng.choice(n, size=max_points, replace=False)
            x = samples[idx]
    else:
        x = samples

    m = x.shape[0]
    diffs = x[:, None, :] - x[None, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=-1))
    tri = np.triu_indices(m, k=1)
    vals = dists[tri]
    if vals.size == 0:
        raise ValueError("Not enough distinct pairs for median heuristic.")
    med = float(np.median(vals))
    if med <= 0.0:
        raise ValueError("Median distance is non-positive.")
    return med


def scott_bandwidth(samples: ArrayF) -> float:
    """Scott's rule-of-thumb bandwidth."""
    if samples.ndim != 2:
        raise ValueError("samples must be 2D.")
    n, d = samples.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    scale = float(np.sqrt(np.mean(np.var(samples, axis=0, ddof=1))))
    if scale <= 0.0:
        raise ValueError("Non-positive scale in Scott bandwidth.")
    return scale * (n ** (-1.0 / (d + 4.0)))


def silverman_bandwidth(samples: ArrayF) -> float:
    """Silverman's rule-of-thumb bandwidth."""
    if samples.ndim != 2:
        raise ValueError("samples must be 2D.")
    n, d = samples.shape
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    scale = float(np.sqrt(np.mean(np.var(samples, axis=0, ddof=1))))
    if scale <= 0.0:
        raise ValueError("Non-positive scale in Silverman bandwidth.")
    return scale * ((n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0)))
