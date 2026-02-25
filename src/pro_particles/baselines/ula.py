from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def run_baseline_ula_constant(
    *,
    init: ArrayF,
    grad_log_target: Callable[[ArrayF], ArrayF],
    dt: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> ArrayF:
    """ULA with constant step size (baseline for MALA comparisons)."""
    if dt <= 0.0:
        raise ValueError("dt must be > 0.")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if burn_in < 0 or burn_in >= n_steps:
        raise ValueError("burn_in must be in [0, n_steps).")
    if thin <= 0:
        raise ValueError("thin must be > 0.")

    x = init.astype(np.float64, copy=True)
    if x.ndim != 1:
        raise ValueError("init must be 1D.")

    rng = np.random.default_rng(seed)
    saved = []

    for t in range(n_steps):
        grad = grad_log_target(x)
        if grad.shape != x.shape:
            raise ValueError("grad_log_target returned wrong shape.")
        if not np.all(np.isfinite(grad)):
            raise ValueError("Non-finite gradient in ULA.")

        x = x + 0.5 * dt * grad + np.sqrt(dt) * rng.normal(size=x.shape)

        if t >= burn_in and ((t - burn_in) % thin == 0):
            saved.append(x.copy())

    return np.asarray(saved, dtype=np.float64)
