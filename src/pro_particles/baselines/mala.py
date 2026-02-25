from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def run_baseline_mala_gaussian_location(
    *,
    init: ArrayF,
    log_target: Callable[[ArrayF], float],
    grad_log_target: Callable[[ArrayF], ArrayF],
    dt: float,
    n_steps: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> ArrayF:
    """Metropolis-adjusted Langevin algorithm (MALA) with constant step size."""
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

    def proposal(x_cur: ArrayF) -> ArrayF:
        grad = grad_log_target(x_cur)
        if grad.shape != x_cur.shape:
            raise ValueError("grad_log_target returned wrong shape.")
        return x_cur + 0.5 * dt * grad + np.sqrt(dt) * rng.normal(size=x_cur.shape)

    def log_q(x_from: ArrayF, x_to: ArrayF) -> float:
        mean = x_from + 0.5 * dt * grad_log_target(x_from)
        diff = x_to - mean
        return -0.5 * np.dot(diff, diff) / dt

    for t in range(n_steps):
        x_prop = proposal(x)
        log_alpha = log_target(x_prop) + log_q(x_prop, x) - log_target(x) - log_q(x, x_prop)
        if np.log(rng.uniform()) < log_alpha:
            x = x_prop

        if t >= burn_in and ((t - burn_in) % thin == 0):
            saved.append(x.copy())

    return np.asarray(saved, dtype=np.float64)
