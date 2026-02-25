from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass
class FuseState:
    """State for Fuse step size schedule.

    Spec reference: docs/fuse_spec.md (Eq. 180–182, Algorithm 1).
    """

    r_eps: float
    x1: Optional[ArrayF] = None
    max_movement: float = 0.0
    sum_grad_norm_sq: float = 0.0
    eta: float = 0.0

    def validate(self) -> None:
        if self.r_eps <= 0.0:
            raise ValueError("r_eps must be > 0.")
        if self.x1 is not None and not np.all(np.isfinite(self.x1)):
            raise ValueError("x1 contains non-finite values.")


def should_update(t: int) -> bool:
    """Fuse update condition.

    Spec reference: docs/fuse_spec.md (Algorithm 1, line 3–4).
    """
    return t >= 1


def update_eta(
    state: FuseState,
    *,
    t: int,
    particles_t: ArrayF,
    particles_prev: Optional[ArrayF],
    grad_t: ArrayF,
) -> float:
    """Update Fuse step size (Eq. 180) and return eta_t.

    Spec reference: docs/fuse_spec.md (Eq. 180, Algorithm 1).
    """
    state.validate()

    if particles_t.ndim != 2:
        raise ValueError("particles_t must be 2D.")
    if grad_t.shape != particles_t.shape:
        raise ValueError("grad_t must match particles_t shape.")
    if particles_prev is not None and particles_prev.shape != particles_t.shape:
        raise ValueError("particles_prev must match particles_t shape.")
    if not np.all(np.isfinite(particles_t)):
        raise ValueError("particles_t contains non-finite values.")
    if not np.all(np.isfinite(grad_t)):
        raise ValueError("grad_t contains non-finite values.")

    if state.eta == 0.0:
        state.eta = state.r_eps

    if not should_update(t):
        return state.eta

    if state.x1 is None:
        raise ValueError("x1 must be set before the first update (t >= 1).")
    if particles_prev is None:
        raise ValueError("particles_prev is required when t >= 1.")

    movement = np.mean(np.sum((state.x1 - particles_prev) ** 2, axis=1))
    if not np.isfinite(movement):
        raise ValueError("Non-finite movement term.")
    state.max_movement = max(state.max_movement, movement, state.r_eps)

    grad_norm_sq = np.mean(np.sum(grad_t ** 2, axis=1))
    if not np.isfinite(grad_norm_sq):
        raise ValueError("Non-finite gradient norm term.")
    state.sum_grad_norm_sq += grad_norm_sq

    if state.sum_grad_norm_sq <= 0.0:
        raise ValueError("sum_grad_norm_sq must be > 0.")

    state.eta = np.sqrt(state.max_movement / state.sum_grad_norm_sq)
    return state.eta
