from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class WassersteinLangevinConfig:
    """Configuration for Wasserstein gradient flow via Langevin dynamics."""

    n_steps: int = 2_000
    step_size: float = 1e-2
    seed: Optional[int] = 0


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def run_wasserstein_langevin(
    *,
    init_particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    cfg: WassersteinLangevinConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Run overdamped Langevin dynamics (Wasserstein gradient flow of KL)."""
    particles = _ensure_2d(init_particles, "init_particles").copy()
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if cfg.step_size <= 0.0:
        raise ValueError("step_size must be > 0.")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    saved: list[ArrayF] = []
    for _ in range(cfg.n_steps):
        grads = np.stack([grad_logp(particles[j]) for j in range(particles.shape[0])], axis=0)
        noise = rng.normal(size=particles.shape)
        particles = particles + cfg.step_size * grads + np.sqrt(2.0 * cfg.step_size) * noise
        saved.append(particles.copy())

    return {
        "final_particles": particles,
        "saved_particles": np.stack(saved, axis=0),
    }
