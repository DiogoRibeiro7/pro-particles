from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pro_particles.algorithms.svgd import svgd_update


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class SVGDLangevinConfig:
    """Configuration for Langevin SVGD (SVGD with stochastic noise)."""

    n_steps: int = 2_000
    step_size: float = 1e-2
    seed: Optional[int] = 0
    lengthscale: float = 1.0


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def run_svgd_langevin(
    *,
    init_particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    cfg: SVGDLangevinConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Run Langevin SVGD updates."""
    particles = _ensure_2d(init_particles, "init_particles").copy()
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if cfg.step_size <= 0.0:
        raise ValueError("step_size must be > 0.")

    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    saved: list[ArrayF] = []
    for _ in range(cfg.n_steps):
        velocity = svgd_update(
            particles=particles,
            grad_logp=grad_logp,
            lengthscale=cfg.lengthscale,
        )
        noise = rng.normal(size=particles.shape)
        particles = particles + cfg.step_size * velocity + np.sqrt(2.0 * cfg.step_size) * noise
        saved.append(particles.copy())

    return {
        "final_particles": particles,
        "saved_particles": np.stack(saved, axis=0),
    }
