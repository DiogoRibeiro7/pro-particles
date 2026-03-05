from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pro_particles.kernels.rbf import gaussian_kernel, grad_gaussian_kernel_wrt_first_arg


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class SVGDConfig:
    """Configuration for SVGD updates."""

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


def svgd_update(
    *,
    particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    lengthscale: float,
) -> ArrayF:
    """Compute SVGD velocity field for particles.

    grad_logp(theta): returns gradient of log target for a single theta (d,).
    """
    particles = _ensure_2d(particles, "particles")
    p, d = particles.shape
    if p < 2:
        raise ValueError("Need at least 2 particles for SVGD.")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")

    grads = np.stack([grad_logp(particles[j]) for j in range(p)], axis=0)
    k_xx = gaussian_kernel(particles[:, None, :], particles[None, :, :], lengthscale)
    grad_k = grad_gaussian_kernel_wrt_first_arg(
        particles[:, None, :], particles[None, :, :], lengthscale
    )

    term1 = k_xx @ grads / float(p)
    term2 = grad_k.mean(axis=1)
    return term1 + term2


def run_svgd(
    *,
    init_particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    cfg: SVGDConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Run SVGD for a target defined by grad_logp."""
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
        particles = particles + cfg.step_size * velocity
        saved.append(particles.copy())

    return {
        "final_particles": particles,
        "saved_particles": np.stack(saved, axis=0),
    }
