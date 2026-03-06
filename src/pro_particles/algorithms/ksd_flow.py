from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from pro_particles.kernels.rbf import gaussian_kernel, grad_gaussian_kernel_wrt_first_arg


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class KSDFlowConfig:
    """Configuration for KSD gradient flow (RBF kernel)."""

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


def ksd_rbf(
    *,
    particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    lengthscale: float,
) -> float:
    """Compute empirical KSD with an RBF kernel.

    This uses the standard Stein kernel for the target defined by grad_logp.
    """
    particles = _ensure_2d(particles, "particles")
    p, _ = particles.shape
    if p < 2:
        raise ValueError("Need at least 2 particles for KSD.")
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")

    grads = np.stack([grad_logp(particles[j]) for j in range(p)], axis=0)
    k_xx = gaussian_kernel(particles[:, None, :], particles[None, :, :], lengthscale)
    grad_k = grad_gaussian_kernel_wrt_first_arg(
        particles[:, None, :], particles[None, :, :], lengthscale
    )
    # Stein kernel trace term
    term1 = (grads @ grads.T) * k_xx
    term2 = np.einsum("ijd,jd->ij", grad_k, grads)
    term3 = np.einsum("ijd,id->ij", grad_k, grads)
    diff = particles[:, None, :] - particles[None, :, :]
    sq = np.sum(diff * diff, axis=-1)
    l2 = lengthscale * lengthscale
    term4 = k_xx * (particles.shape[1] / l2 - sq / (l2 * l2))
    ksd2 = (term1 + term2 + term3 + term4).mean()
    return float(np.sqrt(max(ksd2, 0.0)))


def ksd_flow_update(
    *,
    particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    lengthscale: float,
) -> ArrayF:
    """Gradient flow update using Stein kernel (same structure as SVGD)."""
    particles = _ensure_2d(particles, "particles")
    p, d = particles.shape
    grads = np.stack([grad_logp(particles[j]) for j in range(p)], axis=0)
    k_xx = gaussian_kernel(particles[:, None, :], particles[None, :, :], lengthscale)
    grad_k = grad_gaussian_kernel_wrt_first_arg(
        particles[:, None, :], particles[None, :, :], lengthscale
    )
    term1 = k_xx @ grads / float(p)
    term2 = grad_k.mean(axis=1)
    return term1 + term2


def run_ksd_flow(
    *,
    init_particles: ArrayF,
    grad_logp: Callable[[ArrayF], ArrayF],
    cfg: KSDFlowConfig,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, ArrayF]:
    """Run a KSD gradient flow (deterministic)."""
    particles = _ensure_2d(init_particles, "init_particles").copy()
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if cfg.step_size <= 0.0:
        raise ValueError("step_size must be > 0.")
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    _ = rng  # deterministic flow, seed kept for API parity

    saved: list[ArrayF] = []
    for _ in range(cfg.n_steps):
        velocity = ksd_flow_update(
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
