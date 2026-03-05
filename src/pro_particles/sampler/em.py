"""Convenience API for quick experiments. Delegates to spec_impl internally."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pro_particles.kernels.rbf import grad_gaussian_kernel_wrt_first_arg
from pro_particles.kernels.matern import grad_matern_kernel_wrt_first_arg
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.config import SpecConfig, AdaptiveStepConfig
from pro_particles.spec_impl.run_particle_system import run_particle_system


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class SamplerConfig:
    """Configuration for Euler–Maruyama particle simulation."""

    n_steps: int = 20_000
    burn_in: int = 5_000
    dt: float = 1e-3
    thin: int = 10
    seed: Optional[int] = 0
    adaptive_step: bool = False
    max_drift_step: float = 0.5
    dt_min: float = 1e-6
    dt_max: float = 1e-1
    adapt_eps: float = 1e-12


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


def sample_pro_posterior(
    *,
    init_particles: ArrayF,  # (p, d)
    x_obs: ArrayF,          # (n, xdim)
    lam_n: float,
    prior: GaussianPrior,
    cfg: SamplerConfig,
    mode: Literal["log_score", "mmd2", "energy_score", "crps"],
    logpdf: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
    grad_logpdf_theta: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
    sigma: float = 1.0,
    lengthscale: float = 1.0,
    m: int = 64,
    kernel: Literal["rbf", "matern12", "matern32", "matern52"] = "rbf",
    eps: float = 1e-8,
) -> Tuple[ArrayF, ArrayF]:
    """Run the interacting particle system and return samples after burn-in."""
    particles = _ensure_2d(init_particles, "init_particles").copy()
    x_obs2 = _ensure_2d(x_obs, "x_obs")

    if lam_n <= 0.0:
        raise ValueError("lam_n must be > 0.")
    if cfg.n_steps <= 0:
        raise ValueError("cfg.n_steps must be > 0.")
    if cfg.burn_in < 0 or cfg.burn_in >= cfg.n_steps:
        raise ValueError("cfg.burn_in must be in [0, n_steps).")
    if cfg.dt <= 0.0:
        raise ValueError("cfg.dt must be > 0.")
    if cfg.thin <= 0:
        raise ValueError("cfg.thin must be > 0.")
    if mode == "log_score" and (logpdf is None or grad_logpdf_theta is None):
        raise ValueError("logpdf and grad_logpdf_theta are required for mode='log_score'.")
    if mode in {"energy_score", "crps"} and sigma <= 0.0:
        raise ValueError("sigma must be > 0 for mode='energy_score' or mode='crps'.")
    if mode in {"mmd2", "energy_score", "crps"} and m < 2:
        raise ValueError("m must be >= 2 for Monte Carlo gradients.")
    if eps <= 0.0:
        raise ValueError("eps must be > 0.")

    rng = np.random.default_rng(cfg.seed)
    p, d = particles.shape

    adaptive_cfg = None
    if cfg.adaptive_step:
        adaptive_cfg = AdaptiveStepConfig(
            enabled=True,
            max_drift_step=cfg.max_drift_step,
            dt_min=cfg.dt_min,
            dt_max=cfg.dt_max,
            eps=cfg.adapt_eps,
        )

    spec_cfg = SpecConfig(
        p=p,
        dt_t=lambda _step: cfg.dt,
        B=cfg.burn_in,
        K=cfg.n_steps,
        thin=cfg.thin,
        seed=cfg.seed,
        lam_n=lam_n,
        adaptive_step=adaptive_cfg,
    )

    grad_L_mmd = None
    grad_L_energy = None
    grad_L_crps = None
    if mode == "mmd2":
        if x_obs2.shape[1] != d:
            raise ValueError("x_obs second dimension must match particle dimension d.")

        if kernel == "rbf":
            def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                return grad_gaussian_kernel_wrt_first_arg(y, z, lengthscale)
        elif kernel == "matern12":
            def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                return grad_matern_kernel_wrt_first_arg(y, z, lengthscale, nu=0.5)
        elif kernel == "matern32":
            def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                return grad_matern_kernel_wrt_first_arg(y, z, lengthscale, nu=1.5)
        elif kernel == "matern52":
            def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                return grad_matern_kernel_wrt_first_arg(y, z, lengthscale, nu=2.5)
        else:
            raise ValueError("kernel must be one of {'rbf','matern12','matern32','matern52'}.")

        def grad_L_mmd(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
            eps = rng.normal(size=(m, d))
            eps2 = rng.normal(size=(m, d))
            y = theta[None, :] + sigma * eps
            y2 = theta_prime[None, :] + sigma * eps2
            grad_k_yy2 = grad_kernel(y[:, None, :], y2[None, :, :]).mean(axis=(0, 1))
            grad_k_yx = grad_kernel(y, x_i[None, :]).mean(axis=0)
            return grad_k_yy2 - grad_k_yx
    if mode == "energy_score":
        if x_obs2.shape[1] != d:
            raise ValueError("x_obs second dimension must match particle dimension d.")

        def grad_L_energy(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
            eps1 = rng.normal(size=(m, d))
            eps2 = rng.normal(size=(m, d))
            y = theta[None, :] + sigma * eps1
            y2 = theta_prime[None, :] + sigma * eps2
            diff_yx = y - x_i[None, :]
            norm_yx = np.sqrt(np.sum(diff_yx * diff_yx, axis=1) + eps)
            grad_term1 = diff_yx / norm_yx[:, None]
            diff_yy2 = y - y2
            norm_yy2 = np.sqrt(np.sum(diff_yy2 * diff_yy2, axis=1) + eps)
            grad_term2 = diff_yy2 / norm_yy2[:, None]
            return grad_term1.mean(axis=0) - grad_term2.mean(axis=0)

    if mode == "crps":
        if d != 1 or x_obs2.shape[1] != 1:
            raise ValueError("crps mode requires d=1 and x_obs shape (n, 1).")

        def grad_L_crps(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
            eps1 = rng.normal(size=(m, 1))
            eps2 = rng.normal(size=(m, 1))
            y = theta[None, :] + sigma * eps1
            y2 = theta_prime[None, :] + sigma * eps2
            diff_yx = y[:, 0] - float(x_i[0])
            grad_term1 = np.where(diff_yx >= 0.0, 1.0, -1.0)
            diff_yy2 = y[:, 0] - y2[:, 0]
            grad_term2 = np.where(diff_yy2 >= 0.0, 1.0, -1.0)
            return np.array([float(grad_term1.mean() - grad_term2.mean())], dtype=np.float64)

    out = run_particle_system(
        init_particles=particles,
        x_obs=x_obs2,
        cfg=spec_cfg,
        prior=prior,
        rule=mode,
        use_fuse=False,
        leave_one_out=True,
        logpdf=logpdf,
        grad_logpdf_theta=grad_logpdf_theta,
        grad_L_mmd=grad_L_mmd,
        grad_L_energy=grad_L_energy,
        grad_L_crps=grad_L_crps,
        rng=rng,
    )

    return out["final_particles"], out["time_avg_samples"]
