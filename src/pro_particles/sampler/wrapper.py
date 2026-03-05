from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from pro_particles.diagnostics.convergence import (
    effective_sample_size,
    running_mean,
    split_rhat,
    trace_plot_data,
    trajectory_summary,
)
from pro_particles.kernels.matern import grad_matern_kernel_wrt_first_arg
from pro_particles.kernels.rbf import grad_gaussian_kernel_wrt_first_arg
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl.config import SpecConfig, AdaptiveStepConfig
from pro_particles.spec_impl.run_particle_system import run_particle_system
from pro_particles.sampler.em import SamplerConfig


ArrayF = NDArray[np.float64]


@dataclass(frozen=True)
class SamplerDiagnostics:
    trace: Dict[str, ArrayF]
    running_mean: ArrayF
    ess: ArrayF
    rhat: ArrayF
    trajectory: Dict[str, ArrayF]


@dataclass(frozen=True)
class SamplerResult:
    final_particles: ArrayF
    saved_particles: ArrayF
    time_avg_samples: ArrayF
    diagnostics: SamplerDiagnostics


def _ensure_2d(x: ArrayF, name: str) -> ArrayF:
    if x.ndim == 1:
        return x.reshape(-1, 1).astype(np.float64, copy=False)
    if x.ndim == 2:
        return x.astype(np.float64, copy=False)
    raise ValueError(f"{name} must be 1D or 2D, got shape {x.shape}.")


class ProSampler:
    """Higher-level sampler wrapper with config, seeding, and diagnostics."""

    def __init__(
        self,
        *,
        mode: Literal["log_score", "mmd2", "energy_score", "crps"],
        lam_n: float,
        prior: GaussianPrior,
        cfg: SamplerConfig,
        logpdf: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
        grad_logpdf_theta: Optional[Callable[[ArrayF, ArrayF], ArrayF]] = None,
        sigma: float = 1.0,
        lengthscale: float = 1.0,
        m: int = 64,
        kernel: Literal["rbf", "matern12", "matern32", "matern52"] = "rbf",
        eps: float = 1e-8,
    ) -> None:
        self.mode = mode
        self.lam_n = lam_n
        self.prior = prior
        self.cfg = cfg
        self.logpdf = logpdf
        self.grad_logpdf_theta = grad_logpdf_theta
        self.sigma = sigma
        self.lengthscale = lengthscale
        self.m = m
        self.kernel = kernel
        self.eps = eps

        if lam_n <= 0.0:
            raise ValueError("lam_n must be > 0.")
        if mode == "log_score" and (logpdf is None or grad_logpdf_theta is None):
            raise ValueError("logpdf and grad_logpdf_theta are required for mode='log_score'.")
        if mode in {"mmd2", "energy_score", "crps"} and m < 2:
            raise ValueError("m must be >= 2 for Monte Carlo gradients.")
        if mode in {"energy_score", "crps"} and sigma <= 0.0:
            raise ValueError("sigma must be > 0 for mode='energy_score' or mode='crps'.")
        if eps <= 0.0:
            raise ValueError("eps must be > 0.")

    def sample(
        self,
        *,
        init_particles: ArrayF,
        x_obs: ArrayF,
    ) -> SamplerResult:
        particles = _ensure_2d(init_particles, "init_particles").copy()
        x_obs2 = _ensure_2d(x_obs, "x_obs")

        rng = np.random.default_rng(self.cfg.seed)
        p, d = particles.shape

        adaptive_cfg = None
        if self.cfg.adaptive_step:
            adaptive_cfg = AdaptiveStepConfig(
                enabled=True,
                max_drift_step=self.cfg.max_drift_step,
                dt_min=self.cfg.dt_min,
                dt_max=self.cfg.dt_max,
                eps=self.cfg.adapt_eps,
            )

        spec_cfg = SpecConfig(
            p=p,
            dt_t=lambda _step: self.cfg.dt,
            B=self.cfg.burn_in,
            K=self.cfg.n_steps,
            thin=self.cfg.thin,
            seed=self.cfg.seed,
            lam_n=self.lam_n,
            adaptive_step=adaptive_cfg,
        )

        grad_L_mmd = None
        grad_L_energy = None
        grad_L_crps = None

        if self.mode == "mmd2":
            if x_obs2.shape[1] != d:
                raise ValueError("x_obs second dimension must match particle dimension d.")

            if self.kernel == "rbf":
                def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                    return grad_gaussian_kernel_wrt_first_arg(y, z, self.lengthscale)
            elif self.kernel == "matern12":
                def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                    return grad_matern_kernel_wrt_first_arg(y, z, self.lengthscale, nu=0.5)
            elif self.kernel == "matern32":
                def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                    return grad_matern_kernel_wrt_first_arg(y, z, self.lengthscale, nu=1.5)
            elif self.kernel == "matern52":
                def grad_kernel(y: ArrayF, z: ArrayF) -> ArrayF:
                    return grad_matern_kernel_wrt_first_arg(y, z, self.lengthscale, nu=2.5)
            else:
                raise ValueError("kernel must be one of {'rbf','matern12','matern32','matern52'}.")

            def grad_L_mmd(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
                eps = rng.normal(size=(self.m, d))
                eps2 = rng.normal(size=(self.m, d))
                y = theta[None, :] + self.sigma * eps
                y2 = theta_prime[None, :] + self.sigma * eps2
                grad_k_yy2 = grad_kernel(y[:, None, :], y2[None, :, :]).mean(axis=(0, 1))
                grad_k_yx = grad_kernel(y, x_i[None, :]).mean(axis=0)
                return grad_k_yy2 - grad_k_yx

        if self.mode == "energy_score":
            if x_obs2.shape[1] != d:
                raise ValueError("x_obs second dimension must match particle dimension d.")

            def grad_L_energy(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
                eps1 = rng.normal(size=(self.m, d))
                eps2 = rng.normal(size=(self.m, d))
                y = theta[None, :] + self.sigma * eps1
                y2 = theta_prime[None, :] + self.sigma * eps2
                diff_yx = y - x_i[None, :]
                norm_yx = np.sqrt(np.sum(diff_yx * diff_yx, axis=1) + self.eps)
                grad_term1 = diff_yx / norm_yx[:, None]
                diff_yy2 = y - y2
                norm_yy2 = np.sqrt(np.sum(diff_yy2 * diff_yy2, axis=1) + self.eps)
                grad_term2 = diff_yy2 / norm_yy2[:, None]
                return grad_term1.mean(axis=0) - grad_term2.mean(axis=0)

        if self.mode == "crps":
            if d != 1 or x_obs2.shape[1] != 1:
                raise ValueError("crps mode requires d=1 and x_obs shape (n, 1).")

            def grad_L_crps(theta: ArrayF, theta_prime: ArrayF, x_i: ArrayF) -> ArrayF:
                eps1 = rng.normal(size=(self.m, 1))
                eps2 = rng.normal(size=(self.m, 1))
                y = theta[None, :] + self.sigma * eps1
                y2 = theta_prime[None, :] + self.sigma * eps2
                diff_yx = y[:, 0] - float(x_i[0])
                grad_term1 = np.where(diff_yx >= 0.0, 1.0, -1.0)
                diff_yy2 = y[:, 0] - y2[:, 0]
                grad_term2 = np.where(diff_yy2 >= 0.0, 1.0, -1.0)
                return np.array([float(grad_term1.mean() - grad_term2.mean())], dtype=np.float64)

        out = run_particle_system(
            init_particles=particles,
            x_obs=x_obs2,
            cfg=spec_cfg,
            prior=self.prior,
            rule=self.mode,
            use_fuse=False,
            leave_one_out=True,
            logpdf=self.logpdf,
            grad_logpdf_theta=self.grad_logpdf_theta,
            grad_L_mmd=grad_L_mmd,
            grad_L_energy=grad_L_energy,
            grad_L_crps=grad_L_crps,
            rng=rng,
        )

        saved_particles = out["saved_particles"]
        trace = trace_plot_data(saved_particles) if saved_particles.size else {
            "step": np.empty((0,), dtype=np.float64),
            "particle_means": np.empty((0, particles.shape[1]), dtype=np.float64),
            "particle_stds": np.empty((0, particles.shape[1]), dtype=np.float64),
        }
        run_mean = running_mean(saved_particles) if saved_particles.size else np.empty(
            (0, particles.shape[1]), dtype=np.float64
        )
        if out["time_avg_samples"].shape[0] >= 2:
            ess = effective_sample_size(out["time_avg_samples"])
        else:
            ess = np.full((particles.shape[1],), np.nan, dtype=np.float64)

        if out["time_avg_samples"].shape[0] >= 4:
            rhat = split_rhat(out["time_avg_samples"])
        else:
            rhat = np.full((particles.shape[1],), np.nan, dtype=np.float64)

        trajectory = trajectory_summary(saved_particles) if saved_particles.size else {
            "step_means": np.empty((0, particles.shape[1]), dtype=np.float64),
            "mean": np.empty((particles.shape[1],), dtype=np.float64),
            "std": np.empty((particles.shape[1],), dtype=np.float64),
            "min": np.empty((particles.shape[1],), dtype=np.float64),
            "max": np.empty((particles.shape[1],), dtype=np.float64),
        }

        diagnostics = SamplerDiagnostics(
            trace=trace, running_mean=run_mean, ess=ess, rhat=rhat, trajectory=trajectory
        )
        return SamplerResult(
            final_particles=out["final_particles"],
            saved_particles=saved_particles,
            time_avg_samples=out["time_avg_samples"],
            diagnostics=diagnostics,
        )
