from __future__ import annotations

# ruff: noqa: E402

import json
from pathlib import Path
import sys

# Path setup for standalone execution (not needed under `poetry run`)
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

import numpy as np

from pro_particles.kernels.rbf import grad_gaussian_kernel_wrt_first_arg
from pro_particles.models.normal_location import normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.spec_impl import SpecConfig, run_particle_system
from pro_particles.fast_impl.run_mmd2 import run_particle_system_mmd2_fast
from pro_particles.metrics.metrics import elpd_from_lppd, lppd_from_logpdf
from pro_particles.experiments.data import make_d1_normal_location_data
from pro_particles.schedules.fuse import FuseState

from experiments.utils import (
    env_info,
    git_commit,
    is_dirty,
    median_heuristic_lengthscale,
    save_json,
    sha256_array,
    sha256_file,
    utc_now,
)


def grad_L_mmd_gaussian_location(
    theta: np.ndarray,
    theta2: np.ndarray,
    x: np.ndarray,
    *,
    sigma: float,
    lengthscale: float,
    m: int,
    rng: np.random.Generator,
) -> np.ndarray:
    eps1 = rng.normal(size=(m, theta.shape[0]))
    eps2 = rng.normal(size=(m, theta.shape[0]))
    y = theta[None, :] + sigma * eps1
    y2 = theta2[None, :] + sigma * eps2

    y_exp = y[:, None, :]
    y2_exp = y2[None, :, :]
    term1 = grad_gaussian_kernel_wrt_first_arg(y_exp, y2_exp, lengthscale).mean(axis=(0, 1))

    term2 = grad_gaussian_kernel_wrt_first_arg(y, x[None, :], lengthscale).mean(axis=0)
    return term1 - term2


def main() -> None:
    exp_dir = Path(__file__).resolve().parent
    cfg_path = exp_dir / "config.json"
    if len(sys.argv) > 1:
        cfg_path = Path(sys.argv[1])
    cfg = json.loads(cfg_path.read_text())

    seed = int(cfg["data"]["seed"])
    n = int(cfg["data"]["n"])
    sigma = float(cfg["data"]["sigma"])

    x_obs = make_d1_normal_location_data(
        seed=seed,
        regime=cfg["data"]["regime"],
        n=n,
        sigma=sigma,
    )

    lengthscale = median_heuristic_lengthscale(x_obs)

    lam_n = float(n)
    rng = np.random.default_rng(seed)
    init_particles = rng.normal(size=(cfg["method"]["p"], 1)).astype(np.float64)

    spec_cfg = SpecConfig(
        p=cfg["method"]["p"],
        dt_t=lambda step: float(cfg["method"].get("dt", 1.0)),
        B=cfg["method"]["B"],
        K=cfg["method"]["K"],
        thin=cfg["method"]["thin"],
        seed=seed,
        lam_n=lam_n,
    )

    fuse_state = FuseState(r_eps=cfg["method"]["r_eps"]) if cfg["method"]["use_fuse"] else None

    def grad_L(theta, theta2, x):
        return grad_L_mmd_gaussian_location(
            theta,
            theta2,
            x,
            sigma=sigma,
            lengthscale=lengthscale,
            m=cfg["method"]["m_mmd"],
            rng=rng,
        )

    def fuse_grad_fn(particles, x, lam_n, prior):
        p = particles.shape[0]
        n = x.shape[0]
        wq = np.zeros_like(particles)
        for j in range(p):
            acc = np.zeros((n, particles.shape[1]), dtype=np.float64)
            for ell in range(p):
                if ell == j:
                    continue
                for i in range(n):
                    acc[i] += grad_L(particles[j], particles[ell], x[i])
            acc /= float(p - 1)
            wq[j] = acc.mean(axis=0)
        prior_grad = np.array([prior.grad_log_pdf(theta) for theta in particles])
        return lam_n * wq - prior_grad

    if cfg["method"].get("impl", "fast") == "fast":
        out = run_particle_system_mmd2_fast(
            init_particles=init_particles,
            x_obs=x_obs,
            y_obs=None,
            cfg=spec_cfg,
            prior=GaussianPrior(prior_var=10.0),
            sigma=sigma,
            lengthscale=lengthscale,
            m=cfg["method"]["m_mmd"],
            use_fuse=cfg["method"]["use_fuse"],
            leave_one_out=True,
            fuse_state=fuse_state,
            rng=rng,
            model="gaussian_location",
        )
    else:
        out = run_particle_system(
            init_particles=init_particles,
            x_obs=x_obs,
            cfg=spec_cfg,
            prior=GaussianPrior(prior_var=10.0),
            rule="mmd2",
            use_fuse=cfg["method"]["use_fuse"],
            leave_one_out=True,
            grad_L_mmd=grad_L,
            fuse_state=fuse_state,
            fuse_grad_fn=fuse_grad_fn if cfg["method"]["use_fuse"] else None,
            rng=rng,
        )

    samples = out["time_avg_samples"]
    samples_path = exp_dir / "samples_method.npz"
    np.savez(samples_path, samples=samples)

    logpdf = np.vstack([normal_logpdf(theta, x_obs, sigma=sigma) for theta in samples])
    lppd = lppd_from_logpdf(logpdf)
    elpd = elpd_from_lppd(lppd)

    result = {
        "schema_version": "1.0.0",
        "paper_id": cfg["paper_id"],
        "exp_id": cfg["exp_id"],
        "exp_name": cfg["exp_name"],
        "created_at_utc": utc_now(),
        "code": {"repo": "pro-particles", "git_commit": git_commit(), "dirty": is_dirty()},
        "environment": env_info(),
        "data": {
            "generator": "make_d1_normal_location_data",
            "seed": seed,
            "n": n,
            "d_x": 1,
            "notes": cfg["data"]["regime"],
            "params": {"sigma": sigma},
            "hash": sha256_array(x_obs),
        },
        "configs": {
            "method": {
                "name": "pro_particles",
                "rule": cfg["method"]["rule"],
                "lam_n": {"type": "paper_default", "value": lam_n},
                "prior": {"type": "gaussian_isotropic", "prior_var": 10.0},
                "schedule": {"type": "fuse", "params": {"r_eps": cfg["method"]["r_eps"]}},
                "sampler": {
                    "p": cfg["method"]["p"],
                    "burn_in": cfg["method"]["B"],
                    "n_steps": cfg["method"]["K"],
                    "thin": cfg["method"]["thin"],
                },
                "model": {"family": "normal_location", "sigma": sigma},
            },
            "baselines": [],
        },
        "runs": [
            {
                "run_id": "method",
                "kind": "method",
                "name": "pro_particles",
                "seed": seed,
                "status": "ok",
                "timing": {},
                "samples": {
                    "shape": list(samples.shape),
                    "n_saved": int(samples.shape[0]),
                    "path": str(samples_path.name),
                    "hash": sha256_file(samples_path),
                },
                "diagnostics": {},
                "metrics": {
                    "posterior_mean": samples.mean(axis=0).tolist(),
                    "posterior_std": samples.std(axis=0).tolist(),
                    "elpd": elpd,
                },
            }
        ],
        "summary": {},
    }

    save_json(exp_dir / "results.json", result)


if __name__ == "__main__":
    main()
