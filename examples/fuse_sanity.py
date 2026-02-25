from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf
from pro_particles.priors.gaussian import GaussianPrior
from pro_particles.schedules.fuse import FuseState
from pro_particles.spec_impl import SpecConfig
from pro_particles.spec_impl.run import run_em_fuse


def drift_and_grad(particles, x_obs, lam_n, prior):
    logp = np.array([normal_logpdf(theta, x_obs, sigma=1.0) for theta in particles])
    gradlogp = np.array([normal_grad_logpdf_theta(theta, x_obs, sigma=1.0) for theta in particles])
    dP = np.exp(logp)
    grad_dP = dP[:, :, None] * gradlogp

    denom = dP.mean(axis=0)
    wq = (grad_dP / denom[None, :, None]).mean(axis=1)
    drift = -(lam_n * wq - np.array([prior.grad_log_pdf(theta) for theta in particles]))
    return drift, drift


def main() -> None:
    rng = np.random.default_rng(0)
    x_obs = rng.normal(loc=0.5, scale=1.0, size=(20, 1)).astype(np.float64)
    init_particles = rng.normal(size=(8, 1)).astype(np.float64)

    cfg = SpecConfig(
        p=8,
        dt_t=lambda step: 1.0,  # Fuse provides eta
        B=10,
        K=200,
        thin=5,
        seed=0,
        lam_n=1.0,
        sqrt2=np.sqrt(2.0),
    )

    fuse_state = FuseState(r_eps=1e-3)

    out = run_em_fuse(
        init_particles=init_particles,
        x_obs=x_obs,
        cfg=cfg,
        prior=GaussianPrior(prior_var=10.0),
        drift_and_grad_fn=drift_and_grad,
        fuse_state=fuse_state,
        rng=rng,
    )

    etas = fuse_state.eta
    final_particles = out["final_particles"]
    print("eta_final:", etas)
    print("final_particles_finite:", np.isfinite(final_particles).all())


if __name__ == "__main__":
    main()
