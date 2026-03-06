from __future__ import annotations

import numpy as np

from pro_particles.algorithms.ksd_flow import KSDFlowConfig, ksd_rbf, run_ksd_flow
from pro_particles.models.normal_location import normal_grad_logpdf_theta


def main() -> None:
    rng = np.random.default_rng(123)

    n = 300
    x = rng.normal(0.0, 1.0, size=n)
    x_obs = x.reshape(-1, 1).astype(np.float64)

    prior_var = 9.0
    sigma = 1.0

    def grad_logp(theta: np.ndarray) -> np.ndarray:
        grad_ll = normal_grad_logpdf_theta(theta, x_obs, sigma).sum(axis=0)
        grad_prior = -theta / prior_var
        return grad_ll + grad_prior

    p = 48
    init_particles = rng.normal(0.0, 1.0, size=(p, 1)).astype(np.float64)
    cfg = KSDFlowConfig(n_steps=1_500, step_size=1e-2, seed=7, lengthscale=1.0)

    out = run_ksd_flow(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)
    ksd = ksd_rbf(particles=out["final_particles"], grad_logp=grad_logp, lengthscale=1.0)
    print("Final particles shape:", out["final_particles"].shape)
    print("KSD (approx):", float(ksd))


if __name__ == "__main__":
    main()
