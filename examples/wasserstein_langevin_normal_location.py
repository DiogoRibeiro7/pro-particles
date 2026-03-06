from __future__ import annotations

import numpy as np

from pro_particles.algorithms.wasserstein_langevin import (
    WassersteinLangevinConfig,
    run_wasserstein_langevin,
)
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

    p = 64
    init_particles = rng.normal(0.0, 1.0, size=(p, 1)).astype(np.float64)
    cfg = WassersteinLangevinConfig(n_steps=2_000, step_size=5e-3, seed=7)

    out = run_wasserstein_langevin(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)
    print("Final particles shape:", out["final_particles"].shape)
    print("Posterior mean (approx):", float(out["final_particles"].mean()))


if __name__ == "__main__":
    main()
