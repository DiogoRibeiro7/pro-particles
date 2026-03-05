from __future__ import annotations

import numpy as np

from pro_particles.algorithms.svgd import SVGDConfig, run_svgd
from pro_particles.models.logistic_regression import logistic_grad_logpdf_theta


def main() -> None:
    rng = np.random.default_rng(123)

    # Synthetic logistic regression data
    n = 400
    d = 3
    x = rng.normal(size=(n, d)).astype(np.float64)
    true_beta = np.array([0.8, -1.1, 0.4], dtype=np.float64)
    logits = x @ true_beta
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.float64)
    x_obs = np.concatenate([x, y[:, None]], axis=1)

    # Simple isotropic Gaussian prior on beta
    prior_var = 9.0

    def grad_logp(theta: np.ndarray) -> np.ndarray:
        grad_ll = logistic_grad_logpdf_theta(theta, x_obs).sum(axis=0)
        grad_prior = -theta / prior_var
        return grad_ll + grad_prior

    p = 64
    init_particles = rng.normal(0.0, 1.0, size=(p, d)).astype(np.float64)
    cfg = SVGDConfig(n_steps=2_000, step_size=5e-3, seed=7, lengthscale=1.0)

    out = run_svgd(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)
    print("Final particles shape:", out["final_particles"].shape)
    print("Posterior mean (approx):", out["final_particles"].mean(axis=0))


if __name__ == "__main__":
    main()
