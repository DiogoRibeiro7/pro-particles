from __future__ import annotations

import numpy as np

from pro_particles.models.hierarchical_gaussian import (
    hierarchical_gaussian_grad_logpdf_theta,
    hierarchical_gaussian_logpdf,
)
from pro_particles.priors.hierarchical_gaussian import HierarchicalGaussianPrior
from pro_particles.sampler.em import SamplerConfig, sample_pro_posterior


def main() -> None:
    rng = np.random.default_rng(123)

    # Synthetic hierarchical Gaussian data
    n_groups = 6
    n_per_group = 80
    sigma_y = 0.7
    sigma_group = 1.0
    alpha_true = 0.5

    mu_true = rng.normal(alpha_true, sigma_group, size=n_groups)
    y_list = []
    for g in range(n_groups):
        y_g = rng.normal(mu_true[g], sigma_y, size=n_per_group)
        group_ids = np.full(n_per_group, g)
        y_list.append(np.column_stack([group_ids, y_g]))
    x_obs = np.concatenate(y_list, axis=0).astype(np.float64)

    d = 1 + n_groups
    p = 32
    prior = HierarchicalGaussianPrior(
        n_groups=n_groups, sigma_alpha=2.0, sigma_group=sigma_group
    )
    init_particles = rng.normal(0.0, 1.0, size=(p, d)).astype(np.float64)

    cfg = SamplerConfig(n_steps=10_000, burn_in=2_000, dt=8e-4, thin=10, seed=7)
    lam_n = float(np.sqrt(x_obs.shape[0]))

    _, samples = sample_pro_posterior(
        init_particles=init_particles,
        x_obs=x_obs,
        lam_n=lam_n,
        prior=prior,  # type: ignore[arg-type]
        cfg=cfg,
        mode="log_score",
        logpdf=lambda th, xx: hierarchical_gaussian_logpdf(th, xx, sigma_y),
        grad_logpdf_theta=lambda th, xx: hierarchical_gaussian_grad_logpdf_theta(
            th, xx, sigma_y
        ),
    )

    mean = samples.mean(axis=0)
    print("Collected samples shape:", samples.shape)
    print("Posterior mean alpha (approx):", float(mean[0]))
    print("Posterior mean group means (approx):", mean[1:])


if __name__ == "__main__":
    main()
