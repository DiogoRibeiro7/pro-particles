from __future__ import annotations

import numpy as np

from pro_particles.baselines.gaussian_conjugate import gaussian_conjugate_posterior
from pro_particles.baselines.mala import run_baseline_mala_gaussian_location
from pro_particles.baselines.ula import run_baseline_ula_constant
from pro_particles.models.normal_location import normal_grad_logpdf_theta, normal_logpdf


def test_gaussian_conjugate_moments() -> None:
    x_obs = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    res = gaussian_conjugate_posterior(
        x_obs=x_obs,
        sigma=1.0,
        prior_mean=np.array([0.0]),
        prior_var=1.0,
    )
    assert res.mean.shape == (1,)
    assert res.cov.shape == (1, 1)
    assert np.isfinite(res.mean).all()
    assert np.isfinite(res.cov).all()


def test_ula_smoke() -> None:
    x_obs = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

    def log_target(theta: np.ndarray) -> float:
        return normal_logpdf(theta, x_obs, sigma=1.0).sum()

    def grad_log_target(theta: np.ndarray) -> np.ndarray:
        return normal_grad_logpdf_theta(theta, x_obs, sigma=1.0).sum(axis=0)

    samples = run_baseline_ula_constant(
        init=np.array([0.0]),
        grad_log_target=grad_log_target,
        dt=1e-3,
        n_steps=500,
        burn_in=100,
        thin=10,
        seed=0,
    )
    assert samples.shape[1] == 1
    assert np.isfinite(samples).all()


def test_mala_smoke() -> None:
    x_obs = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)

    def log_target(theta: np.ndarray) -> float:
        return normal_logpdf(theta, x_obs, sigma=1.0).sum()

    def grad_log_target(theta: np.ndarray) -> np.ndarray:
        return normal_grad_logpdf_theta(theta, x_obs, sigma=1.0).sum(axis=0)

    samples = run_baseline_mala_gaussian_location(
        init=np.array([0.0]),
        log_target=log_target,
        grad_log_target=grad_log_target,
        dt=1e-3,
        n_steps=500,
        burn_in=100,
        thin=10,
        seed=0,
    )
    assert samples.shape[1] == 1
    assert np.isfinite(samples).all()
