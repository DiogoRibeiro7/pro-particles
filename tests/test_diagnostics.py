import numpy as np

from pro_particles.diagnostics.convergence import (
    effective_sample_size,
    running_mean,
    split_rhat,
    trajectory_summary,
)


def test_running_mean_constant() -> None:
    saved = np.ones((5, 3, 2), dtype=np.float64) * 2.5
    rm = running_mean(saved)
    assert rm.shape == (5, 2)
    assert np.allclose(rm, 2.5)


def test_effective_sample_size_iid() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(500, 2)).astype(np.float64)
    ess = effective_sample_size(samples)
    assert ess.shape == (2,)
    assert np.all(ess > 250.0)


def test_effective_sample_size_correlated() -> None:
    n = 200
    base = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    samples = np.repeat(base, 2, axis=1)
    ess = effective_sample_size(samples)
    assert ess.shape == (2,)
    assert np.all(ess <= 5.0)


def test_split_rhat_basic() -> None:
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(60, 2)).astype(np.float64)
    rhat = split_rhat(samples)
    assert rhat.shape == (2,)
    assert np.isfinite(rhat).all()


def test_trajectory_summary_shapes() -> None:
    rng = np.random.default_rng(2)
    saved = rng.normal(size=(10, 4, 3)).astype(np.float64)
    summary = trajectory_summary(saved)
    assert summary["step_means"].shape == (10, 3)
    assert summary["mean"].shape == (3,)
    assert summary["std"].shape == (3,)
    assert summary["min"].shape == (3,)
    assert summary["max"].shape == (3,)
