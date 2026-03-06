from __future__ import annotations

import numpy as np

from pro_particles.algorithms.ksd_flow import KSDFlowConfig, run_ksd_flow
from pro_particles.algorithms.svgd_langevin import SVGDLangevinConfig, run_svgd_langevin
from pro_particles.algorithms.wasserstein_langevin import (
    WassersteinLangevinConfig,
    run_wasserstein_langevin,
)


def test_svgd_langevin_smoke() -> None:
    rng = np.random.default_rng(0)
    init_particles = rng.normal(size=(12, 2)).astype(np.float64)

    def grad_logp(theta: np.ndarray) -> np.ndarray:
        return -theta

    cfg = SVGDLangevinConfig(n_steps=30, step_size=1e-2, seed=0, lengthscale=1.0)
    out = run_svgd_langevin(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)
    assert out["final_particles"].shape == (12, 2)
    assert np.isfinite(out["final_particles"]).all()


def test_ksd_flow_smoke() -> None:
    rng = np.random.default_rng(1)
    init_particles = rng.normal(size=(10, 1)).astype(np.float64)

    def grad_logp(theta: np.ndarray) -> np.ndarray:
        return -theta

    cfg = KSDFlowConfig(n_steps=20, step_size=1e-2, seed=1, lengthscale=1.0)
    out = run_ksd_flow(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)
    assert out["final_particles"].shape == (10, 1)
    assert np.isfinite(out["final_particles"]).all()


def test_wasserstein_langevin_smoke() -> None:
    rng = np.random.default_rng(2)
    init_particles = rng.normal(size=(10, 2)).astype(np.float64)

    def grad_logp(theta: np.ndarray) -> np.ndarray:
        return -theta

    cfg = WassersteinLangevinConfig(n_steps=20, step_size=1e-2, seed=2)
    out = run_wasserstein_langevin(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)
    assert out["final_particles"].shape == (10, 2)
    assert np.isfinite(out["final_particles"]).all()
