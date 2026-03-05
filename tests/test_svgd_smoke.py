from __future__ import annotations

import numpy as np

from pro_particles.algorithms.svgd import SVGDConfig, run_svgd


def test_svgd_smoke() -> None:
    rng = np.random.default_rng(0)
    init_particles = rng.normal(size=(10, 2)).astype(np.float64)

    def grad_logp(theta: np.ndarray) -> np.ndarray:
        return -theta

    cfg = SVGDConfig(n_steps=50, step_size=1e-2, seed=0, lengthscale=1.0)
    out = run_svgd(init_particles=init_particles, grad_logp=grad_logp, cfg=cfg)

    assert out["final_particles"].shape == (10, 2)
    assert out["saved_particles"].shape[0] == cfg.n_steps
    assert np.isfinite(out["final_particles"]).all()
