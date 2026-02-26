from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.fast_impl.mmd2 import wq_mmd2_gaussian_location_fast


def test_fast_mmd2_leave_one_out_shape_non_square() -> None:
    rng = np.random.default_rng(0)
    p, d = 5, 3
    n = 4

    particles = rng.normal(size=(p, d)).astype(np.float64)
    x_obs = rng.normal(size=(n, d)).astype(np.float64)

    wq = wq_mmd2_gaussian_location_fast(
        particles=particles,
        x_obs=x_obs,
        sigma=1.0,
        lengthscale=1.5,
        m=8,
        rng=rng,
        leave_one_out=True,
    )

    assert wq.shape == (p, d)
    assert np.all(np.isfinite(wq))
