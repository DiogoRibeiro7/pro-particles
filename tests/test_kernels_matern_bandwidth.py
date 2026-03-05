from __future__ import annotations

import numpy as np

from pro_particles.kernels.matern import matern_kernel, grad_matern_kernel_wrt_first_arg
from pro_particles.kernels.bandwidth import (
    median_heuristic_lengthscale,
    scott_bandwidth,
    silverman_bandwidth,
)


def test_matern_kernel_shapes_and_grad_zero_at_equal() -> None:
    y = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    z = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)

    for nu in (0.5, 1.5, 2.5):
        k = matern_kernel(y, z, lengthscale=1.2, nu=nu)
        assert k.shape == (2,)
        grad = grad_matern_kernel_wrt_first_arg(y, z, lengthscale=1.2, nu=nu)
        assert grad.shape == (2, 2)
        assert np.allclose(grad, 0.0)


def test_bandwidth_utilities_positive() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(200, 3)).astype(np.float64)

    med = median_heuristic_lengthscale(samples, max_points=128)
    assert med > 0.0

    scott = scott_bandwidth(samples)
    silverman = silverman_bandwidth(samples)
    assert scott > 0.0
    assert silverman > 0.0
