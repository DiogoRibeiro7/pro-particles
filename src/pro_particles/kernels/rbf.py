from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def gaussian_kernel(y: ArrayF, z: ArrayF, lengthscale: float) -> ArrayF:
    """Gaussian/RBF kernel k(y,z) = exp(-||y-z||^2 / (2 l^2))."""
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")
    diff = y - z
    return np.exp(-0.5 * np.sum(diff * diff, axis=-1) / (lengthscale * lengthscale))


def grad_gaussian_kernel_wrt_first_arg(y: ArrayF, z: ArrayF, lengthscale: float) -> ArrayF:
    """Gradient wrt first argument of the RBF kernel.

    For RBF:
        ∇_y k(y,z) = k(y,z) * (z - y) / l^2
    """
    k = gaussian_kernel(y, z, lengthscale)
    return (k[..., None] * (z - y)) / (lengthscale * lengthscale)
