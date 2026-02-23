from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def normal_logpdf(theta: ArrayF, x: ArrayF, sigma: float) -> ArrayF:
    """Log N(x; mean=theta, var=sigma^2), returns shape (n,).

    Parameters
    ----------
    theta:
        Shape (d,), intended d=1 for this example.
    x:
        Shape (n, 1).
    sigma:
        Positive standard deviation.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if theta.ndim != 1:
        raise ValueError("theta must be 1D (d,).")
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("x must be 2D with shape (n, 1).")

    x1 = x[:, 0]
    z = (x1 - float(theta[0])) / sigma
    return -0.5 * (z * z) - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)


def normal_grad_logpdf_theta(theta: ArrayF, x: ArrayF, sigma: float) -> ArrayF:
    """∇_theta log N(x; theta, sigma^2) = (x - theta) / sigma^2.

    Returns shape (n, d) with d=1 for this example.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if theta.ndim != 1:
        raise ValueError("theta must be 1D (d,).")
    if x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("x must be 2D with shape (n, 1).")

    grad = (x[:, 0] - float(theta[0])) / (sigma * sigma)
    return grad.reshape(-1, 1).astype(np.float64, copy=False)
