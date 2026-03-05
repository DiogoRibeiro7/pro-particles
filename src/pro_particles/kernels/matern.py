from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def matern_kernel(y: ArrayF, z: ArrayF, lengthscale: float, nu: float) -> ArrayF:
    """Matérn kernel for nu in {1/2, 3/2, 5/2}.

    k(r) with r = ||y - z||:
    - nu = 1/2: exp(-r / l)
    - nu = 3/2: (1 + a r) exp(-a r), a = sqrt(3) / l
    - nu = 5/2: (1 + a r + b r^2) exp(-a r), a = sqrt(5) / l, b = 5 / (3 l^2)
    """
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")

    diff = y - z
    r = np.sqrt(np.sum(diff * diff, axis=-1))

    if nu == 0.5:
        a = 1.0 / lengthscale
        return np.exp(-a * r)
    if nu == 1.5:
        a = np.sqrt(3.0) / lengthscale
        return (1.0 + a * r) * np.exp(-a * r)
    if nu == 2.5:
        a = np.sqrt(5.0) / lengthscale
        b = 5.0 / (3.0 * lengthscale * lengthscale)
        return (1.0 + a * r + b * r * r) * np.exp(-a * r)

    raise ValueError("nu must be one of {0.5, 1.5, 2.5}.")


def grad_matern_kernel_wrt_first_arg(
    y: ArrayF, z: ArrayF, lengthscale: float, nu: float
) -> ArrayF:
    """Gradient wrt first argument of the Matérn kernel (nu in {1/2, 3/2, 5/2})."""
    if lengthscale <= 0.0:
        raise ValueError("lengthscale must be > 0.")

    diff = y - z
    r = np.sqrt(np.sum(diff * diff, axis=-1))

    if nu == 0.5:
        a = 1.0 / lengthscale
        k = np.exp(-a * r)
        kprime = -a * k
    elif nu == 1.5:
        a = np.sqrt(3.0) / lengthscale
        k = (1.0 + a * r) * np.exp(-a * r)
        kprime = -(a * a) * r * np.exp(-a * r)
    elif nu == 2.5:
        a = np.sqrt(5.0) / lengthscale
        b = 5.0 / (3.0 * lengthscale * lengthscale)
        exp_term = np.exp(-a * r)
        k = (1.0 + a * r + b * r * r) * exp_term
        kprime = exp_term * (2.0 * b * r - (a * a) * r - a * b * r * r)
    else:
        raise ValueError("nu must be one of {0.5, 1.5, 2.5}.")

    coeff = np.where(r > 0.0, kprime / r, 0.0)
    return coeff[..., None] * diff
