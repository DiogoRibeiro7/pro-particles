from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


def logistic_logpdf(theta: ArrayF, x_obs: ArrayF) -> ArrayF:
    """Log-likelihood per observation for logistic regression.

    Parameters
    ----------
    theta:
        Shape (d,), regression coefficients.
    x_obs:
        Shape (n, d + 1), with last column containing y in {0,1}.
    """
    if theta.ndim != 1:
        raise ValueError("theta must be 1D (d,).")
    if x_obs.ndim != 2:
        raise ValueError("x_obs must be 2D (n, d+1).")
    if x_obs.shape[1] != theta.shape[0] + 1:
        raise ValueError("x_obs must have d+1 columns (features + label).")

    x = x_obs[:, :-1]
    y = x_obs[:, -1]
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must be in {0,1}.")

    logits = x @ theta
    # log sigmoid in a stable way
    log_sigmoid = -np.logaddexp(0.0, -logits)
    log_one_minus = -np.logaddexp(0.0, logits)
    return y * log_sigmoid + (1.0 - y) * log_one_minus


def logistic_grad_logpdf_theta(theta: ArrayF, x_obs: ArrayF) -> ArrayF:
    """Gradient of log-likelihood per observation, shape (n, d)."""
    if theta.ndim != 1:
        raise ValueError("theta must be 1D (d,).")
    if x_obs.ndim != 2:
        raise ValueError("x_obs must be 2D (n, d+1).")
    if x_obs.shape[1] != theta.shape[0] + 1:
        raise ValueError("x_obs must have d+1 columns (features + label).")

    x = x_obs[:, :-1]
    y = x_obs[:, -1]
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y must be in {0,1}.")

    logits = x @ theta
    probs = 1.0 / (1.0 + np.exp(-logits))
    grad = (y - probs)[:, None] * x
    return grad.astype(np.float64, copy=False)
