from __future__ import annotations

import math

import numpy as np

from pro_particles.schedules.fuse import FuseState, update_eta


def test_fuse_rule_matches_spec() -> None:
    # Two particles, one dimension.
    x0 = np.array([[0.0], [1.0]])
    x1 = np.array([[0.5], [1.5]])
    x_prev = x0.copy()
    x_t = x1.copy()

    grad_t = np.array([[1.0], [2.0]])
    r_eps = 0.1

    state = FuseState(r_eps=r_eps, x1=x1.copy())

    # t=1 update
    eta_1 = update_eta(
        state,
        t=1,
        particles_t=x_t,
        particles_prev=x_prev,
        grad_t=grad_t,
    )

    movement = np.mean((x1 - x0) ** 2)
    denom = np.mean(grad_t ** 2)
    expected = math.sqrt(max(r_eps, movement) / denom)

    assert math.isclose(eta_1, expected, rel_tol=1e-12, abs_tol=0.0)

    # t=2 update with new data
    x2 = np.array([[1.0], [2.0]])
    grad_2 = np.array([[2.0], [4.0]])
    eta_2 = update_eta(
        state,
        t=2,
        particles_t=x2,
        particles_prev=x_t,
        grad_t=grad_2,
    )

    movement_2 = np.mean((x1 - x_t) ** 2)
    max_movement = max(r_eps, movement, movement_2)
    denom_2 = denom + np.mean(grad_2 ** 2)
    expected_2 = math.sqrt(max_movement / denom_2)

    assert math.isclose(eta_2, expected_2, rel_tol=1e-12, abs_tol=0.0)
