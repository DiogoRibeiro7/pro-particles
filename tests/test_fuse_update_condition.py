from __future__ import annotations

from pro_particles.schedules.fuse import should_update


def test_should_update() -> None:
    assert not should_update(0)
    assert should_update(1)
    assert should_update(2)
