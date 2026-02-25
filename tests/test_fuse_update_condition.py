from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pro_particles.schedules.fuse import should_update


def test_should_update() -> None:
    assert not should_update(0)
    assert should_update(1)
    assert should_update(2)
