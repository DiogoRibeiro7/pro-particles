import json
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
class TestExperimentsFast:
    def _run_exp(self, exp_id: str) -> dict:
        exp_dir = REPO_ROOT / "experiments" / exp_id
        cfg_path = exp_dir / "config_fast.json"
        subprocess.check_call(
            ["python", str(exp_dir / "run.py"), str(cfg_path)],
            cwd=str(REPO_ROOT),
        )
        results_path = exp_dir / "results.json"
        assert results_path.exists()
        return json.loads(results_path.read_text())

    def test_exp01_fast(self):
        r = self._run_exp("exp01")
        assert r["runs"][0]["status"] == "ok"
        assert all(
            v is not None
            for v in [r["runs"][0]["metrics"]["elpd"]]
        )

    def test_exp02_fast(self):
        r = self._run_exp("exp02")
        assert r["runs"][0]["status"] == "ok"
        metrics = r["runs"][0]["metrics"]
        assert metrics["elpd"] is not None
        assert metrics["crps_mean"] is not None

    def test_exp03_fast(self):
        r = self._run_exp("exp03")
        assert r["runs"][0]["status"] == "ok"
        assert r["runs"][0]["metrics"]["elpd"] is not None
