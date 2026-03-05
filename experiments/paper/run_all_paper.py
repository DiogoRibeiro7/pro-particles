from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from experiments.utils import env_info, git_commit, is_dirty, utc_now


def run_exp(exp_dir: Path, cfg_path: Path | None) -> None:
    cmd = ["python", str(exp_dir / "run.py")]
    if cfg_path is not None:
        cmd.append(str(cfg_path))
    subprocess.check_call(cmd)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    exp_dirs = [root / "exp01", root / "exp02", root / "exp03"]

    for exp in exp_dirs:
        run_exp(exp, exp / "config.json")

    results: List[Dict[str, Any]] = []
    for exp in exp_dirs:
        res = json.loads((exp / "results.json").read_text())
        results.append(res)

    payload = {
        "created_at": utc_now(),
        "git_commit": git_commit(),
        "dirty": is_dirty(),
        "env": env_info(),
        "results": results,
    }

    out_dir = root.parent / "results" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_summary.json").write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
