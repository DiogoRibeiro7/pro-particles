from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_exp(exp_dir: Path, cfg_path: Path | None) -> None:
    cmd = ["python", str(exp_dir / "run.py")]
    if cfg_path is not None:
        cmd.append(str(cfg_path))
    subprocess.check_call(cmd)


def main() -> None:
    root = Path(__file__).resolve().parent
    exp_dirs = [root / "exp01", root / "exp02", root / "exp03"]
    use_fast = False
    if len(sys.argv) > 1 and sys.argv[1] == "--fast":
        use_fast = True

    for exp in exp_dirs:
        cfg_path = None
        if use_fast:
            cfg_path = exp / "config_fast.json"
        run_exp(exp, cfg_path)

    results = []
    for exp in exp_dirs:
        res = json.loads((exp / "results.json").read_text())
        results.append(res)

    out_dir = root.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
