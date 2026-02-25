from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run_exp(exp_dir: Path) -> None:
    subprocess.check_call(["python", str(exp_dir / "run.py")])


def main() -> None:
    root = Path(__file__).resolve().parent
    exp_dirs = [root / "exp01", root / "exp02", root / "exp03"]

    for exp in exp_dirs:
        run_exp(exp)

    results = []
    for exp in exp_dirs:
        res = json.loads((exp / "results.json").read_text())
        results.append(res)

    out_dir = root.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
