"""CLI for pro-particles."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="pro-particles CLI")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run an experiment")
    run_parser.add_argument("exp_id", help="Experiment ID (e.g., exp01)")
    run_parser.add_argument("--fast", action="store_true", help="Use fast config")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("list", help="List available experiments")

    args = parser.parse_args()

    if args.command == "list":
        _list_experiments()
    elif args.command == "run":
        if args.verbose:
            logging.basicConfig(level=logging.INFO)
        _run_experiment(args.exp_id, fast=args.fast)
    else:
        parser.print_help()


def _list_experiments() -> None:
    exp_root = Path(__file__).resolve().parents[2] / "experiments"
    for d in sorted(exp_root.iterdir()):
        cfg = d / "config.json"
        if cfg.exists():
            name = json.loads(cfg.read_text()).get("exp_name", d.name)
            print(f"  {d.name}: {name}")


def _run_experiment(exp_id: str, *, fast: bool) -> None:
    exp_root = Path(__file__).resolve().parents[2] / "experiments"
    exp_dir = exp_root / exp_id
    if not exp_dir.exists():
        print(f"Experiment {exp_id} not found.")
        sys.exit(1)
    cfg_path = exp_dir / ("config_fast.json" if fast else "config.json")
    subprocess.check_call([sys.executable, str(exp_dir / "run.py"), str(cfg_path)])
