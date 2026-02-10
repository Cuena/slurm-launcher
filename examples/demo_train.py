# examples/demo_train.py
# What: Demo training entrypoint that prints received args/env as JSON.
# Why: Serves as a minimal runnable workload for launcher dry-run and submission examples.
# RELEVANT FILES: examples/demo_eval.py, examples/remote_launcher_config.demo.py, examples/remote_launcher_config.mn5.example.py, README.md

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo training script")
    parser.add_argument("--config", required=True, help="Path to experiment config")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "script": "demo_train",
        "cwd": str(Path.cwd()),
        "config": args.config,
        "epochs": args.epochs,
        "lr": args.lr,
        "experiment": os.getenv("EXPERIMENT_NAME", ""),
    }
    console.print_json(data=payload)


if __name__ == "__main__":
    main()
