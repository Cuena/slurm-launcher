# examples/demo_eval.py
# What: Demo evaluation entrypoint that prints received args/env as JSON.
# Why: Serves as a minimal runnable workload for launcher dry-run and submission examples.
# RELEVANT FILES: examples/demo_train.py, examples/remote_launcher_config.demo.py, examples/remote_launcher_config.mn5.example.py, README.md

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rich.console import Console

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo evaluation script")
    parser.add_argument("--config", required=True, help="Path to evaluation config")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "script": "demo_eval",
        "cwd": str(Path.cwd()),
        "config": args.config,
        "checkpoint": args.ckpt,
        "experiment": os.getenv("EXPERIMENT_NAME", ""),
    }
    console.print_json(data=payload)


if __name__ == "__main__":
    main()
