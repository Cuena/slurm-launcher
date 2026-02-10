# launcher/tracking.py
# What: Shared helpers to locate launcher tracking files under slurm_output.
# Why: Keeps CLI and utility scripts aligned on how latest jobs.json is discovered.
# RELEVANT FILES: launcher/cli.py, scripts/download_logs.py, launcher/core.py, README.md

from __future__ import annotations

from pathlib import Path


def resolve_tracking_file(path_arg: str | None) -> Path | None:
    if path_arg:
        candidate = Path(path_arg)
        return candidate if candidate.exists() else None

    latest = Path("slurm_output/latest_jobs.json")
    if latest.exists():
        return latest

    candidates = sorted(
        Path("slurm_output").glob("*/jobs.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None
