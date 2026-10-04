"""Execution and destination safety for already-planned downloads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def destination_component(value: str, label: str) -> str:
    """Accept only a single ordinary local path component."""
    if not value or value in {".", ".."} or "/" in value or "\0" in value:
        raise ValueError(f"Invalid {label} for download destination: {value!r}")
    return value


def run_downloads(
    entries: list[dict[str, object]], *, dry_run: bool, quiet: bool = False
) -> int:
    """Execute each planned transfer once, retaining process diagnostics."""
    failures = 0
    for entry in entries:
        remote_path = str(entry["remote_path"])
        destination = Path(str(entry["destination"]))
        if not quiet:
            print(f"{remote_path} -> {destination}")
            print(f"  $ {entry['command']}")
        if dry_run:
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                list(entry["argv"]), check=False, capture_output=True, text=True
            )
            entry["returncode"] = result.returncode
            entry["stderr"] = result.stderr.strip()[:4096]
            if not quiet and result.stdout:
                print(result.stdout, end="")
            if not quiet and result.stderr:
                print(result.stderr, end="", file=sys.stderr)
            failed = result.returncode != 0
        except OSError as exc:
            entry["returncode"] = None
            entry["stderr"] = str(exc)
            failed = True
        if failed:
            failures += 1
            if not quiet:
                print(
                    f"ERROR: download failed ({entry['returncode']}) for {remote_path}: "
                    f"{entry['stderr']}",
                    file=sys.stderr,
                )
    return failures
