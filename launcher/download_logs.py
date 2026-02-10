# launcher/download_logs.py
# What: Provides download-logs CLI helpers to fetch tracked remote SLURM stdout/stderr files.
# Why: Enables a first-class launcher subcommand without requiring script-path execution.
# RELEVANT FILES: launcher/cli.py, scripts/download_logs.py, launcher/tracking.py, README.md

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .tracking import resolve_tracking_file


def add_download_logs_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-file",
        help=(
            "Path to a jobs.json file. Defaults to slurm_output/latest_jobs.json, "
            "or the most recent slurm_output/*/jobs.json."
        ),
    )
    parser.add_argument(
        "--job-name",
        action="append",
        default=[],
        help="Download only matching job name(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--job-id",
        action="append",
        default=[],
        help="Download only matching SLURM job id(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Local destination directory. "
            "Default: slurm_output/downloaded_logs/<job_folder>/"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rsync commands without executing them.",
    )


def _select_records(
    records: list[dict[str, Any]], job_names: set[str], job_ids: set[str]
) -> list[dict[str, Any]]:
    if not job_names and not job_ids:
        return records

    selected: list[dict[str, Any]] = []
    for record in records:
        record_name = str(record.get("job_name", ""))
        record_id = str(record.get("job_id", ""))
        if job_names and record_name in job_names:
            selected.append(record)
            continue
        if job_ids and record_id in job_ids:
            selected.append(record)
    return selected


def _collect_downloads(records: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    downloads: list[tuple[str, str, str]] = []
    for record in records:
        job_name = str(record.get("job_name", "") or "unknown_job")
        stdout_path = str(record.get("stdout", "") or "")
        stderr_path = str(record.get("stderr", "") or "")
        if stdout_path:
            downloads.append((job_name, "stdout", stdout_path))
        if stderr_path and stderr_path != stdout_path:
            downloads.append((job_name, "stderr", stderr_path))
    return downloads


def _run_downloads(
    cluster_login: str,
    downloads: list[tuple[str, str, str]],
    output_dir: Path,
    *,
    dry_run: bool,
) -> int:
    failures = 0
    for job_name, stream, remote_path in downloads:
        destination_dir = output_dir / job_name
        destination_file = destination_dir / Path(remote_path).name
        source = f"{cluster_login}:{remote_path}"
        cmd = ["rsync", "-az"]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([source, str(destination_file)])

        print(f"[{job_name}] {stream}: {remote_path}")
        print(f"  -> {destination_file}")
        print(f"  $ {shlex.join(cmd)}")

        if dry_run:
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            print(
                f"ERROR: rsync failed ({result.returncode}) for {job_name} {stream}: {remote_path}",
                file=sys.stderr,
            )
    return failures


def run_download_logs(args: argparse.Namespace) -> int:
    tracking_file = resolve_tracking_file(args.tracking_file)
    if tracking_file is None:
        print(
            "ERROR: No tracking file found. "
            "Run a non-dry submission first or pass --tracking-file.",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(tracking_file.read_text(encoding="utf-8"))
    cluster_login = str(payload.get("cluster_login", "") or "")
    if not cluster_login:
        print(f"ERROR: Missing cluster_login in {tracking_file}", file=sys.stderr)
        return 1

    records = payload.get("jobs", [])
    if not isinstance(records, list):
        print(f"ERROR: Invalid tracking file format: {tracking_file}", file=sys.stderr)
        return 1

    selected_records = _select_records(
        [record for record in records if isinstance(record, dict)],
        set(args.job_name),
        set(args.job_id),
    )
    if not selected_records:
        print("No matching jobs in tracking file.")
        return 0

    downloads = _collect_downloads(selected_records)
    if not downloads:
        print("No log paths found in selected jobs.")
        return 0

    job_folder = str(
        payload.get("job_folder", "unknown_job_folder") or "unknown_job_folder"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("slurm_output") / "downloaded_logs" / job_folder
    )

    print(f"Tracking file: {tracking_file}")
    print(f"Cluster: {cluster_login}")
    print(f"Jobs selected: {len(selected_records)}")
    print(f"Log files to download: {len(downloads)}")
    print(f"Local destination: {output_dir}")
    if args.dry_run:
        print("Dry-run mode: commands will not be executed.")

    failures = _run_downloads(
        cluster_login, downloads, output_dir, dry_run=args.dry_run
    )
    if failures:
        print(f"Completed with {failures} failed download(s).", file=sys.stderr)
        return 1

    print("Download complete.")
    return 0


def parse_download_logs_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download remote logs tracked by slurm-launcher. "
            "Defaults to all jobs in the latest tracking file."
        )
    )
    add_download_logs_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_download_logs(parse_download_logs_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
