from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from .core import build_rsync_ssh_command
from .tracking import (
    JobRecord,
    TrackingError,
    load_tracking_payload,
    resolve_tracking_file,
)


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


def _collect_downloads(jobs: list[JobRecord]) -> list[tuple[str, str, str]]:
    downloads: list[tuple[str, str, str]] = []
    for job in jobs:
        name = job.job_name or "unknown_job"
        if job.stdout:
            downloads.append((name, "stdout", job.stdout))
        if job.stderr and job.stderr != job.stdout:
            downloads.append((name, "stderr", job.stderr))
    return downloads


def _run_downloads(
    cluster_login: str,
    downloads: list[tuple[str, str, str]],
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> int:
    failures = 0
    for job_name, stream, remote_path in downloads:
        destination_dir = output_dir / job_name
        destination_file = destination_dir / Path(remote_path).name
        source = f"{cluster_login}:{remote_path}"
        cmd = [
            "rsync",
            "-az",
            "-e",
            build_rsync_ssh_command(ssh_config_file, ssh_options),
        ]
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
    tracking_path = resolve_tracking_file(args.tracking_file)
    if tracking_path is None:
        print(
            "ERROR: No tracking file found. "
            "Run a non-dry submission first or pass --tracking-file.",
            file=sys.stderr,
        )
        return 1

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not payload.cluster_login:
        print(f"ERROR: Missing cluster_login in {tracking_path}", file=sys.stderr)
        return 1

    selected = payload.filter_jobs(
        names=set(args.job_name) or None,
        ids=set(args.job_id) or None,
    )
    if not selected:
        print("No matching jobs in tracking file.")
        return 0

    downloads = _collect_downloads(selected)
    if not downloads:
        print("No log paths found in selected jobs.")
        return 0

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("slurm_output") / "downloaded_logs" / payload.job_folder
    )

    print(f"Tracking file: {tracking_path}")
    print(f"Cluster: {payload.cluster_login}")
    print(f"Jobs selected: {len(selected)}")
    print(f"Log files to download: {len(downloads)}")
    print(f"Local destination: {output_dir}")
    if args.dry_run:
        print("Dry-run mode: commands will not be executed.")

    failures = _run_downloads(
        payload.cluster_login,
        downloads,
        output_dir,
        dry_run=args.dry_run,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
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
