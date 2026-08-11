from __future__ import annotations

import argparse
import json
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
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


def _download_entry(
    cluster_login: str,
    job_name: str,
    stream: str,
    remote_path: str,
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> dict[str, object]:
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
    return {
        "job_name": job_name,
        "stream": stream,
        "remote_path": remote_path,
        "destination": str(destination_file),
        "command": shlex.join(cmd),
        "argv": cmd,
    }


def _download_entries(
    cluster_login: str,
    downloads: list[tuple[str, str, str]],
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[dict[str, object]]:
    return [
        _download_entry(
            cluster_login,
            job_name,
            stream,
            remote_path,
            output_dir,
            dry_run=dry_run,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        for job_name, stream, remote_path in downloads
    ]


def _run_downloads(
    cluster_login: str,
    downloads: list[tuple[str, str, str]],
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    quiet: bool = False,
) -> int:
    failures = 0
    entries = _download_entries(
        cluster_login,
        downloads,
        output_dir,
        dry_run=dry_run,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    for entry in entries:
        job_name = str(entry["job_name"])
        stream = str(entry["stream"])
        remote_path = str(entry["remote_path"])
        destination_file = Path(str(entry["destination"]))
        cmd = list(entry["argv"])

        if not quiet:
            print(f"[{job_name}] {stream}: {remote_path}")
            print(f"  -> {destination_file}")
            print(f"  $ {entry['command']}")

        if dry_run:
            continue

        destination_file.parent.mkdir(parents=True, exist_ok=True)
        if quiet:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            if not quiet:
                print(
                    f"ERROR: rsync failed ({result.returncode}) for {job_name} {stream}: {remote_path}",
                    file=sys.stderr,
                )
    return failures


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_json_error(message: str, **extra: object) -> int:
    payload: dict[str, object] = {"ok": False, "error": message}
    payload.update(extra)
    _print_json(payload)
    return 1


def run_download_logs(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False))
    tracking_path = resolve_tracking_file(args.tracking_file)
    if tracking_path is None:
        message = (
            "No tracking file found. "
            "Run a non-dry submission first or pass --tracking-file."
        )
        if json_output:
            return _emit_json_error(message, tracking_file=args.tracking_file)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        if json_output:
            return _emit_json_error(str(exc), tracking_file=str(tracking_path))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not payload.cluster_login:
        if json_output:
            return _emit_json_error(
                f"Missing cluster_login in {tracking_path}",
                tracking_file=str(tracking_path),
            )
        print(f"ERROR: Missing cluster_login in {tracking_path}", file=sys.stderr)
        return 1

    selected = payload.filter_jobs(
        names=set(args.job_name) or None,
        ids=set(args.job_id) or None,
    )
    if not selected:
        if json_output:
            _print_json(
                {
                    "ok": True,
                    "tracking_file": str(tracking_path),
                    "cluster_login": payload.cluster_login,
                    "selected_jobs": [],
                    "downloads": [],
                    "commands": [],
                    "output_dir": None,
                    "dry_run": bool(args.dry_run),
                    "failures": 0,
                }
            )
            return 0
        print("No matching jobs in tracking file.")
        return 0

    downloads = _collect_downloads(selected)
    if not downloads:
        if json_output:
            _print_json(
                {
                    "ok": True,
                    "tracking_file": str(tracking_path),
                    "cluster_login": payload.cluster_login,
                    "selected_jobs": [
                        {"job_name": job.job_name, "job_id": job.job_id}
                        for job in selected
                    ],
                    "downloads": [],
                    "commands": [],
                    "output_dir": None,
                    "dry_run": bool(args.dry_run),
                    "failures": 0,
                }
            )
            return 0
        print("No log paths found in selected jobs.")
        return 0

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("slurm_output") / "downloaded_logs" / payload.job_folder
    )

    entries = _download_entries(
        payload.rsync_login or payload.cluster_login,
        downloads,
        output_dir,
        dry_run=args.dry_run,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
    )

    if json_output:
        failures = _run_downloads(
            payload.rsync_login or payload.cluster_login,
            downloads,
            output_dir,
            dry_run=args.dry_run,
            ssh_config_file=payload.ssh_config_file,
            ssh_options=payload.ssh_options,
            quiet=True,
        )
        _print_json(
            {
                "ok": failures == 0,
                "tracking_file": str(tracking_path),
                "cluster_login": payload.cluster_login,
                "selected_jobs": [
                    {"job_name": job.job_name, "job_id": job.job_id} for job in selected
                ],
                "downloads": [
                    {
                        "job_name": entry["job_name"],
                        "stream": entry["stream"],
                        "remote_path": entry["remote_path"],
                        "destination": entry["destination"],
                    }
                    for entry in entries
                ],
                "commands": [str(entry["command"]) for entry in entries],
                "output_dir": str(output_dir),
                "dry_run": bool(args.dry_run),
                "failures": failures,
            }
        )
        return 0 if failures == 0 else 1

    print(f"Tracking file: {tracking_path}")
    print(f"Cluster: {payload.cluster_login}")
    print(f"Jobs selected: {len(selected)}")
    print(f"Log files to download: {len(downloads)}")
    print(f"Local destination: {output_dir}")
    if args.dry_run:
        print("Dry-run mode: commands will not be executed.")

    failures = _run_downloads(
        payload.rsync_login or payload.cluster_login,
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
