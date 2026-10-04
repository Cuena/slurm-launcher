from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .transport import build_rsync_ssh_command
from .transfers import destination_component, run_downloads
from .tracking import (
    JobRecord,
    TrackingError,
    load_tracking_payload,
    resolve_tracking_file,
)


def add_download_logs_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run",
        dest="tracking_file",
        help="Run ID, tracking path, or latest (default).",
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


def _collect_downloads(jobs: list[JobRecord]) -> list[tuple[str, str, str, str]]:
    downloads: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for job in jobs:
        name = destination_component(job.job_name or "unknown_job", "tracked job name")
        job_id = destination_component(job.job_id or "unknown", "tracked job ID")
        for stream, path in (("stdout", job.stdout), ("stderr", job.stderr)):
            if path and (name, job_id, path) not in seen:
                downloads.append((name, job_id, stream, path))
                seen.add((name, job_id, path))
    return downloads


def _download_entry(
    cluster_login: str,
    job_name: str,
    job_id: str,
    stream: str,
    remote_path: str,
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> dict[str, object]:
    destination_dir = (
        output_dir
        / destination_component(job_name, "tracked job name")
        / destination_component(job_id, "tracked job ID")
        / destination_component(stream, "log stream")
    )
    basename = destination_component(Path(remote_path).name, "log basename")
    if ".." in Path(remote_path).parts:
        raise ValueError("Log paths cannot contain traversal components.")
    destination_file = destination_dir / basename
    source = f"{cluster_login}:{remote_path}"
    cmd = [
        "rsync",
        "-az",
        "-e",
        build_rsync_ssh_command(ssh_config_file, ssh_options),
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--protect-args", source, str(destination_file)])
    return {
        "job_name": job_name,
        "job_id": job_id,
        "stream": stream,
        "remote_path": remote_path,
        "destination": str(destination_file),
        "command": shlex.join(cmd),
        "argv": cmd,
    }


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
            "Run a submission first or pass --run ID or a tracking path."
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
    try:
        downloads = _collect_downloads(selected)
        output_dir = None
        entries = []
        if downloads:
            output_dir = (
                Path(args.output_dir)
                if args.output_dir
                else Path("slurm_output")
                / "downloaded_logs"
                / destination_component(payload.job_folder, "tracked job folder")
            )
            entries = [
                _download_entry(
                    payload.rsync_login or payload.cluster_login,
                    job_name,
                    job_id,
                    stream,
                    remote_path,
                    output_dir,
                    dry_run=args.dry_run,
                    ssh_config_file=payload.ssh_config_file,
                    ssh_options=payload.ssh_options,
                )
                for job_name, job_id, stream, remote_path in downloads
            ]
    except ValueError as exc:
        if json_output:
            return _emit_json_error(str(exc))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not json_output and downloads:
        print(f"Tracking file: {tracking_path}")
        print(f"Cluster: {payload.cluster_login}")
        print(f"Jobs selected: {len(selected)}")
        print(f"Log files to download: {len(downloads)}")
        print(f"Local destination: {output_dir}")
        if args.dry_run:
            print("Dry-run mode: commands will not be executed.")

    failures = run_downloads(entries, dry_run=args.dry_run, quiet=json_output)
    if json_output:
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
                        key: value
                        for key, value in entry.items()
                        if key not in {"argv", "command"}
                    }
                    for entry in entries
                ],
                "commands": [str(entry["command"]) for entry in entries],
                "output_dir": str(output_dir) if output_dir is not None else None,
                "dry_run": bool(args.dry_run),
                "failures": failures,
            }
        )
    elif not selected:
        print("No matching jobs in tracking file.")
    elif not downloads:
        print("No log paths found in selected jobs.")
    elif failures:
        print(f"Completed with {failures} failed download(s).", file=sys.stderr)
    else:
        print("Download complete.")
    return 0 if failures == 0 else 1
