"""Project-scoped artifact discovery and download."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import build_rsync_ssh_command
from .tracking import JobRecord, TrackingError, TrackingPayload, load_tracking_payload, resolve_tracking_file

console = Console()
err_console = Console(stderr=True)


def _resolve_remote_artifact_path(remote_workdir: str, artifact_path: str) -> str:
    if artifact_path.startswith("/"):
        return artifact_path
    return f"{remote_workdir.rstrip('/')}/{artifact_path.lstrip('/')}"


def _local_artifact_destination(
    output_dir: Path,
    job: JobRecord,
    artifact_path: str,
) -> Path:
    """Return local destination for an artifact.

    Layout: output_dir / job_name / job_id / artifact_path
    """
    job_label = job.job_name or "unknown_job"
    job_id = job.job_id or "unknown"
    return output_dir / job_label / job_id / Path(artifact_path.lstrip("/"))


def _artifact_entries(
    cluster_login: str,
    remote_workdir: str,
    job: JobRecord,
    artifact_paths: list[str],
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for artifact_path in artifact_paths:
        remote_path = _resolve_remote_artifact_path(remote_workdir, artifact_path)
        local_path = _local_artifact_destination(output_dir, job, artifact_path)
        source = f"{cluster_login}:{remote_path}"
        cmd = [
            "rsync",
            "-az",
            "-e",
            build_rsync_ssh_command(ssh_config_file, ssh_options),
        ]
        if dry_run:
            cmd.append("--dry-run")
        cmd.extend([source, str(local_path)])
        entries.append(
            {
                "job_name": job.job_name,
                "job_id": job.job_id,
                "path": artifact_path,
                "remote_path": remote_path,
                "destination": str(local_path),
                "command": shlex.join(cmd),
                "argv": cmd,
            }
        )
    return entries


def _run_downloads(
    cluster_login: str,
    entries: list[dict[str, object]],
    *,
    dry_run: bool,
    quiet: bool = False,
) -> int:
    failures = 0
    for entry in entries:
        artifact_path = str(entry["path"])
        remote_path = str(entry["remote_path"])
        destination = Path(str(entry["destination"]))
        cmd = list(entry["argv"])

        if not quiet:
            print(f"{artifact_path} -> {remote_path}")
            print(f"  -> {destination}")
            print(f"  $ {entry['command']}")

        if dry_run:
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if quiet:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            if not quiet:
                print(
                    f"ERROR: rsync failed ({result.returncode}) for {artifact_path}: {remote_path}",
                    file=sys.stderr,
                )
    return failures


def _collect_job_artifacts(payload: TrackingPayload, job: JobRecord) -> list[str]:
    """Artifact paths for a job, preferring job-specific declarations."""
    paths = job.artifacts if job.artifacts else payload.artifact_paths
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def list_artifacts(
    payload: TrackingPayload,
    output_dir: Path,
    *,
    selected_jobs: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build artifact list entries without running downloads."""
    jobs = payload.filter_jobs(names=set(selected_jobs) if selected_jobs else None)
    entries: list[dict[str, object]] = []
    for job in jobs:
        artifact_paths = _collect_job_artifacts(payload, job)
        for artifact_path in artifact_paths:
            remote_path = _resolve_remote_artifact_path(
                payload.remote_workdir or "", artifact_path
            )
            local_path = _local_artifact_destination(output_dir, job, artifact_path)
            entries.append(
                {
                    "job_name": job.job_name,
                    "job_id": job.job_id,
                    "path": artifact_path,
                    "remote_path": remote_path,
                    "destination": str(local_path),
                }
            )
    return entries


def _payload(
    tracking_file: Path | None,
    output_dir: Path,
    entries: list[dict[str, object]],
    dry_run: bool,
    failures: int = 0,
) -> dict[str, Any]:
    return {
        "ok": failures == 0,
        "tracking_file": str(tracking_file) if tracking_file else None,
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "artifacts": [
            {
                "job_name": entry.get("job_name"),
                "job_id": entry.get("job_id"),
                "path": entry["path"],
                "remote_path": entry["remote_path"],
                "destination": entry["destination"],
            }
            for entry in entries
        ],
        "commands": [str(entry.get("command", "")) for entry in entries],
        "failures": failures,
    }


def print_artifact_table(entries: list[dict[str, object]]) -> None:
    if not entries:
        console.print("No artifacts found.", style="yellow")
        return

    table = Table()
    table.add_column("Job")
    table.add_column("Job ID")
    table.add_column("Path")
    table.add_column("Remote")
    table.add_column("Local")
    for entry in entries:
        table.add_row(
            str(entry.get("job_name", "")),
            str(entry.get("job_id", "")),
            str(entry["path"]),
            str(entry["remote_path"]),
            str(entry["destination"]),
        )
    console.print(table)


def run_artifacts(
    *,
    subcommand: str,
    tracking_file: str | None = None,
    output_dir: str | None = None,
    selected_jobs: list[str] | None = None,
    dry_run: bool = False,
    json_output: bool = False,
) -> int:
    tracking_path = resolve_tracking_file(tracking_file)
    if tracking_path is None:
        message = (
            "No tracking file found. Run a non-dry submission first or pass --tracking-file."
        )
        if json_output:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        message = f"Cannot load tracking file: {exc}"
        if json_output:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    if not payload.cluster_login:
        message = f"Missing cluster_login in {tracking_path}"
        if json_output:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    if not payload.remote_workdir:
        message = f"Missing remote_workdir in {tracking_path}"
        if json_output:
            print(json.dumps({"ok": False, "error": message}, indent=2))
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    effective_output_dir = (
        Path(output_dir)
        if output_dir
        else Path("slurm_output") / "downloaded_artifacts" / payload.job_folder
    )

    jobs = payload.filter_jobs(names=set(selected_jobs) if selected_jobs else None)
    if not jobs:
        if json_output:
            print(
                json.dumps(
                    _payload(tracking_path, effective_output_dir, [], dry_run),
                    indent=2,
                )
            )
        else:
            console.print("No matching jobs.", style="yellow")
        return 0

    if subcommand == "list":
        entries = list_artifacts(
            payload,
            effective_output_dir,
            selected_jobs=selected_jobs,
        )
        if json_output:
            print(
                json.dumps(
                    _payload(tracking_path, effective_output_dir, entries, dry_run),
                    indent=2,
                )
            )
            return 0
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"[bold]Tracking file:[/bold] {tracking_path}",
                        f"[bold]Cluster:[/bold] {payload.cluster_login}",
                        f"[bold]Remote workdir:[/bold] {payload.remote_workdir}",
                        f"[bold]Local output:[/bold] {effective_output_dir}",
                    ]
                ),
                title="Artifacts",
                border_style="cyan",
            )
        )
        print_artifact_table(entries)
        return 0

    # download
    entries: list[dict[str, object]] = []
    for job in jobs:
        artifact_paths = _collect_job_artifacts(payload, job)
        if not artifact_paths:
            continue
        entries.extend(
            _artifact_entries(
                payload.cluster_login,
                payload.remote_workdir,
                job,
                artifact_paths,
                effective_output_dir,
                dry_run=dry_run,
                ssh_config_file=payload.ssh_config_file,
                ssh_options=payload.ssh_options,
            )
        )

    if json_output:
        failures = _run_downloads(
            payload.cluster_login,
            entries,
            dry_run=dry_run,
            quiet=True,
        )
        print(
            json.dumps(
                _payload(
                    tracking_path,
                    effective_output_dir,
                    entries,
                    dry_run,
                    failures=failures,
                ),
                indent=2,
            )
        )
        return 0 if failures == 0 else 1

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Tracking file:[/bold] {tracking_path}",
                    f"[bold]Cluster:[/bold] {payload.cluster_login}",
                    f"[bold]Remote workdir:[/bold] {payload.remote_workdir}",
                    f"[bold]Local output:[/bold] {effective_output_dir}",
                ]
            ),
            title="Download Artifacts",
            border_style="cyan",
        )
    )
    if dry_run:
        console.print("Dry-run mode: commands will not be executed.", style="yellow")

    failures = _run_downloads(
        payload.cluster_login,
        entries,
        dry_run=dry_run,
    )
    if failures:
        err_console.print(
            f"Completed with {failures} failed download(s).", style="bold red"
        )
        return 1

    console.print("Download complete.", style="green")
    return 0


def add_artifacts_parser(subparsers: Any) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "artifacts",
        help="List or download job artifacts tracked by slurm-launcher.",
    )
    sub = parser.add_subparsers(dest="artifacts_command", required=True)

    list_parser = sub.add_parser("list", help="List discovered artifacts.")
    list_parser.add_argument(
        "--tracking-file",
        help=(
            "Path to a jobs.json file. Defaults to slurm_output/latest_jobs.json, "
            "or the most recent slurm_output/*/jobs.json."
        ),
    )
    list_parser.add_argument(
        "--only",
        nargs="+",
        help="Limit to the specified job names.",
    )
    list_parser.add_argument(
        "--output-dir",
        help="Local destination root. Default: slurm_output/downloaded_artifacts/<job_folder>/",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )

    download_parser = sub.add_parser("download", help="Download discovered artifacts.")
    download_parser.add_argument(
        "--tracking-file",
        help=(
            "Path to a jobs.json file. Defaults to slurm_output/latest_jobs.json, "
            "or the most recent slurm_output/*/jobs.json."
        ),
    )
    download_parser.add_argument(
        "--only",
        nargs="+",
        help="Limit to the specified job names.",
    )
    download_parser.add_argument(
        "--output-dir",
        help="Local destination root. Default: slurm_output/downloaded_artifacts/<job_folder>/",
    )
    download_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rsync commands without executing them.",
    )
    download_parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )
    return parser


def dispatch_artifacts(args: argparse.Namespace) -> int:
    return run_artifacts(
        subcommand=args.artifacts_command,
        tracking_file=args.tracking_file,
        output_dir=args.output_dir,
        selected_jobs=args.only,
        dry_run=bool(getattr(args, "dry_run", False)),
        json_output=bool(args.json),
    )
