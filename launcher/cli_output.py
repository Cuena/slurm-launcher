"""Shared CLI output and submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.panel import Panel
from rich.table import Table

from .core import (
    JobSpec,
    LauncherSettings,
    RemotePaths,
    build_job_record,
    format_ssh_command,
    submit_job,
)
from .tracking import JobRecord


def selected_job_names(jobs: list[JobSpec]) -> list[str]:
    return [job.name for job in jobs]


def monitor_command(
    cluster_login: str,
    job_ids: list[str],
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> str:
    remote_command = f"squeue -j {','.join(job_ids)}" if job_ids else "squeue -u $USER"
    return format_ssh_command(
        cluster_login,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
        remote_command=remote_command,
    )


def collect_submission_results(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    jobs: list[JobSpec],
    *,
    dry_run: bool,
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    submitted_jobs: list[dict[str, Any]] = []
    job_records: list[dict[str, Any]] = []
    commands: list[str] = []
    for job in jobs:
        submission = submit_job(
            settings,
            remote_paths,
            job,
            dry_run=dry_run,
            quiet=quiet,
        )
        commands.extend(submission.commands)
        record = build_job_record(job, submission, settings)
        submitted_jobs.append(record)
        if not dry_run:
            job_records.append(record)
    return submitted_jobs, job_records, commands


def print_execution_panel(
    console: Any,
    title: str,
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    *,
    extra_lines: list[str] | None = None,
    note: str | None = None,
) -> None:
    lines = [
        f"[bold]Cluster:[/bold] {settings.cluster_login}",
        f"[bold]Workspace:[/bold] {settings.workspace_mode}",
        f"[bold]Job folder:[/bold] {remote_paths.job_folder}",
    ]
    if extra_lines:
        lines.extend(extra_lines)
    console.print()
    console.print(
        Panel.fit(
            "\n".join(lines),
            title=title,
            border_style="cyan",
        )
    )
    if settings.workspace_mode == "fixed":
        console.print(
            "Using REMOTE_WORKSPACE_DIR as the execution directory.",
            style="yellow",
        )
    if note:
        console.print(note, style="yellow")


def print_execution_details(
    console: Any,
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    *,
    include_archive_dirs: bool = False,
) -> None:
    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", settings.workspace_mode)
    details_table.add_row("Remote workdir", remote_paths.workdir)
    details_table.add_row("Remote logdir", remote_paths.logdir)
    if include_archive_dirs and settings.remote_slurm_dashboard_log_archive_dir:
        details_table.add_row(
            "Remote slurm-dashboard archive dir",
            settings.remote_slurm_dashboard_log_archive_dir,
        )
    if include_archive_dirs and settings.remote_slurm_dashboard_log_view_dir:
        details_table.add_row(
            "Remote slurm-dashboard view dir",
            settings.remote_slurm_dashboard_log_view_dir,
        )
    console.print()
    console.print(details_table)


def collect_job_ids(records: list[dict[str, Any]]) -> list[str]:
    return [
        str(record.get("job_id", "") or "")
        for record in records
        if str(record.get("job_id", "") or "") not in {"", "unknown", "dry-run"}
    ]


def print_job_logs_from_records(
    console: Any, jobs: list[JobRecord | dict[str, object]]
) -> None:
    console.print()
    console.print(Panel.fit("Remote Logs", border_style="cyan"))
    for index, job in enumerate(jobs):
        if isinstance(job, JobRecord):
            job_name = job.job_name
            job_id = job.job_id
            stdout_path = job.stdout or ""
            stderr_path = job.stderr or ""
        else:
            job_name = str(job.get("job_name", "") or "")
            job_id = str(job.get("job_id", "") or "")
            stdout_path = str(job.get("stdout", "") or "")
            stderr_path = str(job.get("stderr", "") or "")
        job_label = f"{job_name} ({job_id})" if job_id else job_name
        console.print(f"[bold]{job_label}[/bold]")
        console.print(f"stdout: {stdout_path or '-'}", soft_wrap=True)
        console.print(
            f"stderr: {stderr_path if stderr_path and stderr_path != stdout_path else '-'}",
            soft_wrap=True,
        )
        if index < len(jobs) - 1:
            console.print()


def print_submission_tracking(
    console: Any,
    tracking_file: Path | None,
    job_records: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    if tracking_file is not None:
        console.print()
        console.print(f"Saved job metadata to {tracking_file}", style="green")
        submitted_table = Table(title="Submitted Jobs")
        submitted_table.add_column("Job")
        submitted_table.add_column("Job ID")
        for record in job_records:
            submitted_table.add_row(
                str(record.get("job_name", "")),
                str(record.get("job_id", "")),
            )
        console.print(submitted_table)
        print_job_logs_from_records(console, job_records)
        return
    if dry_run:
        console.print()
        console.print(
            "Skipped job metadata tracking because --dry-run was used.",
            style="yellow",
        )


def emit_submission_result(
    console: Any,
    *,
    json_output: bool,
    payload: dict[str, Any],
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    tracking_file: Path | None,
    job_records: list[dict[str, Any]],
    monitor_cmd: str,
    dry_run: bool,
    include_archive_dirs: bool = False,
) -> int:
    if json_output:
        console.print_json(data=payload)
        return 0

    print_submission_tracking(
        console,
        tracking_file,
        job_records,
        dry_run=dry_run,
    )
    print_execution_details(
        console,
        settings,
        remote_paths,
        include_archive_dirs=include_archive_dirs,
    )
    console.print("Monitor jobs with:")
    console.print(monitor_cmd, style="bold")
    return 0
