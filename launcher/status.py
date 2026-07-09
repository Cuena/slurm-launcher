"""Project-scoped job status queries for tracked submissions."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import build_ssh_command
from .job_tools import _normalized_text
from .tracking import JobRecord, TrackingPayload, load_tracking_payload, resolve_tracking_file

console = Console()
err_console = Console(stderr=True)


@dataclass(frozen=True)
class JobStatus:
    """Current SLURM state for one tracked job."""

    job_id: str
    job_name: str
    state: str | None
    exit_code: str | None
    submit_time: str | None
    start_time: str | None
    end_time: str | None
    elapsed: str | None
    partition: str | None
    derived_state: str
    tracking: JobRecord | None = None


def _run_ssh_capture(
    cluster_login: str,
    script: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            *build_ssh_command(
                cluster_login,
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            ),
            "bash",
            "-s",
        ],
        input=script.rstrip() + "\n",
        capture_output=True,
        text=True,
        check=False,
    )


def _parse_sacct_status(output: str, job_ids: set[str]) -> dict[str, dict[str, str]]:
    """Parse sacct output into a mapping of job_id -> fields."""
    results: dict[str, dict[str, str]] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 8:
            continue
        job_id = parts[0]
        if job_id not in job_ids:
            continue
        results[job_id] = {
            "job_name": parts[1],
            "state": parts[2],
            "exit_code": parts[3],
            "submit": parts[4],
            "start": parts[5],
            "end": parts[6],
            "elapsed": parts[7],
            "partition": parts[8] if len(parts) > 8 else "",
        }
    return results


def _derive_state(state: str | None, exit_code: str | None) -> str:
    if not state:
        return "UNKNOWN"
    token = state.strip().upper().split()[0]
    if token == "COMPLETED":
        return "DONE"
    if token in {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}:
        return "FAILED"
    if token in {"RUNNING", "COMPLETING"}:
        return "RUNNING"
    if token in {"PENDING", "CONFIGURING", "RESIZING"}:
        return "PENDING"
    return token


def query_job_statuses(
    cluster_login: str,
    jobs: list[JobRecord],
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[JobStatus]:
    """Query sacct/squeue for the given tracked jobs and return status records."""
    runnable = [job for job in jobs if job.job_id and job.job_id not in {"", "unknown", "dry-run"}]
    if not runnable:
        return []

    job_ids = [job.job_id for job in runnable]
    id_set = set(job_ids)
    id_expr = ",".join(shlex.quote(job_id) for job_id in job_ids)

    # Prefer sacct for completed/finished state; fall back to squeue for running jobs.
    script = "\n".join(
        [
            "set -euo pipefail",
            "if command -v sacct >/dev/null 2>&1; then",
            '  echo "__SOURCE__=sacct"',
            f"  sacct -X -n -P -j {id_expr} ",
            "    --format JobIDRaw,JobName,State,ExitCode,Submit,Start,End,Elapsed,Partition",
            "else",
            '  echo "__SOURCE__=squeue"',
            f"  squeue -h -j {id_expr} -o \"%i|%j|%T|%P|-.|-.|-.|%M|%P\"",
            "fi",
        ]
    )
    result = _run_ssh_capture(
        cluster_login,
        script,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )

    parsed: dict[str, dict[str, str]] = {}
    if result.returncode == 0:
        lines = result.stdout.splitlines()
        if lines and lines[0].startswith("__SOURCE__="):
            output = "\n".join(lines[1:])
        else:
            output = result.stdout
        parsed = _parse_sacct_status(output, id_set)

    # Build status records, preserving tracking file order.
    statuses: list[JobStatus] = []
    for job in runnable:
        fields = parsed.get(job.job_id, {})
        state = _normalized_text(fields.get("state"))
        exit_code = _normalized_text(fields.get("exit_code"))
        derived = _derive_state(state, exit_code)
        statuses.append(
            JobStatus(
                job_id=job.job_id,
                job_name=_normalized_text(fields.get("job_name")) or job.job_name,
                state=state,
                exit_code=exit_code,
                submit_time=_normalized_text(fields.get("submit")),
                start_time=_normalized_text(fields.get("start")),
                end_time=_normalized_text(fields.get("end")),
                elapsed=_normalized_text(fields.get("elapsed")),
                partition=_normalized_text(fields.get("partition")),
                derived_state=derived,
                tracking=job,
            )
        )
    return statuses


def _status_payload(
    tracking_file: Path | None,
    cluster_login: str | None,
    statuses: list[JobStatus],
) -> dict[str, Any]:
    return {
        "ok": True,
        "tracking_file": str(tracking_file) if tracking_file else None,
        "cluster_login": cluster_login,
        "jobs": [
            {
                "job_id": status.job_id,
                "job_name": status.job_name,
                "state": status.state,
                "derived_state": status.derived_state,
                "exit_code": status.exit_code,
                "submit_time": status.submit_time,
                "start_time": status.start_time,
                "end_time": status.end_time,
                "elapsed": status.elapsed,
                "partition": status.partition,
            }
            for status in statuses
        ],
    }


def print_status_table(
    tracking_file: Path | None,
    cluster_login: str | None,
    statuses: list[JobStatus],
) -> None:
    if tracking_file:
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"[bold]Tracking file:[/bold] {tracking_file}",
                        f"[bold]Cluster:[/bold] {cluster_login or '-'}",
                    ]
                ),
                title="Status",
                border_style="cyan",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"[bold]Cluster:[/bold] {cluster_login or '-'}",
                title="Status",
                border_style="cyan",
            )
        )

    if not statuses:
        console.print("No runnable jobs found.", style="yellow")
        return

    table = Table()
    table.add_column("Job ID")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("Derived")
    table.add_column("Elapsed")
    table.add_column("Exit Code")
    for status in statuses:
        style = None
        if status.derived_state == "DONE":
            style = "green"
        elif status.derived_state == "FAILED":
            style = "red"
        elif status.derived_state == "RUNNING":
            style = "cyan"
        table.add_row(
            status.job_id,
            status.job_name,
            status.state or "-",
            status.derived_state,
            status.elapsed or "-",
            status.exit_code or "-",
            style=style,
        )
    console.print(table)


def run_status(
    *,
    tracking_file: str | None = None,
    job_id: str | None = None,
    cluster_login: str | None = None,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    json_output: bool = False,
) -> int:
    """Project-scoped status command.

    Either queries a single job by id (using the provided cluster login) or
    resolves the latest tracking file and queries all tracked jobs.
    """
    resolved_tracking: Path | None = None
    payload: TrackingPayload | None = None

    if job_id and cluster_login:
        # Direct cluster query for a single job id.
        jobs = [JobRecord(job_name="", job_id=job_id)]
        effective_login = cluster_login
        # Try to enrich with tracking file if available.
        resolved_tracking = resolve_tracking_file(tracking_file)
        if resolved_tracking is not None:
            try:
                payload = load_tracking_payload(resolved_tracking)
                matched = payload.filter_jobs(ids={job_id})
                if matched:
                    jobs = matched
                    effective_login = payload.cluster_login or cluster_login
                    ssh_config_file = ssh_config_file or payload.ssh_config_file
                    ssh_options = ssh_options or payload.ssh_options
            except Exception:
                pass
        statuses = query_job_statuses(
            effective_login,
            jobs,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        if json_output:
            console.print_json(data=_status_payload(resolved_tracking, effective_login, statuses))
            return 0
        print_status_table(resolved_tracking, effective_login, statuses)
        return 0

    resolved_tracking = resolve_tracking_file(tracking_file)
    if resolved_tracking is None:
        message = "No tracking file found. Run a submission first or pass --tracking-file."
        if json_output:
            console.print_json(data={"ok": False, "error": message})
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    try:
        payload = load_tracking_payload(resolved_tracking)
    except Exception as exc:
        message = f"Cannot load tracking file: {exc}"
        if json_output:
            console.print_json(data={"ok": False, "error": message})
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    if not payload.cluster_login:
        message = f"Missing cluster_login in {resolved_tracking}"
        if json_output:
            console.print_json(data={"ok": False, "error": message})
        else:
            err_console.print(f"ERROR: {message}", style="bold red")
        return 1

    statuses = query_job_statuses(
        payload.cluster_login,
        payload.jobs,
        ssh_config_file=ssh_config_file or payload.ssh_config_file,
        ssh_options=ssh_options or payload.ssh_options,
    )
    if json_output:
        console.print_json(data=_status_payload(resolved_tracking, payload.cluster_login, statuses))
        return 0
    print_status_table(resolved_tracking, payload.cluster_login, statuses)
    return 0
