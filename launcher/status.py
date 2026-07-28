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
from .tracking import (
    JobRecord,
    TrackingPayload,
    load_tracking_payload,
    resolve_tracking_file,
)

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
    source: str | None = None
    tracking: JobRecord | None = None


@dataclass(frozen=True)
class StatusProbe:
    """Outcome of one remote SLURM status source."""

    source: str
    returncode: int
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True)
class StatusQueryResult:
    """Statuses plus enough diagnostics to distinguish UNKNOWN from query failure."""

    statuses: list[JobStatus]
    probes: list[StatusProbe]
    unresolved_job_ids: list[str]

    @property
    def ok(self) -> bool:
        if not self.unresolved_job_ids:
            return True
        return all(probe.ok for probe in self.probes)


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


def _parse_status_output(output: str, job_ids: set[str]) -> dict[str, dict[str, str]]:
    """Parse normalized sacct/squeue output into a mapping of job_id -> fields."""
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


def _build_sacct_script(job_ids: list[str]) -> str:
    id_expr = ",".join(shlex.quote(job_id) for job_id in job_ids)
    return "\n".join(
        [
            "set -euo pipefail",
            "command -v sacct >/dev/null 2>&1",
            (
                f"sacct -X -n -P -j {id_expr} "
                "--format JobIDRaw,JobName,State,ExitCode,Submit,Start,End,Elapsed,Partition"
            ),
        ]
    )


def _build_squeue_script(job_ids: list[str]) -> str:
    id_expr = ",".join(shlex.quote(job_id) for job_id in job_ids)
    return "\n".join(
        [
            "set -euo pipefail",
            "command -v squeue >/dev/null 2>&1",
            f'squeue -h -j {id_expr} -o "%i|%j|%T|-|%V|%S|-|%M|%P"',
        ]
    )


def query_job_statuses(
    cluster_login: str,
    jobs: list[JobRecord],
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> StatusQueryResult:
    """Query sacct/squeue for the given tracked jobs and return status records."""
    runnable = [
        job
        for job in jobs
        if job.job_id and job.job_id not in {"", "unknown", "dry-run"}
    ]
    if not runnable:
        return StatusQueryResult(statuses=[], probes=[], unresolved_job_ids=[])

    job_ids = [job.job_id for job in runnable]
    id_set = set(job_ids)
    sacct_script = _build_sacct_script(job_ids)
    sacct_result = _run_ssh_capture(
        cluster_login,
        sacct_script,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )

    probes = [
        StatusProbe(
            source="sacct",
            returncode=sacct_result.returncode,
            stderr=sacct_result.stderr.strip(),
        )
    ]
    parsed: dict[str, dict[str, str]] = {}
    if sacct_result.returncode == 0:
        parsed = _parse_status_output(sacct_result.stdout, id_set)
        for fields in parsed.values():
            fields["source"] = "sacct"

    # sacct can omit jobs that are still live or report UNKNOWN, especially
    # immediately after submission. Query squeue for only those unresolved
    # IDs and merge the result without replacing richer accounting data.
    unresolved_ids = [
        job_id
        for job_id in job_ids
        if job_id not in parsed
        or _derive_state(parsed[job_id].get("state"), parsed[job_id].get("exit_code"))
        == "UNKNOWN"
    ]
    if unresolved_ids:
        squeue_script = _build_squeue_script(unresolved_ids)
        squeue_result = _run_ssh_capture(
            cluster_login,
            squeue_script,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        probes.append(
            StatusProbe(
                source="squeue",
                returncode=squeue_result.returncode,
                stderr=squeue_result.stderr.strip(),
            )
        )
        if squeue_result.returncode == 0:
            squeue_parsed = _parse_status_output(
                squeue_result.stdout, set(unresolved_ids)
            )
            for fields in squeue_parsed.values():
                fields["source"] = "squeue"
            parsed.update(squeue_parsed)

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
                source=_normalized_text(fields.get("source")),
                tracking=job,
            )
        )
    unresolved_job_ids = [
        status.job_id for status in statuses if status.derived_state == "UNKNOWN"
    ]
    return StatusQueryResult(
        statuses=statuses,
        probes=probes,
        unresolved_job_ids=unresolved_job_ids,
    )


def _status_payload(
    tracking_file: Path | None,
    cluster_login: str | None,
    result: StatusQueryResult,
) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "tracking_file": str(tracking_file) if tracking_file else None,
        "cluster_login": cluster_login,
        "probes": [
            {
                "source": probe.source,
                "ok": probe.ok,
                "returncode": probe.returncode,
                "error": None if probe.ok else (probe.stderr or None),
            }
            for probe in result.probes
        ],
        "unresolved_job_ids": result.unresolved_job_ids,
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
                "source": status.source,
            }
            for status in result.statuses
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


def _print_probe_errors(result: StatusQueryResult) -> None:
    if result.ok:
        return
    for probe in result.probes:
        if probe.ok:
            continue
        detail = probe.stderr or f"exit code {probe.returncode}"
        err_console.print(
            f"ERROR: {probe.source} status probe failed: {detail}",
            style="bold red",
        )


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
                    if payload.cluster_login:
                        # A tracking file owns its complete SSH context. In
                        # particular, ssh_config_file=None means use the
                        # caller's normal SSH config; do not combine that alias
                        # with transport settings from an unrelated config.
                        effective_login = payload.cluster_login
                        ssh_config_file = payload.ssh_config_file
                        ssh_options = payload.ssh_options
            except Exception:
                pass
        result = query_job_statuses(
            effective_login,
            jobs,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        if json_output:
            console.print_json(
                data=_status_payload(resolved_tracking, effective_login, result)
            )
            return 0 if result.ok else 1
        print_status_table(resolved_tracking, effective_login, result.statuses)
        _print_probe_errors(result)
        return 0 if result.ok else 1

    resolved_tracking = resolve_tracking_file(tracking_file)
    if resolved_tracking is None:
        message = (
            "No tracking file found. Run a submission first or pass --tracking-file."
        )
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

    result = query_job_statuses(
        payload.cluster_login,
        payload.jobs,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
    )
    if json_output:
        console.print_json(
            data=_status_payload(resolved_tracking, payload.cluster_login, result)
        )
        return 0 if result.ok else 1
    print_status_table(resolved_tracking, payload.cluster_login, result.statuses)
    _print_probe_errors(result)
    return 0 if result.ok else 1
