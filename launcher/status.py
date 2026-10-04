"""Project-scoped job status queries for tracked submissions."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .job_tools import _normalized_text
from .tracking import (
    JobRecord,
    load_tracking_payload,
    resolve_tracking_file,
)
from .transport import run_ssh_capture

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
    tasks: list[JobStatus] = field(default_factory=list)
    array_complete: bool | None = None


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


def _array_indices(expression: str) -> set[int] | None:
    """Expand Slurm's comma/range/stride notation, excluding the throttle."""
    expression = expression.strip("[]").split("%", 1)[0]
    indices: set[int] = set()
    for part in expression.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+)(?::(\d+))?)?", part)
        if match is None:
            return None
        first = int(match[1])
        last = int(match[2] or match[1])
        step = int(match[3] or 1)
        if last < first or step < 1:
            return None
        indices.update(range(first, last + 1, step))
    return indices


def _parse_status_output(
    output: str, job_ids: set[str], *, source: str
) -> dict[str, dict[str, str]]:
    """Keep task identities, never mistake raw allocation IDs for selectors."""
    results: dict[str, dict[str, str]] = {}
    for raw_line in output.splitlines():
        parts = [part.strip() for part in raw_line.split("|")]
        if len(parts) < 8:
            continue
        identity = parts[0]
        if source == "squeue" and len(parts) > 10:
            parent, index = parts[9:11]
            if parent.isdecimal() and index not in {"", "N/A", "4294967294"}:
                identity = f"{parent}_{index}"
        if "." in identity:
            continue  # Job steps are not array tasks or allocations.
        parent, separator, expression = identity.partition("_")
        indices = _array_indices(expression) if separator else None
        identities = (
            [f"{parent}_{index}" for index in sorted(indices)]
            if indices is not None
            else [identity]
        )
        for job_id in identities:
            if job_id not in job_ids and parent not in job_ids:
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
                "source": source,
            }
    return results


def _derive_state(state: str | None, exit_code: str | None) -> str:
    if not state:
        return "UNKNOWN"
    token = state.strip().upper().split()[0].rstrip("+")
    if token == "COMPLETED":
        code = _normalized_text(exit_code)
        if code is None:
            return "UNKNOWN"
        return "DONE" if code == "0:0" else "FAILED"
    if token in {
        "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
        "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED",
    }:
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
                f"sacct --array -X -n -P -j {id_expr} "
                "--format JobID%128,JobName,State%32,ExitCode,Submit,Start,End,Elapsed,Partition"
            ),
        ]
    )


def _build_squeue_script(job_ids: list[str]) -> str:
    id_expr = ",".join(shlex.quote(job_id) for job_id in job_ids)
    return "\n".join(
        [
            "set -euo pipefail",
            "command -v squeue >/dev/null 2>&1",
            f'squeue -r -h -j {id_expr} -o "%i|%j|%T|-|%V|%S|-|%M|%P|%F|%K"',
        ]
    )


def _make_status(
    job_id: str, fields: dict[str, str], tracking: JobRecord | None = None
) -> JobStatus:
    state = _normalized_text(fields.get("state"))
    exit_code = _normalized_text(fields.get("exit_code"))
    return JobStatus(
        job_id=job_id,
        job_name=_normalized_text(fields.get("job_name"))
        or (tracking.job_name if tracking else ""),
        state=state,
        exit_code=exit_code,
        submit_time=_normalized_text(fields.get("submit")),
        start_time=_normalized_text(fields.get("start")),
        end_time=_normalized_text(fields.get("end")),
        elapsed=_normalized_text(fields.get("elapsed")),
        partition=_normalized_text(fields.get("partition")),
        derived_state=_derive_state(state, exit_code),
        source=_normalized_text(fields.get("source")),
        tracking=tracking,
    )


def _aggregate_array(
    job: JobRecord, parsed: dict[str, dict[str, str]], *, queue_ok: bool
) -> JobStatus:
    tasks = [
        _make_status(identity, fields)
        for identity, fields in parsed.items()
        if identity.startswith(f"{job.job_id}_")
    ]
    spec = job.array_spec
    expected = _array_indices(spec) if spec else None
    if expected is not None:
        present = {task.job_id for task in tasks}
        for index in sorted(expected):
            identity = f"{job.job_id}_{index}"
            if identity not in present:
                tasks.append(_make_status(identity, {}))
    tasks.sort(key=lambda task: (
        int(task.job_id.split("_", 1)[1])
        if task.job_id.split("_", 1)[1].isdecimal() else -1
    ))
    complete = (
        expected is not None
        and queue_ok
        and all(task.derived_state != "UNKNOWN" for task in tasks)
        and {task.job_id for task in tasks}
        == {f"{job.job_id}_{index}" for index in expected}
    )
    states = {task.derived_state for task in tasks}
    if "FAILED" in states:
        derived = "FAILED"
    elif states & {"RUNNING", "SUSPENDED", "STOPPED"}:
        derived = "RUNNING"
    elif "PENDING" in states:
        derived = "PENDING"
    elif states == {"DONE"} and complete:
        derived = "DONE"
    else:
        derived = "UNKNOWN"
    sources = sorted({task.source for task in tasks if task.source})
    return JobStatus(
        job_id=job.job_id,
        job_name=job.job_name or (tasks[0].job_name if tasks else ""),
        state={"DONE": "COMPLETED"}.get(derived, derived),
        exit_code=None,
        submit_time=None,
        start_time=None,
        end_time=None,
        elapsed=None,
        partition=None,
        derived_state=derived,
        source="+".join(sources) or None,
        tracking=job,
        tasks=tasks,
        array_complete=complete,
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
    sacct_result = run_ssh_capture(
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
        parsed = _parse_status_output(sacct_result.stdout, id_set, source="sacct")

    # A completed allocation is not evidence that its array siblings finished.
    # Probe every detected/known array parent, including terminal accounting rows.
    array_parents = {
        job.job_id for job in runnable
        if "_" not in job.job_id
        and (job.array_spec or any(
            identity.startswith(f"{job.job_id}_") for identity in parsed
        ))
    }
    unresolved_ids = [
        job_id for job_id in job_ids
        if job_id in array_parents or job_id not in parsed
        or _derive_state(parsed[job_id].get("state"), parsed[job_id].get("exit_code"))
        == "UNKNOWN"
    ]
    queue_ok = False
    if unresolved_ids:
        squeue_script = _build_squeue_script(unresolved_ids)
        squeue_result = run_ssh_capture(
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
            queue_ok = True
            squeue_parsed = _parse_status_output(
                squeue_result.stdout, set(unresolved_ids), source="squeue"
            )
            for identity, fields in squeue_parsed.items():
                previous = parsed.get(identity, {})
                if _derive_state(previous.get("state"), previous.get("exit_code")) != "FAILED":
                    parsed[identity] = fields

    statuses: list[JobStatus] = []
    for job in runnable:
        is_array = "_" not in job.job_id and (
            job.job_id in array_parents
            or any(identity.startswith(f"{job.job_id}_") for identity in parsed)
        )
        if is_array:
            statuses.append(_aggregate_array(job, parsed, queue_ok=queue_ok))
        else:
            statuses.append(_make_status(job.job_id, parsed.get(job.job_id, {}), job))
    unresolved_job_ids = [
        status.job_id for status in statuses if status.derived_state == "UNKNOWN"
    ]
    return StatusQueryResult(
        statuses=statuses,
        probes=probes,
        unresolved_job_ids=unresolved_job_ids,
    )


def status_payload(status: JobStatus) -> dict[str, Any]:
    """Serialize scheduler evidence without duplicating saved tracking metadata."""
    result = {
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
    if status.array_complete is not None:
        result["array_complete"] = status.array_complete
        result["tasks"] = [status_payload(task) for task in status.tasks]
    return result


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
        "jobs": [status_payload(status) for status in result.statuses],
    }


def print_status_table(
    tracking_file: Path | None,
    cluster_login: str | None,
    statuses: list[JobStatus],
) -> None:
    heading = f"[bold]Cluster:[/bold] {cluster_login or '-'}"
    if tracking_file:
        heading = f"[bold]Tracking file:[/bold] {tracking_file}\n{heading}"
    console.print(Panel.fit(heading, title="Status", border_style="cyan"))

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
    selected_jobs: list[str] | None = None,
    json_output: bool = False,
) -> int:
    """Project-scoped status command.

    Either queries a single job by id (using the provided cluster login) or
    resolves the latest tracking file and queries all tracked jobs.
    """
    resolved_tracking: Path | None = None
    if job_id and cluster_login:
        jobs = [JobRecord(job_name="", job_id=job_id)]
    else:
        resolved_tracking = resolve_tracking_file(tracking_file)
        if resolved_tracking is None:
            message = "Run not found. Pass --run ID, a tracking path, or latest."
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

        jobs = payload.filter_jobs(names=set(selected_jobs) if selected_jobs else None)
        missing = set(selected_jobs or ()) - {job.job_name for job in jobs}
        if missing:
            message = f"Jobs not found in run: {', '.join(sorted(missing))}"
            if json_output:
                console.print_json(data={"ok": False, "error": message})
            else:
                err_console.print(message)
            return 1
        cluster_login = payload.cluster_login
        ssh_config_file = payload.ssh_config_file
        ssh_options = payload.ssh_options

    result = query_job_statuses(
        cluster_login,
        jobs,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if json_output:
        console.print_json(
            data=_status_payload(resolved_tracking, cluster_login, result)
        )
    else:
        print_status_table(resolved_tracking, cluster_login, result.statuses)
        _print_probe_errors(result)
    return 0 if result.ok else 1
