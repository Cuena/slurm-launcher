from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import build_ssh_command, resolve_log_path
from .tracking import all_tracking_files, load_tracking_payload

console = Console()
err_console = Console(stderr=True)
DEFAULT_ARCHIVE_DIR = Path.home() / ".slurm-dashboard" / "logs"
LAUNCHER_METADATA_PREFIX = "# slurm-launcher-metadata:"


@dataclass(frozen=True)
class RecentJob:
    job_id: str
    job_name: str
    state: str
    partition: str
    submit: str
    start: str
    end: str
    elapsed: str
    source: str


@dataclass(frozen=True)
class JobLogInfo:
    job_id: str
    job_name: str
    state: str
    stdout: str | None
    stderr: str | None
    source: str


@dataclass(frozen=True)
class LauncherInfo:
    managed: bool
    runtime_kind: str | None
    runtime_artifact: str | None
    entry_command: str | None


@dataclass(frozen=True)
class JobDetails:
    job_id: str
    job_name: str | None
    state: str | None
    partition: str | None
    command: str | None
    work_dir: str | None
    stdout: str | None
    stderr: str | None
    node_list: str | None
    num_nodes: str | None
    gres: str | None
    submit_time: str | None
    start_time: str | None
    end_time: str | None
    source: str
    detail_level: str = "full"
    launcher: LauncherInfo | None = None


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"(null)", "None", "N/A"}:
        return None
    return text


def _parse_one_line_fields(output: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in output.strip().split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields[key] = value
    return fields


def _gres_from_fields(fields: dict[str, str]) -> str | None:
    for key in ("Gres", "ReqGRES", "TresPerNode", "TresPerTask", "TRES"):
        value = _normalized_text(fields.get(key))
        if value:
            return value
    return None


def _job_details_from_scontrol(output: str, job_id: str) -> JobDetails | None:
    fields = _parse_one_line_fields(output)
    if not fields:
        return None
    return JobDetails(
        job_id=_normalized_text(fields.get("JobId")) or job_id,
        job_name=_normalized_text(fields.get("JobName")),
        state=_normalized_text(fields.get("JobState")),
        partition=_normalized_text(fields.get("Partition")),
        command=_normalized_text(fields.get("Command")),
        work_dir=_normalized_text(fields.get("WorkDir")),
        stdout=resolve_log_path(
            _normalized_text(fields.get("StdOut")),
            job_id,
        ),
        stderr=resolve_log_path(
            _normalized_text(fields.get("StdErr")),
            job_id,
        ),
        node_list=_normalized_text(fields.get("NodeList")),
        num_nodes=_normalized_text(fields.get("NumNodes")),
        gres=_gres_from_fields(fields),
        submit_time=_normalized_text(fields.get("SubmitTime")),
        start_time=_normalized_text(fields.get("StartTime")),
        end_time=_normalized_text(fields.get("EndTime")),
        source="scontrol",
    )


def _job_details_from_sacct(output: str, job_id: str) -> JobDetails | None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|", 13)]
        if len(parts) < 13:
            continue
        if parts[0] != job_id:
            continue
        command = _normalized_text(parts[13]) if len(parts) > 13 else None
        return JobDetails(
            job_id=parts[0],
            job_name=_normalized_text(parts[1]),
            state=_normalized_text(parts[2]),
            partition=_normalized_text(parts[3]),
            command=command,
            work_dir=_normalized_text(parts[10]),
            stdout=resolve_log_path(_normalized_text(parts[11]), job_id),
            stderr=resolve_log_path(_normalized_text(parts[12]), job_id),
            node_list=_normalized_text(parts[7]),
            num_nodes=_normalized_text(parts[8]),
            gres=_normalized_text(parts[9]),
            submit_time=_normalized_text(parts[4]),
            start_time=_normalized_text(parts[5]),
            end_time=_normalized_text(parts[6]),
            source="sacct",
        )
    return None


def _merge_job_details(primary: JobDetails, secondary: JobDetails) -> JobDetails:
    def pick(first: str | None, second: str | None) -> str | None:
        return first if first else second

    source = primary.source
    if secondary.source and secondary.source != primary.source:
        source = f"{primary.source}+{secondary.source}"
    return JobDetails(
        job_id=pick(primary.job_id, secondary.job_id) or secondary.job_id,
        job_name=pick(primary.job_name, secondary.job_name),
        state=pick(primary.state, secondary.state),
        partition=pick(primary.partition, secondary.partition),
        command=pick(primary.command, secondary.command),
        work_dir=pick(primary.work_dir, secondary.work_dir),
        stdout=pick(primary.stdout, secondary.stdout),
        stderr=pick(primary.stderr, secondary.stderr),
        node_list=pick(primary.node_list, secondary.node_list),
        num_nodes=pick(primary.num_nodes, secondary.num_nodes),
        gres=pick(primary.gres, secondary.gres),
        submit_time=pick(primary.submit_time, secondary.submit_time),
        start_time=pick(primary.start_time, secondary.start_time),
        end_time=pick(primary.end_time, secondary.end_time),
        source=source,
        detail_level=(
            "full"
            if "full" in {primary.detail_level, secondary.detail_level}
            else primary.detail_level
        ),
        launcher=primary.launcher or secondary.launcher,
    )


def _job_details_from_log_info(info: JobLogInfo) -> JobDetails:
    return JobDetails(
        job_id=info.job_id,
        job_name=info.job_name or None,
        state=info.state or None,
        partition=None,
        command=None,
        work_dir=None,
        stdout=info.stdout,
        stderr=info.stderr,
        node_list=None,
        num_nodes=None,
        gres=None,
        submit_time=None,
        start_time=None,
        end_time=None,
        source=info.source,
        detail_level="log-resolution",
    )


def _launcher_info_from_payload(payload: dict[str, Any]) -> LauncherInfo | None:
    managed = bool(payload.get("managed"))
    if not managed:
        return None
    return LauncherInfo(
        managed=True,
        runtime_kind=_normalized_text(payload.get("runtime_kind")),
        runtime_artifact=_normalized_text(payload.get("runtime_artifact")),
        entry_command=_normalized_text(payload.get("entry_command")),
    )


def _launcher_info_from_tracking(job_id: str) -> LauncherInfo | None:
    for tracking_file in all_tracking_files():
        try:
            payload = load_tracking_payload(tracking_file)
        except Exception:
            continue
        for job in payload.jobs:
            if job.job_id.strip() != job_id:
                continue
            if job.launcher:
                parsed = _launcher_info_from_payload(job.launcher)
                if parsed is not None:
                    return parsed
            runtime_artifact = _normalized_text(
                payload.singularity_image_path
            ) or _normalized_text(payload.venv_python_executable)
            return LauncherInfo(
                managed=True,
                runtime_kind=_normalized_text(payload.runtime_mode),
                runtime_artifact=runtime_artifact,
                entry_command=None,
            )
    return None


def _launcher_info_from_script_text(output: str) -> LauncherInfo | None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line.startswith(LAUNCHER_METADATA_PREFIX):
            continue
        try:
            payload = json.loads(line.split(":", 1)[1].strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        return _launcher_info_from_payload(payload)
    return None


def _launcher_info_from_script(
    cluster_login: str,
    command: str | None,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> LauncherInfo | None:
    command_path = _normalized_text(command)
    if not command_path or not command_path.endswith(".sbatch"):
        return None
    result = _run_ssh_capture(
        cluster_login,
        f"test -f {shlex.quote(command_path)} && sed -n '1,20p' {shlex.quote(command_path)}",
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if result.returncode != 0:
        return None
    return _launcher_info_from_script_text(result.stdout)


def resolve_job_details(
    cluster_login: str,
    job_id: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    enrich_launcher: bool = True,
) -> JobDetails | None:
    scontrol_details: JobDetails | None = None
    sacct_details: JobDetails | None = None

    scontrol_result = _run_ssh_capture(
        cluster_login,
        f"scontrol show job -o {shlex.quote(job_id)}",
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if scontrol_result.returncode == 0:
        scontrol_details = _job_details_from_scontrol(scontrol_result.stdout, job_id)

    sacct_result = _run_ssh_capture(
        cluster_login,
        (
            "command -v sacct >/dev/null 2>&1 && "
            f"sacct -X -n -P -j {shlex.quote(job_id)} "
            "--format JobIDRaw,JobName,State,Partition,Submit,Start,End,"
            "NodeList,NNodes,ReqGRES,WorkDir,StdOut,StdErr,SubmitLine"
        ),
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if sacct_result.returncode == 0:
        sacct_details = _job_details_from_sacct(sacct_result.stdout, job_id)

    details = scontrol_details or sacct_details
    if details is None:
        scontrol_log_info = (
            _job_log_info_from_scontrol(scontrol_result.stdout, job_id)
            if scontrol_result.returncode == 0
            else None
        )
        if scontrol_log_info is not None:
            details = _job_details_from_log_info(scontrol_log_info)
        else:
            sacct_log_result = _run_ssh_capture(
                cluster_login,
                (
                    "command -v sacct >/dev/null 2>&1 && "
                    f"sacct -X -n -P -j {shlex.quote(job_id)} "
                    "--format JobIDRaw,JobName,State,StdOut,StdErr"
                ),
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            )
            if sacct_log_result.returncode == 0:
                sacct_log_info = _job_log_info_from_sacct(
                    sacct_log_result.stdout,
                    job_id,
                )
                if sacct_log_info is not None:
                    details = _job_details_from_log_info(sacct_log_info)
        if details is None:
            return None
    if scontrol_details is not None and sacct_details is not None:
        details = _merge_job_details(scontrol_details, sacct_details)
    if not enrich_launcher:
        return details

    launcher = _launcher_info_from_tracking(job_id)
    if launcher is None:
        launcher = _launcher_info_from_script(
            cluster_login,
            details.command,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
    return JobDetails(
        job_id=details.job_id,
        job_name=details.job_name,
        state=details.state,
        partition=details.partition,
        command=details.command,
        work_dir=details.work_dir,
        stdout=details.stdout,
        stderr=details.stderr,
        node_list=details.node_list,
        num_nodes=details.num_nodes,
        gres=details.gres,
        submit_time=details.submit_time,
        start_time=details.start_time,
        end_time=details.end_time,
        source=details.source,
        detail_level=details.detail_level,
        launcher=launcher,
    )


def _job_details_payload(details: JobDetails) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": details.job_id,
        "resolved_via": details.source,
        "detail_level": details.detail_level,
    }
    optional_fields = {
        "job_name": details.job_name,
        "state": details.state,
        "partition": details.partition,
        "command": details.command,
        "work_dir": details.work_dir,
        "stdout": details.stdout,
        "stderr": details.stderr,
        "node_list": details.node_list,
        "num_nodes": details.num_nodes,
        "gres": details.gres,
        "submit_time": details.submit_time,
        "start_time": details.start_time,
        "end_time": details.end_time,
    }
    for field_name, value in optional_fields.items():
        if value is not None:
            payload[field_name] = value
    launcher = None
    if details.launcher is not None:
        launcher = {
            "managed": details.launcher.managed,
            "runtime_kind": details.launcher.runtime_kind,
            "runtime_artifact": details.launcher.runtime_artifact,
            "entry_command": details.launcher.entry_command,
        }
    if launcher is not None:
        payload["launcher"] = launcher
    return payload


def effective_archive_dir(archive_dir: str | None) -> tuple[str, str]:
    if archive_dir and archive_dir.strip():
        return archive_dir.strip(), "config"
    return str(DEFAULT_ARCHIVE_DIR), "default"


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


def _parse_recent_jobs(output: str, *, source: str) -> list[RecentJob]:
    jobs: list[RecentJob] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if source == "sacct":
            if len(parts) < 8:
                continue
            jobs.append(
                RecentJob(
                    job_id=parts[0],
                    job_name=parts[1],
                    state=parts[2],
                    partition=parts[3],
                    submit=parts[4],
                    start=parts[5],
                    end=parts[6],
                    elapsed=parts[7],
                    source=source,
                )
            )
            continue
        if len(parts) < 5:
            continue
        jobs.append(
            RecentJob(
                job_id=parts[0],
                job_name=parts[1],
                state=parts[2],
                partition=parts[3],
                submit="-",
                start="-",
                end="-",
                elapsed=parts[4],
                source=source,
            )
        )
    return jobs


def _normalized_state_token(state: str) -> str:
    normalized = state.strip().upper()
    if not normalized:
        return ""
    return normalized.split()[0]


def _filter_recent_jobs(
    jobs: list[RecentJob],
    *,
    states: set[str] | None,
) -> list[RecentJob]:
    if not states:
        return jobs
    normalized_states = {_normalized_state_token(state) for state in states if state}
    return [
        job
        for job in jobs
        if _normalized_state_token(job.state) in normalized_states
    ]


def _recent_jobs_sort_key(job: RecentJob) -> tuple[str, str, str, str]:
    return (job.submit, job.start, job.end, job.job_id)


def list_recent_jobs(
    cluster_login: str,
    *,
    user: str | None,
    hours: int,
    limit: int,
    states: set[str] | None,
    json_output: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> int:
    if hours < 1:
        err_console.print("ERROR: --hours must be at least 1.", style="bold red")
        return 1
    if limit < 1:
        err_console.print("ERROR: --limit must be at least 1.", style="bold red")
        return 1

    user_expr = shlex.quote(user) if user else '"${USER:-$(whoami)}"'
    script = "\n".join(
        [
            "set -euo pipefail",
            f"user_name={user_expr}",
            "if command -v sacct >/dev/null 2>&1; then",
            '  echo "__SOURCE__=sacct"',
            (
                f"  sacct -X -n -P --starttime now-{int(hours)}hours "
                "--format JobIDRaw,JobName,State,Partition,Submit,Start,End,Elapsed "
                '-u "$user_name"'
            ),
            "else",
            '  echo "__SOURCE__=squeue"',
            '  squeue -h -u "$user_name" -o "%i|%j|%T|%P|%M"',
            "fi",
        ]
    )
    result = _run_ssh_capture(
        cluster_login,
        script,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if result.returncode != 0:
        err_console.print(
            f"ERROR: Failed to query jobs on {cluster_login}: {result.stderr.strip()}",
            style="bold red",
        )
        return result.returncode or 1

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if lines and lines[0].startswith("__SOURCE__="):
        source = lines[0].split("=", 1)[1].strip() or "unknown"
        output = "\n".join(lines[1:])
    else:
        source = "unknown"
        output = result.stdout

    jobs = _parse_recent_jobs(output, source=source)
    jobs = _filter_recent_jobs(jobs, states=states)
    jobs.sort(key=_recent_jobs_sort_key, reverse=True)
    jobs = jobs[:limit]

    payload = {
        "cluster_login": cluster_login,
        "user": user or "$USER",
        "hours": hours,
        "limit": limit,
        "states": sorted(states) if states else [],
        "source": source,
        "jobs": [
            {
                "job_id": job.job_id,
                "job_name": job.job_name,
                "state": job.state,
                "partition": job.partition,
                "submit": job.submit,
                "start": job.start,
                "end": job.end,
                "elapsed": job.elapsed,
            }
            for job in jobs
        ],
    }
    if json_output:
        console.print_json(data=payload)
        return 0

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Cluster:[/bold] {cluster_login}",
                    f"[bold]User:[/bold] {user or '$USER'}",
                    f"[bold]Source:[/bold] {source}",
                    f"[bold]Window:[/bold] last {hours}h",
                    f"[bold]States:[/bold] {', '.join(sorted(states)) if states else 'all'}",
                ]
            ),
            title="Recent Jobs",
            border_style="cyan",
        )
    )

    if not jobs:
        console.print("No jobs found.", style="yellow")
        return 0

    table = Table()
    table.add_column("Job ID")
    table.add_column("Name")
    table.add_column("State")
    table.add_column("Partition")
    table.add_column("Submit")
    table.add_column("Elapsed")
    for job in jobs:
        table.add_row(
            job.job_id,
            job.job_name,
            job.state,
            job.partition,
            job.submit,
            job.elapsed,
        )
    console.print(table)
    return 0


def show_job_details(
    cluster_login: str,
    job_id: str,
    *,
    json_output: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> int:
    details = resolve_job_details(
        cluster_login,
        job_id,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if details is None:
        err_console.print(
            f"ERROR: Could not resolve job details for job {job_id}.",
            style="bold red",
        )
        return 1

    payload = _job_details_payload(details)
    if json_output:
        console.print_json(data=payload)
        return 0

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Cluster:[/bold] {cluster_login}",
                    f"[bold]Job ID:[/bold] {details.job_id}",
                    f"[bold]Resolved via:[/bold] {details.source}",
                ]
            ),
            title="Job Show",
            border_style="cyan",
        )
    )

    table = Table(show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for field_name, value in payload.items():
        if field_name == "launcher":
            rendered = json.dumps(value) if value is not None else "null"
        else:
            rendered = str(value) if value is not None else "-"
        table.add_row(field_name, rendered)
    console.print(table)
    return 0


def _job_log_info_from_scontrol(output: str, job_id: str) -> JobLogInfo | None:
    fields = _parse_one_line_fields(output)
    if not fields:
        return None
    return JobLogInfo(
        job_id=job_id,
        job_name=_normalized_text(fields.get("JobName")) or "",
        state=_normalized_text(fields.get("JobState")) or "",
        stdout=resolve_log_path(_normalized_text(fields.get("StdOut")), job_id),
        stderr=resolve_log_path(_normalized_text(fields.get("StdErr")), job_id),
        source="scontrol",
    )


def _job_log_info_from_sacct(output: str, job_id: str) -> JobLogInfo | None:
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 5:
            continue
        record_job_id = parts[0]
        if record_job_id != job_id:
            continue
        return JobLogInfo(
            job_id=record_job_id,
            job_name=parts[1],
            state=parts[2],
            stdout=resolve_log_path(parts[3] or None, job_id),
            stderr=resolve_log_path(parts[4] or None, job_id),
            source="sacct",
        )
    return None


def resolve_job_log_info(
    cluster_login: str,
    job_id: str,
    *,
    archive_dir: str | None,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> JobLogInfo | None:
    probe_errors: list[str] = []

    scontrol_result = _run_ssh_capture(
        cluster_login,
        f"scontrol show job -o {shlex.quote(job_id)}",
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if scontrol_result.returncode == 0:
        info = _job_log_info_from_scontrol(scontrol_result.stdout, job_id)
        if info and (info.stdout or info.stderr):
            return info
        probe_errors.append("scontrol returned no log paths")
    else:
        probe_errors.append(f"scontrol failed (rc={scontrol_result.returncode})")

    sacct_result = _run_ssh_capture(
        cluster_login,
        (
            "command -v sacct >/dev/null 2>&1 && "
            f"sacct -X -n -P -j {shlex.quote(job_id)} "
            "--format JobIDRaw,JobName,State,StdOut,StdErr"
        ),
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if sacct_result.returncode == 0:
        info = _job_log_info_from_sacct(sacct_result.stdout, job_id)
        if info and (info.stdout or info.stderr):
            return info
        probe_errors.append("sacct returned no log paths")
    else:
        probe_errors.append(f"sacct failed (rc={sacct_result.returncode})")

    archive_root, archive_source = effective_archive_dir(archive_dir)
    archive_root = archive_root.rstrip("/")
    fallback_detail = "; ".join(probe_errors)
    return JobLogInfo(
        job_id=job_id,
        job_name="",
        state="",
        stdout=f"{archive_root}/{job_id}.out",
        stderr=f"{archive_root}/{job_id}.err",
        source=f"archive:{archive_source} (fallback: {fallback_detail})",
    )


def show_job_log(
    cluster_login: str,
    job_id: str,
    *,
    stream: str,
    lines: int,
    follow: bool,
    full: bool,
    path_only: bool,
    json_output: bool,
    archive_dir: str | None,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> int:
    if lines < 1:
        err_console.print("ERROR: --lines must be at least 1.", style="bold red")
        return 1

    info = resolve_job_log_info(
        cluster_login,
        job_id,
        archive_dir=archive_dir,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if info is None:
        err_console.print(
            f"ERROR: Could not resolve log paths for job {job_id}.",
            style="bold red",
        )
        return 1

    target_path = info.stdout if stream == "stdout" else info.stderr
    if not target_path:
        err_console.print(
            f"ERROR: No {stream} path available for job {job_id}.",
            style="bold red",
        )
        return 1

    payload = {
        "job_id": info.job_id,
        "job_name": info.job_name or None,
        "state": info.state or None,
        "stream": stream,
        "path": target_path,
        "resolved_via": info.source,
    }
    if json_output:
        console.print_json(data=payload)
        return 0

    if path_only:
        console.print(target_path)
        return 0

    is_fallback = info.source.startswith("archive:")
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Cluster:[/bold] {cluster_login}",
                    f"[bold]Job ID:[/bold] {job_id}",
                    f"[bold]Job name:[/bold] {info.job_name or '-'}",
                    f"[bold]State:[/bold] {info.state or '-'}",
                    f"[bold]Stream:[/bold] {stream}",
                    f"[bold]Path:[/bold] {target_path}",
                    f"[bold]Resolved via:[/bold] {info.source}",
                ]
            ),
            title="Job Log",
            border_style="cyan",
        )
    )
    if is_fallback:
        err_console.print(
            f"WARNING: Log path for job {job_id} is a best-guess archive fallback. "
            "scontrol/sacct could not resolve the actual path.",
            style="yellow",
        )

    if full:
        remote_command = f"cat {shlex.quote(target_path)}"
    else:
        tail_args = ["tail", "-n", str(lines)]
        if follow:
            tail_args.append("-f")
        tail_args.append(target_path)
        remote_command = shlex.join(tail_args)

    result = subprocess.run(
        [
            *build_ssh_command(
                cluster_login,
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            ),
            remote_command,
        ],
        check=False,
    )
    if result.returncode != 0:
        err_console.print(
            f"ERROR: Failed to read {stream} for job {job_id}.",
            style="bold red",
        )
        return result.returncode or 1
    return 0
