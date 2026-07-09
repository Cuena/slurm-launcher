"""Post-run summary generation and updates."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .core import LauncherSettings, RemotePaths
from .status import JobStatus, query_job_statuses
from .tracking import JobRecord, TrackingPayload, load_tracking_payload


@dataclass(frozen=True)
class JobSummary:
    """Rich summary for one job."""

    job_name: str
    job_id: str
    state: str | None
    elapsed: str | None
    remote_output_dir: str | None
    local_artifact_dir: str | None
    command: str | None
    image: str | None
    git_commit: str | None
    started_at: str | None
    finished_at: str | None
    exit_code: str | None


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _local_artifact_dir(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job_name: str,
    job_id: str,
) -> str:
    root = settings.local_artifact_root or (
        settings.project_root / "slurm_output" / "downloaded_artifacts"
    )
    return str(root / remote_paths.job_folder / job_name / job_id)


def build_job_summary(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job: JobRecord,
    status: JobStatus | None = None,
) -> JobSummary:
    """Build a summary for one job, optionally enriched with current status."""
    launcher = job.launcher or {}
    command = launcher.get("entry_command")
    if command is None:
        command = job.sbatch_command
    runtime_artifact = launcher.get("runtime_artifact")
    image = None
    if launcher.get("runtime_kind") == "singularity":
        image = runtime_artifact
    remote_output_dir = None
    if job.stdout:
        remote_output_dir = str(Path(job.stdout).parent)
    return JobSummary(
        job_name=job.job_name,
        job_id=job.job_id,
        state=status.state if status else None,
        elapsed=status.elapsed if status else None,
        remote_output_dir=remote_output_dir,
        local_artifact_dir=_local_artifact_dir(
            settings, remote_paths, job.job_name, job.job_id
        ),
        command=str(command) if command else None,
        image=str(image) if image else None,
        git_commit=_git_commit(settings.project_root),
        started_at=status.submit_time if status else job.submitted_at,
        finished_at=status.end_time if status else None,
        exit_code=status.exit_code if status else None,
    )


def summary_to_dict(summary: JobSummary) -> dict[str, Any]:
    return {
        "job_name": summary.job_name,
        "job_id": summary.job_id,
        "state": summary.state,
        "elapsed": summary.elapsed,
        "remote_output_dir": summary.remote_output_dir,
        "local_artifact_dir": summary.local_artifact_dir,
        "command": summary.command,
        "image": summary.image,
        "git_commit": summary.git_commit,
        "started_at": summary.started_at,
        "finished_at": summary.finished_at,
        "exit_code": summary.exit_code,
    }


def write_submission_summary(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    payload: TrackingPayload,
    *,
    statuses: list[JobStatus] | None = None,
) -> Path:
    """Write local and remote summary files after a submission or status update."""
    status_by_id = {status.job_id: status for status in (statuses or [])}
    summaries = [
        summary_to_dict(
            build_job_summary(
                settings,
                remote_paths,
                job,
                status=status_by_id.get(job.job_id),
            )
        )
        for job in payload.jobs
    ]
    meta = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "cluster_login": payload.cluster_login,
        "remote_workdir": payload.remote_workdir,
        "job_folder": payload.job_folder,
        "summaries": summaries,
    }

    # Local summary
    local_dir = settings.project_root / "slurm_output" / remote_paths.job_folder
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / "summary.json"
    local_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Remote summary
    remote_path = f"{remote_paths.workdir.rstrip('/')}/.slurm_run/summary.json"
    remote_dir = f"{remote_paths.workdir.rstrip('/')}/.slurm_run"
    script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(remote_dir)}",
            f"cat > {shlex.quote(remote_path)} <<'SUMMARY_JSON'",
            json.dumps(meta, indent=2),
            "SUMMARY_JSON",
        ]
    )
    from .core import ssh_script

    ssh_script(
        settings.cluster_login,
        script,
        dry_run=False,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
        quiet=True,
    )
    return local_path


def update_summary_from_status(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    payload: TrackingPayload,
) -> Path:
    """Refresh the summary file using current SLURM status."""
    statuses = query_job_statuses(
        payload.cluster_login,
        payload.jobs,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
    )
    return write_submission_summary(
        settings, remote_paths, payload, statuses=statuses
    )


def load_and_update_summary(
    tracking_file: Path,
    settings: LauncherSettings,
    remote_paths: RemotePaths | None = None,
) -> Path:
    """Load a tracking file and update its summary."""
    payload = load_tracking_payload(tracking_file)
    if remote_paths is None:
        from .core import RemotePaths

        remote_paths = RemotePaths(
            job_folder=payload.job_folder,
            workdir=payload.remote_workdir or "",
            logdir=payload.remote_logdir or "",
            slurm_output_dir=payload.remote_slurm_output_dir or "",
        )
    return update_summary_from_status(settings, remote_paths, payload)
