"""Shared machine-readable payload builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .tracking import JobRecord, TrackingPayload


def error_payload(message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": message}
    payload.update(extra)
    return payload


def stage_payload(
    *,
    config_path: Path | None,
    workspace_mode: str | None,
    remote_workdir: str | None,
    job_folder: str | None,
    commands: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        "config_path": str(config_path) if config_path else None,
        "workspace_mode": workspace_mode,
        "remote_workdir": remote_workdir,
        "job_folder": job_folder,
        "commands": commands,
        "dry_run": dry_run,
    }


def submission_payload(
    *,
    config_path: Path | None,
    workspace_mode: str | None,
    remote_workdir: str | None,
    job_folder: str | None,
    selected_jobs: list[str],
    submitted_jobs: list[dict[str, Any]],
    tracking_file: Path | None,
    commands: list[str],
    monitor_command: str,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "ok": True,
        "config_path": str(config_path) if config_path else None,
        "workspace_mode": workspace_mode,
        "remote_workdir": remote_workdir,
        "job_folder": job_folder,
        "selected_jobs": selected_jobs,
        "submitted_jobs": submitted_jobs,
        "tracking_file": str(tracking_file) if tracking_file else None,
        "commands": commands,
        "monitor_command": monitor_command,
        "dry_run": dry_run,
    }


def validate_payload(
    *,
    ok: bool,
    config_path: Path | None,
    workspace_mode: str | None,
    selected_jobs: list[str],
    warnings: list[str],
    errors: list[str],
    ssh_checked: bool,
    remote_checks: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": ok,
        "valid": ok,
        "config_path": str(config_path) if config_path else None,
        "workspace_mode": workspace_mode,
        "selected_jobs": selected_jobs,
        "warnings": warnings,
        "errors": errors,
        "ssh_checked": ssh_checked,
        "remote_checks": remote_checks,
    }


def render_payload(
    *,
    config_path: Path,
    workspace_mode: str,
    selected_jobs: list[str],
    rendered_jobs: list[dict[str, Any]],
    job_scripts: dict[str, str],
    sbatch_scripts: dict[str, str],
) -> dict[str, Any]:
    return {
        "ok": True,
        "config_path": str(config_path),
        "workspace_mode": workspace_mode,
        "selected_jobs": selected_jobs,
        "rendered_jobs": rendered_jobs,
        "job_scripts": job_scripts,
        "sbatch_scripts": sbatch_scripts,
    }


def monitor_payload(
    *,
    ok: bool,
    tracking_file: Path | None,
    job_ids: list[str],
    command: str | None,
    dry_run: bool,
    returncode: int | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "tracking_file": str(tracking_file) if tracking_file else None,
        "job_ids": job_ids,
        "command": command,
        "dry_run": dry_run,
    }
    if returncode is not None:
        payload["returncode"] = returncode
    if stdout is not None:
        payload["stdout"] = stdout
    if stderr is not None:
        payload["stderr"] = stderr
    return payload


def doctor_payload(
    *,
    cluster_login: str,
    config_path: Path | None,
    ssh_config_file: str | None,
    ssh_options: list[str],
    archive_dir: str,
    archive_dir_source: str,
    ssh_ok: bool | None = None,
    remote_tools: dict[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cluster_login": cluster_login,
        "config_path": str(config_path) if config_path else None,
        "ssh_config_file": ssh_config_file,
        "ssh_options": ssh_options,
        "archive_dir": archive_dir,
        "archive_dir_source": archive_dir_source,
    }
    if ssh_ok is not None:
        payload["ssh_ok"] = ssh_ok
    if remote_tools is not None:
        payload["remote_tools"] = remote_tools
    return payload


def tracking_payload_to_dict(
    payload: TrackingPayload,
    jobs: list[JobRecord | dict[str, object]],
) -> dict[str, Any]:
    job_dicts = []
    for job in jobs:
        if isinstance(job, JobRecord):
            entry: dict[str, Any] = {
                "job_name": job.job_name,
                "job_id": job.job_id,
                "stdout": job.stdout,
                "stderr": job.stderr,
            }
            if job.sbatch_command:
                entry["sbatch_command"] = job.sbatch_command
            if job.remote_sbatch:
                entry["remote_sbatch"] = job.remote_sbatch
            if job.submitted_at:
                entry["submitted_at"] = job.submitted_at
            if job.launcher:
                entry["launcher"] = job.launcher
            if job.artifacts:
                entry["artifacts"] = job.artifacts
            if job.requires:
                entry["requires"] = job.requires
            if job.validators:
                entry["validators"] = job.validators
            job_dicts.append(entry)
            continue
        job_dicts.append(job)
    return {
        "created_at": payload.created_at,
        "cluster_login": payload.cluster_login,
        "ssh_config_file": payload.ssh_config_file,
        "ssh_options": payload.ssh_options,
        "job_folder": payload.job_folder,
        "remote_workdir": payload.remote_workdir,
        "remote_logdir": payload.remote_logdir,
        "remote_slurm_output_dir": payload.remote_slurm_output_dir,
        "jobs": job_dicts,
    }
