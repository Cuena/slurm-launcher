"""Shared machine-readable payload builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def error_payload(message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"ok": False, "error": message}
    payload.update(extra)
    return payload


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
