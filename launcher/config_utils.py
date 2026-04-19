"""Shared config and job normalization helpers for the CLI."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from .core import JobSpec, LauncherSettings, resolve_local_project_path

WORKSPACE_MODES = {"per-run", "fixed"}


def load_config(config_path: Path) -> ModuleType:
    config_path = config_path.resolve()
    spec = importlib.util.spec_from_file_location("remote_launcher_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def normalize_workspace_mode(value: Any, *, setting_name: str) -> str:
    mode = str(value).strip().lower()
    if mode in WORKSPACE_MODES:
        return mode
    raise SystemExit(f"ERROR: {setting_name} must be one of: per-run, fixed.")


def build_settings(
    config: ModuleType,
    config_path: Path,
    *,
    workspace_mode_override: str | None = None,
) -> LauncherSettings:
    cluster_login = getattr(config, "CLUSTER_LOGIN", None)
    ssh_config_file = getattr(config, "SSH_CONFIG_FILE", None)
    ssh_options = ensure_list(getattr(config, "SSH_OPTIONS", []))
    remote_workspace_base = getattr(config, "REMOTE_WORKSPACE_BASE", None)
    remote_workspace_dir = getattr(config, "REMOTE_WORKSPACE_DIR", None)
    configured_workspace_mode = getattr(config, "WORKSPACE_MODE", "per-run")
    workspace_mode = normalize_workspace_mode(
        workspace_mode_override or configured_workspace_mode,
        setting_name="WORKSPACE_MODE",
    )

    if not cluster_login:
        raise SystemExit("ERROR: Config must define CLUSTER_LOGIN.")

    remote_log_base_path = getattr(config, "REMOTE_LOG_BASE_PATH", None)
    if not remote_log_base_path:
        remote_log_base_path = remote_workspace_base or remote_workspace_dir
    if not remote_log_base_path:
        raise SystemExit(
            "ERROR: Config must define REMOTE_LOG_BASE_PATH and one workspace path "
            "(REMOTE_WORKSPACE_BASE/REMOTE_WORKSPACE_DIR)."
        )
    if workspace_mode == "per-run" and not remote_workspace_base:
        raise SystemExit(
            "ERROR: REMOTE_WORKSPACE_BASE is required for WORKSPACE_MODE='per-run'."
        )
    if workspace_mode == "fixed" and not remote_workspace_dir:
        raise SystemExit(
            "ERROR: REMOTE_WORKSPACE_DIR is required for WORKSPACE_MODE='fixed'."
        )

    remote_slurm_dashboard_log_archive_dir = getattr(
        config, "REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR", None
    )
    remote_slurm_dashboard_log_view_dir = getattr(
        config, "REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR", None
    )
    local_root = Path(getattr(config, "LOCAL_ROOT", config_path.parent)).resolve()
    project_prefix = getattr(
        config,
        "PROJECT_NAME",
        local_root.name.replace(" ", "_") or "project",
    )

    runtime_mode = str(getattr(config, "RUNTIME_MODE", "native")).lower()
    allowed_runtimes = {"native", "venv", "singularity"}
    if runtime_mode not in allowed_runtimes:
        raise SystemExit(
            "ERROR: RUNTIME_MODE must be one of: native, venv, singularity."
        )

    venv_python = getattr(config, "VENV_PYTHON_EXECUTABLE", None)
    singularity_image = getattr(config, "SINGULARITY_IMAGE_PATH", None)
    if hasattr(config, "SINGULARITY_EXTRA_ARGS"):
        raise SystemExit(
            "ERROR: SINGULARITY_EXTRA_ARGS was removed. "
            "Rename it to SINGULARITY_EXEC_FLAGS."
        )
    singularity_exec_flags = [
        str(arg) for arg in getattr(config, "SINGULARITY_EXEC_FLAGS", [])
    ]

    if runtime_mode == "venv":
        if not venv_python:
            raise SystemExit(
                "ERROR: Set VENV_PYTHON_EXECUTABLE when RUNTIME_MODE='venv'."
            )
    elif runtime_mode == "singularity":
        if not singularity_image:
            raise SystemExit(
                "ERROR: Set SINGULARITY_IMAGE_PATH when RUNTIME_MODE='singularity'."
            )
    default_env = dict(getattr(config, "DEFAULT_ENV", {}))
    default_sbatch = dict(getattr(config, "DEFAULT_SBATCH", {}))
    extra_rsync_excludes = [
        str(item) for item in getattr(config, "EXTRA_RSYNC_EXCLUDES", [])
    ]
    extra_rsync_args = [str(item) for item in getattr(config, "EXTRA_RSYNC_ARGS", [])]
    artifact_paths = ensure_list(getattr(config, "ARTIFACT_PATHS", []))
    verbose = bool(getattr(config, "VERBOSE", False))

    return LauncherSettings(
        cluster_login=cluster_login,
        ssh_config_file=(str(ssh_config_file) if ssh_config_file else None),
        ssh_options=ssh_options,
        remote_workspace_base=(
            str(remote_workspace_base) if remote_workspace_base else None
        ),
        remote_log_base_path=str(remote_log_base_path),
        workspace_mode=workspace_mode,
        remote_workspace_dir=(
            str(remote_workspace_dir) if remote_workspace_dir else None
        ),
        project_root=local_root,
        project_prefix=project_prefix,
        venv_python_executable=(str(venv_python) if venv_python else None),
        default_env=default_env,
        default_sbatch=default_sbatch,
        extra_rsync_excludes=extra_rsync_excludes,
        extra_rsync_args=extra_rsync_args,
        remote_slurm_dashboard_log_archive_dir=(
            str(remote_slurm_dashboard_log_archive_dir)
            if remote_slurm_dashboard_log_archive_dir
            else None
        ),
        remote_slurm_dashboard_log_view_dir=(
            str(remote_slurm_dashboard_log_view_dir)
            if remote_slurm_dashboard_log_view_dir
            else None
        ),
        runtime_mode=runtime_mode,
        singularity_image_path=(str(singularity_image) if singularity_image else None),
        singularity_exec_flags=singularity_exec_flags,
        artifact_paths=artifact_paths,
        verbose=verbose,
    )


def coerce_job(entry: Any) -> JobSpec:
    if isinstance(entry, JobSpec):
        return entry
    if isinstance(entry, dict):
        name = str(entry["name"])
        if "python" in entry:
            raise ValueError(
                f"Job '{name}' uses unsupported key 'python'. "
                "Use a single explicit 'command' string."
            )
        if "script" in entry or "entrypoint" in entry:
            raise ValueError(
                f"Job '{name}' uses unsupported keys ('script'/'entrypoint'). "
                "Use a single explicit 'command' string."
            )
        if "args" in entry or "shell" in entry or "interpreter" in entry:
            raise ValueError(
                f"Job '{name}' uses unsupported keys ('args'/'shell'/'interpreter'). "
                "Use a single explicit 'command' string."
            )
        has_command = "command" in entry and bool(str(entry.get("command", "")).strip())
        has_sbatch_file = "sbatch_file" in entry and bool(
            str(entry.get("sbatch_file", "")).strip()
        )
        if has_command == has_sbatch_file:
            raise ValueError(
                f"Job '{name}' must define exactly one of 'command' or 'sbatch_file'."
            )
        command = str(entry["command"]) if has_command else None
        sbatch_file = str(entry["sbatch_file"]) if has_sbatch_file else None
        return JobSpec(
            name=name,
            command=command,
            sbatch_file=sbatch_file,
            sbatch_args=ensure_list(entry.get("sbatch_args")),
            env=dict(entry.get("env") or {}),
            sbatch=dict(entry.get("sbatch") or {}),
            setup=ensure_list(entry.get("setup")),
        )
    raise TypeError(f"Unsupported job entry: {entry!r}")


def select_jobs(jobs: list[JobSpec], run_only: list[str] | None) -> list[JobSpec]:
    if not run_only:
        return jobs
    wanted = set(run_only)
    available = {job.name for job in jobs}
    missing = wanted.difference(available)
    if missing:
        raise SystemExit(f"ERROR: Requested jobs not found: {sorted(missing)}")
    return [job for job in jobs if job.name in wanted]


def prepare_jobs(
    config: ModuleType, run_only: list[str] | None, default_env: dict[str, Any]
) -> list[JobSpec]:
    raw_jobs = getattr(config, "JOBS", None)
    if not raw_jobs:
        raise SystemExit("ERROR: Config must define JOBS.")
    jobs = [coerce_job(entry) for entry in raw_jobs]
    for job in jobs:
        if job.uses_sbatch_file():
            continue
        job.env = {**default_env, **job.env}
    return select_jobs(jobs, run_only)


def fail_duplicate_jobs(jobs: list[JobSpec]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for job in jobs:
        if job.name in seen:
            duplicates.add(job.name)
        seen.add(job.name)
    if duplicates:
        raise SystemExit(f"ERROR: Duplicate job names found: {sorted(duplicates)}")


def fail_if_not_absolute(label: str, value: str | None) -> None:
    if value is None:
        return
    if not str(value).startswith("/"):
        raise SystemExit(f"ERROR: {label} must be an absolute path. Got: {value!r}")


def resolve_local_sbatch_file_path(
    settings: LauncherSettings, sbatch_file: str
) -> Path | None:
    return resolve_local_project_path(settings.project_root, sbatch_file)


def validate_predefined_sbatch_file_job(
    settings: LauncherSettings, job: JobSpec
) -> None:
    if not job.sbatch_file:
        return
    local_path = resolve_local_sbatch_file_path(settings, job.sbatch_file)
    if local_path is None:
        raise SystemExit(
            f"ERROR: Job '{job.name}' sbatch_file must stay inside LOCAL_ROOT. "
            f"Got: {job.sbatch_file!r}"
        )
    if not local_path.exists():
        raise SystemExit(
            f"ERROR: Job '{job.name}' sbatch_file not found in LOCAL_ROOT: {local_path}"
        )


def validate_predefined_sbatch_jobs(
    settings: LauncherSettings, jobs: list[JobSpec]
) -> None:
    for job in jobs:
        if job.uses_sbatch_file():
            validate_predefined_sbatch_file_job(settings, job)


def remote_runtime_checks(settings: LauncherSettings) -> list[str]:
    commands: list[str] = []
    if settings.runtime_mode == "venv" and settings.venv_python_executable:
        venv_python = settings.venv_python_executable
        activate = str(Path(venv_python).parent / "activate")
        commands.extend(
            [
                f"test -f {activate}",
                f"test -x {venv_python}",
            ]
        )
    if settings.runtime_mode == "singularity" and settings.singularity_image_path:
        commands.extend(
            [
                "command -v singularity >/dev/null 2>&1",
                f"test -f {settings.singularity_image_path}",
            ]
        )
    return commands
