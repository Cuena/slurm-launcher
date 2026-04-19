"""CLI entry point for the remote SLURM launcher."""

from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .core import (
    JobSpec,
    LauncherSettings,
    RemotePaths,
    build_ssh_command,
    build_predefined_sbatch_command,
    build_job_record,
    build_job_script,
    build_launcher_metadata,
    build_sbatch_script,
    format_sbatch_options,
    format_ssh_command,
    resolve_local_project_path,
    resolve_remote_paths,
    resolve_remote_paths_for_job_folder,
    ssh_script,
    submit_job,
    sync_project,
    test_ssh_connection,
    write_job_tracking_file,
)
from .download_artifacts import add_download_artifacts_args, run_download_artifacts
from .download_logs import add_download_logs_args, run_download_logs
from .init_wizard import init_config
from .job_tools import (
    effective_archive_dir,
    list_recent_jobs,
    show_job_details,
    show_job_log,
)
from .tracking import (
    JobRecord,
    TrackingError,
    TrackingPayload,
    load_tracking_payload,
    resolve_tracking_file,
)

console = Console()
err_console = Console(stderr=True)
WORKSPACE_MODES = {"per-run", "fixed"}
GENERIC_CONFIG_PATH = Path.home() / ".config" / "slurm-launcher" / "config.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs on a remote cluster"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    init_parser = subparsers.add_parser(
        "init", help="Initialize launcher config in current directory"
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config file"
    )
    init_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Copy the template without prompting (still updates .gitignore).",
    )

    logs_parser = subparsers.add_parser(
        "logs", help="Show tracked log file paths from a previous submission"
    )
    _add_logs_args(logs_parser)

    download_logs_parser = subparsers.add_parser(
        "download-logs",
        help="Download tracked .out/.err files from a previous submission",
    )
    add_download_logs_args(download_logs_parser)

    download_artifacts_parser = subparsers.add_parser(
        "download-artifacts",
        help="Download tracked artifact paths from a previous submission",
    )
    add_download_artifacts_args(download_artifacts_parser)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Run squeue for tracked jobs from a previous submission"
    )
    _add_monitor_args(monitor_parser)

    jobs_parser = subparsers.add_parser(
        "jobs", help="Show recent SLURM jobs on the cluster"
    )
    _add_jobs_args(jobs_parser)

    job_show_parser = subparsers.add_parser(
        "job-show", help="Show generic SLURM details for one job id"
    )
    _add_job_show_args(job_show_parser)

    job_log_parser = subparsers.add_parser(
        "job-log", help="Read stdout or stderr for a SLURM job id"
    )
    _add_job_log_args(job_log_parser)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check generic config resolution and SSH/tool availability"
    )
    _add_doctor_args(doctor_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate the launcher config without submitting jobs",
    )
    _add_validate_args(validate_parser)

    render_parser = subparsers.add_parser(
        "render",
        help="Render generated sbatch scripts without submitting jobs",
    )
    _add_render_args(render_parser)

    stage_parser = subparsers.add_parser(
        "stage",
        help="Sync project files to the remote cluster without submitting jobs",
    )
    _add_stage_args(stage_parser)

    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit jobs without syncing code first",
    )
    _add_submit_args(submit_parser)

    sbatch_parser = subparsers.add_parser(
        "sbatch",
        help="Stage code and submit one existing sbatch file",
    )
    _add_sbatch_args(sbatch_parser)

    run_parser = subparsers.add_parser(
        "run", help="Stage code and submit jobs (default if no command provided)"
    )
    _add_run_args(run_parser)

    raw_args = sys.argv[1:]
    if not raw_args:
        return run_parser.parse_args([])
    if raw_args[0] in {
        "init",
        "logs",
        "download-logs",
        "download-artifacts",
        "monitor",
        "jobs",
        "job-show",
        "job-log",
        "doctor",
        "validate",
        "render",
        "stage",
        "submit",
        "sbatch",
        "run",
    }:
        return parser.parse_args(raw_args)
    if raw_args[0] in {"-h", "--help"}:
        return parser.parse_args(raw_args)
    return run_parser.parse_args(raw_args)


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        help=(
            "Path to the launcher configuration module. "
            "Default search order: .slurm/remote_launcher_config.mn5.py, "
            "then remote_launcher_config.py."
        ),
    )
    parser.add_argument(
        "--workspace",
        choices=["per-run", "fixed"],
        help=(
            "Remote workspace strategy. "
            "'per-run' creates a unique workdir under REMOTE_WORKSPACE_BASE. "
            "'fixed' reuses REMOTE_WORKSPACE_DIR."
        ),
    )


def _add_job_selection_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only the specified job names (overrides RUN_JOBS)",
    )


def _add_cluster_target_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cluster-login",
        help="Remote SSH login (user@host). Overrides CLUSTER_LOGIN from config.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Optional launcher config path used to resolve CLUSTER_LOGIN and "
            "generic log settings. Default lookup: repo config, then "
            "~/.config/slurm-launcher/config.py."
        ),
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SSH/rsync/sbatch",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _add_stage_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SSH/rsync",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _add_submit_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--job-folder",
        help=(
            "Existing job folder to submit from when --workspace per-run. "
            "Required for per-run submit-only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SSH/sbatch",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _add_sbatch_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    parser.add_argument(
        "sbatch_file",
        help=(
            "Path to an existing sbatch file. Relative paths are resolved from "
            "LOCAL_ROOT and submitted from the staged remote workdir."
        ),
    )
    parser.add_argument(
        "--name",
        help="Tracking name for this submission (default: sbatch file stem).",
    )
    parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Extra argument passed to sbatch (repeatable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SSH/rsync/sbatch",
    )


def _add_validate_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--ssh",
        action="store_true",
        help="Also test SSH connectivity.",
    )
    parser.add_argument(
        "--check-remote-paths",
        action="store_true",
        help="With --ssh, check remote runtime paths (no writes).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--job-script",
        action="store_true",
        help="Also print the per-job script (without #SBATCH directives).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _add_logs_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-file",
        help=(
            "Path to a jobs.json file. Defaults to slurm_output/latest_jobs.json, "
            "or the most recent slurm_output/*/jobs.json."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Show only the specified job names",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON payload",
    )


def _add_monitor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-file",
        help=(
            "Path to a jobs.json file. Defaults to slurm_output/latest_jobs.json, "
            "or the most recent slurm_output/*/jobs.json."
        ),
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Monitor only the specified job names",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ssh+squeue command without running it.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _add_jobs_args(parser: argparse.ArgumentParser) -> None:
    _add_cluster_target_args(parser)
    parser.add_argument(
        "--user",
        help="Cluster username to query. Defaults to the remote SSH user.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="How far back to look when sacct is available. Default: 24.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of jobs to show. Default: 20.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw job list as JSON.",
    )


def _add_job_log_args(parser: argparse.ArgumentParser) -> None:
    _add_cluster_target_args(parser)
    parser.add_argument("job_id", help="SLURM job id to inspect.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print structured log resolution as JSON.",
    )
    parser.add_argument(
        "--stream",
        choices=["stdout", "stderr"],
        default="stdout",
        help="Which log stream to read. Default: stdout.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="How many lines to tail when not using --full. Default: 50.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow the selected log with tail -f.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print the full file instead of tailing it.",
    )
    parser.add_argument(
        "--path-only",
        action="store_true",
        help="Print the resolved remote log path without reading the file.",
    )


def _add_job_show_args(parser: argparse.ArgumentParser) -> None:
    _add_cluster_target_args(parser)
    parser.add_argument("job_id", help="SLURM job id to inspect.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print generic job details as JSON.",
    )


def _add_doctor_args(parser: argparse.ArgumentParser) -> None:
    _add_cluster_target_args(parser)
    parser.add_argument(
        "--ssh",
        action="store_true",
        help="Also test SSH connectivity and remote SLURM tool availability.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print doctor output as JSON.",
    )


def load_config(config_path: Path) -> ModuleType:
    config_path = config_path.resolve()
    spec = importlib.util.spec_from_file_location("remote_launcher_config", config_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"ERROR: Unable to load config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_workspace_mode(value: Any, *, setting_name: str) -> str:
    mode = str(value).strip().lower()
    if mode in WORKSPACE_MODES:
        return mode
    raise SystemExit(f"ERROR: {setting_name} must be one of: per-run, fixed.")


def _workspace_mode_from_args(args: argparse.Namespace) -> str | None:
    workspace = getattr(args, "workspace", None)
    if workspace:
        return _normalize_workspace_mode(workspace, setting_name="--workspace")
    return None


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
    workspace_mode = _normalize_workspace_mode(
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


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


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


def select_jobs(jobs: list[JobSpec], run_only: list[str] | None) -> list[JobSpec]:
    if not run_only:
        return jobs
    wanted = set(run_only)
    available = {job.name for job in jobs}
    missing = wanted.difference(available)
    if missing:
        raise SystemExit(f"ERROR: Requested jobs not found: {sorted(missing)}")
    return [job for job in jobs if job.name in wanted]


def do_init(args: argparse.Namespace) -> int:
    template_path = Path(__file__).parent / "templates" / "config.py.template"
    slurm_dir = Path.cwd() / ".slurm"
    dest_path = slurm_dir / "remote_launcher_config.mn5.py"
    example_path = slurm_dir / "remote_launcher_config.mn5.example.py"

    interactive = sys.stdin.isatty() and not args.non_interactive
    try:
        created_path, answers = init_config(
            cwd=Path.cwd(),
            template_path=template_path,
            dest_path=dest_path,
            force=bool(args.force),
            interactive=interactive,
        )
    except FileExistsError:
        err_console.print(
            f"Config file already exists at {dest_path}. Use --force to overwrite.",
            style="bold red",
        )
        return 1
    except FileNotFoundError:
        err_console.print(
            f"Template file not found at {template_path}", style="bold red"
        )
        return 1
    except RuntimeError as exc:
        err_console.print(f"ERROR: {exc}", style="bold red")
        return 1

    console.print(f"Created {created_path}", style="bold green")
    if args.force or not example_path.exists():
        example_path.parent.mkdir(parents=True, exist_ok=True)
        example_path.write_text(
            template_path.read_text(encoding="utf-8").rstrip() + "\n",
            encoding="utf-8",
        )
        console.print(f"Created {example_path}", style="green")
    if answers is not None and interactive:
        console.print(
            f"Applied wizard answers to {created_path}. "
            f"{example_path.name} keeps template defaults for reference.",
            style="dim",
        )
    console.print("Added .slurm/*.py to .gitignore", style="green")
    console.print("Added !.slurm/*.example.py to .gitignore", style="green")
    if answers is None and not interactive:
        console.print(
            "Non-interactive mode used; please edit the config.", style="yellow"
        )
    else:
        console.print("Please review and adjust values as needed.", style="yellow")
    return 0


def _load_run_config(
    args: argparse.Namespace, *, quiet_errors: bool = False
) -> tuple[ModuleType, Path] | None:
    config_arg = str(args.config) if args.config else None
    config_path = _resolve_config_path(config_arg)
    if config_path is None:
        if not quiet_errors:
            if config_arg:
                err_console.print(
                    f"Config file not found: {config_arg}", style="bold red"
                )
                err_console.print("Pass a valid --config PATH.")
            else:
                err_console.print(
                    "Config file not found. Checked .slurm/remote_launcher_config.mn5.py "
                    "and remote_launcher_config.py.",
                    style="bold red",
                )
                err_console.print(
                    "Pass --config PATH or run 'uv run slurm-launcher init' to create one."
                )
        return None
    return load_config(config_path), config_path


def _resolve_cluster_context(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, list[str], Path | None] | None:
    config_arg = str(args.config) if getattr(args, "config", None) else None
    config_path = _resolve_config_path(
        config_arg, extra_candidates=[GENERIC_CONFIG_PATH]
    )
    config = None
    if config_arg and config_path is None:
        err_console.print(f"Config file not found: {config_arg}", style="bold red")
        err_console.print("Pass a valid --config PATH or use --cluster-login.")
        return None
    if config_path is not None:
        config = load_config(config_path)

    cluster_login = str(
        getattr(args, "cluster_login", None)
        or getattr(config, "CLUSTER_LOGIN", None)
        or ""
    ).strip()
    if not cluster_login:
        err_console.print(
            "ERROR: Pass --cluster-login or use a repo config or "
            "~/.config/slurm-launcher/config.py that defines CLUSTER_LOGIN.",
            style="bold red",
        )
        return None

    archive_dir = getattr(config, "REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR", None)
    archive_dir_text = str(archive_dir).strip() if archive_dir else None
    ssh_config_file = getattr(config, "SSH_CONFIG_FILE", None)
    ssh_config_file_text = str(ssh_config_file).strip() if ssh_config_file else None
    ssh_options = ensure_list(getattr(config, "SSH_OPTIONS", [])) if config else []
    return (
        cluster_login,
        archive_dir_text or None,
        ssh_config_file_text,
        ssh_options,
        config_path,
    )


def _emit_command_error(
    message: str,
    *,
    json_output: bool,
    payload: dict[str, Any] | None = None,
) -> int:
    err_console.print(message, style="bold red")
    if json_output:
        error_payload: dict[str, Any] = {"ok": False, "error": message}
        if payload:
            error_payload.update(payload)
        console.print_json(data=error_payload)
    return 1


def _selected_job_names(jobs: list[JobSpec]) -> list[str]:
    return [job.name for job in jobs]


def _monitor_command(
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


def _collect_submission_results(
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


def do_run(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    loaded = _load_run_config(args, quiet_errors=json_output)
    if loaded is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    config, config_path = loaded
    commands: list[str] = []
    try:
        settings = build_settings(
            config,
            config_path,
            workspace_mode_override=_workspace_mode_from_args(args),
        )
        run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
        jobs = prepare_jobs(config, run_jobs, settings.default_env)
        _validate_predefined_sbatch_jobs(settings, jobs)
        remote_paths = resolve_remote_paths(settings)
        if not json_output:
            console.print()
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"[bold]Cluster:[/bold] {settings.cluster_login}",
                            f"[bold]Workspace:[/bold] {settings.workspace_mode}",
                            f"[bold]Job folder:[/bold] {remote_paths.job_folder}",
                        ]
                    ),
                    title="Remote Launcher",
                    border_style="cyan",
                )
            )
            if settings.workspace_mode == "fixed":
                console.print(
                    "Using REMOTE_WORKSPACE_DIR as the execution directory.",
                    style="yellow",
                )
        test_ssh_connection(
            settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=settings.ssh_config_file,
            ssh_options=settings.ssh_options,
            quiet=json_output,
        )
        commands.extend(
            sync_project(
                settings,
                remote_paths,
                dry_run=args.dry_run,
                quiet=json_output,
            )
        )
        submitted_jobs, job_records, submit_commands = _collect_submission_results(
            settings,
            remote_paths,
            jobs,
            dry_run=args.dry_run,
            quiet=json_output,
        )
        commands.extend(submit_commands)
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(config_path),
                "commands": commands,
                "dry_run": bool(args.dry_run),
            },
        )

    tracking_file: Path | None = None
    if job_records:
        tracking_file = write_job_tracking_file(settings, remote_paths, job_records)

    job_ids = _collect_job_ids(job_records)
    monitor_cmd = _monitor_command(
        settings.cluster_login,
        job_ids,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
    )

    if json_output:
        console.print_json(
            data={
                "ok": True,
                "config_path": str(config_path),
                "workspace_mode": settings.workspace_mode,
                "remote_workdir": remote_paths.workdir,
                "job_folder": remote_paths.job_folder,
                "selected_jobs": _selected_job_names(jobs),
                "submitted_jobs": submitted_jobs,
                "tracking_file": str(tracking_file) if tracking_file else None,
                "commands": commands,
                "monitor_command": monitor_cmd,
                "dry_run": bool(args.dry_run),
            }
        )
        return 0

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
        _print_job_logs_from_records(job_records)
    elif args.dry_run:
        console.print()
        console.print(
            "Skipped job metadata tracking because --dry-run was used.",
            style="yellow",
        )

    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", settings.workspace_mode)
    details_table.add_row("Remote workdir", remote_paths.workdir)
    details_table.add_row("Remote logdir", remote_paths.logdir)
    if settings.remote_slurm_dashboard_log_archive_dir:
        details_table.add_row(
            "Remote slurm-dashboard archive dir",
            settings.remote_slurm_dashboard_log_archive_dir,
        )
    if settings.remote_slurm_dashboard_log_view_dir:
        details_table.add_row(
            "Remote slurm-dashboard view dir",
            settings.remote_slurm_dashboard_log_view_dir,
        )
    console.print()
    console.print(details_table)
    console.print("Monitor jobs with:")
    console.print(monitor_cmd, style="bold")
    return 0


def do_stage(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    loaded = _load_run_config(args, quiet_errors=json_output)
    if loaded is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    config, config_path = loaded
    try:
        settings = build_settings(
            config,
            config_path,
            workspace_mode_override=_workspace_mode_from_args(args),
        )
        remote_paths = resolve_remote_paths(settings)
        if not json_output:
            console.print()
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"[bold]Cluster:[/bold] {settings.cluster_login}",
                            f"[bold]Workspace:[/bold] {settings.workspace_mode}",
                            f"[bold]Job folder:[/bold] {remote_paths.job_folder}",
                        ]
                    ),
                    title="Stage",
                    border_style="cyan",
                )
            )
            if settings.workspace_mode == "fixed":
                console.print(
                    "Using REMOTE_WORKSPACE_DIR as the execution directory.",
                    style="yellow",
                )
        test_ssh_connection(
            settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=settings.ssh_config_file,
            ssh_options=settings.ssh_options,
            quiet=json_output,
        )
        commands = sync_project(
            settings,
            remote_paths,
            dry_run=args.dry_run,
            include_logging_dirs=False,
            quiet=json_output,
        )
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(config_path),
                "dry_run": bool(args.dry_run),
            },
        )

    if json_output:
        console.print_json(
            data={
                "ok": True,
                "config_path": str(config_path),
                "workspace_mode": settings.workspace_mode,
                "remote_workdir": remote_paths.workdir,
                "job_folder": remote_paths.job_folder,
                "commands": commands,
                "dry_run": bool(args.dry_run),
            }
        )
        return 0

    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", settings.workspace_mode)
    details_table.add_row("Remote workdir", remote_paths.workdir)
    if settings.workspace_mode == "per-run":
        details_table.add_row("Job folder", remote_paths.job_folder)
    console.print()
    console.print(details_table)
    return 0


def do_submit(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    loaded = _load_run_config(args, quiet_errors=json_output)
    if loaded is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    config, config_path = loaded
    try:
        settings = build_settings(
            config,
            config_path,
            workspace_mode_override=_workspace_mode_from_args(args),
        )
        if settings.workspace_mode == "per-run" and not args.job_folder:
            raise SystemExit(
                "ERROR: --job-folder is required for submit-only when --workspace per-run."
            )
        run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
        jobs = prepare_jobs(config, run_jobs, settings.default_env)
        _validate_predefined_sbatch_jobs(settings, jobs)
        remote_paths = resolve_remote_paths_for_job_folder(
            settings,
            job_folder=args.job_folder,
        )
        if not json_output:
            console.print()
            console.print(
                Panel.fit(
                    "\n".join(
                        [
                            f"[bold]Cluster:[/bold] {settings.cluster_login}",
                            f"[bold]Workspace:[/bold] {settings.workspace_mode}",
                            f"[bold]Job folder:[/bold] {remote_paths.job_folder}",
                        ]
                    ),
                    title="Submit",
                    border_style="cyan",
                )
            )
            console.print(
                "Skipping stage step. Assuming code is already present on the remote workdir.",
                style="yellow",
            )
        test_ssh_connection(
            settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=settings.ssh_config_file,
            ssh_options=settings.ssh_options,
            quiet=json_output,
        )
        submitted_jobs, job_records, commands = _collect_submission_results(
            settings,
            remote_paths,
            jobs,
            dry_run=args.dry_run,
            quiet=json_output,
        )
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(config_path),
                "dry_run": bool(args.dry_run),
            },
        )

    tracking_file: Path | None = None
    if job_records:
        tracking_file = write_job_tracking_file(settings, remote_paths, job_records)

    job_ids = _collect_job_ids(job_records)
    monitor_cmd = _monitor_command(
        settings.cluster_login,
        job_ids,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
    )

    if json_output:
        console.print_json(
            data={
                "ok": True,
                "config_path": str(config_path),
                "workspace_mode": settings.workspace_mode,
                "remote_workdir": remote_paths.workdir,
                "job_folder": remote_paths.job_folder,
                "selected_jobs": _selected_job_names(jobs),
                "submitted_jobs": submitted_jobs,
                "tracking_file": str(tracking_file) if tracking_file else None,
                "commands": commands,
                "monitor_command": monitor_cmd,
                "dry_run": bool(args.dry_run),
            }
        )
        return 0

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
        _print_job_logs_from_records(job_records)
    elif args.dry_run:
        console.print()
        console.print(
            "Skipped job metadata tracking because --dry-run was used.",
            style="yellow",
        )

    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", settings.workspace_mode)
    details_table.add_row("Remote workdir", remote_paths.workdir)
    details_table.add_row("Remote logdir", remote_paths.logdir)
    console.print()
    console.print(details_table)
    console.print("Monitor jobs with:")
    console.print(monitor_cmd, style="bold")
    return 0


def do_sbatch(args: argparse.Namespace) -> int:
    loaded = _load_run_config(args)
    if loaded is None:
        return 1
    config, config_path = loaded

    try:
        settings = build_settings(
            config,
            config_path,
            workspace_mode_override=_workspace_mode_from_args(args),
        )

        job_name = (
            args.name or Path(str(args.sbatch_file)).stem or "sbatch_job"
        ).strip()
        if not job_name:
            raise ValueError("--name cannot be empty.")
        job = JobSpec(
            name=job_name,
            sbatch_file=str(args.sbatch_file),
            sbatch_args=ensure_list(args.sbatch_arg),
        )
        _validate_predefined_sbatch_file_job(settings, job)

        test_ssh_connection(
            settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=settings.ssh_config_file,
            ssh_options=settings.ssh_options,
        )
        remote_paths = resolve_remote_paths(settings)

        console.print()
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"[bold]Cluster:[/bold] {settings.cluster_login}",
                        f"[bold]Workspace:[/bold] {settings.workspace_mode}",
                        f"[bold]Job folder:[/bold] {remote_paths.job_folder}",
                        f"[bold]Sbatch file:[/bold] {job.sbatch_file}",
                    ]
                ),
                title="Sbatch",
                border_style="cyan",
            )
        )

        if settings.workspace_mode == "fixed":
            console.print(
                "Using REMOTE_WORKSPACE_DIR as the execution directory.",
                style="yellow",
            )
        sync_project(settings, remote_paths, dry_run=args.dry_run)

        submission = submit_job(settings, remote_paths, job, dry_run=args.dry_run)
    except (RuntimeError, SystemExit, ValueError) as exc:
        err_console.print(str(exc), style="bold red")
        return 1

    job_records: list[dict[str, Any]] = []
    if not args.dry_run:
        job_records.append(build_job_record(job, submission, settings))

    if job_records:
        tracking_file = write_job_tracking_file(settings, remote_paths, job_records)
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
        _print_job_logs_from_records(job_records)
    elif args.dry_run:
        console.print()
        console.print(
            "Skipped job metadata tracking because --dry-run was used.",
            style="yellow",
        )

    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", settings.workspace_mode)
    details_table.add_row("Remote workdir", remote_paths.workdir)
    details_table.add_row("Remote logdir", remote_paths.logdir)
    console.print()
    console.print(details_table)
    job_ids = _collect_job_ids(job_records)
    monitor_cmd = _monitor_command(
        settings.cluster_login,
        job_ids,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
    )
    console.print("Monitor jobs with:")
    console.print(monitor_cmd, style="bold")
    return 0


def _fail_duplicate_jobs(jobs: list[JobSpec]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for job in jobs:
        if job.name in seen:
            duplicates.add(job.name)
        seen.add(job.name)
    if duplicates:
        raise SystemExit(f"ERROR: Duplicate job names found: {sorted(duplicates)}")


def _fail_if_not_absolute(label: str, value: str | None) -> None:
    if value is None:
        return
    if not str(value).startswith("/"):
        raise SystemExit(f"ERROR: {label} must be an absolute path. Got: {value!r}")


def _resolve_local_sbatch_file_path(
    settings: LauncherSettings, sbatch_file: str
) -> Path | None:
    return resolve_local_project_path(settings.project_root, sbatch_file)


def _validate_predefined_sbatch_file_job(
    settings: LauncherSettings, job: JobSpec
) -> None:
    if not job.sbatch_file:
        return
    local_path = _resolve_local_sbatch_file_path(settings, job.sbatch_file)
    if local_path is None:
        raise SystemExit(
            f"ERROR: Job '{job.name}' sbatch_file must stay inside LOCAL_ROOT. "
            f"Got: {job.sbatch_file!r}"
        )
    if not local_path.exists():
        raise SystemExit(
            f"ERROR: Job '{job.name}' sbatch_file not found in LOCAL_ROOT: {local_path}"
        )


def _validate_predefined_sbatch_jobs(
    settings: LauncherSettings, jobs: list[JobSpec]
) -> None:
    for job in jobs:
        if job.uses_sbatch_file():
            _validate_predefined_sbatch_file_job(settings, job)


def _remote_runtime_checks(settings: LauncherSettings) -> list[str]:
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


def do_validate(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    if args.check_remote_paths and not args.ssh:
        return _emit_command_error(
            "ERROR: --check-remote-paths requires --ssh.",
            json_output=json_output,
            payload={
                "config_path": str(args.config) if args.config else None,
                "workspace_mode": _workspace_mode_from_args(args),
                "selected_jobs": list(args.only or []),
                "warnings": [],
                "errors": ["ERROR: --check-remote-paths requires --ssh."],
                "ssh_checked": bool(args.ssh),
                "remote_checks": {
                    "requested": bool(args.check_remote_paths),
                    "checks": [],
                    "ok": False,
                },
            },
        )

    config_arg = str(args.config) if args.config else None
    config_path = _resolve_config_path(config_arg)
    if config_path is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH.",
            json_output=json_output,
            payload={
                "config_path": None,
                "workspace_mode": _workspace_mode_from_args(args),
                "selected_jobs": list(args.only or []),
                "warnings": [],
                "errors": ["Config file not found. Pass --config PATH."],
                "ssh_checked": bool(args.ssh),
                "remote_checks": {
                    "requested": bool(args.check_remote_paths),
                    "checks": [],
                    "ok": False,
                },
            },
        )

    selected_jobs = list(args.only or [])
    workspace_mode = _workspace_mode_from_args(args)
    remote_checks: dict[str, Any] = {
        "requested": bool(args.check_remote_paths),
        "checks": [],
        "ok": None,
    }

    try:
        config = load_config(config_path)
        settings = build_settings(
            config,
            config_path,
            workspace_mode_override=workspace_mode,
        )
        workspace_mode = settings.workspace_mode
        run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
        jobs = prepare_jobs(config, run_jobs, settings.default_env)
        selected_jobs = _selected_job_names(jobs)
        _fail_duplicate_jobs(jobs)

        _fail_if_not_absolute("REMOTE_LOG_BASE_PATH", settings.remote_log_base_path)
        if settings.workspace_mode == "per-run":
            _fail_if_not_absolute(
                "REMOTE_WORKSPACE_BASE", settings.remote_workspace_base
            )
        if settings.workspace_mode == "fixed":
            _fail_if_not_absolute("REMOTE_WORKSPACE_DIR", settings.remote_workspace_dir)
        if settings.runtime_mode == "venv":
            _fail_if_not_absolute(
                "VENV_PYTHON_EXECUTABLE", settings.venv_python_executable
            )
        if settings.runtime_mode == "singularity":
            _fail_if_not_absolute(
                "SINGULARITY_IMAGE_PATH", settings.singularity_image_path
            )

        remote_paths = resolve_remote_paths(settings)
        for job in jobs:
            if job.uses_sbatch_file():
                _validate_predefined_sbatch_file_job(settings, job)
                continue
            format_sbatch_options(job, settings, remote_paths)

        if args.ssh:
            test_ssh_connection(
                settings.cluster_login,
                dry_run=False,
                ssh_config_file=settings.ssh_config_file,
                ssh_options=settings.ssh_options,
                quiet=json_output,
            )

            checks = _remote_runtime_checks(settings) if args.check_remote_paths else []
            remote_checks["checks"] = checks
            if args.check_remote_paths:
                if checks:
                    script = "set -euo pipefail\n" + "\n".join(checks) + "\necho OK\n"
                    stdout, _ = ssh_script(
                        settings.cluster_login,
                        script,
                        dry_run=False,
                        ssh_config_file=settings.ssh_config_file,
                        ssh_options=settings.ssh_options,
                        quiet=json_output,
                    )
                    if "OK" not in stdout:
                        raise SystemExit("ERROR: Remote checks did not return OK.")
                remote_checks["ok"] = True
        if args.check_remote_paths and remote_checks["ok"] is None:
            remote_checks["ok"] = True
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(config_path),
                "workspace_mode": workspace_mode,
                "selected_jobs": selected_jobs,
                "warnings": [],
                "errors": [str(exc)],
                "ssh_checked": bool(args.ssh),
                "remote_checks": remote_checks,
            },
        )

    if json_output:
        console.print_json(
            data={
                "ok": True,
                "config_path": str(config_path),
                "workspace_mode": settings.workspace_mode,
                "selected_jobs": selected_jobs,
                "warnings": [],
                "errors": [],
                "ssh_checked": bool(args.ssh),
                "remote_checks": remote_checks,
            }
        )
        return 0

    console.print()
    console.print(Panel.fit("Config OK", border_style="green"))
    summary = Table.grid(padding=(0, 1))
    summary.add_row("Config", str(config_path))
    summary.add_row("Cluster", settings.cluster_login)
    summary.add_row("Workspace", settings.workspace_mode)
    summary.add_row("Runtime mode", settings.runtime_mode)
    summary.add_row("Job folder", remote_paths.job_folder)
    summary.add_row("Remote workdir", remote_paths.workdir)
    summary.add_row("Remote logdir", remote_paths.logdir)
    summary.add_row("Remote slurm_output", remote_paths.slurm_output_dir)
    summary.add_row("Jobs", ", ".join(selected_jobs))
    console.print(summary)
    if args.check_remote_paths:
        console.print("Remote checks OK", style="green")
    return 0


def do_render(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    config_arg = str(args.config) if args.config else None
    config_path = _resolve_config_path(config_arg)
    if config_path is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH.",
            json_output=json_output,
            payload={
                "config_path": None,
                "workspace_mode": _workspace_mode_from_args(args),
                "selected_jobs": list(args.only or []),
            },
        )

    try:
        config = load_config(config_path)
        settings = build_settings(
            config,
            config_path,
            workspace_mode_override=_workspace_mode_from_args(args),
        )
        run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
        jobs = prepare_jobs(config, run_jobs, settings.default_env)
        _fail_duplicate_jobs(jobs)
        _validate_predefined_sbatch_jobs(settings, jobs)
        remote_paths = resolve_remote_paths(settings)
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(config_path),
                "workspace_mode": _workspace_mode_from_args(args),
                "selected_jobs": list(args.only or []),
            },
        )

    rendered_jobs: list[dict[str, Any]] = []
    job_scripts: dict[str, str] = {}
    sbatch_scripts: dict[str, str] = {}

    for job in jobs:
        if job.uses_sbatch_file():
            remote_sbatch_path, sbatch_command = build_predefined_sbatch_command(
                settings, remote_paths, job
            )
            job_payload: dict[str, Any] = {
                "job_name": job.name,
                "job_type": "sbatch_file",
                "sbatch_file": str(job.sbatch_file),
                "remote_sbatch_path": remote_sbatch_path,
                "sbatch_command": sbatch_command,
            }
            if args.job_script:
                local_path = _resolve_local_sbatch_file_path(
                    settings, str(job.sbatch_file)
                )
                if local_path and local_path.exists():
                    job_payload["job_script"] = local_path.read_text(encoding="utf-8")
                else:
                    job_payload["warning"] = (
                        "render cannot preview local contents for sbatch_file outside LOCAL_ROOT."
                    )
            rendered_jobs.append(job_payload)
            continue

        sbatch_options = format_sbatch_options(job, settings, remote_paths)
        job_script = build_job_script(job, settings, remote_paths)
        sbatch_script = build_sbatch_script(
            job_script,
            sbatch_options,
            launcher_metadata=build_launcher_metadata(job, settings),
        )
        job_scripts[job.name] = job_script
        sbatch_scripts[job.name] = sbatch_script
        rendered_jobs.append(
            {
                "job_name": job.name,
                "job_type": "command",
                "job_script": job_script,
                "sbatch_script": sbatch_script,
            }
        )

    if json_output:
        console.print_json(
            data={
                "ok": True,
                "config_path": str(config_path),
                "workspace_mode": settings.workspace_mode,
                "selected_jobs": _selected_job_names(jobs),
                "rendered_jobs": rendered_jobs,
                "job_scripts": job_scripts,
                "sbatch_scripts": sbatch_scripts,
            }
        )
        return 0

    console.print()
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Config:[/bold] {config_path}",
                    f"[bold]Cluster:[/bold] {settings.cluster_login}",
                    f"[bold]Workspace:[/bold] {settings.workspace_mode}",
                    f"[bold]Runtime:[/bold] {settings.runtime_mode}",
                    f"[bold]Job folder:[/bold] {remote_paths.job_folder}",
                ]
            ),
            title="Render",
            border_style="cyan",
        )
    )

    for job_payload in rendered_jobs:
        console.print()
        console.rule(f"[cyan]{job_payload['job_name']} sbatch")
        if job_payload["job_type"] == "sbatch_file":
            console.print(Syntax(str(job_payload["sbatch_command"]), "bash"))
            if args.job_script and "job_script" in job_payload:
                console.print()
                console.rule(f"[cyan]{job_payload['job_name']} sbatch file")
                console.print(Syntax(str(job_payload["job_script"]), "bash"))
            elif "warning" in job_payload:
                console.print(str(job_payload["warning"]), style="yellow")
            continue
        console.print(Syntax(str(job_payload["sbatch_script"]).rstrip(), "bash"))
        if args.job_script:
            console.print()
            console.rule(f"[cyan]{job_payload['job_name']} job script")
            console.print(Syntax(str(job_payload["job_script"]).rstrip(), "bash"))
    return 0


_RUN_CONFIG_CANDIDATES = [
    Path(".slurm/remote_launcher_config.mn5.py"),
    Path("remote_launcher_config.py"),
]


def _resolve_config_path(
    path_arg: str | None,
    extra_candidates: list[Path] | None = None,
) -> Path | None:
    if path_arg:
        candidate = Path(path_arg)
        return candidate if candidate.exists() else None

    candidates = _RUN_CONFIG_CANDIDATES + (extra_candidates or [])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def do_logs(args: argparse.Namespace) -> int:
    tracking_path = resolve_tracking_file(args.tracking_file)
    if tracking_path is None:
        err_console.print(
            "No tracking file found. Run a non-dry submission first "
            "or pass --tracking-file.",
            style="bold red",
        )
        return 1

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        err_console.print(str(exc), style="bold red")
        return 1

    selected = payload.filter_jobs(names=set(args.only) if args.only else None)

    if args.json:
        console.print_json(data=_tracking_payload_to_dict(payload, selected))
        return 0

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Tracking file:[/bold] {tracking_path}",
                    f"[bold]Cluster:[/bold] {payload.cluster_login}",
                    f"[bold]Job folder:[/bold] {payload.job_folder}",
                ]
            ),
            title="Tracked Submission",
            border_style="cyan",
        )
    )

    if not selected:
        console.print("No matching jobs.", style="yellow")
        return 0

    _print_job_logs_from_records(selected)
    return 0


def _collect_job_ids(records: list[dict[str, Any]]) -> list[str]:
    return [
        str(record.get("job_id", "") or "")
        for record in records
        if str(record.get("job_id", "") or "") not in {"", "unknown", "dry-run"}
    ]


def _tracking_payload_to_dict(
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
            job_dicts.append(entry)
        else:
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


def _print_job_logs_from_records(jobs: list[JobRecord | dict[str, object]]) -> None:
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


def do_monitor(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    tracking_path = resolve_tracking_file(args.tracking_file)
    if tracking_path is None:
        return _emit_command_error(
            "No tracking file found. Run a non-dry submission first or pass --tracking-file.",
            json_output=json_output,
            payload={
                "tracking_file": None,
                "job_ids": [],
                "command": None,
                "dry_run": bool(args.dry_run),
            },
        )

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "tracking_file": str(tracking_path),
                "job_ids": [],
                "command": None,
                "dry_run": bool(args.dry_run),
            },
        )

    if not payload.cluster_login:
        return _emit_command_error(
            f"Missing cluster_login in tracking file: {tracking_path}",
            json_output=json_output,
            payload={
                "tracking_file": str(tracking_path),
                "job_ids": [],
                "command": None,
                "dry_run": bool(args.dry_run),
            },
        )

    selected = payload.filter_jobs(names=set(args.only) if args.only else None)
    job_ids = payload.runnable_job_ids(selected)
    if not job_ids:
        return _emit_command_error(
            "No runnable job IDs found in tracking file selection.",
            json_output=json_output,
            payload={
                "tracking_file": str(tracking_path),
                "job_ids": [],
                "command": None,
                "dry_run": bool(args.dry_run),
            },
        )

    remote_command = f"squeue -j {','.join(job_ids)}"
    ssh_cmd = build_ssh_command(
        payload.cluster_login,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
    )
    ssh_cmd.append(remote_command)
    command = shlex.join(ssh_cmd)

    if json_output:
        result_payload: dict[str, Any] = {
            "ok": True,
            "tracking_file": str(tracking_path),
            "job_ids": job_ids,
            "command": command,
            "dry_run": bool(args.dry_run),
        }
        if not args.dry_run:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            result_payload["ok"] = result.returncode == 0
            result_payload["returncode"] = result.returncode
            result_payload["stdout"] = result.stdout
            result_payload["stderr"] = result.stderr
            console.print_json(data=result_payload)
            return result.returncode
        console.print_json(data=result_payload)
        return 0

    console.print("Monitor jobs with:", style="cyan")
    console.print(command, style="bold")

    if args.dry_run:
        return 0
    return subprocess.run(ssh_cmd, check=False).returncode


def do_doctor(args: argparse.Namespace) -> int:
    resolved = _resolve_cluster_context(args)
    if resolved is None:
        return 1
    (
        cluster_login,
        archive_dir,
        ssh_config_file,
        ssh_options,
        config_path,
    ) = resolved
    effective_archive, archive_source = effective_archive_dir(archive_dir)
    payload: dict[str, Any] = {
        "cluster_login": cluster_login,
        "config_path": str(config_path) if config_path else None,
        "ssh_config_file": ssh_config_file,
        "ssh_options": ssh_options,
        "archive_dir": effective_archive,
        "archive_dir_source": archive_source,
    }

    if args.ssh:
        try:
            test_ssh_connection(
                cluster_login,
                dry_run=False,
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            )
        except SystemExit as exc:
            err_console.print(str(exc), style="bold red")
            return 1

        script = "\n".join(
            [
                "set -euo pipefail",
                "for tool in sacct scontrol squeue; do",
                '  if command -v "$tool" >/dev/null 2>&1; then',
                '    echo "$tool=ok"',
                "  else",
                '    echo "$tool=missing"',
                "  fi",
                "done",
            ]
        )
        try:
            stdout, _ = ssh_script(
                cluster_login,
                script,
                dry_run=False,
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            )
        except RuntimeError as exc:
            err_console.print(
                f"ERROR: SSH doctor checks failed: {exc}", style="bold red"
            )
            return 1
        remote_tools: dict[str, str] = {}
        for line in stdout.splitlines():
            if "=" not in line:
                continue
            name, status = line.split("=", 1)
            remote_tools[name.strip()] = status.strip()
        payload["ssh_ok"] = True
        payload["remote_tools"] = remote_tools

    if args.json:
        console.print_json(data=payload)
        return 0

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Cluster:[/bold] {cluster_login}",
                    f"[bold]Config:[/bold] {config_path or '-'}",
                    f"[bold]SSH config file:[/bold] {ssh_config_file or '-'}",
                    f"[bold]SSH options:[/bold] {', '.join(ssh_options) or '-'}",
                    f"[bold]Archive dir:[/bold] {effective_archive}",
                    f"[bold]Archive source:[/bold] {archive_source}",
                ]
            ),
            title="Doctor",
            border_style="cyan",
        )
    )
    if args.ssh:
        remote_tools = payload.get("remote_tools", {})
        table = Table(title="Remote Tools")
        table.add_column("Tool")
        table.add_column("Status")
        for tool_name in ("sacct", "scontrol", "squeue"):
            table.add_row(tool_name, str(remote_tools.get(tool_name, "unknown")))
        console.print(table)
    return 0


def do_jobs(args: argparse.Namespace) -> int:
    resolved = _resolve_cluster_context(args)
    if resolved is None:
        return 1
    cluster_login, _, ssh_config_file, ssh_options, _ = resolved
    return list_recent_jobs(
        cluster_login,
        user=args.user,
        hours=args.hours,
        limit=args.limit,
        json_output=args.json,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )


def do_job_show(args: argparse.Namespace) -> int:
    resolved = _resolve_cluster_context(args)
    if resolved is None:
        return 1
    cluster_login, _, ssh_config_file, ssh_options, _ = resolved
    return show_job_details(
        cluster_login,
        args.job_id,
        json_output=args.json,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )


def do_job_log(args: argparse.Namespace) -> int:
    resolved = _resolve_cluster_context(args)
    if resolved is None:
        return 1
    cluster_login, archive_dir, ssh_config_file, ssh_options, _ = resolved
    return show_job_log(
        cluster_login,
        args.job_id,
        stream=args.stream,
        lines=args.lines,
        follow=args.follow,
        full=args.full,
        path_only=args.path_only,
        json_output=args.json,
        archive_dir=archive_dir,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )


def do_download_logs(args: argparse.Namespace) -> int:
    return run_download_logs(args)


def do_download_artifacts(args: argparse.Namespace) -> int:
    return run_download_artifacts(args)


def main() -> int:
    args = parse_args()
    if hasattr(args, "command") and args.command == "init":
        return do_init(args)
    if hasattr(args, "command") and args.command == "logs":
        return do_logs(args)
    if hasattr(args, "command") and args.command == "download-logs":
        return do_download_logs(args)
    if hasattr(args, "command") and args.command == "download-artifacts":
        return do_download_artifacts(args)
    if hasattr(args, "command") and args.command == "monitor":
        return do_monitor(args)
    if hasattr(args, "command") and args.command == "jobs":
        return do_jobs(args)
    if hasattr(args, "command") and args.command == "job-show":
        return do_job_show(args)
    if hasattr(args, "command") and args.command == "job-log":
        return do_job_log(args)
    if hasattr(args, "command") and args.command == "doctor":
        return do_doctor(args)
    if hasattr(args, "command") and args.command == "validate":
        return do_validate(args)
    if hasattr(args, "command") and args.command == "render":
        return do_render(args)
    if hasattr(args, "command") and args.command == "stage":
        return do_stage(args)
    if hasattr(args, "command") and args.command == "submit":
        return do_submit(args)
    if hasattr(args, "command") and args.command == "sbatch":
        return do_sbatch(args)
    return do_run(args)
