"""CLI entry point for the remote SLURM launcher."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
    build_job_record,
    build_job_script,
    build_sbatch_script,
    format_sbatch_options,
    resolve_remote_paths,
    resolve_remote_paths_for_job_folder,
    ssh_script,
    submit_job,
    sync_project,
    test_ssh_connection,
    write_job_tracking_file,
)
from .download_logs import add_download_logs_args, run_download_logs
from .init_wizard import init_config
from .tracking import resolve_tracking_file

console = Console()
err_console = Console(stderr=True)
WORKSPACE_MODES = {"per-run", "fixed"}


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

    monitor_parser = subparsers.add_parser(
        "monitor", help="Run squeue for tracked jobs from a previous submission"
    )
    _add_monitor_args(monitor_parser)

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
        "monitor",
        "validate",
        "render",
        "stage",
        "submit",
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


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SSH/rsync/sbatch",
    )


def _add_stage_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running SSH/rsync",
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


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    _add_config_args(parser)
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--job-script",
        action="store_true",
        help="Also print the per-job script (without #SBATCH directives).",
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
    verbose = bool(getattr(config, "VERBOSE", False))

    return LauncherSettings(
        cluster_login=cluster_login,
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
        if "command" not in entry:
            raise ValueError(f"Job '{name}' must define 'command'.")
        return JobSpec(
            name=name,
            command=str(entry["command"]),
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


def _load_run_config(args: argparse.Namespace) -> tuple[ModuleType, Path] | None:
    config_arg = str(args.config) if args.config else None
    config_path = _resolve_run_config_path(config_arg)
    if config_path is None:
        if config_arg:
            err_console.print(f"Config file not found: {config_arg}", style="bold red")
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


def do_run(args: argparse.Namespace) -> int:
    loaded = _load_run_config(args)
    if loaded is None:
        return 1
    config, config_path = loaded
    settings = build_settings(
        config,
        config_path,
        workspace_mode_override=_workspace_mode_from_args(args),
    )

    run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
    jobs = prepare_jobs(config, run_jobs, settings.default_env)

    test_ssh_connection(settings.cluster_login, dry_run=args.dry_run)
    remote_paths = resolve_remote_paths(settings)

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
    sync_project(settings, remote_paths, dry_run=args.dry_run)

    job_records: list[dict[str, Any]] = []
    for job in jobs:
        submission = submit_job(settings, remote_paths, job, dry_run=args.dry_run)
        if not args.dry_run:
            job_records.append(build_job_record(job, submission))

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
        _print_job_logs(job_records)
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
    job_ids = _collect_job_ids(job_records)
    if job_ids:
        monitor_cmd = f"ssh {settings.cluster_login} 'squeue -j {','.join(job_ids)}'"
    else:
        monitor_cmd = f"ssh {settings.cluster_login} 'squeue -u $USER'"
    console.print("Monitor jobs with:")
    console.print(monitor_cmd, style="bold")
    return 0


def do_stage(args: argparse.Namespace) -> int:
    loaded = _load_run_config(args)
    if loaded is None:
        return 1
    config, config_path = loaded
    settings = build_settings(
        config,
        config_path,
        workspace_mode_override=_workspace_mode_from_args(args),
    )

    test_ssh_connection(settings.cluster_login, dry_run=args.dry_run)
    remote_paths = resolve_remote_paths(settings)

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
    sync_project(
        settings,
        remote_paths,
        dry_run=args.dry_run,
        include_logging_dirs=False,
    )

    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", settings.workspace_mode)
    details_table.add_row("Remote workdir", remote_paths.workdir)
    if settings.workspace_mode == "per-run":
        details_table.add_row("Job folder", remote_paths.job_folder)
    console.print()
    console.print(details_table)
    return 0


def do_submit(args: argparse.Namespace) -> int:
    loaded = _load_run_config(args)
    if loaded is None:
        return 1
    config, config_path = loaded
    settings = build_settings(
        config,
        config_path,
        workspace_mode_override=_workspace_mode_from_args(args),
    )

    if settings.workspace_mode == "per-run" and not args.job_folder:
        err_console.print(
            "ERROR: --job-folder is required for submit-only when --workspace per-run.",
            style="bold red",
        )
        return 1

    run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
    jobs = prepare_jobs(config, run_jobs, settings.default_env)

    test_ssh_connection(settings.cluster_login, dry_run=args.dry_run)
    remote_paths = resolve_remote_paths_for_job_folder(
        settings,
        job_folder=args.job_folder,
    )

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

    job_records: list[dict[str, Any]] = []
    for job in jobs:
        submission = submit_job(settings, remote_paths, job, dry_run=args.dry_run)
        if not args.dry_run:
            job_records.append(build_job_record(job, submission))

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
        _print_job_logs(job_records)
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
    if job_ids:
        monitor_cmd = f"ssh {settings.cluster_login} 'squeue -j {','.join(job_ids)}'"
    else:
        monitor_cmd = f"ssh {settings.cluster_login} 'squeue -u $USER'"
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
    if args.check_remote_paths and not args.ssh:
        err_console.print(
            "ERROR: --check-remote-paths requires --ssh.", style="bold red"
        )
        return 1

    config_arg = str(args.config) if args.config else None
    config_path = _resolve_run_config_path(config_arg)
    if config_path is None:
        err_console.print(
            "Config file not found. Pass --config PATH.", style="bold red"
        )
        return 1

    config = load_config(config_path)
    settings = build_settings(
        config,
        config_path,
        workspace_mode_override=_workspace_mode_from_args(args),
    )

    run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
    jobs = prepare_jobs(config, run_jobs, settings.default_env)
    _fail_duplicate_jobs(jobs)

    _fail_if_not_absolute("REMOTE_LOG_BASE_PATH", settings.remote_log_base_path)
    if settings.workspace_mode == "per-run":
        _fail_if_not_absolute(
            "REMOTE_WORKSPACE_BASE", settings.remote_workspace_base
        )
    if settings.workspace_mode == "fixed":
        _fail_if_not_absolute("REMOTE_WORKSPACE_DIR", settings.remote_workspace_dir)
    if settings.runtime_mode == "venv":
        _fail_if_not_absolute("VENV_PYTHON_EXECUTABLE", settings.venv_python_executable)
    if settings.runtime_mode == "singularity":
        _fail_if_not_absolute("SINGULARITY_IMAGE_PATH", settings.singularity_image_path)

    remote_paths = resolve_remote_paths(settings)
    for job in jobs:
        format_sbatch_options(job, settings, remote_paths)

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
    summary.add_row("Jobs", ", ".join(job.name for job in jobs))
    console.print(summary)

    if args.ssh:
        try:
            test_ssh_connection(settings.cluster_login, dry_run=False)
        except SystemExit as exc:
            err_console.print(str(exc), style="bold red")
            return 1

        if args.check_remote_paths:
            checks = _remote_runtime_checks(settings)
            if checks:
                script = "set -euo pipefail\n" + "\n".join(checks) + "\necho OK\n"
                try:
                    stdout, _ = ssh_script(
                        settings.cluster_login, script, dry_run=False
                    )
                except RuntimeError as exc:
                    err_console.print(
                        f"ERROR: Remote checks failed: {exc}", style="bold red"
                    )
                    return 1
                if "OK" not in stdout:
                    err_console.print(
                        "ERROR: Remote checks did not return OK.", style="bold red"
                    )
                    return 1
                console.print("Remote checks OK", style="green")
    return 0


def do_render(args: argparse.Namespace) -> int:
    config_arg = str(args.config) if args.config else None
    config_path = _resolve_run_config_path(config_arg)
    if config_path is None:
        err_console.print(
            "Config file not found. Pass --config PATH.", style="bold red"
        )
        return 1

    config = load_config(config_path)
    settings = build_settings(
        config,
        config_path,
        workspace_mode_override=_workspace_mode_from_args(args),
    )

    run_jobs = args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None
    jobs = prepare_jobs(config, run_jobs, settings.default_env)
    _fail_duplicate_jobs(jobs)

    remote_paths = resolve_remote_paths(settings)
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

    for job in jobs:
        sbatch_options = format_sbatch_options(job, settings, remote_paths)
        job_script = build_job_script(job, settings, remote_paths)
        sbatch_script = build_sbatch_script(job_script, sbatch_options)
        console.print()
        console.rule(f"[cyan]{job.name} sbatch")
        console.print(Syntax(sbatch_script.rstrip(), "bash"))
        if args.job_script:
            console.print()
            console.rule(f"[cyan]{job.name} job script")
            console.print(Syntax(job_script.rstrip(), "bash"))
    return 0


def _resolve_run_config_path(path_arg: str | None) -> Path | None:
    if path_arg:
        candidate = Path(path_arg)
        return candidate if candidate.exists() else None

    candidates = [
        Path(".slurm/remote_launcher_config.mn5.py"),
        Path("remote_launcher_config.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def do_logs(args: argparse.Namespace) -> int:
    tracking_file = resolve_tracking_file(args.tracking_file)
    if tracking_file is None:
        err_console.print(
            "No tracking file found. Run a non-dry submission first "
            "or pass --tracking-file.",
            style="bold red",
        )
        return 1

    payload = json.loads(tracking_file.read_text(encoding="utf-8"))
    records = payload.get("jobs", [])
    if not isinstance(records, list):
        err_console.print(
            f"Invalid tracking file format: {tracking_file}",
            style="bold red",
        )
        return 1

    if args.only:
        wanted = set(args.only)
        records = [
            record for record in records if str(record.get("job_name", "")) in wanted
        ]

    if args.json:
        filtered_payload = dict(payload)
        filtered_payload["jobs"] = records
        console.print_json(data=filtered_payload)
        return 0

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Tracking file:[/bold] {tracking_file}",
                    f"[bold]Cluster:[/bold] {payload.get('cluster_login', 'unknown')}",
                    f"[bold]Job folder:[/bold] {payload.get('job_folder', 'unknown')}",
                ]
            ),
            title="Tracked Submission",
            border_style="cyan",
        )
    )

    if not records:
        console.print("No matching jobs.", style="yellow")
        return 0

    _print_job_logs(records)
    return 0


def _filter_records_by_name(
    records: list[dict[str, Any]], only: list[str] | None
) -> list[dict[str, Any]]:
    if not only:
        return records
    wanted = set(only)
    return [record for record in records if str(record.get("job_name", "")) in wanted]


def _collect_job_ids(records: list[dict[str, Any]]) -> list[str]:
    return [
        str(record.get("job_id", "") or "")
        for record in records
        if str(record.get("job_id", "") or "") not in {"", "unknown", "dry-run"}
    ]


def do_monitor(args: argparse.Namespace) -> int:
    tracking_file = resolve_tracking_file(args.tracking_file)
    if tracking_file is None:
        err_console.print(
            "No tracking file found. Run a non-dry submission first "
            "or pass --tracking-file.",
            style="bold red",
        )
        return 1

    payload = json.loads(tracking_file.read_text(encoding="utf-8"))
    cluster_login = str(payload.get("cluster_login", "") or "")
    if not cluster_login:
        err_console.print(
            f"Missing cluster_login in tracking file: {tracking_file}",
            style="bold red",
        )
        return 1

    raw_records = payload.get("jobs", [])
    if not isinstance(raw_records, list):
        err_console.print(
            f"Invalid tracking file format: {tracking_file}",
            style="bold red",
        )
        return 1

    records = _filter_records_by_name(
        [record for record in raw_records if isinstance(record, dict)],
        args.only,
    )
    job_ids = _collect_job_ids(records)
    if not job_ids:
        err_console.print(
            "No runnable job IDs found in tracking file selection.",
            style="bold red",
        )
        return 1

    remote_command = f"squeue -j {','.join(job_ids)}"
    ssh_cmd = ["ssh", cluster_login, remote_command]
    console.print("Monitor jobs with:", style="cyan")
    console.print(f"ssh {cluster_login} '{remote_command}'", style="bold")

    if args.dry_run:
        return 0
    return subprocess.run(ssh_cmd, check=False).returncode


def do_download_logs(args: argparse.Namespace) -> int:
    return run_download_logs(args)


def _print_job_logs(records: list[dict[str, Any]]) -> None:
    console.print()
    console.print(Panel.fit("Remote Logs", border_style="cyan"))
    for index, record in enumerate(records):
        job_name = str(record.get("job_name", "") or "")
        job_id = str(record.get("job_id", "") or "")
        stdout_path = str(record.get("stdout", "") or "")
        stderr_path = str(record.get("stderr", "") or "")
        job_label = f"{job_name} ({job_id})" if job_id else job_name
        console.print(f"[bold]{job_label}[/bold]")
        console.print(f"stdout: {stdout_path or '-'}", soft_wrap=True)
        console.print(
            f"stderr: {stderr_path if stderr_path and stderr_path != stdout_path else '-'}",
            soft_wrap=True,
        )
        if index < len(records) - 1:
            console.print()


def main() -> int:
    args = parse_args()
    if hasattr(args, "command") and args.command == "init":
        return do_init(args)
    if hasattr(args, "command") and args.command == "logs":
        return do_logs(args)
    if hasattr(args, "command") and args.command == "download-logs":
        return do_download_logs(args)
    if hasattr(args, "command") and args.command == "monitor":
        return do_monitor(args)
    if hasattr(args, "command") and args.command == "validate":
        return do_validate(args)
    if hasattr(args, "command") and args.command == "render":
        return do_render(args)
    if hasattr(args, "command") and args.command == "stage":
        return do_stage(args)
    if hasattr(args, "command") and args.command == "submit":
        return do_submit(args)
    return do_run(args)
