"""CLI entry point for the remote SLURM launcher."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .cli_output import (
    collect_job_ids,
    collect_submission_results,
    emit_submission_result,
    monitor_command,
    print_execution_panel,
    print_job_logs_from_records,
    selected_job_names,
)
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
    enforce_clean_git,
    format_sbatch_options,
    resolve_remote_paths,
    resolve_remote_paths_for_job_folder,
    ssh_script,
    submit_job,
    sync_project,
    test_ssh_connection,
    write_job_tracking_file,
)
from .command_specs import COMMAND_NAMES, COMMAND_SPECS, DEFAULT_COMMAND
from .config_utils import (
    build_settings,
    ensure_list,
    fail_duplicate_jobs,
    fail_if_not_absolute,
    load_config,
    normalize_workspace_mode,
    prepare_jobs,
    remote_runtime_checks,
    resolve_local_sbatch_file_path,
    validate_predefined_sbatch_file_job,
    validate_predefined_sbatch_jobs,
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
    load_tracking_payload,
    resolve_tracking_file,
)
from .payloads import (
    doctor_payload,
    error_payload,
    monitor_payload,
    render_payload,
    stage_payload,
    submission_payload,
    tracking_payload_to_dict,
    validate_payload,
)

console = Console()
err_console = Console(stderr=True)
GENERIC_CONFIG_PATH = Path.home() / ".config" / "slurm-launcher" / "config.py"


@dataclass(frozen=True)
class ExecutionContext:
    config: ModuleType
    config_path: Path
    settings: LauncherSettings
    remote_paths: RemotePaths


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs on a remote cluster"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    init_parser = subparsers.add_parser(
        "init", help=COMMAND_SPECS["init"].summary
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
        "logs", help=COMMAND_SPECS["logs"].summary
    )
    _add_logs_args(logs_parser)

    download_logs_parser = subparsers.add_parser(
        "download-logs",
        help=COMMAND_SPECS["download-logs"].summary,
    )
    add_download_logs_args(download_logs_parser)

    download_artifacts_parser = subparsers.add_parser(
        "download-artifacts",
        help=COMMAND_SPECS["download-artifacts"].summary,
    )
    add_download_artifacts_args(download_artifacts_parser)

    monitor_parser = subparsers.add_parser(
        "monitor", help=COMMAND_SPECS["monitor"].summary
    )
    _add_monitor_args(monitor_parser)

    jobs_parser = subparsers.add_parser(
        "jobs", help=COMMAND_SPECS["jobs"].summary
    )
    _add_jobs_args(jobs_parser)

    job_show_parser = subparsers.add_parser(
        "job-show", help=COMMAND_SPECS["job-show"].summary
    )
    _add_job_show_args(job_show_parser)

    job_log_parser = subparsers.add_parser(
        "job-log", help=COMMAND_SPECS["job-log"].summary
    )
    _add_job_log_args(job_log_parser)

    doctor_parser = subparsers.add_parser(
        "doctor", help=COMMAND_SPECS["doctor"].summary
    )
    _add_doctor_args(doctor_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help=COMMAND_SPECS["validate"].summary,
    )
    _add_validate_args(validate_parser)

    render_parser = subparsers.add_parser(
        "render",
        help=COMMAND_SPECS["render"].summary,
    )
    _add_render_args(render_parser)

    stage_parser = subparsers.add_parser(
        "stage",
        help=COMMAND_SPECS["stage"].summary,
    )
    _add_stage_args(stage_parser)

    submit_parser = subparsers.add_parser(
        "submit",
        help=COMMAND_SPECS["submit"].summary,
    )
    _add_submit_args(submit_parser)

    sbatch_parser = subparsers.add_parser(
        "sbatch",
        help=COMMAND_SPECS["sbatch"].summary,
    )
    _add_sbatch_args(sbatch_parser)

    run_parser = subparsers.add_parser(
        "run", help=COMMAND_SPECS["run"].summary
    )
    _add_run_args(run_parser)
    return parser, run_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser, run_parser = _build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        return run_parser.parse_args([])
    if raw_args[0] in COMMAND_NAMES:
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
        "--require-clean-git",
        action="store_true",
        help="Fail before staging if LOCAL_ROOT is not a clean git checkout.",
    )
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
        "--require-clean-git",
        action="store_true",
        help="Fail before staging if LOCAL_ROOT is not a clean git checkout.",
    )
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
        "--require-clean-git",
        action="store_true",
        help="Fail before staging if LOCAL_ROOT is not a clean git checkout.",
    )
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
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
        "--state",
        action="append",
        default=[],
        help=(
            "Filter to one or more job states. Matches the leading state token, "
            "so '--state cancelled' also matches 'CANCELLED by <uid>'."
        ),
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


def _normalize_workspace_mode(value: Any, *, setting_name: str) -> str:
    return normalize_workspace_mode(value, setting_name=setting_name)


def _workspace_mode_from_args(args: argparse.Namespace) -> str | None:
    workspace = getattr(args, "workspace", None)
    if workspace:
        return _normalize_workspace_mode(workspace, setting_name="--workspace")
    return None


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


def _configured_run_only(
    config: ModuleType, args: argparse.Namespace
) -> list[str] | None:
    return args.only or ensure_list(getattr(config, "RUN_JOBS", None)) or None


def _prepare_configured_jobs(
    config: ModuleType,
    settings: LauncherSettings,
    args: argparse.Namespace,
    *,
    fail_duplicate_names: bool = False,
    validate_predefined_jobs: bool = True,
) -> list[JobSpec]:
    jobs = prepare_jobs(config, _configured_run_only(config, args), settings.default_env)
    if fail_duplicate_names:
        _fail_duplicate_jobs(jobs)
    if validate_predefined_jobs:
        _validate_predefined_sbatch_jobs(settings, jobs)
    return jobs


def _load_execution_context(
    args: argparse.Namespace,
    *,
    json_output: bool,
    existing_job_folder: bool = False,
) -> ExecutionContext | None:
    loaded = _load_run_config(args, quiet_errors=json_output)
    if loaded is None:
        return None
    config, config_path = loaded
    settings = build_settings(
        config,
        config_path,
        workspace_mode_override=_workspace_mode_from_args(args),
    )
    if existing_job_folder:
        remote_paths = resolve_remote_paths_for_job_folder(
            settings,
            job_folder=args.job_folder,
        )
    else:
        remote_paths = resolve_remote_paths(settings)
    return ExecutionContext(
        config=config,
        config_path=config_path,
        settings=settings,
        remote_paths=remote_paths,
    )


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
        console.print_json(data=error_payload(message, **(payload or {})))
    return 1


def _selected_job_names(jobs: list[JobSpec]) -> list[str]:
    return selected_job_names(jobs)


def _monitor_command(
    cluster_login: str,
    job_ids: list[str],
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> str:
    return monitor_command(
        cluster_login,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
        job_ids=job_ids,
    )


def _collect_submission_results(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    jobs: list[JobSpec],
    *,
    dry_run: bool,
    quiet: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    return collect_submission_results(
        settings,
        remote_paths,
        jobs,
        dry_run=dry_run,
        quiet=quiet,
    )


def _print_execution_panel(
    title: str,
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    *,
    extra_lines: list[str] | None = None,
    note: str | None = None,
) -> None:
    print_execution_panel(
        console,
        title,
        settings,
        remote_paths,
        extra_lines=extra_lines,
        note=note,
    )


def _emit_submission_result(
    *,
    json_output: bool,
    payload: dict[str, Any],
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    tracking_file: Path | None,
    job_records: list[dict[str, Any]],
    monitor_cmd: str,
    dry_run: bool,
    include_archive_dirs: bool = False,
) -> int:
    return emit_submission_result(
        console,
        json_output=json_output,
        payload=payload,
        settings=settings,
        remote_paths=remote_paths,
        tracking_file=tracking_file,
        job_records=job_records,
        monitor_cmd=monitor_cmd,
        dry_run=dry_run,
        include_archive_dirs=include_archive_dirs,
    )


def do_run(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    context = _load_execution_context(args, json_output=json_output)
    if context is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    commands: list[str] = []
    try:
        jobs = _prepare_configured_jobs(
            context.config,
            context.settings,
            args,
            fail_duplicate_names=True,
        )
        if not json_output:
            _print_execution_panel(
                "Remote Launcher",
                context.settings,
                context.remote_paths,
            )
        enforce_clean_git(
            context.settings,
            require_clean_git=bool(getattr(args, "require_clean_git", False)),
        )
        test_ssh_connection(
            context.settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=context.settings.ssh_config_file,
            ssh_options=context.settings.ssh_options,
            quiet=json_output,
        )
        commands.extend(
            sync_project(
                context.settings,
                context.remote_paths,
                dry_run=args.dry_run,
                quiet=json_output,
            )
        )
        submitted_jobs, job_records, submit_commands = _collect_submission_results(
            context.settings,
            context.remote_paths,
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
                "config_path": str(context.config_path),
                "commands": commands,
                "dry_run": bool(args.dry_run),
            },
        )

    tracking_file: Path | None = None
    if job_records:
        tracking_file = write_job_tracking_file(
            context.settings, context.remote_paths, job_records
        )

    job_ids = _collect_job_ids(job_records)
    monitor_cmd = _monitor_command(
        context.settings.cluster_login,
        job_ids,
        ssh_config_file=context.settings.ssh_config_file,
        ssh_options=context.settings.ssh_options,
    )
    payload = submission_payload(
        config_path=context.config_path,
        workspace_mode=context.settings.workspace_mode,
        remote_workdir=context.remote_paths.workdir,
        job_folder=context.remote_paths.job_folder,
        selected_jobs=_selected_job_names(jobs),
        submitted_jobs=submitted_jobs,
        tracking_file=tracking_file,
        commands=commands,
        monitor_command=monitor_cmd,
        dry_run=bool(args.dry_run),
    )
    return _emit_submission_result(
        json_output=json_output,
        payload=payload,
        settings=context.settings,
        remote_paths=context.remote_paths,
        tracking_file=tracking_file,
        job_records=job_records,
        monitor_cmd=monitor_cmd,
        dry_run=bool(args.dry_run),
        include_archive_dirs=True,
    )


def do_stage(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    context = _load_execution_context(args, json_output=json_output)
    if context is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    try:
        if not json_output:
            _print_execution_panel(
                "Stage",
                context.settings,
                context.remote_paths,
            )
        enforce_clean_git(
            context.settings,
            require_clean_git=bool(getattr(args, "require_clean_git", False)),
        )
        test_ssh_connection(
            context.settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=context.settings.ssh_config_file,
            ssh_options=context.settings.ssh_options,
            quiet=json_output,
        )
        commands = sync_project(
            context.settings,
            context.remote_paths,
            dry_run=args.dry_run,
            include_logging_dirs=False,
            quiet=json_output,
        )
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(context.config_path),
                "dry_run": bool(args.dry_run),
            },
        )

    if json_output:
        console.print_json(
            data=stage_payload(
                config_path=context.config_path,
                workspace_mode=context.settings.workspace_mode,
                remote_workdir=context.remote_paths.workdir,
                job_folder=context.remote_paths.job_folder,
                commands=commands,
                dry_run=bool(args.dry_run),
            )
        )
        return 0

    details_table = Table.grid(padding=(0, 1))
    details_table.add_row("Workspace", context.settings.workspace_mode)
    details_table.add_row("Remote workdir", context.remote_paths.workdir)
    if context.settings.workspace_mode == "per-run":
        details_table.add_row("Job folder", context.remote_paths.job_folder)
    console.print()
    console.print(details_table)
    return 0


def do_submit(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    context = _load_execution_context(
        args,
        json_output=json_output,
        existing_job_folder=True,
    )
    if context is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    try:
        if context.settings.workspace_mode == "per-run" and not args.job_folder:
            raise SystemExit(
                "ERROR: --job-folder is required for submit-only when --workspace per-run."
            )
        jobs = _prepare_configured_jobs(
            context.config,
            context.settings,
            args,
            fail_duplicate_names=True,
        )
        if not json_output:
            _print_execution_panel(
                "Submit",
                context.settings,
                context.remote_paths,
                note=(
                    "Skipping stage step. Assuming code is already present on the remote workdir."
                ),
            )
        test_ssh_connection(
            context.settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=context.settings.ssh_config_file,
            ssh_options=context.settings.ssh_options,
            quiet=json_output,
        )
        submitted_jobs, job_records, commands = _collect_submission_results(
            context.settings,
            context.remote_paths,
            jobs,
            dry_run=args.dry_run,
            quiet=json_output,
        )
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(context.config_path),
                "dry_run": bool(args.dry_run),
            },
        )

    tracking_file: Path | None = None
    if job_records:
        tracking_file = write_job_tracking_file(
            context.settings, context.remote_paths, job_records
        )

    job_ids = _collect_job_ids(job_records)
    monitor_cmd = _monitor_command(
        context.settings.cluster_login,
        job_ids,
        ssh_config_file=context.settings.ssh_config_file,
        ssh_options=context.settings.ssh_options,
    )
    payload = submission_payload(
        config_path=context.config_path,
        workspace_mode=context.settings.workspace_mode,
        remote_workdir=context.remote_paths.workdir,
        job_folder=context.remote_paths.job_folder,
        selected_jobs=_selected_job_names(jobs),
        submitted_jobs=submitted_jobs,
        tracking_file=tracking_file,
        commands=commands,
        monitor_command=monitor_cmd,
        dry_run=bool(args.dry_run),
    )
    return _emit_submission_result(
        json_output=json_output,
        payload=payload,
        settings=context.settings,
        remote_paths=context.remote_paths,
        tracking_file=tracking_file,
        job_records=job_records,
        monitor_cmd=monitor_cmd,
        dry_run=bool(args.dry_run),
    )


def do_sbatch(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False))
    context = _load_execution_context(args, json_output=json_output)
    if context is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH or run 'slurm-launcher init'.",
            json_output=json_output,
            payload={"config_path": None, "dry_run": bool(args.dry_run)},
        )

    try:
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
        _validate_predefined_sbatch_file_job(context.settings, job)

        enforce_clean_git(
            context.settings,
            require_clean_git=bool(getattr(args, "require_clean_git", False)),
        )
        test_ssh_connection(
            context.settings.cluster_login,
            dry_run=args.dry_run,
            ssh_config_file=context.settings.ssh_config_file,
            ssh_options=context.settings.ssh_options,
            quiet=json_output,
        )
        if not json_output:
            _print_execution_panel(
                "Sbatch",
                context.settings,
                context.remote_paths,
                extra_lines=[f"[bold]Sbatch file:[/bold] {job.sbatch_file}"],
            )
        stage_commands = sync_project(
            context.settings,
            context.remote_paths,
            dry_run=args.dry_run,
            quiet=json_output,
        )
        submission = submit_job(
            context.settings,
            context.remote_paths,
            job,
            dry_run=args.dry_run,
            quiet=json_output,
        )
    except (RuntimeError, SystemExit, ValueError) as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload={
                "config_path": str(context.config_path),
                "dry_run": bool(args.dry_run),
            },
        )

    job_records: list[dict[str, Any]] = []
    if not args.dry_run:
        job_records.append(build_job_record(job, submission, context.settings))

    tracking_file: Path | None = None
    if job_records:
        tracking_file = write_job_tracking_file(
            context.settings, context.remote_paths, job_records
        )
    job_ids = _collect_job_ids(job_records)
    monitor_cmd = _monitor_command(
        context.settings.cluster_login,
        job_ids,
        ssh_config_file=context.settings.ssh_config_file,
        ssh_options=context.settings.ssh_options,
    )
    submitted_jobs = [build_job_record(job, submission, context.settings)]
    payload = submission_payload(
        config_path=context.config_path,
        workspace_mode=context.settings.workspace_mode,
        remote_workdir=context.remote_paths.workdir,
        job_folder=context.remote_paths.job_folder,
        selected_jobs=[job.name],
        submitted_jobs=submitted_jobs,
        tracking_file=tracking_file,
        commands=[*stage_commands, *submission.commands],
        monitor_command=monitor_cmd,
        dry_run=bool(args.dry_run),
    )
    return _emit_submission_result(
        json_output=json_output,
        payload=payload,
        settings=context.settings,
        remote_paths=context.remote_paths,
        tracking_file=tracking_file,
        job_records=job_records,
        monitor_cmd=monitor_cmd,
        dry_run=bool(args.dry_run),
    )


def _fail_duplicate_jobs(jobs: list[JobSpec]) -> None:
    fail_duplicate_jobs(jobs)


def _fail_if_not_absolute(label: str, value: str | None) -> None:
    fail_if_not_absolute(label, value)


def _resolve_local_sbatch_file_path(
    settings: LauncherSettings, sbatch_file: str
) -> Path | None:
    return resolve_local_sbatch_file_path(settings, sbatch_file)


def _validate_predefined_sbatch_file_job(
    settings: LauncherSettings, job: JobSpec
) -> None:
    validate_predefined_sbatch_file_job(settings, job)


def _validate_predefined_sbatch_jobs(
    settings: LauncherSettings, jobs: list[JobSpec]
) -> None:
    validate_predefined_sbatch_jobs(settings, jobs)


def _remote_runtime_checks(settings: LauncherSettings) -> list[str]:
    return remote_runtime_checks(settings)


def do_validate(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    if args.check_remote_paths and not args.ssh:
        return _emit_command_error(
            "ERROR: --check-remote-paths requires --ssh.",
            json_output=json_output,
            payload=validate_payload(
                ok=False,
                config_path=Path(str(args.config)) if args.config else None,
                workspace_mode=_workspace_mode_from_args(args),
                selected_jobs=list(args.only or []),
                warnings=[],
                errors=["ERROR: --check-remote-paths requires --ssh."],
                ssh_checked=bool(args.ssh),
                remote_checks={
                    "requested": bool(args.check_remote_paths),
                    "checks": [],
                    "ok": False,
                },
            ),
        )

    config_arg = str(args.config) if args.config else None
    config_path = _resolve_config_path(config_arg)
    if config_path is None:
        return _emit_command_error(
            "Config file not found. Pass --config PATH.",
            json_output=json_output,
            payload=validate_payload(
                ok=False,
                config_path=None,
                workspace_mode=_workspace_mode_from_args(args),
                selected_jobs=list(args.only or []),
                warnings=[],
                errors=["Config file not found. Pass --config PATH."],
                ssh_checked=bool(args.ssh),
                remote_checks={
                    "requested": bool(args.check_remote_paths),
                    "checks": [],
                    "ok": False,
                },
            ),
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
        jobs = _prepare_configured_jobs(
            config,
            settings,
            args,
            fail_duplicate_names=True,
            validate_predefined_jobs=False,
        )
        selected_jobs = _selected_job_names(jobs)

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
            payload=validate_payload(
                ok=False,
                config_path=config_path,
                workspace_mode=workspace_mode,
                selected_jobs=selected_jobs,
                warnings=[],
                errors=[str(exc)],
                ssh_checked=bool(args.ssh),
                remote_checks=remote_checks,
            ),
        )

    if json_output:
        console.print_json(
            data=validate_payload(
                ok=True,
                config_path=config_path,
                workspace_mode=settings.workspace_mode,
                selected_jobs=selected_jobs,
                warnings=[],
                errors=[],
                ssh_checked=bool(args.ssh),
                remote_checks=remote_checks,
            )
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
        jobs = _prepare_configured_jobs(
            config,
            settings,
            args,
            fail_duplicate_names=True,
        )
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
            data=render_payload(
                config_path=config_path,
                workspace_mode=settings.workspace_mode,
                selected_jobs=_selected_job_names(jobs),
                rendered_jobs=rendered_jobs,
                job_scripts=job_scripts,
                sbatch_scripts=sbatch_scripts,
            )
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
        console.print_json(data=tracking_payload_to_dict(payload, selected))
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
    return collect_job_ids(records)


def _print_job_logs_from_records(jobs: list[JobRecord | dict[str, object]]) -> None:
    print_job_logs_from_records(console, jobs)


def do_monitor(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    tracking_path = resolve_tracking_file(args.tracking_file)
    if tracking_path is None:
        return _emit_command_error(
            "No tracking file found. Run a non-dry submission first or pass --tracking-file.",
            json_output=json_output,
            payload=monitor_payload(
                ok=False,
                tracking_file=None,
                job_ids=[],
                command=None,
                dry_run=bool(args.dry_run),
            ),
        )

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        return _emit_command_error(
            str(exc),
            json_output=json_output,
            payload=monitor_payload(
                ok=False,
                tracking_file=tracking_path,
                job_ids=[],
                command=None,
                dry_run=bool(args.dry_run),
            ),
        )

    if not payload.cluster_login:
        return _emit_command_error(
            f"Missing cluster_login in tracking file: {tracking_path}",
            json_output=json_output,
            payload=monitor_payload(
                ok=False,
                tracking_file=tracking_path,
                job_ids=[],
                command=None,
                dry_run=bool(args.dry_run),
            ),
        )

    selected = payload.filter_jobs(names=set(args.only) if args.only else None)
    job_ids = payload.runnable_job_ids(selected)
    if not job_ids:
        return _emit_command_error(
            "No runnable job IDs found in tracking file selection.",
            json_output=json_output,
            payload=monitor_payload(
                ok=False,
                tracking_file=tracking_path,
                job_ids=[],
                command=None,
                dry_run=bool(args.dry_run),
            ),
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
        result_payload = monitor_payload(
            ok=True,
            tracking_file=tracking_path,
            job_ids=job_ids,
            command=command,
            dry_run=bool(args.dry_run),
        )
        if not args.dry_run:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            console.print_json(
                data=monitor_payload(
                    ok=result.returncode == 0,
                    tracking_file=tracking_path,
                    job_ids=job_ids,
                    command=command,
                    dry_run=bool(args.dry_run),
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
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
    payload = doctor_payload(
        cluster_login=cluster_login,
        config_path=config_path,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
        archive_dir=effective_archive,
        archive_dir_source=archive_source,
    )

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
        payload = doctor_payload(
            cluster_login=cluster_login,
            config_path=config_path,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
            archive_dir=effective_archive,
            archive_dir_source=archive_source,
            ssh_ok=True,
            remote_tools=remote_tools,
        )

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
    states = {str(state) for state in args.state if str(state).strip()} or None
    return list_recent_jobs(
        cluster_login,
        user=args.user,
        hours=args.hours,
        limit=args.limit,
        states=states,
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


COMMAND_HANDLERS = {
    "doctor": do_doctor,
    "download-artifacts": do_download_artifacts,
    "download-logs": do_download_logs,
    "init": do_init,
    "job-log": do_job_log,
    "job-show": do_job_show,
    "jobs": do_jobs,
    "logs": do_logs,
    "monitor": do_monitor,
    "render": do_render,
    "run": do_run,
    "sbatch": do_sbatch,
    "stage": do_stage,
    "submit": do_submit,
    "validate": do_validate,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = getattr(args, "command", None) or DEFAULT_COMMAND
    return COMMAND_HANDLERS[command](args)
