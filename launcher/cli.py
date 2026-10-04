"""CLI entry point for the remote SLURM launcher."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .execution import do_preflight, do_run, do_sbatch, do_stage, do_submit
from .logs import add_logs_args, run_logs
from .core import (
    JobSpec,
    LauncherSettings,
    build_job_script,
    build_launcher_metadata,
    build_sbatch_script,
    format_sbatch_options,
    resolve_remote_paths,
    ssh_script,
    submit_job,
    test_ssh_connection,
)
from .command_specs import COMMAND_SPECS
from .config_utils import (
    build_settings,
    collect_config_warnings,
    configured_run_only,
    ensure_list,
    resolve_config_path,
    load_config,
    normalize_workspace_mode,
    prepare_jobs,
    remote_runtime_checks,
    resolve_local_sbatch_file_path,
    validate_jobs,
    validate_settings,
)
from .artifacts import add_artifacts_parser, dispatch_artifacts
from .download_logs import add_download_logs_args, run_download_logs
from .init_wizard import init_config
from .job_tools import (
    effective_archive_dir,
    list_recent_jobs,
    show_job_details,
)
from .status import run_status
from .summary import run_summary
from .payloads import (
    doctor_payload,
    error_payload,
    render_payload,
    validate_payload,
)

console = Console()
err_console = Console(stderr=True)
GENERIC_CONFIG_PATH = Path.home() / ".config" / "slurm-launcher" / "config.py"

try:
    PACKAGE_VERSION = version("slurm-launcher")
except PackageNotFoundError:
    PACKAGE_VERSION = "unknown"


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(
        description="Submit SLURM jobs on a remote cluster"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"slurm-launcher {PACKAGE_VERSION}",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    init_parser = subparsers.add_parser("init", help=COMMAND_SPECS["init"].summary)
    init_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing config file"
    )
    init_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Copy the template without prompting (still updates .gitignore).",
    )

    status_parser = subparsers.add_parser(
        "status", help=COMMAND_SPECS["status"].summary
    )
    _add_status_args(status_parser)

    logs_parser = subparsers.add_parser("logs", help=COMMAND_SPECS["logs"].summary)
    add_logs_args(logs_parser)

    add_artifacts_parser(subparsers)

    download_logs_parser = subparsers.add_parser(
        "download-logs",
        help=COMMAND_SPECS["download-logs"].summary,
    )
    add_download_logs_args(download_logs_parser)

    jobs_parser = subparsers.add_parser("jobs", help=COMMAND_SPECS["jobs"].summary)
    _add_jobs_args(jobs_parser)

    job_show_parser = subparsers.add_parser(
        "job-show", help=COMMAND_SPECS["job-show"].summary
    )
    _add_job_show_args(job_show_parser)

    doctor_parser = subparsers.add_parser(
        "doctor", help=COMMAND_SPECS["doctor"].summary
    )
    _add_doctor_args(doctor_parser)

    validate_parser = subparsers.add_parser(
        "validate",
        help=COMMAND_SPECS["validate"].summary,
    )
    _add_validate_args(validate_parser)

    preflight_parser = subparsers.add_parser(
        "preflight",
        help=COMMAND_SPECS["preflight"].summary,
    )
    _add_preflight_args(preflight_parser)

    summary_parser = subparsers.add_parser(
        "summary",
        help=COMMAND_SPECS["summary"].summary,
    )
    _add_summary_args(summary_parser)

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

    run_parser = subparsers.add_parser("run", help=COMMAND_SPECS["run"].summary)
    _add_run_args(run_parser)
    for name, command in subparsers.choices.items():
        spec = COMMAND_SPECS[name]
        command.description = spec.summary
        command.epilog = "Examples:\n" + "\n".join(
            f"  slurm-launcher {example}" for example in spec.examples
        )
        command.formatter_class = argparse.RawDescriptionHelpFormatter
    return parser, run_parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser, _ = _build_parser()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args:
        parser.print_help()
        parser.exit()
    return parser.parse_args(raw_args)


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
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only", nargs="+", help="Select these job names.")
    selection.add_argument(
        "--all",
        dest="all_jobs",
        action="store_true",
        help="Explicitly select every configured job.",
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
    _add_job_selection_arg(parser)
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
    parser.add_argument(
        "--run", required=True, help="Staged run ID, tracking path, or latest."
    )
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the frozen submission without SSH.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable result."
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


def _add_preflight_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run", required=True, help="Staged run ID, tracking path, or latest."
    )
    _add_job_selection_arg(parser)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print prerequisite checks without SSH."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a machine-readable result."
    )


def _add_summary_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", help="Run ID, tracking path, or latest (default).")
    parser.add_argument(
        "--json", action="store_true", help="Print a read-only run summary."
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


def _add_status_args(parser: argparse.ArgumentParser) -> None:
    _add_cluster_target_args(parser)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--run", help="Run ID, tracking path, or latest (default).")
    target.add_argument("--job-id", help="Query one scheduler job directly.")
    parser.add_argument("--only", nargs="+", help="Select tracked job names.")
    parser.add_argument(
        "--json", action="store_true", help="Print state, exit codes, and probe errors."
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


def _add_job_show_args(parser: argparse.ArgumentParser) -> None:
    _add_cluster_target_args(parser)
    parser.add_argument("job_id", help="SLURM job id to inspect.")
    parser.add_argument(
        "--sbatch",
        action="store_true",
        help="Also retrieve the exact batch script submitted for the job.",
    )
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


def _workspace_mode_from_args(args: argparse.Namespace) -> str | None:
    workspace = getattr(args, "workspace", None)
    if workspace:
        return normalize_workspace_mode(workspace, setting_name="--workspace")
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


def _prepare_configured_jobs(
    config: ModuleType,
    settings: LauncherSettings,
    args: argparse.Namespace,
) -> list[JobSpec]:
    validate_settings(settings)
    jobs = prepare_jobs(
        config,
        configured_run_only(config, args, require_selection=False),
        settings.default_env,
    )
    validate_jobs(settings, jobs)
    return jobs


def _resolve_cluster_context(
    args: argparse.Namespace,
) -> tuple[str, str | None, str | None, list[str], Path | None] | None:
    config_arg = str(args.config) if getattr(args, "config", None) else None
    explicit_login = str(getattr(args, "cluster_login", None) or "").strip()
    # An explicit destination is a complete request to use the caller's
    # normal SSH resolution unless --config was also explicitly supplied.
    # Loading the generic config here can otherwise turn `ssh acc` into
    # `ssh -F /dev/null acc` and silently disable the requested alias.
    config_path = (
        resolve_config_path(config_arg, extra_candidates=[GENERIC_CONFIG_PATH])
        if config_arg or not explicit_login
        else None
    )
    config = None
    if config_arg and config_path is None:
        raise ValueError(
            f"Config not found: {config_arg}. Pass --config PATH or --cluster-login."
        )
    if config_path is not None:
        config = load_config(config_path)

    cluster_login = str(
        explicit_login or getattr(config, "CLUSTER_LOGIN", None) or ""
    ).strip()
    if not cluster_login:
        raise ValueError("Pass --cluster-login or configure CLUSTER_LOGIN.")

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
    config_path = resolve_config_path(config_arg)
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
        jobs = _prepare_configured_jobs(config, settings, args)
        selected_jobs = [job.name for job in jobs]

        remote_paths = resolve_remote_paths(settings)

        if args.ssh:
            test_ssh_connection(
                settings.cluster_login,
                dry_run=False,
                ssh_config_file=settings.ssh_config_file,
                ssh_options=settings.ssh_options,
                quiet=json_output,
            )

            checks = remote_runtime_checks(settings) if args.check_remote_paths else []
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

        warnings = collect_config_warnings(settings, jobs)
    except (OSError, RuntimeError, SystemExit, TypeError, ValueError) as exc:
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
                warnings=warnings,
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
    if warnings:
        console.print("\n[bold yellow]Warnings:[/bold yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")
    if args.check_remote_paths:
        console.print("Remote checks OK", style="green")
    return 0


def do_render(args: argparse.Namespace) -> int:
    json_output = bool(args.json)
    config_arg = str(args.config) if args.config else None
    config_path = resolve_config_path(config_arg)
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
        jobs = _prepare_configured_jobs(config, settings, args)
        remote_paths = resolve_remote_paths(settings)
    except (OSError, RuntimeError, SystemExit, TypeError, ValueError) as exc:
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
            preview = submit_job(settings, remote_paths, job, dry_run=True, quiet=True)
            job_payload: dict[str, Any] = {
                "job_name": job.name,
                "job_type": "sbatch_file",
                "sbatch_file": str(job.sbatch_file),
                "remote_sbatch_path": preview.remote_sbatch_path,
                "sbatch_command": preview.sbatch_command,
            }
            if args.job_script:
                local_path = resolve_local_sbatch_file_path(
                    settings, str(job.sbatch_file)
                )
                job_payload["job_script"] = local_path.read_bytes().decode("utf-8")
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
                selected_jobs=[job.name for job in jobs],
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


def do_status(args: argparse.Namespace) -> int:
    context = None
    if args.job_id:
        if args.only:
            return _emit_command_error(
                "--only selects tracked jobs, not a direct job ID.",
                json_output=args.json,
            )
        context = _resolve_cluster_context(args)
        if context is None:
            return 1
    elif args.config or args.cluster_login:
        return _emit_command_error(
            "Tracked status uses its saved cluster context. Use --job-id for a direct query.",
            json_output=args.json,
        )
    return run_status(
        tracking_file=args.run,
        job_id=args.job_id,
        cluster_login=context[0] if context else None,
        ssh_config_file=context[2] if context else None,
        ssh_options=context[3] if context else None,
        selected_jobs=args.only,
        json_output=args.json,
    )


def do_logs(args: argparse.Namespace) -> int:
    context = None
    if args.job_id:
        context = _resolve_cluster_context(args)
        if context is None:
            return 1
    return run_logs(args, cluster_context=context)


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
                quiet=bool(args.json),
            )
        except (RuntimeError, SystemExit) as exc:
            return _emit_command_error(str(exc), json_output=bool(args.json))

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
        stdout, _ = ssh_script(
            cluster_login,
            script,
            dry_run=False,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
            quiet=bool(args.json),
        )
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
        include_sbatch=args.sbatch,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )


COMMAND_HANDLERS = {
    "doctor": do_doctor,
    "download-logs": run_download_logs,
    "init": do_init,
    "job-show": do_job_show,
    "jobs": do_jobs,
    "logs": do_logs,
    "render": do_render,
    "run": do_run,
    "artifacts": dispatch_artifacts,
    "preflight": do_preflight,
    "sbatch": do_sbatch,
    "stage": do_stage,
    "status": do_status,
    "submit": do_submit,
    "summary": run_summary,
    "validate": do_validate,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = args.command
    try:
        return COMMAND_HANDLERS[command](args)
    except (OSError, RuntimeError, SystemExit, TypeError, ValueError) as exc:
        return _emit_command_error(
            str(exc), json_output=bool(getattr(args, "json", False))
        )
