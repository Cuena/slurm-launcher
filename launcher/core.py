"""Core logic for the remote SLURM launcher."""

from __future__ import annotations

import base64
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from rich.console import Console
from rich.syntax import Syntax

from . import transport

DEFAULT_RSYNC_EXCLUDES = [
    ".git/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.egg-info/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".uv-cache/",
    ".cache/",
    ".venv/",
    "venv/",
    ".idea/",
    ".vscode/",
    "slurm_output/",
    "slurm/",
    "outputs/",
    "logs/",
]

console = Console()


@dataclass
class JobSpec:
    """Single job declaration coming from the config file."""

    name: str
    command: str | None = None
    sbatch_file: str | None = None
    sbatch_args: list[str] = field(default_factory=list)
    env: dict[str, Any] = field(default_factory=dict)
    sbatch: dict[str, Any] = field(default_factory=dict)
    setup: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.command is not None:
            self.command = str(self.command).strip()
        if self.sbatch_file is not None:
            self.sbatch_file = str(self.sbatch_file).strip()
        has_command = bool(self.command)
        has_sbatch_file = bool(self.sbatch_file)
        if has_command == has_sbatch_file:
            raise ValueError(
                f"Job '{self.name}' must define exactly one of 'command' or 'sbatch_file'."
            )
        if has_sbatch_file and (self.env or self.sbatch or self.setup):
            raise ValueError(
                f"Job '{self.name}' with 'sbatch_file' cannot define "
                "'env', 'sbatch', or 'setup'. Put those settings in the sbatch file."
            )
        self.setup = [str(cmd) for cmd in self.setup]
        self.sbatch_args = [str(arg) for arg in self.sbatch_args]
        self.artifacts = [str(path) for path in self.artifacts]
        self.requires = [str(path) for path in self.requires]

    def render_command(self) -> str:
        if self.command is None:
            raise ValueError(
                f"Job '{self.name}' does not define 'command' (uses 'sbatch_file')."
            )
        return self.command

    def uses_sbatch_file(self) -> bool:
        return self.sbatch_file is not None


@dataclass(frozen=True)
class LauncherSettings:
    cluster_login: str
    ssh_config_file: str | None
    ssh_options: list[str]
    remote_workspace_base: str | None
    remote_log_base_path: str
    workspace_mode: str
    remote_workspace_dir: str | None
    project_root: Path
    project_prefix: str
    venv_python_executable: str | None
    default_env: dict[str, Any]
    default_sbatch: dict[str, Any]
    extra_rsync_excludes: list[str]
    extra_rsync_args: list[str]
    remote_slurm_dashboard_log_archive_dir: str | None
    remote_slurm_dashboard_log_view_dir: str | None
    runtime_mode: str
    singularity_image_path: str | None
    singularity_exec_flags: list[str]
    artifact_paths: list[str]
    require_clean_git: bool
    sync_symlinks: str
    rsync_login: str | None = None


@dataclass(frozen=True)
class RemotePaths:
    job_folder: str
    workdir: str
    logdir: str
    slurm_output_dir: str


@dataclass(frozen=True)
class SubmissionResult:
    job_id: str
    sbatch_command: str
    sbatch_options: dict[str, Any]
    remote_sbatch_path: str
    commands: list[str] = field(default_factory=list)


class SubmissionError(RuntimeError):
    """A rejected or uncertain dispatch, retaining its transport evidence."""

    def __init__(
        self,
        message: str,
        *,
        rejected: bool = False,
        stdout: str = "",
        stderr: str = "",
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.rejected = rejected
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def resolve_local_project_path(project_root: Path, configured_path: str) -> Path | None:
    candidate = Path(configured_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (project_root / candidate).resolve()
    try:
        resolved.relative_to(project_root.resolve())
    except ValueError:
        return None
    return resolved


def ssh_script(
    cluster_login: str,
    script: str,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    quiet: bool = False,
) -> tuple[str, str]:
    script = script.rstrip() + "\n"
    if dry_run:
        if not quiet:
            console.print(
                f"[yellow]dry-run[/yellow] "
                f"{transport.format_ssh_command(cluster_login, ssh_config_file=ssh_config_file, ssh_options=ssh_options)} <<'EOF'"
            )
            console.print(Syntax(script.rstrip(), "bash"))
            console.print("EOF")
        return "", ""
    result = transport.run_ssh_capture(
        cluster_login,
        script,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if result.returncode:
        cause = subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
        raise RuntimeError(
            f"SSH command failed with exit code {result.returncode}: {result.stderr.strip()}"
        ) from cause
    return result.stdout, result.stderr


def test_ssh_connection(
    cluster_login: str,
    dry_run: bool,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    quiet: bool = False,
) -> None:
    if dry_run:
        if not quiet:
            console.print(
                f"[yellow]dry-run[/yellow] skip SSH connectivity check for {cluster_login}"
            )
        return
    stdout, _ = ssh_script(
        cluster_login,
        "echo SSH_OK",
        dry_run=dry_run,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
        quiet=quiet,
    )
    if "SSH_OK" not in stdout:
        raise SystemExit("ERROR: SSH test failed. Check your SSH setup.")
    if not quiet:
        console.print("SSH connection OK", style="green")


def ensure_remote_directories(
    settings: LauncherSettings,
    paths: list[str],
    dry_run: bool,
    *,
    quiet: bool = False,
) -> list[str]:
    unique_paths = sorted(set(paths))
    if not unique_paths:
        return []

    mkdir_cmd = f"mkdir -p {' '.join(shlex.quote(p) for p in unique_paths)}"
    command = transport.format_ssh_command(
        settings.cluster_login,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
        remote_command=mkdir_cmd,
    )
    if dry_run:
        if not quiet:
            console.print(f"[yellow]dry-run[/yellow] {command}")
        return [command]
    ssh_script(
        settings.cluster_login,
        mkdir_cmd,
        dry_run=False,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
        quiet=quiet,
    )
    return [command]


def sync_project(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    dry_run: bool,
    *,
    include_logging_dirs: bool = True,
    quiet: bool = False,
    source_state: SourceState | None = None,
) -> list[str]:
    source_state = source_state or inspect_source_state(settings.project_root)
    remote_directories = [remote_paths.workdir]
    if include_logging_dirs:
        remote_directories.extend([remote_paths.logdir, remote_paths.slurm_output_dir])
        if settings.remote_slurm_dashboard_log_archive_dir:
            remote_directories.append(settings.remote_slurm_dashboard_log_archive_dir)
        if settings.remote_slurm_dashboard_log_view_dir:
            remote_directories.append(settings.remote_slurm_dashboard_log_view_dir)

    commands = ensure_remote_directories(
        settings,
        remote_directories,
        dry_run,
        quiet=quiet,
    )

    excludes = DEFAULT_RSYNC_EXCLUDES + settings.extra_rsync_excludes
    destination = (
        f"{settings.rsync_login or settings.cluster_login}:{remote_paths.workdir}/"
    )
    cmd = [
        "rsync",
        "-az",
        "--info=progress2",
        "-e",
        transport.build_rsync_ssh_command(settings.ssh_config_file, settings.ssh_options),
    ]
    if settings.sync_symlinks == "copy-links":
        cmd.append("--copy-links")
    if dry_run:
        cmd.append("--dry-run")
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    cmd.extend(settings.extra_rsync_args)
    cmd.extend([f"{settings.project_root}/", destination])
    rsync_command = shlex.join(cmd)
    commands.append(rsync_command)

    if not quiet:
        console.print(f"Syncing project to {destination}")
    if dry_run:
        if not quiet:
            console.print("dry-run rsync command:", style="yellow")
            console.print(rsync_command, style="dim")
            console.print("dry-run skipping rsync execution", style="yellow")
        commands.append(
            format_source_metadata_command(settings, remote_paths, source_state)
        )
        return commands
    subprocess.run(cmd, check=True, stdout=sys.stderr if quiet else None)
    commands.extend(write_remote_source_metadata(settings, remote_paths, source_state))
    if not quiet:
        console.print("Sync complete", style="green")
    return commands


def build_job_script(
    job: JobSpec, settings: LauncherSettings, remote_paths: RemotePaths
) -> str:
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"cd {shlex.quote(remote_paths.workdir)}",
    ]
    for key, value in job.env.items():
        lines.append(f"export {key}={shlex.quote(str(value))}")
    if settings.runtime_mode == "venv":
        venv_python = settings.venv_python_executable
        if not venv_python:
            raise SystemExit(
                "ERROR: venv runtime selected but VENV_PYTHON_EXECUTABLE is missing."
            )
        venv_bin = Path(venv_python).parent
        activate = venv_bin / "activate"
        lines.extend(
            [
                f"test -f {shlex.quote(str(activate))} || {{ printf '%s\\n' {shlex.quote(f'ERROR: venv activate script not found: {activate}')} >&2; exit 1; }}",
                f"source {shlex.quote(str(activate))}",
            ]
        )
    lines.extend(job.setup)
    lines.append(render_runtime_command(job, settings))
    return "\n".join(lines).rstrip() + "\n"


def build_launcher_metadata(
    job: JobSpec,
    settings: LauncherSettings,
) -> dict[str, Any]:
    runtime_artifact: str | None = None
    if settings.runtime_mode == "venv":
        runtime_artifact = settings.venv_python_executable
    elif settings.runtime_mode == "singularity":
        runtime_artifact = settings.singularity_image_path
    return {
        "managed": True,
        "runtime_kind": settings.runtime_mode,
        "runtime_artifact": runtime_artifact,
        "entry_command": job.command,
    }


def render_sbatch_directives(options: dict[str, Any]) -> list[str]:
    directives: list[str] = []
    for key, value in options.items():
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                directives.append(f"#SBATCH {flag}")
            continue
        directives.append(f"#SBATCH {flag}={shlex.quote(str(value))}")
    return directives


def build_sbatch_script(
    job_script: str,
    sbatch_options: dict[str, Any],
    *,
    launcher_metadata: dict[str, Any] | None = None,
) -> str:
    script_lines = job_script.splitlines()
    if script_lines and script_lines[0].startswith("#!"):
        shebang = script_lines[0]
        body = script_lines[1:]
    else:
        shebang = "#!/bin/bash"
        body = script_lines

    lines: list[str] = [shebang]
    if launcher_metadata is not None:
        lines.append(
            "# slurm-launcher-metadata: "
            + json.dumps(launcher_metadata, sort_keys=True)
        )
    lines.extend(render_sbatch_directives(sbatch_options))
    lines.extend(body)
    return "\n".join(lines).rstrip() + "\n"


def format_sbatch_options(
    job: JobSpec, settings: LauncherSettings, remote_paths: RemotePaths
) -> dict[str, Any]:
    options = {**settings.default_sbatch, **job.sbatch}
    if "chdir" in options or "ch_dir" in options:
        raise ValueError(
            "ERROR: sbatch 'chdir' is not supported. "
            "The launcher always runs from its managed remote workdir."
        )

    options.setdefault("job-name", job.name)
    archive_dir = settings.remote_slurm_dashboard_log_archive_dir
    if archive_dir:
        options.setdefault("output", f"{archive_dir}/%j.out")
        options.setdefault("error", f"{archive_dir}/%j.err")
    else:
        job_label = str(options.get("job-name") or job.name).replace(" ", "_")
        options.setdefault(
            "output", f"{remote_paths.slurm_output_dir}/{job_label}-%j.out"
        )
        options.setdefault(
            "error", f"{remote_paths.slurm_output_dir}/{job_label}-%j.err"
        )
    return options


def parse_job_id(output: str) -> str:
    """Accept only sbatch --parsable's job ID (and optional cluster suffix)."""
    match = re.fullmatch(r"([1-9][0-9]*)(?:;[A-Za-z0-9_.-]+)?", output.strip())
    if match is None:
        raise ValueError(f"Ambiguous sbatch response: {output!r}")
    return match.group(1)


def _sanitize_log_view_component(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    cleaned = normalized.strip("._-")
    return cleaned or fallback


def create_log_view_symlinks(
    settings: LauncherSettings,
    job: JobSpec,
    submission: SubmissionResult,
    *,
    quiet: bool = False,
) -> None:
    archive_dir = settings.remote_slurm_dashboard_log_archive_dir
    view_dir = settings.remote_slurm_dashboard_log_view_dir
    if not archive_dir or not view_dir:
        return
    if submission.job_id in {"", "unknown", "dry-run"}:
        return
    stdout_path = resolve_log_path(
        submission.sbatch_options.get("output"), submission.job_id
    )
    stderr_path = resolve_log_path(
        submission.sbatch_options.get("error"), submission.job_id
    )
    if not stdout_path and not stderr_path:
        return

    project_label = _sanitize_log_view_component(settings.project_prefix, "project")
    date_label = datetime.now().strftime("%Y-%m-%d")
    job_label = _sanitize_log_view_component(job.name, "job")
    view_root = view_dir.rstrip("/")
    view_subdir = f"{view_root}/{project_label}/{date_label}"
    link_script_lines = [
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(view_subdir)}",
    ]
    if stdout_path:
        dst_stdout = f"{view_subdir}/{job_label}-{submission.job_id}.out"
        link_script_lines.append(
            f"ln -sfn {shlex.quote(stdout_path)} {shlex.quote(dst_stdout)}"
        )
    if stderr_path:
        dst_stderr = f"{view_subdir}/{job_label}-{submission.job_id}.err"
        link_script_lines.append(
            f"ln -sfn {shlex.quote(stderr_path)} {shlex.quote(dst_stderr)}"
        )
    link_script = "\n".join(link_script_lines)
    try:
        ssh_script(
            settings.cluster_login,
            link_script,
            dry_run=False,
            ssh_config_file=settings.ssh_config_file,
            ssh_options=settings.ssh_options,
            quiet=quiet,
        )
    except RuntimeError as exc:
        if not quiet:
            console.print(
                "WARNING: Failed to create slurm-dashboard view symlinks "
                f"for job {submission.job_id}: {exc}",
                style="yellow",
            )


def submit_job(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job: JobSpec,
    *,
    dry_run: bool,
    quiet: bool = False,
    frozen_script: str | None = None,
    frozen_options: dict[str, Any] | None = None,
    on_acknowledged: Callable[[SubmissionResult], None] | None = None,
) -> SubmissionResult:
    """Dispatch one exact script; persist acknowledgment before optional probes."""
    if not job.name or Path(job.name).name != job.name or job.name in {".", ".."}:
        raise ValueError(f"Job name must be a single path component: {job.name!r}")
    options = frozen_options
    if options is None:
        options = (
            {}
            if job.uses_sbatch_file()
            else format_sbatch_options(job, settings, remote_paths)
        )
    script = frozen_script
    if script is None:
        if job.sbatch_file:
            local_path = resolve_local_project_path(
                settings.project_root, job.sbatch_file
            )
            if local_path is None:
                raise ValueError("sbatch_file must stay inside LOCAL_ROOT")
            script = local_path.read_bytes().decode("utf-8")
        else:
            script = build_sbatch_script(
                build_job_script(job, settings, remote_paths),
                options,
                launcher_metadata=build_launcher_metadata(job, settings),
            )
    remote_sbatch_path = f"{remote_paths.logdir}/{job.name}.sbatch"
    sbatch_cmd = shlex.join(
        ["sbatch", "--parsable", *job.sbatch_args, remote_sbatch_path]
    )
    # Base64 preserves exact bytes and cannot collide with a script heredoc delimiter.
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    frame = f"__SLURM_LAUNCHER_{uuid4().hex}__"
    directories = [remote_paths.logdir, remote_paths.slurm_output_dir]
    if settings.remote_slurm_dashboard_log_archive_dir:
        directories.append(settings.remote_slurm_dashboard_log_archive_dir)
    submission_script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {' '.join(shlex.quote(path) for path in directories)}",
            f"printf %s {shlex.quote(encoded)} | base64 -d > {shlex.quote(remote_sbatch_path)}",
            f"cd {shlex.quote(remote_paths.workdir)}",
            # exec makes this status belong to sbatch, never a subshell EXIT trap.
            f"if sbatch_stdout=$(trap - EXIT ERR; exec {sbatch_cmd}); then",
            "    sbatch_status=0",
            "else",
            "    sbatch_status=$?",
            "fi",
            "printf '%s\\n' \"$sbatch_stdout\"",
            "sbatch_encoded=$(trap - EXIT ERR; printf %s \"$sbatch_stdout\" | base64 | tr -d '\\n')",
            f"printf '\\n{frame}:%s:%s\\n' \"$sbatch_status\" \"$sbatch_encoded\"",
        ]
    )
    command = transport.format_ssh_script_command(
        settings.cluster_login,
        submission_script,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
    )
    if dry_run:
        return SubmissionResult(
            "dry-run", sbatch_cmd, options, remote_sbatch_path, [command]
        )
    try:
        result = transport.run_ssh_capture(
            settings.cluster_login,
            submission_script,
            ssh_config_file=settings.ssh_config_file,
            ssh_options=settings.ssh_options,
        )
    except subprocess.CalledProcessError as exc:
        result = subprocess.CompletedProcess(
            exc.cmd, exc.returncode, exc.stdout or "", exc.stderr or ""
        )
    frames = re.findall(
        rf"^{re.escape(frame)}:([0-9]+):([A-Za-z0-9+/=]*)$",
        result.stdout,
        flags=re.MULTILINE,
    )
    evidence = dict(
        stdout=result.stdout, stderr=result.stderr, returncode=result.returncode
    )
    if len(frames) != 1:
        raise SubmissionError(
            "Submission outcome is unknown: missing or ambiguous sbatch acknowledgement.",
            **evidence,
        )
    status_text, encoded_response = frames[0]
    try:
        response = base64.b64decode(encoded_response, validate=True).decode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise SubmissionError(
            "Submission outcome is unknown: invalid sbatch acknowledgement.", **evidence
        ) from exc
    status = int(status_text)
    try:
        job_id = parse_job_id(response)
    except ValueError as exc:
        if status and not response.strip():
            raise SubmissionError(
                f"sbatch rejected the submission (exit {status}): {result.stderr.strip()}",
                rejected=True,
                **evidence,
            ) from exc
        raise SubmissionError(
            f"Submission outcome is unknown: {exc}", **evidence
        ) from exc
    submission = SubmissionResult(
        job_id,
        sbatch_cmd,
        dict(options),
        remote_sbatch_path,
        [command],
    )
    if on_acknowledged is not None:
        on_acknowledged(submission)
    try:
        if job.uses_sbatch_file():
            from .job_tools import resolve_job_log_info

            info = resolve_job_log_info(
                settings.cluster_login,
                submission.job_id,
                archive_dir=None,
                ssh_config_file=settings.ssh_config_file,
                ssh_options=settings.ssh_options,
            )
            if info.stdout:
                submission.sbatch_options["output"] = info.stdout
            if info.stderr:
                submission.sbatch_options["error"] = info.stderr
        create_log_view_symlinks(settings, job, submission, quiet=quiet)
    except Exception as exc:
        if not quiet:
            console.print(
                f"WARNING: Submitted {submission.job_id}; log enrichment failed: {exc}",
                style="yellow",
            )
    if not quiet:
        console.print(
            f"Submitted {job.name} -> {submission.job_id}", style="bold green"
        )
    return submission


def resolve_log_path(template: str | None, job_id: str) -> str | None:
    if not template:
        return None
    path = str(template)
    if job_id and job_id != "dry-run":
        path = path.replace("%j", job_id).replace("%J", job_id)
    return path


def build_job_record(
    job: JobSpec,
    submission: SubmissionResult,
    settings: LauncherSettings,
) -> dict[str, Any]:
    stdout_path = resolve_log_path(
        submission.sbatch_options.get("output"), submission.job_id
    )
    stderr_path = resolve_log_path(
        submission.sbatch_options.get("error"), submission.job_id
    )
    launcher = build_launcher_metadata(job, settings)
    if job.uses_sbatch_file():
        launcher["runtime_kind"] = "sbatch_file"
        launcher["runtime_artifact"] = job.sbatch_file
    return {
        "job_name": job.name,
        "job_id": submission.job_id,
        "stdout": stdout_path,
        "stderr": stderr_path,
        "sbatch_command": submission.sbatch_command,
        "remote_sbatch": submission.remote_sbatch_path,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "launcher": launcher,
        "artifacts": job.artifacts,
        "requires": job.requires,
        "array_spec": (
            str(submission.sbatch_options["array"])
            if submission.sbatch_options.get("array") is not None
            else None
        ),
    }


def write_job_tracking_file(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job_records: list[dict[str, Any]],
    *,
    run_dir: Path | None = None,
) -> Path:
    """Atomically merge progress without dropping earlier acknowledgments."""
    from .tracking import atomic_write_json

    root = settings.project_root / "slurm_output"
    selected_run_dir = run_dir or root / remote_paths.job_folder
    output_path = selected_run_dir / "jobs.json"
    payload = (
        json.loads(output_path.read_text())
        if output_path.exists()
        else {
            "created_at": datetime.now().isoformat(),
            "run_id": remote_paths.job_folder,
            "cluster_login": settings.cluster_login,
            "rsync_login": settings.rsync_login or settings.cluster_login,
            "ssh_config_file": settings.ssh_config_file,
            "ssh_options": settings.ssh_options,
            "job_folder": remote_paths.job_folder,
            "remote_workdir": remote_paths.workdir,
            "remote_logdir": remote_paths.logdir,
            "remote_slurm_output_dir": remote_paths.slurm_output_dir,
            "remote_slurm_dashboard_log_archive_dir": settings.remote_slurm_dashboard_log_archive_dir,
            "remote_slurm_dashboard_log_view_dir": settings.remote_slurm_dashboard_log_view_dir,
            "runtime_mode": settings.runtime_mode,
            "venv_python_executable": settings.venv_python_executable,
            "singularity_image_path": settings.singularity_image_path,
            "artifact_paths": settings.artifact_paths,
            "sync_symlinks": settings.sync_symlinks,
            "workspace_mode": settings.workspace_mode,
            "workspace_mutable": settings.workspace_mode == "fixed",
            "jobs": [],
        }
    )
    records = {record["job_name"]: record for record in payload["jobs"]}
    for record in job_records:
        previous = records.get(record["job_name"], {})
        if previous.get("job_id") and previous.get("job_id") not in {
            "unknown",
            "dry-run",
        }:
            if record.get("job_id") != previous["job_id"]:
                raise ValueError(
                    f"Refusing to replace acknowledged job {record['job_name']}"
                )
        records[record["job_name"]] = {**previous, **record}
    payload["jobs"] = list(records.values())
    atomic_write_json(output_path, payload)
    if selected_run_dir.parent.name == "slurm_output":
        atomic_write_json(selected_run_dir.parent / "latest_jobs.json", payload)
    return output_path


def render_runtime_command(job: JobSpec, settings: LauncherSettings) -> str:
    base_command = job.render_command()
    if settings.runtime_mode != "singularity":
        return base_command
    if not settings.singularity_image_path:
        raise SystemExit(
            "ERROR: SINGULARITY_IMAGE_PATH missing while RUNTIME_MODE='singularity'."
        )
    return shlex.join(
        [
            "singularity",
            "exec",
            *settings.singularity_exec_flags,
            settings.singularity_image_path,
            "bash",
            "-euo",
            "pipefail",
            "-c",
            base_command,
        ]
    )


def create_job_folder_name(
    prefix: str, repo_root: Path, *, source_state: SourceState | None = None
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    source_state = source_state or inspect_source_state(repo_root)
    git_hash = source_state.git_short_commit or "nogit"
    suffix = "_dirty" if source_state.git_dirty else ""
    return f"{prefix}_{timestamp}_{git_hash}{suffix}_{uuid4().hex[:8]}"


@dataclass(frozen=True)
class SourceState:
    git_available: bool
    git_commit: str | None
    git_short_commit: str | None
    git_branch: str | None
    git_dirty: bool
    git_status_porcelain: str
    git_diff_stat: str
    untracked_files: list[str]
    captured_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


def _git_output(repo_root: Path, args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def inspect_source_state(repo_root: Path) -> SourceState:
    commit = _git_output(repo_root, ["rev-parse", "HEAD"])
    short_commit = _git_output(repo_root, ["rev-parse", "--short", "HEAD"])
    branch = _git_output(repo_root, ["branch", "--show-current"])
    status = _git_output(
        repo_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )
    diff_stat = _git_output(repo_root, ["diff", "--stat"])
    untracked = _git_output(repo_root, ["ls-files", "--others", "--exclude-standard"])
    git_available = commit is not None and status is not None
    status_text = status or ""
    return SourceState(
        git_available=git_available,
        git_commit=commit,
        git_short_commit=short_commit,
        git_branch=branch or None,
        git_dirty=bool(status_text.strip()) if git_available else False,
        git_status_porcelain=status_text,
        git_diff_stat=diff_stat or "",
        untracked_files=untracked.splitlines() if untracked else [],
    )


def enforce_clean_git(
    settings: LauncherSettings,
    *,
    require_clean_git: bool = False,
    source_state: SourceState | None = None,
) -> None:
    if not (settings.require_clean_git or require_clean_git):
        return
    source_state = source_state or inspect_source_state(settings.project_root)
    if not source_state.git_available:
        raise SystemExit(
            "ERROR: Git state is unavailable and clean git state is required."
        )
    if not source_state.git_dirty:
        return
    dirty_files = source_state.git_status_porcelain.strip()
    raise SystemExit(
        "ERROR: Git working tree is dirty and clean git state is required.\n"
        "Commit, stash, or rerun without --require-clean-git.\n"
        f"Dirty files:\n{dirty_files}"
    )


def build_source_metadata(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    source_state: SourceState,
) -> dict[str, Any]:
    return {
        "created_at": source_state.captured_at,
        "job_folder": remote_paths.job_folder,
        "remote_workdir": remote_paths.workdir,
        "local_project_root": str(settings.project_root),
        "workspace_mode": settings.workspace_mode,
        "project_prefix": settings.project_prefix,
        "git": {
            "available": source_state.git_available,
            "commit": source_state.git_commit,
            "short_commit": source_state.git_short_commit,
            "branch": source_state.git_branch,
            "dirty": source_state.git_dirty,
            "status_porcelain": source_state.git_status_porcelain,
            "diff_stat": source_state.git_diff_stat,
            "untracked_files": source_state.untracked_files,
        },
    }


def format_source_metadata_command(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    source_state: SourceState,
) -> str:
    remote_dir = f"{remote_paths.workdir.rstrip('/')}/.slurm_run"
    remote_path = f"{remote_dir}/source.json"
    metadata = json.dumps(
        build_source_metadata(settings, remote_paths, source_state),
        indent=2,
    )
    script = "\n".join(
        [
            f"mkdir -p {shlex.quote(remote_dir)}",
            f"cat > {shlex.quote(remote_path)} <<'SOURCE_METADATA_JSON'",
            metadata,
            "SOURCE_METADATA_JSON",
        ]
    )
    return transport.format_ssh_script_command(
        settings.cluster_login,
        script,
        ssh_config_file=settings.ssh_config_file,
        ssh_options=settings.ssh_options,
    )


def write_remote_source_metadata(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    source_state: SourceState,
) -> list[str]:
    metadata = build_source_metadata(settings, remote_paths, source_state)
    remote_dir = f"{remote_paths.workdir.rstrip('/')}/.slurm_run"
    remote_path = f"{remote_dir}/source.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
        handle.flush()
        command = [
            "rsync",
            "-az",
            "-e",
            transport.build_rsync_ssh_command(settings.ssh_config_file, settings.ssh_options),
            handle.name,
            f"{settings.rsync_login or settings.cluster_login}:{remote_path}",
        ]
        ensure_remote_directories(settings, [remote_dir], dry_run=False, quiet=True)
        subprocess.run(command, check=True)
    return [shlex.join(command)]


def resolve_remote_paths(
    settings: LauncherSettings,
    *,
    job_folder: str | None = None,
    source_state: SourceState | None = None,
) -> RemotePaths:
    effective_job_folder = job_folder or create_job_folder_name(
        settings.project_prefix, settings.project_root, source_state=source_state
    )
    remote_log_base = settings.remote_log_base_path.rstrip("/")
    if settings.workspace_mode == "fixed":
        if not settings.remote_workspace_dir:
            raise SystemExit(
                "ERROR: REMOTE_WORKSPACE_DIR is required when WORKSPACE_MODE='fixed'."
            )
        workdir = settings.remote_workspace_dir.rstrip("/")
    else:
        if not settings.remote_workspace_base:
            raise SystemExit(
                "ERROR: REMOTE_WORKSPACE_BASE is required when WORKSPACE_MODE='per-run'."
            )
        remote_base = settings.remote_workspace_base.rstrip("/")
        workdir = f"{remote_base}/{effective_job_folder}"
    logdir = f"{remote_log_base}/{effective_job_folder}"
    return RemotePaths(
        job_folder=effective_job_folder,
        workdir=workdir,
        logdir=logdir,
        slurm_output_dir=f"{logdir}/slurm_output",
    )
