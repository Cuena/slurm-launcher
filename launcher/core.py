"""Core logic for the remote SLURM launcher."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.syntax import Syntax

DEFAULT_RSYNC_EXCLUDES = [
    ".git/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.egg-info/",
    ".mypy_cache/",
    ".pytest_cache/",
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
    command: str
    env: dict[str, Any] = field(default_factory=dict)
    sbatch: dict[str, Any] = field(default_factory=dict)
    setup: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.command = str(self.command).strip()
        if not self.command:
            raise ValueError(f"Job '{self.name}' must define a non-empty 'command'.")
        self.setup = [str(cmd) for cmd in self.setup]

    def render_command(self) -> str:
        return self.command


@dataclass(frozen=True)
class LauncherSettings:
    cluster_login: str
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
    verbose: bool


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


def ssh_script(cluster_login: str, script: str, *, dry_run: bool) -> tuple[str, str]:
    script = script.rstrip() + "\n"
    if dry_run:
        console.print(f"[yellow]dry-run[/yellow] ssh {cluster_login} <<'EOF'")
        console.print(Syntax(script.rstrip(), "bash"))
        console.print("EOF")
        return "", ""
    try:
        result = subprocess.run(
            ["ssh", cluster_login, "bash", "-s"],
            input=script,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"SSH command failed with exit code {exc.returncode}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout, result.stderr


def test_ssh_connection(cluster_login: str, dry_run: bool) -> None:
    if dry_run:
        console.print(
            f"[yellow]dry-run[/yellow] skip SSH connectivity check for {cluster_login}"
        )
        return
    stdout, _ = ssh_script(cluster_login, "echo SSH_OK", dry_run=dry_run)
    if "SSH_OK" not in stdout:
        raise SystemExit("ERROR: SSH test failed. Check your SSH setup.")
    console.print("SSH connection OK", style="green")


def ensure_remote_directories(
    settings: LauncherSettings, paths: list[str], dry_run: bool
) -> None:
    unique_paths = sorted(set(paths))
    if not unique_paths:
        return

    mkdir_cmd = f"mkdir -p {' '.join(shlex.quote(p) for p in unique_paths)}"
    if dry_run:
        console.print(
            f"[yellow]dry-run[/yellow] ssh {settings.cluster_login} '{mkdir_cmd}'"
        )
        return
    ssh_script(settings.cluster_login, mkdir_cmd, dry_run=False)


def sync_project(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    dry_run: bool,
    *,
    include_logging_dirs: bool = True,
) -> None:
    remote_directories = [remote_paths.workdir]
    if include_logging_dirs:
        remote_directories.extend(
            [remote_paths.logdir, remote_paths.slurm_output_dir]
        )
        if settings.remote_slurm_dashboard_log_archive_dir:
            remote_directories.append(settings.remote_slurm_dashboard_log_archive_dir)
        if settings.remote_slurm_dashboard_log_view_dir:
            remote_directories.append(settings.remote_slurm_dashboard_log_view_dir)

    ensure_remote_directories(
        settings,
        remote_directories,
        dry_run,
    )

    excludes = DEFAULT_RSYNC_EXCLUDES + settings.extra_rsync_excludes
    destination = f"{settings.cluster_login}:{remote_paths.workdir}/"
    cmd = ["rsync", "-az", "--info=progress2"]
    if dry_run:
        cmd.append("--dry-run")
    for pattern in excludes:
        cmd.extend(["--exclude", pattern])
    cmd.extend(settings.extra_rsync_args)
    cmd.extend([f"{settings.project_root}/", destination])

    console.print(f"Syncing project to {destination}")
    if dry_run:
        console.print("dry-run rsync command:", style="yellow")
        console.print(shlex.join(cmd), style="dim")
        console.print("dry-run skipping rsync execution", style="yellow")
        return
    subprocess.run(cmd, check=True)
    console.print("Sync complete", style="green")


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
                f"test -f {shlex.quote(str(activate))} || (echo 'ERROR: venv activate script not found: {shlex.quote(str(activate))}' && exit 1)",
                f"source {shlex.quote(str(activate))}",
            ]
        )
    lines.extend(job.setup)
    lines.append(render_runtime_command(job, settings))
    return "\n".join(lines).rstrip() + "\n"


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


def build_sbatch_script(job_script: str, sbatch_options: dict[str, Any]) -> str:
    script_lines = job_script.splitlines()
    if script_lines and script_lines[0].startswith("#!"):
        shebang = script_lines[0]
        body = script_lines[1:]
    else:
        shebang = "#!/bin/bash"
        body = script_lines

    lines: list[str] = [shebang]
    lines.extend(render_sbatch_directives(sbatch_options))
    lines.extend(body)
    return "\n".join(lines).rstrip() + "\n"


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def format_sbatch_options(
    job: JobSpec, settings: LauncherSettings, remote_paths: RemotePaths
) -> dict[str, Any]:
    options = {**settings.default_sbatch, **job.sbatch}
    if "chdir" in options or "ch_dir" in options:
        raise SystemExit(
            "ERROR: sbatch 'chdir' is not supported. "
            "The launcher always runs from its managed remote workdir."
        )

    # Ensure ntasks is compatible with nodes x ntasks-per-node when both are set.
    nodes = _parse_int(options.get("nodes"))
    ntasks_per_node = _parse_int(
        options.get("ntasks-per-node", options.get("ntasks_per_node"))
    )
    ntasks = _parse_int(options.get("ntasks"))
    if nodes and ntasks_per_node:
        expected = nodes * ntasks_per_node
        if ntasks is None or ntasks < expected:
            options["ntasks"] = expected

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
    for line in output.splitlines():
        line = line.strip()
        if "Submitted batch job" in line:
            parts = line.split()
            if parts:
                return parts[-1]
    return output.strip() or "unknown"


def _sanitize_log_view_component(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.+-]+", "_", value.strip())
    cleaned = normalized.strip("._-")
    return cleaned or fallback


def create_log_view_symlinks(
    settings: LauncherSettings,
    job: JobSpec,
    submission: SubmissionResult,
) -> None:
    archive_dir = settings.remote_slurm_dashboard_log_archive_dir
    view_dir = settings.remote_slurm_dashboard_log_view_dir
    if not archive_dir or not view_dir:
        return
    if submission.job_id in {"", "unknown", "dry-run"}:
        return

    project_label = _sanitize_log_view_component(settings.project_prefix, "project")
    date_label = datetime.now().strftime("%Y-%m-%d")
    job_label = _sanitize_log_view_component(job.name, "job")
    archive_root = archive_dir.rstrip("/")
    view_root = view_dir.rstrip("/")
    view_subdir = f"{view_root}/{project_label}/{date_label}"
    src_stdout = f"{archive_root}/{submission.job_id}.out"
    src_stderr = f"{archive_root}/{submission.job_id}.err"
    dst_stdout = f"{view_subdir}/{job_label}-{submission.job_id}.out"
    dst_stderr = f"{view_subdir}/{job_label}-{submission.job_id}.err"
    link_script = "\n".join(
        [
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(view_subdir)}",
            f"ln -sfn {shlex.quote(src_stdout)} {shlex.quote(dst_stdout)}",
            f"ln -sfn {shlex.quote(src_stderr)} {shlex.quote(dst_stderr)}",
        ]
    )
    try:
        ssh_script(settings.cluster_login, link_script, dry_run=False)
    except RuntimeError as exc:
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
) -> SubmissionResult:
    sbatch_options = format_sbatch_options(job, settings, remote_paths)
    job_script = build_job_script(job, settings, remote_paths)
    sbatch_script = build_sbatch_script(job_script, sbatch_options)
    remote_sbatch_path = f"{remote_paths.logdir}/{job.name}.sbatch"
    sbatch_cmd = " ".join(["sbatch", shlex.quote(remote_sbatch_path)])

    if settings.verbose or dry_run:
        console.print()
        console.rule(f"[cyan]{job.name} script")
        console.print(Syntax(sbatch_script.rstrip(), "bash"))
        console.rule(f"[cyan]{job.name} sbatch")
        console.print(Syntax(sbatch_cmd, "bash"))

    if not dry_run:
        write_local_submission_artifacts(
            settings,
            remote_paths,
            job,
            job_script=job_script,
            sbatch_script=sbatch_script,
            sbatch_command=sbatch_cmd,
        )

    if dry_run:
        console.print(
            f"dry-run would upload script to {remote_sbatch_path} "
            f"and run sbatch via ssh {settings.cluster_login}",
            style="yellow",
        )
        return SubmissionResult(
            job_id="dry-run",
            sbatch_command=sbatch_cmd,
            sbatch_options=sbatch_options,
            remote_sbatch_path=remote_sbatch_path,
        )

    mkdir_targets = [remote_paths.slurm_output_dir]
    if settings.remote_slurm_dashboard_log_archive_dir:
        mkdir_targets.append(settings.remote_slurm_dashboard_log_archive_dir)
    if settings.remote_slurm_dashboard_log_view_dir:
        mkdir_targets.append(settings.remote_slurm_dashboard_log_view_dir)
    mkdir_command = (
        f"mkdir -p {' '.join(shlex.quote(path) for path in sorted(set(mkdir_targets)))}"
    )

    script_lines = [
        "set -euo pipefail",
        mkdir_command,
        f"cat <<'SBATCH_SCRIPT' > {shlex.quote(remote_sbatch_path)}",
        sbatch_script.rstrip("\n"),
        "SBATCH_SCRIPT",
        f"chmod +x {shlex.quote(remote_sbatch_path)}",
        sbatch_cmd,
    ]
    stdout, _ = ssh_script(
        settings.cluster_login, "\n".join(script_lines), dry_run=False
    )
    job_id = parse_job_id(stdout)
    console.print(f"Submitted {job.name} -> {job_id}", style="bold green")
    submission = SubmissionResult(
        job_id=job_id,
        sbatch_command=sbatch_cmd,
        sbatch_options=sbatch_options,
        remote_sbatch_path=remote_sbatch_path,
    )
    create_log_view_symlinks(settings, job, submission)
    return submission


def resolve_log_path(template: Any, job_id: str) -> str | None:
    if not template:
        return None
    path = str(template)
    if job_id and job_id != "dry-run":
        path = path.replace("%j", job_id).replace("%J", job_id)
    return path


def build_job_record(job: JobSpec, submission: SubmissionResult) -> dict[str, Any]:
    stdout_path = resolve_log_path(
        submission.sbatch_options.get("output"), submission.job_id
    )
    stderr_path = resolve_log_path(
        submission.sbatch_options.get("error"), submission.job_id
    )
    return {
        "job_name": job.name,
        "job_id": submission.job_id,
        "stdout": stdout_path,
        "stderr": stderr_path,
        "sbatch_command": submission.sbatch_command,
        "remote_sbatch": submission.remote_sbatch_path,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_local_submission_artifacts(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job: JobSpec,
    *,
    job_script: str,
    sbatch_script: str,
    sbatch_command: str,
) -> Path:
    artifacts_dir = settings.project_root / "slurm_output" / remote_paths.job_folder
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"{job.name}.sh").write_text(job_script, encoding="utf-8")
    (artifacts_dir / f"{job.name}.sbatch").write_text(sbatch_script, encoding="utf-8")
    (artifacts_dir / f"{job.name}.sbatch.cmd").write_text(
        sbatch_command.rstrip() + "\n", encoding="utf-8"
    )
    return artifacts_dir


def write_job_tracking_file(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job_records: list[dict[str, Any]],
) -> Path:
    root_tracking_dir = settings.project_root / "slurm_output"
    tracking_dir = root_tracking_dir / remote_paths.job_folder
    tracking_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cluster_login": settings.cluster_login,
        "job_folder": remote_paths.job_folder,
        "remote_workdir": remote_paths.workdir,
        "remote_logdir": remote_paths.logdir,
        "remote_slurm_output_dir": remote_paths.slurm_output_dir,
        "remote_slurm_dashboard_log_archive_dir": settings.remote_slurm_dashboard_log_archive_dir,
        "remote_slurm_dashboard_log_view_dir": settings.remote_slurm_dashboard_log_view_dir,
        "local_project_root": str(settings.project_root),
        "jobs": job_records,
    }
    output_path = tracking_dir / "jobs.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    latest_tracking_path = root_tracking_dir / "latest_jobs.json"
    latest_tracking_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    latest_run_path = root_tracking_dir / "latest_run.txt"
    latest_run_path.write_text(remote_paths.job_folder + "\n", encoding="utf-8")
    return output_path


def render_runtime_command(job: JobSpec, settings: LauncherSettings) -> str:
    base_command = job.render_command()
    if settings.runtime_mode != "singularity":
        return base_command
    if not settings.singularity_image_path:
        raise SystemExit(
            "ERROR: SINGULARITY_IMAGE_PATH missing while RUNTIME_MODE='singularity'."
        )
    parts = ["singularity", "exec"]
    parts.extend(shlex.quote(arg) for arg in settings.singularity_exec_flags)
    parts.append(shlex.quote(settings.singularity_image_path))
    parts.append(base_command)
    return " ".join(parts)


def create_job_folder_name(prefix: str, repo_root: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_hash = query_git_hash(repo_root)
    return f"{prefix}_{timestamp}_{git_hash}"


def query_git_hash(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo_root,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def resolve_remote_paths(settings: LauncherSettings) -> RemotePaths:
    return resolve_remote_paths_for_job_folder(settings, job_folder=None)


def resolve_remote_paths_for_job_folder(
    settings: LauncherSettings,
    job_folder: str | None,
) -> RemotePaths:
    effective_job_folder = job_folder or create_job_folder_name(
        settings.project_prefix, settings.project_root
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
