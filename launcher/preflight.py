"""Remote preflight checks for tracked jobs."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import LauncherSettings, RemotePaths, build_ssh_command

console = Console()
err_console = Console(stderr=True)


@dataclass(frozen=True)
class PreflightCheck:
    """One preflight check result."""

    kind: str
    path: str
    remote_path: str
    ok: bool
    message: str


@dataclass(frozen=True)
class PreflightResult:
    """Result of preflight checks for a job."""

    job_name: str
    ok: bool
    checks: list[PreflightCheck] = field(default_factory=list)


def _run_ssh_capture(
    cluster_login: str,
    script: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            *build_ssh_command(
                cluster_login,
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            ),
            "bash",
            "-s",
        ],
        input=script.rstrip() + "\n",
        capture_output=True,
        text=True,
        check=False,
    )


def build_remote_check_script(
    remote_workdir: str,
    requirements: list[str],
) -> str:
    """Build a bash script that checks generic requirements on the remote workspace.

    For each requirement:
      - If it contains a glob character (*, ?, [), count matches and fail on zero.
      - Otherwise check the path exists and is not a broken symlink.
    """
    lines = ["set -euo pipefail", f'cd {shlex.quote(remote_workdir)}']
    lines.append('failed=0')
    for req in requirements:
        quoted = shlex.quote(req)
        lines.append(f'echo "CHECK_START|{req}"')
        # Distinguish globs from plain paths.
        lines.append(
            f'if [[ {quoted} == *"*"* || {quoted} == *"?"* || {quoted} == *"["* ]]; then'
        )
        lines.append(f'  count=$(compgen -G {quoted} 2>/dev/null | wc -l)')
        lines.append('  if [ "$count" -eq 0 ]; then')
        lines.append(f'    echo "CHECK_FAIL|{req}|glob matched 0 files"')
        lines.append('    failed=1')
        lines.append('  else')
        lines.append(f'    echo "CHECK_OK|{req}|matched $count"')
        lines.append('  fi')
        lines.append('else')
        # Detect broken symlinks: -L without -e means a symlink pointing to a missing target.
        lines.append(f'  if [ -e {quoted} ]; then')
        lines.append(f'    echo "CHECK_OK|{req}|exists"')
        lines.append(f'  elif [ -L {quoted} ]; then')
        lines.append(f'    echo "CHECK_FAIL|{req}|broken symlink"')
        lines.append('    failed=1')
        lines.append('  else')
        lines.append(f'    echo "CHECK_FAIL|{req}|missing"')
        lines.append('    failed=1')
        lines.append('  fi')
        lines.append('fi')
    lines.append('exit $failed')
    return "\n".join(lines)


def _parse_check_output(output: str) -> dict[str, tuple[bool, str]]:
    """Parse CHECK_OK/CHECK_FAIL lines into {path: (ok, message)}."""
    results: dict[str, tuple[bool, str]] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("CHECK_"):
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        status, path, message = parts
        results[path] = (status == "CHECK_OK", message)
    return results


def run_preflight_for_job(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    job_name: str,
    requirements: list[str],
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> PreflightResult:
    """Run remote preflight checks for one job."""
    checks: list[PreflightCheck] = []
    if not requirements:
        return PreflightResult(job_name=job_name, ok=True, checks=checks)

    script = build_remote_check_script(remote_paths.workdir, requirements)
    result = _run_ssh_capture(
        settings.cluster_login,
        script,
        ssh_config_file=ssh_config_file or settings.ssh_config_file,
        ssh_options=ssh_options or settings.ssh_options,
    )
    parsed = _parse_check_output(result.stdout)
    for req in requirements:
        ok, message = parsed.get(req, (False, "check did not return"))
        checks.append(
            PreflightCheck(
                kind="require",
                path=req,
                remote_path=f"{remote_paths.workdir}/{req.lstrip('/')}",
                ok=ok,
                message=message,
            )
        )

    return PreflightResult(
        job_name=job_name,
        ok=all(check.ok for check in checks),
        checks=checks,
    )


def run_preflight(
    settings: LauncherSettings,
    remote_paths: RemotePaths,
    jobs: list[Any],
    *,
    selected_jobs: list[str] | None = None,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    json_output: bool = False,
    dry_run: bool = False,
) -> int:
    """Run preflight checks for selected jobs."""
    if selected_jobs:
        wanted = set(selected_jobs)
        jobs = [job for job in jobs if getattr(job, "name", "") in wanted]

    dry_run_entries: list[dict[str, Any]] = []
    results: list[PreflightResult] = []
    for job in jobs:
        requirements = list(getattr(job, "requires", []) or [])
        if not requirements:
            continue
        if dry_run:
            script = build_remote_check_script(remote_paths.workdir, requirements)
            if json_output:
                dry_run_entries.append(
                    {
                        "job_name": job.name,
                        "requirements": requirements,
                        "script": script,
                    }
                )
            else:
                console.print(
                    f"[yellow]dry-run[/yellow] preflight for {job.name}", style="dim"
                )
                console.print(script, style="dim")
            continue
        result = run_preflight_for_job(
            settings,
            remote_paths,
            job.name,
            requirements,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        results.append(result)

    if dry_run:
        if json_output:
            console.print_json(
                data={
                    "ok": True,
                    "dry_run": True,
                    "remote_workdir": remote_paths.workdir,
                    "jobs": dry_run_entries,
                }
            )
        return 0

    if json_output:
        console.print_json(
            data={
                "ok": all(result.ok for result in results),
                "remote_workdir": remote_paths.workdir,
                "jobs": [
                    {
                        "job_name": result.job_name,
                        "ok": result.ok,
                        "checks": [
                            {
                                "kind": check.kind,
                                "path": check.path,
                                "remote_path": check.remote_path,
                                "ok": check.ok,
                                "message": check.message,
                            }
                            for check in result.checks
                        ],
                    }
                    for result in results
                ],
            }
        )
        return 0 if all(result.ok for result in results) else 1

    console.print(
        Panel.fit(
            f"[bold]Remote workdir:[/bold] {remote_paths.workdir}",
            title="Preflight",
            border_style="cyan",
        )
    )
    if not results:
        console.print("No jobs with requirements.", style="yellow")
        return 0

    for result in results:
        console.print()
        console.print(
            f"[bold]{result.job_name}[/bold]", style="green" if result.ok else "red"
        )
        if result.checks:
            table = Table()
            table.add_column("Kind")
            table.add_column("Path")
            table.add_column("Status")
            table.add_column("Message")
            for check in result.checks:
                table.add_row(
                    check.kind,
                    check.path,
                    "OK" if check.ok else "FAIL",
                    check.message,
                    style=None if check.ok else "red",
                )
            console.print(table)

    if any(not result.ok for result in results):
        err_console.print("\nPreflight failed. Fix issues before submitting.", style="bold red")
        return 1

    console.print("\nPreflight passed.", style="green")
    return 0
