"""Remote preflight checks for tracked jobs."""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import LauncherSettings, RemotePaths, build_ssh_command
from .validators import ValidationIssue, run_validator

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
    validator_issues: list[ValidationIssue] = field(default_factory=list)


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
    """Build a bash script that checks requirements on the remote workspace.

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
        lines.append(f'  if [ -e {quoted} ]; then')
        lines.append(f'    echo "CHECK_OK|{req}|exists"')
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
    validators: list[str],
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> PreflightResult:
    """Run remote preflight checks for one job."""
    checks: list[PreflightCheck] = []
    if not requirements and not validators:
        return PreflightResult(job_name=job_name, ok=True, checks=checks)

    if requirements:
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

    validator_issues: list[ValidationIssue] = []
    for validator_name in validators:
        # Resolve validator target relative to remote workdir.
        target = validator_name
        if "=" in validator_name:
            name, _, target = validator_name.partition("=")
        else:
            name = target
            target = remote_paths.workdir
        # Local validators cannot inspect remote paths directly, so we only run
        # them when the target is a local path that corresponds to the staged workdir.
        # For remote-only targets, we issue an informational check.
        local_target = Path(target)
        if local_target.is_absolute() and local_target.exists():
            issues = run_validator(name, target)
        elif not local_target.is_absolute():
            candidate = settings.project_root / target
            if candidate.exists():
                issues = run_validator(name, str(candidate))
            else:
                issues = [
                    ValidationIssue(
                        name,
                        target,
                        "Validator skipped: local path not staged yet. Run preflight after staging.",
                    )
                ]
        else:
            issues = [
                ValidationIssue(
                    name,
                    target,
                    "Validator skipped: target is remote-only. Run check on the cluster.",
                )
            ]
        validator_issues.extend(issues)

    all_ok = all(check.ok for check in checks) and not validator_issues
    return PreflightResult(
        job_name=job_name,
        ok=all_ok,
        checks=checks,
        validator_issues=validator_issues,
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

    results: list[PreflightResult] = []
    for job in jobs:
        requirements = list(getattr(job, "requires", []) or [])
        validators = list(getattr(job, "validators", []) or [])
        if not requirements and not validators:
            continue
        if dry_run:
            script = build_remote_check_script(remote_paths.workdir, requirements)
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
            validators,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        results.append(result)

    if dry_run:
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
                        "validator_issues": [
                            {
                                "validator": issue.validator,
                                "path": issue.path,
                                "message": issue.message,
                            }
                            for issue in result.validator_issues
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
        console.print("No jobs with requirements or validators.", style="yellow")
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
        for issue in result.validator_issues:
            console.print(
                f"  [red]VALIDATOR[/red] {issue.validator}: {issue.path} - {issue.message}"
            )

    if any(not result.ok for result in results):
        err_console.print("\nPreflight failed. Fix issues before submitting.", style="bold red")
        return 1

    console.print("\nPreflight passed.", style="green")
    return 0
