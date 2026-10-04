"""Remote preflight checks for tracked jobs."""

from __future__ import annotations

import shlex
from posixpath import join as posix_join
from dataclasses import asdict, dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .core import LauncherSettings, RemotePaths
from .transport import run_ssh_capture

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
    transport_returncode: int | None = None
    stderr: str = ""


def build_remote_check_script(
    remote_workdir: str,
    requirements: list[str],
) -> str:
    """Check literal paths and count only glob matches with existing targets."""
    lines = ["set -euo pipefail", f"cd {shlex.quote(remote_workdir)}", "failed=0"]
    for index, req in enumerate(requirements):
        quoted = shlex.quote(req)
        if any(character in req for character in "*?["):
            lines.extend(
                [
                    "count=0",
                    "while IFS= read -r match; do",
                    '  if test -e "$match"; then count=$((count + 1)); fi',
                    f"done < <(compgen -G {quoted} || true)",
                    'if [ "$count" -gt 0 ]; then',
                    f'  printf \'{index}|true|matched %s\\n\' "$count"',
                    "else",
                    f"  printf '{index}|false|glob matched 0 existing targets\\n'",
                    "  failed=1",
                    "fi",
                ]
            )
        else:
            lines.extend(
                [
                    f"if test -e {quoted}; then",
                    f"  printf '{index}|true|exists\\n'",
                    f"elif test -L {quoted}; then",
                    f"  printf '{index}|false|broken symlink\\n'",
                    "  failed=1",
                    "else",
                    f"  printf '{index}|false|missing\\n'",
                    "  failed=1",
                    "fi",
                ]
            )
    lines.append("exit $failed")
    return "\n".join(lines)


def _parse_check_output(output: str) -> dict[int, tuple[bool, str]]:
    """Read indexed records without exposing literal paths to the protocol."""
    results: dict[int, tuple[bool, str]] = {}
    for line in output.splitlines():
        parts = line.split("|", 2)
        if len(parts) != 3 or not parts[0].isdigit() or parts[1] not in {"true", "false"}:
            continue
        index, status, message = parts
        results[int(index)] = (status == "true", message)
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
    result = run_ssh_capture(
        settings.cluster_login,
        script,
        ssh_config_file=(
            settings.ssh_config_file if ssh_config_file is None else ssh_config_file
        ),
        ssh_options=settings.ssh_options if ssh_options is None else ssh_options,
    )
    parsed = _parse_check_output(result.stdout)
    for index, req in enumerate(requirements):
        ok, message = parsed.get(index, (False, "check did not return"))
        remote_path = (
            req if req.startswith("/") else posix_join(remote_paths.workdir, req)
        )
        checks.append(
            PreflightCheck(
                kind="require",
                path=req,
                remote_path=remote_path,
                ok=ok,
                message=message,
            )
        )

    return PreflightResult(
        job_name=job_name,
        ok=result.returncode == 0 and all(check.ok for check in checks),
        checks=checks,
        transport_returncode=result.returncode,
        stderr=result.stderr.strip()[:2000],
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

    ok = bool(jobs)
    warnings = [] if jobs else ["No jobs were selected for preflight."]
    entries: list[dict[str, Any]] = []
    results: list[PreflightResult] = []
    checks_planned = checks_run = 0
    for job in jobs:
        requirements = list(getattr(job, "requires", []) or [])
        checks_planned += len(requirements)
        if not requirements:
            ok = False
            warning = (
                f"Job '{job.name}' has no 'requires'; preflight cannot validate its "
                "remote prerequisites."
            )
            warnings.append(warning)
            if json_output:
                entries.append(
                    {
                        "job_name": job.name,
                        "ok": False,
                        "status": "not-configured",
                        "requirements" if dry_run else "checks": [],
                        "message": warning,
                    }
                )
            continue
        if dry_run:
            script = build_remote_check_script(remote_paths.workdir, requirements)
            if json_output:
                entries.append(
                    {
                        "job_name": job.name,
                        "ok": True,
                        "status": "planned",
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
        ok = ok and result.ok
        checks_run += len(result.checks)
        if json_output:
            entries.append(
                {**asdict(result), "status": "passed" if result.ok else "failed"}
            )
        else:
            results.append(result)

    if json_output:
        console.print_json(
            data={
                "ok": ok,
                "dry_run": dry_run,
                "remote_workdir": remote_paths.workdir,
                "checks_planned": checks_planned,
                "checks_run": checks_run,
                "warnings": warnings,
                "jobs": entries,
            }
        )
    elif dry_run:
        for warning in warnings:
            err_console.print(warning, style="bold red")
    else:
        console.print(
            Panel.fit(
                f"[bold]Remote workdir:[/bold] {remote_paths.workdir}",
                title="Preflight",
                border_style="cyan",
            )
        )
        for result in results:
            console.print()
            console.print(
                f"[bold]{result.job_name}[/bold]", style="green" if result.ok else "red"
            )
            if result.transport_returncode:
                err_console.print(
                    f"Preflight process exited with code {result.transport_returncode}: "
                    f"{result.stderr or 'no stderr diagnostic'}",
                    style="bold red",
                )
            if result.checks:
                table = Table()
                for column in ("Kind", "Path", "Status", "Message"):
                    table.add_column(column)
                for check in result.checks:
                    table.add_row(
                        check.kind,
                        check.path,
                        "OK" if check.ok else "FAIL",
                        check.message,
                        style=None if check.ok else "red",
                    )
                console.print(table)
        for warning in warnings:
            err_console.print(warning, style="bold red")
        if ok:
            console.print("\nPreflight passed.", style="green")
        else:
            err_console.print(
                "\nPreflight failed. Fix issues before submitting.", style="bold red"
            )
    return 0 if ok else 1
