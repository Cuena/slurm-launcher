"""Run orchestration and durable, config-free submission of frozen plans."""

from __future__ import annotations

import fcntl
import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from .config_utils import (
    build_settings,
    configured_run_only,
    fail_duplicate_jobs,
    load_config,
    prepare_jobs,
    resolve_config_path,
    select_jobs,
    validate_predefined_sbatch_jobs,
)
from .core import (
    JobSpec,
    LauncherSettings,
    RemotePaths,
    SubmissionResult,
    build_job_record,
    build_job_script,
    build_launcher_metadata,
    build_sbatch_script,
    build_source_metadata,
    console,
    enforce_clean_git,
    format_sbatch_options,
    inspect_source_state,
    resolve_local_project_path,
    resolve_remote_paths,
    submit_job,
    sync_project,
    test_ssh_connection,
    write_job_tracking_file,
)
from .preflight import run_preflight
from .tracking import atomic_write, atomic_write_json, resolve_run_directory

err_console = Console(stderr=True)


def _emit(args: Any, payload: dict[str, Any]) -> int:
    payload["job_ids"] = [
        str(record["job_id"])
        for record in payload.get("jobs", [])
        if str(record.get("job_id", "")).isdigit()
    ]
    if getattr(args, "json", False):
        console.print_json(data=payload)
    elif not payload["ok"]:
        err_console.print(str(payload["error"]), style="bold red")
        if payload.get("tracking_file"):
            err_console.print(f"Recovery state: {payload['tracking_file']}")
        if payload["job_ids"]:
            err_console.print(f"Acknowledged job IDs: {', '.join(payload['job_ids'])}")
    else:
        console.print(f"Run: {payload['run_id']}")
        console.print(f"Remote workdir: {payload['remote_workdir']}")
        if payload.get("workspace_mutable"):
            console.print(
                "Fixed workspace: shared mutable code; later staging can replace it.",
                style="yellow",
            )
        if payload.get("tracking_file"):
            console.print(f"Tracking: {payload['tracking_file']}")
        for record in payload.get("jobs", []):
            console.print(
                f"{record['job_name']}: {record.get('state', 'planned')} {record.get('job_id', '')}"
            )
        if payload.get("dry_run"):
            for command in payload.get("commands", []):
                console.print(command, soft_wrap=True)
    return 0 if payload["ok"] else 1


def _context(
    payload: dict[str, Any],
    settings: LauncherSettings,
    paths: RemotePaths,
    plan: dict[str, Any],
) -> None:
    payload.update(
        run_id=paths.job_folder,
        job_folder=paths.job_folder,
        cluster_login=settings.cluster_login,
        rsync_login=settings.rsync_login or settings.cluster_login,
        workspace_mode=settings.workspace_mode,
        workspace_mutable=settings.workspace_mode == "fixed",
        remote_workdir=paths.workdir,
        remote_logdir=paths.logdir,
        remote_slurm_output_dir=paths.slurm_output_dir,
        selected_jobs=[entry["job"]["name"] for entry in plan["jobs"]],
        provenance=plan["provenance"],
    )


def _freeze(
    settings: LauncherSettings, paths: RemotePaths, jobs: list[JobSpec]
) -> dict[str, Any]:
    entries = []
    for index, job in enumerate(jobs):
        if not job.name or Path(job.name).name != job.name or job.name in {".", ".."}:
            raise ValueError(f"Job name must be a single path component: {job.name!r}")
        if job.sbatch_file:
            source = resolve_local_project_path(settings.project_root, job.sbatch_file)
            if source is None:
                raise ValueError("sbatch_file must stay inside LOCAL_ROOT")
            script = source.read_bytes().decode("utf-8")
            options = {}
            job_script = None
        else:
            options = format_sbatch_options(job, settings, paths)
            job_script = build_job_script(job, settings, paths)
            script = build_sbatch_script(
                job_script,
                options,
                launcher_metadata=build_launcher_metadata(job, settings),
            )
        entries.append(
            {
                "job": asdict(job),
                "script": script,
                "job_script": job_script,
                "snapshot": f"scripts/{index:04d}.sbatch",
                "sbatch_options": options,
                "sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
            }
        )
    return json.loads(
        json.dumps(
            {
                "version": 1,
                "settings": asdict(settings),
                "remote_paths": asdict(paths),
                "jobs": entries,
                "provenance": build_source_metadata(
                    settings, paths, inspect_source_state(settings.project_root)
                ),
                "workspace_mutable": settings.workspace_mode == "fixed",
            },
            default=str,
        )
    )


def _load_plan(args: Any) -> tuple[LauncherSettings, RemotePaths, dict[str, Any], Path]:
    run_dir = resolve_run_directory(getattr(args, "run", None)).resolve()
    plan_path = run_dir / "plan.json"
    if not plan_path.is_file():
        raise ValueError(
            "This run has no frozen plan. Legacy tracking is readable, but cannot be submitted; stage a new run."
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("version") != 1:
        raise ValueError("Unsupported frozen run plan.")
    if not isinstance(plan.get("jobs"), list) or not plan["jobs"]:
        raise ValueError("Frozen plan must contain at least one job.")
    data = dict(plan["settings"])
    data["project_root"] = Path(data["project_root"])
    if data.get("local_artifact_root"):
        data["local_artifact_root"] = Path(data["local_artifact_root"])
    settings = LauncherSettings(**data)
    paths = RemotePaths(**plan["remote_paths"])
    if (
        not isinstance(settings.cluster_login, str)
        or not settings.cluster_login
        or settings.cluster_login.startswith("-")
    ):
        raise ValueError("Invalid cluster login in frozen plan.")
    if (
        not isinstance(paths.job_folder, str)
        or Path(paths.job_folder).name != paths.job_folder
        or paths.job_folder in {".", ".."}
    ):
        raise ValueError("Invalid run ID in frozen plan.")
    if any(
        not isinstance(path, str) or not path.startswith("/")
        for path in (paths.workdir, paths.logdir, paths.slurm_output_dir)
    ):
        raise ValueError("Frozen remote paths must be absolute.")
    if run_dir.name != paths.job_folder:
        raise ValueError("Run directory does not match the frozen run ID.")
    # Recovery follows the selected local run, not a stale original checkout path.
    settings = replace(settings, project_root=run_dir.parent.parent)
    jobs = [JobSpec(**entry["job"]) for entry in plan["jobs"]]
    fail_duplicate_jobs(jobs)
    if getattr(args, "all_jobs", False) and getattr(args, "only", None):
        raise ValueError("--only and --all cannot be combined.")
    selected = {job.name for job in select_jobs(jobs, getattr(args, "only", None))}
    plan["jobs"] = [entry for entry in plan["jobs"] if entry["job"]["name"] in selected]
    for entry in plan["jobs"]:
        name = entry["job"]["name"]
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {"", ".", ".."}
        ):
            raise ValueError("Invalid job name in frozen plan.")
        snapshot = (run_dir / entry["snapshot"]).resolve()
        if not snapshot.is_relative_to(run_dir):
            raise ValueError("Frozen snapshot must stay inside the run directory.")
        if (
            hashlib.sha256(entry["script"].encode("utf-8")).hexdigest()
            != entry["sha256"]
        ):
            raise ValueError(f"Frozen script is corrupt: {entry['job']['name']}")
        if snapshot.read_bytes() != entry["script"].encode("utf-8"):
            raise ValueError(f"Frozen snapshot was modified: {entry['snapshot']}")
    return settings, paths, plan, run_dir


def _save_progress(
    settings: LauncherSettings, paths: RemotePaths, payload: dict[str, Any]
) -> None:
    payload["tracking_file"] = str(
        write_job_tracking_file(settings, paths, payload["jobs"])
    )


def _submit(
    settings: LauncherSettings,
    paths: RemotePaths,
    plan: dict[str, Any],
    payload: dict[str, Any],
    *,
    dry_run: bool,
    quiet: bool,
) -> None:
    for entry in plan["jobs"]:
        job = JobSpec(**entry["job"])
        previous = next(
            (record for record in payload["jobs"] if record["job_name"] == job.name),
            None,
        )
        if previous and (
            previous.get("state") in {"submitted", "submitting", "unknown"}
            or str(previous.get("job_id", "")).isdigit()
        ):
            raise ValueError(
                f"Refusing to resubmit {job.name}: {previous.get('state', 'submitted')}. Reconcile with Slurm; stage a new run to submit again."
            )
    for entry in plan["jobs"]:
        job = JobSpec(**entry["job"])
        record = next(
            (record for record in payload["jobs"] if record["job_name"] == job.name),
            None,
        )
        if record is None:
            record = {
                "job_name": job.name,
                "job_id": "",
                "artifacts": job.artifacts,
                "requires": job.requires,
            }
            payload["jobs"].append(record)
        if not dry_run:
            if record.get("state") == "failed":
                record.setdefault("attempts", []).append(
                    {key: value for key, value in record.items() if key != "attempts"}
                )
            record.update(
                state="submitting", error=None, attempted_at=datetime.now().isoformat()
            )
            _save_progress(settings, paths, payload)

        def acknowledge(submission: SubmissionResult) -> None:
            record.update(
                build_job_record(job, submission, settings), state="submitted"
            )
            _save_progress(settings, paths, payload)

        try:
            submission = submit_job(
                settings,
                paths,
                job,
                dry_run=dry_run,
                quiet=quiet,
                frozen_script=entry["script"],
                frozen_options=entry["sbatch_options"],
                on_acknowledged=acknowledge if not dry_run else None,
            )
        except BaseException as exc:
            if not dry_run and record.get("state") != "submitted":
                cause = exc.__cause__ if exc.__cause__ is not None else exc
                rejected = (
                    isinstance(cause, subprocess.CalledProcessError)
                    and 0 < cause.returncode < 255
                )
                record.update(state="failed" if rejected else "unknown", error=str(exc))
                _save_progress(settings, paths, payload)
            raise
        record.update(
            build_job_record(job, submission, settings),
            state="planned" if dry_run else "submitted",
        )
        payload["commands"].extend(submission.commands)
        if not dry_run:
            _save_progress(settings, paths, payload)


def _execute(args: Any, command: str) -> int:
    dry_run = bool(getattr(args, "dry_run", False))
    quiet = bool(getattr(args, "json", False))
    payload: dict[str, Any] = {
        "ok": True,
        "dry_run": dry_run,
        "commands": [],
        "jobs": [],
        "tracking_file": None,
    }
    run_lock = None
    try:
        if command in {"submit", "preflight"}:
            settings, paths, plan, run_dir = _load_plan(args)
            _context(payload, settings, paths, plan)
            payload.update(
                tracking_file=str(run_dir / "jobs.json"),
                plan_file=str(run_dir / "plan.json"),
            )
            tracked = json.loads((run_dir / "jobs.json").read_text())
            payload["jobs"] = tracked["jobs"]
            if tracked.get("stage_state") != "staged":
                raise ValueError(
                    "Run staging did not complete. Stage a new run before submission or preflight."
                )
            if command == "preflight":
                return run_preflight(
                    settings,
                    paths,
                    [JobSpec(**entry["job"]) for entry in plan["jobs"]],
                    ssh_config_file=settings.ssh_config_file,
                    ssh_options=settings.ssh_options,
                    json_output=quiet,
                    dry_run=dry_run,
                )
            if dry_run:
                _submit(settings, paths, plan, payload, dry_run=True, quiet=quiet)
            else:
                with (run_dir / ".submission.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    payload["jobs"] = json.loads((run_dir / "jobs.json").read_text())[
                        "jobs"
                    ]
                    _submit(settings, paths, plan, payload, dry_run=False, quiet=quiet)
        else:
            config_path = resolve_config_path(getattr(args, "config", None))
            if config_path is None:
                raise ValueError(
                    "Config not found. Pass --config PATH or create one with init."
                )
            config = load_config(config_path)
            settings = build_settings(
                config,
                config_path,
                workspace_mode_override=getattr(args, "workspace", None),
            )
            if command == "sbatch":
                file = str(getattr(args, "sbatch_file", "") or "")
                jobs = [
                    JobSpec(
                        name=getattr(args, "name", None) or Path(file).stem,
                        sbatch_file=file,
                        sbatch_args=getattr(args, "sbatch_arg", None) or [],
                    )
                ]
            else:
                jobs = prepare_jobs(
                    config, configured_run_only(config, args), settings.default_env
                )
            fail_duplicate_jobs(jobs)
            enforce_clean_git(
                settings,
                require_clean_git=bool(getattr(args, "require_clean_git", False)),
            )
            if not dry_run:
                prepare = getattr(config, "prepare", None)
                if prepare is not None:
                    if not callable(prepare):
                        raise ValueError("Config prepare must be callable.")
                    prepare()
                enforce_clean_git(
                    settings,
                    require_clean_git=bool(getattr(args, "require_clean_git", False)),
                )
            validate_predefined_sbatch_jobs(settings, jobs)
            paths = resolve_remote_paths(settings)
            plan = _freeze(settings, paths, jobs)
            _context(payload, settings, paths, plan)
            payload["jobs"] = [
                {
                    "job_name": job.name,
                    "job_id": "",
                    "state": "planned",
                    "artifacts": job.artifacts,
                    "requires": job.requires,
                }
                for job in jobs
            ]
            run_dir = settings.project_root / "slurm_output" / paths.job_folder
            if not dry_run:
                run_dir.mkdir(parents=True, exist_ok=False)
                run_lock = (run_dir / ".submission.lock").open("a")
                fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                payload["plan_file"] = str(run_dir / "plan.json")
                atomic_write_json(run_dir / "plan.json", plan)
                for entry in plan["jobs"]:
                    atomic_write(run_dir / entry["snapshot"], entry["script"])
                    if entry["job_script"] is not None:
                        atomic_write(
                            (run_dir / entry["snapshot"]).with_suffix(".sh"),
                            entry["job_script"],
                        )
                _save_progress(settings, paths, payload)
                tracked = json.loads((run_dir / "jobs.json").read_text())
                tracked.update(
                    stage_state="staging",
                    provenance=plan["provenance"],
                    plan_file="plan.json",
                )
                atomic_write_json(run_dir / "jobs.json", tracked)
            test_ssh_connection(
                settings.cluster_login,
                dry_run=dry_run,
                ssh_config_file=settings.ssh_config_file,
                ssh_options=settings.ssh_options,
                quiet=quiet,
            )
            payload["commands"].extend(
                sync_project(settings, paths, dry_run, quiet=quiet)
            )
            if not dry_run:
                tracked["stage_state"] = "staged"
                atomic_write_json(run_dir / "jobs.json", tracked)
                _save_progress(settings, paths, payload)
            if command != "stage":
                _submit(settings, paths, plan, payload, dry_run=dry_run, quiet=quiet)
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        payload.update(ok=False, error=str(exc) or type(exc).__name__)
    finally:
        if run_lock is not None:
            run_lock.close()
    return _emit(args, payload)


def do_run(args: Any) -> int:
    return _execute(args, "run")


def do_stage(args: Any) -> int:
    return _execute(args, "stage")


def do_submit(args: Any) -> int:
    return _execute(args, "submit")


def do_sbatch(args: Any) -> int:
    return _execute(args, "sbatch")


def do_preflight(args: Any) -> int:
    return _execute(args, "preflight")
