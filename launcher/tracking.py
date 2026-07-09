from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class TrackingError(Exception):
    """Raised when a tracking file cannot be read or has invalid structure."""


@dataclass(frozen=True)
class JobRecord:
    """Single job entry inside a tracking payload."""

    job_name: str
    job_id: str
    stdout: str | None = None
    stderr: str | None = None
    sbatch_command: str | None = None
    remote_sbatch: str | None = None
    submitted_at: str | None = None
    launcher: dict[str, object] | None = None
    artifacts: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrackingPayload:
    """Parsed and validated tracking file (jobs.json)."""

    source_path: Path
    created_at: str | None
    cluster_login: str
    ssh_config_file: str | None
    ssh_options: list[str]
    job_folder: str
    remote_workdir: str | None
    remote_logdir: str | None
    remote_slurm_output_dir: str | None
    remote_slurm_dashboard_log_archive_dir: str | None
    remote_slurm_dashboard_log_view_dir: str | None
    runtime_mode: str | None
    venv_python_executable: str | None
    singularity_image_path: str | None
    artifact_paths: list[str]
    sync_symlinks: str | None
    jobs: list[JobRecord] = field(default_factory=list)

    def filter_jobs(
        self,
        *,
        names: set[str] | None = None,
        ids: set[str] | None = None,
    ) -> list[JobRecord]:
        if not names and not ids:
            return list(self.jobs)
        result: list[JobRecord] = []
        for job in self.jobs:
            if names and job.job_name in names:
                result.append(job)
            elif ids and job.job_id in ids:
                result.append(job)
        return result

    def runnable_job_ids(self, jobs: list[JobRecord] | None = None) -> list[str]:
        source = jobs if jobs is not None else self.jobs
        return [
            job.job_id
            for job in source
            if job.job_id and job.job_id not in {"", "unknown", "dry-run"}
        ]


def resolve_tracking_file(path_arg: str | None) -> Path | None:
    if path_arg:
        candidate = Path(path_arg)
        return candidate if candidate.exists() else None

    latest = Path("slurm_output/latest_jobs.json")
    if latest.exists():
        return latest

    candidates = sorted(
        Path("slurm_output").glob("*/jobs.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return None


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _parse_job_record(raw: object) -> JobRecord | None:
    if not isinstance(raw, dict):
        return None
    job_name = str(raw.get("job_name", "") or "")
    job_id = str(raw.get("job_id", "") or "")
    if not job_name and not job_id:
        return None
    launcher_raw = raw.get("launcher")
    launcher = launcher_raw if isinstance(launcher_raw, dict) else None
    return JobRecord(
        job_name=job_name,
        job_id=job_id,
        stdout=_str_or_none(raw.get("stdout")),
        stderr=_str_or_none(raw.get("stderr")),
        sbatch_command=_str_or_none(raw.get("sbatch_command")),
        remote_sbatch=_str_or_none(raw.get("remote_sbatch")),
        submitted_at=_str_or_none(raw.get("submitted_at")),
        launcher=launcher,
        artifacts=_str_list(raw.get("artifacts")),
        requires=_str_list(raw.get("requires")),
    )


def load_tracking_payload(path: Path) -> TrackingPayload:
    """Read and validate a jobs.json tracking file into typed structures.

    Raises TrackingError on I/O or structural problems.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrackingError(f"Cannot read tracking file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise TrackingError(f"Invalid tracking file format: {path}")

    cluster_login = _str_or_none(raw.get("cluster_login")) or ""

    raw_jobs = raw.get("jobs", [])
    if not isinstance(raw_jobs, list):
        raise TrackingError(f"Invalid 'jobs' field in tracking file: {path}")

    jobs = [rec for item in raw_jobs if (rec := _parse_job_record(item)) is not None]

    return TrackingPayload(
        source_path=path,
        created_at=_str_or_none(raw.get("created_at")),
        cluster_login=cluster_login,
        ssh_config_file=_str_or_none(raw.get("ssh_config_file")),
        ssh_options=_str_list(raw.get("ssh_options")),
        job_folder=str(
            raw.get("job_folder", "unknown_job_folder") or "unknown_job_folder"
        ),
        remote_workdir=_str_or_none(raw.get("remote_workdir")),
        remote_logdir=_str_or_none(raw.get("remote_logdir")),
        remote_slurm_output_dir=_str_or_none(raw.get("remote_slurm_output_dir")),
        remote_slurm_dashboard_log_archive_dir=_str_or_none(
            raw.get("remote_slurm_dashboard_log_archive_dir")
        ),
        remote_slurm_dashboard_log_view_dir=_str_or_none(
            raw.get("remote_slurm_dashboard_log_view_dir")
        ),
        runtime_mode=_str_or_none(raw.get("runtime_mode")),
        venv_python_executable=_str_or_none(raw.get("venv_python_executable")),
        singularity_image_path=_str_or_none(raw.get("singularity_image_path")),
        artifact_paths=_str_list(raw.get("artifact_paths")),
        sync_symlinks=_str_or_none(raw.get("sync_symlinks")),
        jobs=jobs,
    )


def _str_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def all_tracking_files() -> list[Path]:
    """Return all known tracking files, most recent first."""
    tracking_root = Path("slurm_output")
    files: list[Path] = []
    latest = tracking_root / "latest_jobs.json"
    if latest.exists():
        files.append(latest)
    if tracking_root.exists():
        candidates = sorted(
            tracking_root.glob("*/jobs.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate not in files:
                files.append(candidate)
    return files
