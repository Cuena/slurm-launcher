"""Remote SLURM launcher package."""

from .core import (
    JobSpec,
    LauncherSettings,
    RemotePaths,
    SubmissionResult,
    submit_job,
    sync_project,
    test_ssh_connection,
)

__all__ = [
    "JobSpec",
    "LauncherSettings",
    "RemotePaths",
    "SubmissionResult",
    "submit_job",
    "sync_project",
    "test_ssh_connection",
]
