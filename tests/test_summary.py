"""Tests for summary command path handling."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from launcher import cli
from launcher.core import LauncherSettings
from launcher.summary import update_summary_from_status
from launcher.tracking import load_tracking_payload
from tests.helpers import write_tracking_file


class TestSummaryUsesTrackedPaths(TestCase):
    def test_summary_uses_payload_remote_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tracking = write_tracking_file(
                tmp_path / "slurm_output" / "run_001" / "jobs.json",
                {
                    "created_at": "2026-01-01T00:00:00",
                    "cluster_login": "user@cluster",
                    "job_folder": "run_001",
                    "remote_workdir": "/remote/work/run_001",
                    "remote_logdir": "/remote/logs/run_001",
                    "remote_slurm_output_dir": "/remote/logs/run_001/slurm_output",
                    "artifact_paths": ["outputs"],
                    "jobs": [
                        {
                            "job_name": "train",
                            "job_id": "12345",
                            "stdout": "/remote/logs/run_001/train-12345.out",
                        }
                    ],
                },
            )
            (tmp_path / "slurm_output" / "latest_jobs.json").write_text(
                tracking.read_text(), encoding="utf-8"
            )

            settings = LauncherSettings(
                cluster_login="user@cluster",
                ssh_config_file=None,
                ssh_options=[],
                remote_workspace_base="/should/not/use",
                remote_log_base_path="/should/not/use",
                workspace_mode="per-run",
                remote_workspace_dir=None,
                project_root=tmp_path,
                project_prefix="project",
                venv_python_executable=None,
                default_env={},
                default_sbatch={},
                extra_rsync_excludes=[],
                extra_rsync_args=[],
                remote_slurm_dashboard_log_archive_dir=None,
                remote_slurm_dashboard_log_view_dir=None,
                runtime_mode="native",
                singularity_image_path=None,
                singularity_exec_flags=[],
                artifact_paths=[],
                require_clean_git=False,
                sync_symlinks="copy-links",
                local_artifact_root=None,
                verbose=False,
            )
            payload = load_tracking_payload(tracking)
            remote_paths = cli.RemotePaths(
                job_folder=payload.job_folder,
                workdir=payload.remote_workdir,
                logdir=payload.remote_logdir,
                slurm_output_dir=payload.remote_slurm_output_dir,
            )

            with mock.patch(
                "launcher.summary.query_job_statuses",
                return_value=[],
            ):
                with mock.patch(
                    "launcher.core.ssh_script",
                    return_value=("", ""),
                ):
                    summary_path = update_summary_from_status(
                        settings, remote_paths, payload
                    )

            self.assertEqual(summary_path.parent.name, "run_001")
            self.assertEqual(summary_path.name, "summary.json")
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(data["remote_workdir"], "/remote/work/run_001")
            self.assertEqual(data["job_folder"], "run_001")
