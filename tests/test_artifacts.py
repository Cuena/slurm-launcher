"""Tests for the artifacts command."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from launcher.artifacts import list_artifacts
from launcher.tracking import JobRecord, TrackingPayload


class TestArtifactDiscovery(TestCase):
    def test_list_artifacts_uses_job_specific_and_payload_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            payload = TrackingPayload(
                source_path=Path("."),
                cluster_login="user@cluster",
                remote_workdir="/work/project",
                job_folder="run_20250101_120000",
                artifact_paths=["outputs"],
                jobs=[
                    JobRecord(
                        job_name="train",
                        job_id="12345",
                        artifacts=["outputs/train", "checkpoints/best.pt"],
                    ),
                    JobRecord(
                        job_name="eval",
                        job_id="12346",
                        artifacts=[],
                    ),
                ],
                created_at=None,
                ssh_config_file=None,
                ssh_options=[],
                remote_logdir=None,
                remote_slurm_output_dir=None,
                remote_slurm_dashboard_log_archive_dir=None,
                remote_slurm_dashboard_log_view_dir=None,
                runtime_mode=None,
                venv_python_executable=None,
                singularity_image_path=None,
                sync_symlinks=None,
            )
            entries = list_artifacts(payload, output_dir)

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["path"], "outputs/train")
        self.assertEqual(entries[0]["remote_path"], "/work/project/outputs/train")
        self.assertIn("train/12345", entries[0]["destination"])

        # eval falls back to payload artifact_paths
        eval_entries = [e for e in entries if e["job_name"] == "eval"]
        self.assertEqual(len(eval_entries), 1)
        self.assertEqual(eval_entries[0]["path"], "outputs")

    def test_list_artifacts_selected_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            payload = TrackingPayload(
                source_path=Path("."),
                cluster_login="user@cluster",
                remote_workdir="/work/project",
                job_folder="run",
                artifact_paths=["outputs"],
                jobs=[
                    JobRecord(job_name="train", job_id="1"),
                    JobRecord(job_name="eval", job_id="2"),
                ],
                created_at=None,
                ssh_config_file=None,
                ssh_options=[],
                remote_logdir=None,
                remote_slurm_output_dir=None,
                remote_slurm_dashboard_log_archive_dir=None,
                remote_slurm_dashboard_log_view_dir=None,
                runtime_mode=None,
                venv_python_executable=None,
                singularity_image_path=None,
                sync_symlinks=None,
            )
            entries = list_artifacts(payload, output_dir, selected_jobs=["train"])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["job_name"], "train")
