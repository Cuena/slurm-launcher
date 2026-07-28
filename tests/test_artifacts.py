"""Tests for the artifacts command."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from launcher.artifacts import list_artifacts, run_artifacts
from launcher.tracking import JobRecord, TrackingPayload
from tests.helpers import write_tracking_file


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

    def test_list_json_is_explicitly_declared_and_not_remote_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracking = write_tracking_file(
                Path(tmp) / "jobs.json",
                {
                    "cluster_login": "",
                    "job_folder": "run",
                    "remote_workdir": "/work/project",
                    "jobs": [
                        {
                            "job_name": "train",
                            "job_id": "1",
                            "artifacts": ["outputs/result.json"],
                        }
                    ],
                },
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_artifacts(
                    subcommand="list",
                    tracking_file=str(tracking),
                    json_output=True,
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["operation"], "list")
        self.assertTrue(payload["declared_only"])
        self.assertFalse(payload["remote_checked"])
        self.assertFalse(payload["copy_attempted"])
        self.assertNotIn("commands", payload)

    @patch("launcher.artifacts.subprocess.run")
    def test_check_json_reports_remote_existence(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="0|true|file|42\n1|false||\n",
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tracking = write_tracking_file(
                Path(tmp) / "jobs.json",
                {
                    "cluster_login": "acc",
                    "job_folder": "run",
                    "remote_workdir": "/work/project",
                    "jobs": [
                        {
                            "job_name": "train",
                            "job_id": "1",
                            "artifacts": [
                                "outputs/result.json",
                                "outputs/missing.json",
                            ],
                        }
                    ],
                },
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_artifacts(
                    subcommand="check",
                    tracking_file=str(tracking),
                    json_output=True,
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["operation"], "check")
        self.assertTrue(payload["remote_checked"])
        self.assertTrue(payload["artifacts"][0]["exists"])
        self.assertEqual(payload["artifacts"][0]["size_bytes"], 42)
        self.assertFalse(payload["artifacts"][1]["exists"])
        command = mock_run.call_args.args[0]
        self.assertEqual(command[:2], ["ssh", "acc"])
