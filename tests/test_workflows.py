from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher import cli
from launcher.download_logs import run_download_logs
from launcher.download_artifacts import run_download_artifacts
from launcher.job_tools import resolve_job_log_info
from tests.helpers import make_settings, write_tracking_file


FULL_TRACKING_PAYLOAD = {
    "created_at": "2026-04-01T12:00:00",
    "cluster_login": "user@cluster",
    "ssh_config_file": "/dev/null",
    "ssh_options": ["-o", "BatchMode=yes"],
    "job_folder": "project_001",
    "remote_workdir": "/remote/work/project_001",
    "remote_logdir": "/remote/logs/project_001",
    "remote_slurm_output_dir": "/remote/logs/project_001/slurm_output",
    "artifact_paths": ["outputs/model.ckpt"],
    "jobs": [
        {
            "job_name": "train",
            "job_id": "12345",
            "stdout": "/logs/train-12345.out",
            "stderr": "/logs/train-12345.err",
            "sbatch_command": "sbatch train.sbatch",
            "submitted_at": "2026-04-01T12:00:00",
        },
        {
            "job_name": "eval",
            "job_id": "12346",
            "stdout": "/logs/eval-12346.out",
            "stderr": "/logs/eval-12346.err",
        },
    ],
}


class DownloadLogsWorkflowTests(unittest.TestCase):
    """Tests download-logs through the canonical tracking boundary."""

    def test_download_logs_reads_tracking_file_and_selects_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                job_name=["train"],
                job_id=[],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_download_logs(args)

        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("Jobs selected: 1", printed)
        self.assertIn("Log files to download: 2", printed)
        self.assertIn("/logs/train-12345.out", printed)
        self.assertIn("/logs/train-12345.err", printed)

    def test_download_logs_filters_by_job_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                job_name=[],
                job_id=["12346"],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_download_logs(args)

        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("Jobs selected: 1", printed)
        self.assertIn("/logs/eval-12346.out", printed)

    def test_download_logs_fails_on_missing_cluster_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {**FULL_TRACKING_PAYLOAD, "cluster_login": ""}
            tracking = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            args = argparse.Namespace(
                tracking_file=str(tracking),
                job_name=[],
                job_id=[],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print"):
                exit_code = run_download_logs(args)

        self.assertEqual(exit_code, 1)

    def test_download_logs_fails_on_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            path.write_text("not json", encoding="utf-8")
            args = argparse.Namespace(
                tracking_file=str(path),
                job_name=[],
                job_id=[],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print"):
                exit_code = run_download_logs(args)

        self.assertEqual(exit_code, 1)

    def test_download_logs_json_dry_run_reports_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                job_name=["train"],
                job_id=[],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
                json=True,
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_download_logs(args)

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["tracking_file"], str(tracking))
        self.assertEqual(payload["selected_jobs"][0]["job_name"], "train")
        self.assertEqual(len(payload["downloads"]), 2)
        self.assertEqual(len(payload["commands"]), 2)
        self.assertIn("rsync -az", payload["commands"][0])
        self.assertEqual(payload["dry_run"], True)


class DownloadArtifactsWorkflowTests(unittest.TestCase):
    """Tests download-artifacts through the canonical tracking boundary."""

    def test_download_artifacts_uses_tracked_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                path=[],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_download_artifacts(args)

        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("Artifact paths to download: 1", printed)
        self.assertIn("outputs/model.ckpt", printed)

    def test_download_artifacts_overrides_with_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                path=["custom/path"],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_download_artifacts(args)

        self.assertEqual(exit_code, 0)
        printed = "\n".join(str(c.args[0]) for c in mock_print.call_args_list)
        self.assertIn("custom/path", printed)

    def test_download_artifacts_fails_on_missing_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {**FULL_TRACKING_PAYLOAD, "remote_workdir": ""}
            tracking = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            args = argparse.Namespace(
                tracking_file=str(tracking),
                path=["something"],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
            )
            with patch("builtins.print"):
                exit_code = run_download_artifacts(args)

        self.assertEqual(exit_code, 1)

    def test_download_artifacts_json_dry_run_reports_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                path=["custom/path"],
                output_dir=str(Path(tmpdir) / "out"),
                dry_run=True,
                json=True,
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_download_artifacts(args)

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["tracking_file"], str(tracking))
        self.assertEqual(payload["artifact_paths"], ["custom/path"])
        self.assertEqual(payload["artifacts"][0]["path"], "custom/path")
        self.assertEqual(len(payload["commands"]), 1)
        self.assertIn("rsync -az", payload["commands"][0])
        self.assertEqual(payload["dry_run"], True)


class SbatchErrorWrapperTests(unittest.TestCase):
    """Tests that do_sbatch surfaces errors consistently."""

    @patch("launcher.cli.build_settings")
    @patch("launcher.cli._load_run_config")
    def test_sbatch_catches_validation_errors(
        self,
        mock_load_run_config,
        mock_build_settings,
    ) -> None:
        settings = make_settings()
        mock_load_run_config.return_value = (object(), Path("/tmp/config.py"))
        mock_build_settings.return_value = settings

        args = argparse.Namespace(
            config=None,
            workspace=None,
            sbatch_file="../outside/file.sbatch",
            name="test",
            sbatch_arg=[],
            dry_run=False,
        )

        exit_code = cli.do_sbatch(args)
        self.assertEqual(exit_code, 1)

    @patch("launcher.cli.sync_project")
    @patch("launcher.cli.test_ssh_connection")
    @patch("launcher.cli.resolve_remote_paths")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli._load_run_config")
    def test_sbatch_catches_ssh_errors(
        self,
        mock_load_run_config,
        mock_build_settings,
        mock_resolve_remote_paths,
        mock_test_ssh_connection,
        mock_sync_project,
    ) -> None:
        settings = make_settings()
        mock_load_run_config.return_value = (object(), Path("/tmp/config.py"))
        mock_build_settings.return_value = settings
        mock_test_ssh_connection.side_effect = RuntimeError("SSH failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            sbatch_file = Path(tmpdir) / "train.sbatch"
            sbatch_file.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
            settings_with_root = make_settings(project_root=Path(tmpdir))
            mock_build_settings.return_value = settings_with_root

            args = argparse.Namespace(
                config=None,
                workspace=None,
                sbatch_file=str(sbatch_file),
                name="train",
                sbatch_arg=[],
                dry_run=False,
            )

            exit_code = cli.do_sbatch(args)

        self.assertEqual(exit_code, 1)


class JobLogProbeTests(unittest.TestCase):
    """Tests that job-log resolution surfaces probe failures."""

    @patch("launcher.job_tools._run_ssh_capture")
    def test_fallback_source_includes_probe_errors(
        self,
        mock_run_ssh_capture,
    ) -> None:
        mock_run_ssh_capture.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err"),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="err"),
        ]

        info = resolve_job_log_info(
            "user@cluster",
            "99999",
            archive_dir=None,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertIsNotNone(info)
        assert info is not None
        self.assertIn("fallback", info.source)
        self.assertIn("scontrol failed", info.source)
        self.assertIn("sacct failed", info.source)

    @patch("launcher.job_tools._run_ssh_capture")
    def test_fallback_when_scontrol_returns_no_paths(
        self,
        mock_run_ssh_capture,
    ) -> None:
        mock_run_ssh_capture.side_effect = [
            subprocess.CompletedProcess(
                args=[], returncode=0, stdout="JobId=99999 JobName=test", stderr=""
            ),
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
        ]

        info = resolve_job_log_info(
            "user@cluster",
            "99999",
            archive_dir="/custom/archive",
        )

        self.assertIsNotNone(info)
        assert info is not None
        self.assertIn("scontrol returned no log paths", info.source)
        self.assertIn("/custom/archive/99999.out", info.stdout or "")


class LogsCommandWorkflowTests(unittest.TestCase):
    """Tests the logs command through the tracking boundary."""

    @patch("launcher.cli.console.print_json")
    def test_logs_json_reads_tracking_file_typed(self, mock_print_json) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                only=["train"],
                config=None,
                workspace=None,
                latest=False,
                job_id=None,
                use_stderr=False,
                follow=False,
                lines=50,
                full=False,
                json=True,
            )
            exit_code = cli.do_logs(args)

        self.assertEqual(exit_code, 0)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["cluster_login"], "user@cluster")
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["job_name"], "train")
        self.assertEqual(payload["jobs"][0]["job_id"], "12345")


class MonitorCommandWorkflowTests(unittest.TestCase):
    """Tests the monitor command through the tracking boundary."""

    @patch("launcher.cli.console.print_json")
    def test_monitor_reads_tracking_and_filters(self, mock_print_json) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            args = argparse.Namespace(
                tracking_file=str(tracking),
                only=["train"],
                dry_run=True,
                json=True,
            )
            exit_code = cli.do_monitor(args)

        self.assertEqual(exit_code, 0)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["job_ids"], ["12345"])
        self.assertIn("squeue -j 12345", payload["command"])


if __name__ == "__main__":
    unittest.main()
