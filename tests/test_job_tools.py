from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher.job_tools import (
    DEFAULT_ARCHIVE_DIR,
    JobLogInfo,
    LauncherInfo,
    _job_log_info_from_sacct,
    _launcher_info_from_tracking,
    effective_archive_dir,
    list_recent_jobs,
    show_job_details,
    show_job_log,
)


class JobToolsTests(unittest.TestCase):
    def test_effective_archive_dir_defaults_when_missing(self) -> None:
        archive_dir, source = effective_archive_dir(None)
        self.assertEqual(archive_dir, str(DEFAULT_ARCHIVE_DIR))
        self.assertEqual(source, "default")

    def test_job_log_info_from_sacct_expands_job_id_placeholders(self) -> None:
        info = _job_log_info_from_sacct(
            "38238485|job|TIMEOUT|/tmp/%j.out|/tmp/%J.err",
            "38238485",
        )
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.stdout, "/tmp/38238485.out")
        self.assertEqual(info.stderr, "/tmp/38238485.err")

    @patch("launcher.job_tools._run_ssh_capture")
    @patch("launcher.job_tools.console.print_json")
    def test_list_recent_jobs_passes_ssh_settings(
        self,
        mock_print_json,
        mock_run_ssh_capture,
    ) -> None:
        mock_run_ssh_capture.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "__SOURCE__=sacct\n"
                "1|job|RUNNING|acc|2026-03-26T12:00:00|2026-03-26T12:00:01|Unknown|00:01:00\n"
            ),
            stderr="",
        )

        exit_code = list_recent_jobs(
            "user@cluster",
            user=None,
            hours=24,
            limit=5,
            states=None,
            json_output=True,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(exit_code, 0)
        mock_run_ssh_capture.assert_called_once()
        _, kwargs = mock_run_ssh_capture.call_args
        self.assertEqual(kwargs["ssh_config_file"], "/dev/null")
        self.assertEqual(kwargs["ssh_options"], ["-o", "BatchMode=yes"])
        mock_print_json.assert_called_once()

    @patch("launcher.job_tools.console.print_json")
    @patch("launcher.job_tools._run_ssh_capture")
    def test_list_recent_jobs_filters_by_leading_state_token(
        self,
        mock_run_ssh_capture,
        mock_print_json,
    ) -> None:
        mock_run_ssh_capture.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "__SOURCE__=sacct\n"
                "1|job-a|RUNNING|acc|2026-03-26T12:00:00|2026-03-26T12:00:01|Unknown|00:01:00\n"
                "2|job-b|CANCELLED by 4840|acc|2026-03-26T11:00:00|2026-03-26T11:01:00|2026-03-26T11:02:00|00:01:00\n"
            ),
            stderr="",
        )

        exit_code = list_recent_jobs(
            "user@cluster",
            user=None,
            hours=24,
            limit=5,
            states={"cancelled"},
            json_output=True,
        )

        self.assertEqual(exit_code, 0)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["states"], ["cancelled"])
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["job_id"], "2")

    @patch("launcher.job_tools._launcher_info_from_script")
    @patch("launcher.job_tools._launcher_info_from_tracking")
    @patch("launcher.job_tools.console.print_json")
    @patch("launcher.job_tools._run_ssh_capture")
    def test_show_job_details_json_prints_generic_payload(
        self,
        mock_run_ssh_capture,
        mock_print_json,
        mock_launcher_info_from_tracking,
        mock_launcher_info_from_script,
    ) -> None:
        mock_run_ssh_capture.side_effect = [
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "JobId=38238485 JobName=train JobState=RUNNING Partition=acc "
                    "Command=/remote/logs/train.sbatch WorkDir=/remote/workdir "
                    "StdOut=/tmp/%j.out StdErr=/tmp/%j.err NodeList=node01 "
                    "NumNodes=1 SubmitTime=2026-03-26T12:00:00 "
                    "StartTime=2026-03-26T12:00:01 EndTime=Unknown"
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=(
                    "38238485|train|RUNNING|acc|2026-03-26T12:00:00|"
                    "2026-03-26T12:00:01|Unknown|node01|1|gpu:1|"
                    "/remote/workdir|/tmp/%j.out|/tmp/%j.err|sbatch train.sbatch"
                ),
                stderr="",
            ),
        ]
        mock_launcher_info_from_tracking.return_value = None
        mock_launcher_info_from_script.return_value = None

        exit_code = show_job_details(
            "user@cluster",
            "38238485",
            json_output=True,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(exit_code, 0)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["job_id"], "38238485")
        self.assertEqual(payload["job_name"], "train")
        self.assertEqual(payload["command"], "/remote/logs/train.sbatch")
        self.assertEqual(payload["stdout"], "/tmp/38238485.out")
        self.assertEqual(payload["stderr"], "/tmp/38238485.err")
        self.assertEqual(payload["gres"], "gpu:1")
        self.assertEqual(payload["resolved_via"], "scontrol+sacct")
        self.assertEqual(payload["detail_level"], "full")
        self.assertNotIn("launcher", payload)

    @patch("launcher.job_tools._launcher_info_from_script")
    @patch("launcher.job_tools._launcher_info_from_tracking")
    @patch("launcher.job_tools.console.print_json")
    @patch("launcher.job_tools._run_ssh_capture")
    def test_show_job_details_falls_back_to_log_resolution_for_finished_jobs(
        self,
        mock_run_ssh_capture,
        mock_print_json,
        mock_launcher_info_from_tracking,
        mock_launcher_info_from_script,
    ) -> None:
        mock_run_ssh_capture.side_effect = [
            subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="38238485|job|FAILED|acc|2026-03-26T12:00:00|2026-03-26T12:00:01|2026-03-26T12:30:00|node01|1|gpu:1|/remote/workdir|/tmp/%j.out",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="38238485|job|FAILED|/tmp/%j.out|/tmp/%j.err",
                stderr="",
            ),
        ]
        mock_launcher_info_from_tracking.return_value = None
        mock_launcher_info_from_script.return_value = None

        exit_code = show_job_details(
            "user@cluster",
            "38238485",
            json_output=True,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(exit_code, 0)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["job_id"], "38238485")
        self.assertEqual(payload["job_name"], "job")
        self.assertEqual(payload["state"], "FAILED")
        self.assertEqual(payload["stdout"], "/tmp/38238485.out")
        self.assertEqual(payload["stderr"], "/tmp/38238485.err")
        self.assertEqual(payload["resolved_via"], "sacct")
        self.assertEqual(payload["detail_level"], "log-resolution")
        self.assertNotIn("partition", payload)
        self.assertNotIn("command", payload)
        self.assertNotIn("launcher", payload)

    def test_launcher_info_from_tracking_reads_latest_tracking_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            previous = Path.cwd()
            os.chdir(tmpdir)
            try:
                tracking_dir = Path("slurm_output") / "run_001"
                tracking_dir.mkdir(parents=True)
                (Path("slurm_output") / "latest_jobs.json").write_text(
                    (
                        '{"runtime_mode":"singularity","singularity_image_path":"'
                        '/remote/images/train.sif","jobs":[{"job_id":"38238485",'
                        '"launcher":{"managed":true,"runtime_kind":"singularity",'
                        '"runtime_artifact":"/remote/images/train.sif",'
                        '"entry_command":"python train.py"}}]}'
                    ),
                    encoding="utf-8",
                )

                launcher = _launcher_info_from_tracking("38238485")
            finally:
                os.chdir(previous)

        self.assertEqual(
            launcher,
            LauncherInfo(
                managed=True,
                runtime_kind="singularity",
                runtime_artifact="/remote/images/train.sif",
                entry_command="python train.py",
            ),
        )

    @patch("launcher.job_tools.subprocess.run")
    @patch("launcher.job_tools.console.print_json")
    @patch("launcher.job_tools.resolve_job_log_info")
    def test_show_job_log_json_prints_resolution_only(
        self,
        mock_resolve_job_log_info,
        mock_print_json,
        mock_subprocess_run,
    ) -> None:
        mock_resolve_job_log_info.return_value = JobLogInfo(
            job_id="38238485",
            job_name="job",
            state="RUNNING",
            stdout="/tmp/job.out",
            stderr="/tmp/job.err",
            source="sacct",
        )

        exit_code = show_job_log(
            "user@cluster",
            "38238485",
            stream="stdout",
            lines=5,
            follow=False,
            full=False,
            path_only=False,
            json_output=True,
            archive_dir=None,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(exit_code, 0)
        mock_subprocess_run.assert_not_called()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["path"], "/tmp/job.out")
        self.assertEqual(payload["resolved_via"], "sacct")
        self.assertTrue(payload["path_verified"])
        self.assertFalse(payload["content_included"])

    @patch("launcher.job_tools.console.print_json")
    @patch("launcher.job_tools.resolve_job_log_info")
    def test_show_job_log_json_fails_when_probes_are_unresolved(
        self,
        mock_resolve_job_log_info,
        mock_print_json,
    ) -> None:
        mock_resolve_job_log_info.return_value = JobLogInfo(
            job_id="38238485",
            job_name="",
            state="",
            stdout=None,
            stderr=None,
            source="unresolved",
            verified=False,
            probe_errors=("scontrol failed (rc=255)", "sacct failed (rc=255)"),
        )

        exit_code = show_job_log(
            "acc",
            "38238485",
            stream="stdout",
            lines=5,
            follow=False,
            full=False,
            path_only=False,
            json_output=True,
            archive_dir="/remote/archive",
        )

        self.assertEqual(exit_code, 1)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["job_id"], "38238485")
        self.assertEqual(len(payload["probe_errors"]), 2)

    @patch("launcher.job_tools.subprocess.run")
    @patch("launcher.job_tools.resolve_job_log_info")
    def test_show_job_log_passes_ssh_settings_to_tail(
        self,
        mock_resolve_job_log_info,
        mock_subprocess_run,
    ) -> None:
        mock_resolve_job_log_info.return_value = JobLogInfo(
            job_id="38238485",
            job_name="job",
            state="RUNNING",
            stdout="/tmp/job.out",
            stderr="/tmp/job.err",
            source="sacct",
        )
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=0,
        )

        exit_code = show_job_log(
            "user@cluster",
            "38238485",
            stream="stdout",
            lines=5,
            follow=False,
            full=False,
            path_only=False,
            json_output=False,
            archive_dir=None,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(exit_code, 0)
        mock_subprocess_run.assert_called_once()
        command = mock_subprocess_run.call_args.args[0]
        self.assertEqual(
            command[:6],
            ["ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "user@cluster"],
        )
        self.assertEqual(command[-1], "tail -n 5 /tmp/job.out")


if __name__ == "__main__":
    unittest.main()
