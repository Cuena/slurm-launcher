"""Tests for preflight script generation and behavior."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from launcher.preflight import build_remote_check_script


class TestRemoteCheckScript(TestCase):
    def test_plain_path_and_glob(self) -> None:
        script = build_remote_check_script(
            "/work/project",
            ["data/processed", "models/*.pt"],
        )
        self.assertIn("cd /work/project", script)
        self.assertIn("CHECK_START|data/processed", script)
        self.assertIn("CHECK_START|models/*.pt", script)
        self.assertIn("compgen -G 'models/*.pt'", script)
        self.assertIn("[ -e data/processed ]", script)

    def test_broken_symlink_detection(self) -> None:
        script = build_remote_check_script(
            "/work/project",
            ["data/model.pt"],
        )
        self.assertIn("[ -L data/model.pt ]", script)
        self.assertIn("broken symlink", script)

    def test_no_match_glob_does_not_exit_before_reporting_failure(self) -> None:
        script = build_remote_check_script("/work/project", ["models/*.pt"])
        self.assertIn(
            "count=$(compgen -G 'models/*.pt' 2>/dev/null | wc -l) || true", script
        )


class TestPreflightDryRunJson(TestCase):
    def test_dry_run_json_emits_valid_payload(self) -> None:
        from launcher.preflight import run_preflight
        from launcher.core import JobSpec, LauncherSettings

        settings = LauncherSettings(
            cluster_login="user@cluster",
            ssh_config_file=None,
            ssh_options=[],
            remote_workspace_base="/work",
            remote_log_base_path="/logs",
            workspace_mode="fixed",
            remote_workspace_dir="/work/project",
            project_root=Path("/tmp/project"),
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
        remote_paths = type(
            "RemotePaths",
            (),
            {
                "workdir": "/work/project",
                "job_folder": "run_001",
                "logdir": "/logs/run_001",
                "slurm_output_dir": "/logs/run_001/slurm_output",
            },
        )()
        jobs = [
            JobSpec(
                name="eval",
                sbatch_file="slurm/eval.sbatch",
                requires=["data/input/*.mp4", "models/model.onnx"],
            ),
        ]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_preflight(
                settings,
                remote_paths,
                jobs,
                json_output=True,
                dry_run=True,
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["remote_workdir"], "/work/project")
        self.assertEqual(payload["checks_planned"], 2)
        self.assertEqual(payload["checks_run"], 0)
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["job_name"], "eval")
        self.assertEqual(payload["jobs"][0]["status"], "planned")
        self.assertIn("data/input/*.mp4", payload["jobs"][0]["requirements"])
        self.assertIn("set -euo pipefail", payload["jobs"][0]["script"])

    def test_live_json_rejects_no_selected_jobs(self) -> None:
        from launcher.preflight import run_preflight

        settings = type("Settings", (), {})()
        remote_paths = type("RemotePaths", (), {"workdir": "/work/project"})()

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_preflight(
                settings,
                remote_paths,
                [],
                json_output=True,
                dry_run=False,
            )

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["checks_run"], 0)
        self.assertEqual(payload["jobs"], [])
        self.assertEqual(payload["warnings"], ["No jobs were selected for preflight."])

    def test_live_json_rejects_job_without_requires_explicitly(self) -> None:
        from launcher.core import JobSpec
        from launcher.preflight import run_preflight

        settings = type("Settings", (), {})()
        remote_paths = type("RemotePaths", (), {"workdir": "/work/project"})()
        jobs = [JobSpec(name="shared", sbatch_file="slurm/shared.sbatch")]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_preflight(
                settings,
                remote_paths,
                jobs,
                json_output=True,
            )

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["checks_planned"], 0)
        self.assertEqual(payload["checks_run"], 0)
        self.assertEqual(payload["jobs"][0]["job_name"], "shared")
        self.assertEqual(payload["jobs"][0]["status"], "not-configured")
        self.assertFalse(payload["jobs"][0]["ok"])
        self.assertIn("no 'requires'", payload["jobs"][0]["message"])

    def test_absolute_requirement_keeps_absolute_remote_path(self) -> None:
        from launcher.preflight import run_preflight_for_job

        settings = type(
            "Settings",
            (),
            {
                "cluster_login": "user@cluster",
                "ssh_config_file": None,
                "ssh_options": [],
            },
        )()
        remote_paths = type("RemotePaths", (), {"workdir": "/work/project"})()
        completed = type(
            "Completed",
            (),
            {
                "stdout": "CHECK_OK|/models/base.pt|exists\n",
                "returncode": 0,
            },
        )()

        with patch("launcher.preflight._run_ssh_capture", return_value=completed):
            result = run_preflight_for_job(
                settings,
                remote_paths,
                "train",
                ["/models/base.pt"],
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.checks[0].remote_path, "/models/base.pt")
