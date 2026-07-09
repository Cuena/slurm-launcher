"""Tests for preflight script generation and behavior."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from unittest import TestCase

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
        remote_paths = type("RemotePaths", (), {
            "workdir": "/work/project",
            "job_folder": "run_001",
            "logdir": "/logs/run_001",
            "slurm_output_dir": "/logs/run_001/slurm_output",
        })()
        jobs = [
            JobSpec(
                name="eval",
                command="python eval.py",
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
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["job_name"], "eval")
        self.assertIn("data/input/*.mp4", payload["jobs"][0]["requirements"])
        self.assertIn("set -euo pipefail", payload["jobs"][0]["script"])
