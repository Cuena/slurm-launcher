# tests/test_command_contracts.py
# What: Covers exact dry-run command rendering for staging, submission, and tracked downloads.
# Why: Locks down the machine-readable command contract that agents rely on to preview or replay actions safely.
# RELEVANT FILES: launcher/core.py, launcher/download_logs.py, launcher/download_artifacts.py, launcher/cli.py

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from launcher.core import (
    JobSpec,
    LauncherSettings,
    RemotePaths,
    build_predefined_sbatch_command,
    submit_job,
    sync_project,
)
from launcher.download_artifacts import _run_downloads as run_artifact_downloads
from launcher.download_logs import _run_downloads as run_log_downloads


class CommandContractTests(unittest.TestCase):
    def _settings(self, **overrides: object) -> LauncherSettings:
        defaults: dict[str, object] = {
            "cluster_login": "user@cluster",
            "ssh_config_file": "/dev/null",
            "ssh_options": ["-o", "BatchMode=yes"],
            "remote_workspace_base": "/remote/workspaces",
            "remote_log_base_path": "/remote/logs",
            "workspace_mode": "per-run",
            "remote_workspace_dir": None,
            "project_root": Path("/tmp/project"),
            "project_prefix": "project",
            "venv_python_executable": None,
            "default_env": {},
            "default_sbatch": {},
            "extra_rsync_excludes": [],
            "extra_rsync_args": [],
            "remote_slurm_dashboard_log_archive_dir": None,
            "remote_slurm_dashboard_log_view_dir": None,
            "runtime_mode": "native",
            "singularity_image_path": None,
            "singularity_exec_flags": [],
            "artifact_paths": [],
            "verbose": False,
        }
        defaults.update(overrides)
        return LauncherSettings(**defaults)

    def _remote_paths(self) -> RemotePaths:
        return RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )

    def test_sync_project_dry_run_returns_exact_commands(self) -> None:
        settings = self._settings()
        commands = sync_project(
            settings,
            self._remote_paths(),
            dry_run=True,
            quiet=True,
        )

        self.assertEqual(len(commands), 2)
        self.assertIn(
            "ssh -F /dev/null -o BatchMode=yes user@cluster",
            commands[0],
        )
        self.assertIn("mkdir -p", commands[0])
        self.assertIn("rsync -az --info=progress2", commands[1])
        self.assertIn("--dry-run", commands[1])
        self.assertIn("-e 'ssh -F /dev/null -o BatchMode=yes'", commands[1])

    def test_submit_job_dry_run_returns_exact_generated_submission_command(
        self,
    ) -> None:
        settings = self._settings(default_sbatch={"time": "00:10:00"})
        job = JobSpec(name="train", command="python train.py")

        submission = submit_job(
            settings,
            self._remote_paths(),
            job,
            dry_run=True,
            quiet=True,
        )

        self.assertEqual(submission.job_id, "dry-run")
        self.assertEqual(len(submission.commands), 1)
        self.assertIn("<<'EOF'", submission.commands[0])
        self.assertIn("cat <<'SBATCH_SCRIPT' >", submission.commands[0])
        self.assertIn(
            "sbatch /remote/logs/project_001/train.sbatch", submission.commands[0]
        )

    def test_submit_job_dry_run_returns_exact_predefined_sbatch_command(self) -> None:
        settings = self._settings()
        job = JobSpec(
            name="shared",
            sbatch_file="slurm/train.sbatch",
            sbatch_args=["--export=ALL,SEED=1"],
        )

        submission = submit_job(
            settings,
            self._remote_paths(),
            job,
            dry_run=True,
            quiet=True,
        )

        self.assertEqual(submission.job_id, "dry-run")
        self.assertIn("cd /remote/workspaces/project_001", submission.commands[0])
        self.assertIn(
            "sbatch --export=ALL,SEED=1 /remote/workspaces/project_001/slurm/train.sbatch",
            submission.commands[0],
        )

    def test_build_predefined_sbatch_command_rejects_path_outside_local_root(
        self,
    ) -> None:
        settings = self._settings(project_root=Path("/tmp/project"))
        job = JobSpec(name="shared", sbatch_file="../shared/train.sbatch")

        with self.assertRaisesRegex(
            ValueError,
            "sbatch_file must stay inside LOCAL_ROOT",
        ):
            build_predefined_sbatch_command(settings, self._remote_paths(), job)

    @patch("launcher.core.create_log_view_symlinks")
    @patch("launcher.core.ssh_script")
    def test_submit_predefined_sbatch_job_resolves_logs_from_scontrol(
        self,
        mock_ssh_script,
        mock_create_log_view_symlinks,
    ) -> None:
        settings = self._settings(
            remote_slurm_dashboard_log_archive_dir="/archive/logs",
            remote_slurm_dashboard_log_view_dir="/archive/view",
        )
        job = JobSpec(name="shared", sbatch_file="slurm/train.sbatch")
        mock_ssh_script.side_effect = [
            ("Submitted batch job 12345\n", ""),
            (
                "JobId=12345 StdOut=/archive/logs/%j.out StdErr=/archive/logs/%j.err\n",
                "",
            ),
        ]

        submission = submit_job(
            settings,
            self._remote_paths(),
            job,
            dry_run=False,
            quiet=True,
        )

        self.assertEqual(submission.job_id, "12345")
        self.assertEqual(submission.sbatch_options["output"], "/archive/logs/12345.out")
        self.assertEqual(submission.sbatch_options["error"], "/archive/logs/12345.err")
        mock_create_log_view_symlinks.assert_called_once()

    @patch("builtins.print")
    def test_download_logs_dry_run_prints_exact_rsync_command(
        self,
        mock_print,
    ) -> None:
        failures = run_log_downloads(
            "user@cluster",
            [("train", "stdout", "/remote/logs/train.out")],
            Path("/tmp/downloaded_logs"),
            dry_run=True,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(failures, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn(
            "rsync -az -e 'ssh -F /dev/null -o BatchMode=yes' --dry-run", printed
        )
        self.assertIn("user@cluster:/remote/logs/train.out", printed)

    @patch("builtins.print")
    def test_download_artifacts_dry_run_prints_exact_rsync_command(
        self,
        mock_print,
    ) -> None:
        failures = run_artifact_downloads(
            "user@cluster",
            "/remote/workspaces/project_001",
            ["outputs/model.ckpt"],
            Path("/tmp/downloaded_artifacts"),
            dry_run=True,
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )

        self.assertEqual(failures, 0)
        printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
        self.assertIn(
            "rsync -az -e 'ssh -F /dev/null -o BatchMode=yes' --dry-run", printed
        )
        self.assertIn(
            "user@cluster:/remote/workspaces/project_001/outputs/model.ckpt", printed
        )


if __name__ == "__main__":
    unittest.main()
