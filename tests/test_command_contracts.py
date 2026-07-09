from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from launcher.command_specs import COMMAND_NAMES, COMMAND_SPECS
from launcher.core import (
    JobSpec,
    RemotePaths,
    build_predefined_sbatch_command,
    submit_job,
    sync_project,
)
from launcher.download_artifacts import _run_downloads as run_artifact_downloads
from launcher.download_logs import _run_downloads as run_log_downloads
from tests.helpers import make_settings

_SSH_DEFAULTS = {"ssh_config_file": "/dev/null", "ssh_options": ["-o", "BatchMode=yes"]}


class CommandContractTests(unittest.TestCase):
    def _remote_paths(self) -> RemotePaths:
        return RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )

    def test_sync_project_dry_run_returns_exact_commands(self) -> None:
        settings = make_settings(**_SSH_DEFAULTS)
        commands = sync_project(
            settings,
            self._remote_paths(),
            dry_run=True,
            quiet=True,
        )

        self.assertEqual(len(commands), 3)
        self.assertIn(
            "ssh -F /dev/null -o BatchMode=yes user@cluster",
            commands[0],
        )
        self.assertIn("mkdir -p", commands[0])
        self.assertIn("rsync -az --info=progress2", commands[1])
        self.assertIn("--dry-run", commands[1])
        self.assertIn("-e 'ssh -F /dev/null -o BatchMode=yes'", commands[1])
        self.assertIn("source.json", commands[2])

    def test_command_specs_cover_public_commands(self) -> None:
        self.assertEqual(
            set(COMMAND_NAMES),
            {
                "artifacts",
                "doctor",
                "download-artifacts",
                "download-logs",
                "init",
                "job-log",
                "job-show",
                "jobs",
                "logs",
                "monitor",
                "preflight",
                "render",
                "run",
                "sbatch",
                "stage",
                "status",
                "submit",
                "summary",
                "validate",
            },
        )

    def test_json_capable_commands_declare_examples_and_fields(self) -> None:
        for name, spec in COMMAND_SPECS.items():
            with self.subTest(command=name):
                self.assertTrue(spec.examples)
                self.assertTrue(spec.agent_recommendation)
                if spec.supports_json:
                    self.assertTrue(spec.json_fields)

    def test_submit_job_dry_run_returns_exact_generated_submission_command(
        self,
    ) -> None:
        settings = make_settings(**_SSH_DEFAULTS, default_sbatch={"time": "00:10:00"})
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
        settings = make_settings(**_SSH_DEFAULTS)
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
        settings = make_settings(**_SSH_DEFAULTS, project_root=Path("/tmp/project"))
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
        settings = make_settings(
            **_SSH_DEFAULTS,
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
