from __future__ import annotations

import argparse
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher import cli
from launcher.core import JobSpec, RemotePaths, SubmissionResult
from tests.helpers import make_settings


class CliTests(unittest.TestCase):
    def test_parse_args_supports_json_for_agent_facing_commands(self) -> None:
        commands = [
            "validate",
            "render",
            "stage",
            "submit",
            "sbatch",
            "run",
            "monitor",
            "doctor",
            "jobs",
            "job-show",
            "job-log",
        ]
        for command in commands:
            argv = ["slurm-launcher", command, "--json"]
            if command == "submit":
                argv.extend(["--job-folder", "run_001"])
            if command == "sbatch":
                argv.append("slurm/train.sbatch")
            if command in {"job-show", "job-log"}:
                argv.append("12345")
            with self.subTest(command=command):
                with patch("sys.argv", argv):
                    args = cli.parse_args()
                self.assertEqual(args.command, command)
                self.assertTrue(args.json)

    def test_validate_predefined_sbatch_job_rejects_path_outside_local_root(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(project_root=Path(tmpdir))
            job = JobSpec(name="shared", sbatch_file="../shared/train.sbatch")

            with self.assertRaisesRegex(
                SystemExit,
                "sbatch_file must stay inside LOCAL_ROOT",
            ):
                cli._validate_predefined_sbatch_file_job(settings, job)

    @patch("launcher.cli.console.print_json")
    @patch("launcher.cli._resolve_cluster_context")
    def test_doctor_reports_default_archive_dir_in_json(
        self,
        mock_resolve_cluster_context,
        mock_print_json,
    ) -> None:
        mock_resolve_cluster_context.return_value = (
            "user@cluster",
            None,
            "/dev/null",
            ["-o", "BatchMode=yes"],
            Path("/tmp/config.py"),
        )
        args = argparse.Namespace(config=None, cluster_login=None, ssh=False, json=True)

        exit_code = cli.do_doctor(args)

        self.assertEqual(exit_code, 0)
        mock_print_json.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["cluster_login"], "user@cluster")
        self.assertEqual(payload["ssh_config_file"], "/dev/null")
        self.assertEqual(payload["ssh_options"], ["-o", "BatchMode=yes"])
        self.assertEqual(payload["archive_dir_source"], "default")
        self.assertTrue(payload["archive_dir"].endswith("/.slurm-dashboard/logs"))

    @patch("launcher.cli.test_ssh_connection")
    @patch("launcher.cli.ssh_script")
    @patch("launcher.cli.format_sbatch_options")
    @patch("launcher.cli.resolve_remote_paths")
    @patch("launcher.cli.prepare_jobs")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli.load_config")
    @patch("launcher.cli._resolve_config_path")
    @patch("launcher.cli.console.print_json")
    def test_validate_json_success_payload(
        self,
        mock_print_json,
        mock_resolve_config_path,
        mock_load_config,
        mock_build_settings,
        mock_prepare_jobs,
        mock_resolve_remote_paths,
        mock_format_sbatch_options,
        mock_ssh_script,
        mock_test_ssh_connection,
    ) -> None:
        settings = make_settings(
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )
        config_path = Path("/tmp/config.py")
        jobs = [JobSpec(name="train", command="python train.py")]
        remote_paths = RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )
        mock_resolve_config_path.return_value = config_path
        mock_load_config.return_value = object()
        mock_build_settings.return_value = settings
        mock_prepare_jobs.return_value = jobs
        mock_resolve_remote_paths.return_value = remote_paths
        mock_format_sbatch_options.return_value = {"job-name": "train"}
        mock_ssh_script.return_value = ("OK\n", "")
        args = argparse.Namespace(
            config=None,
            workspace=None,
            only=None,
            ssh=True,
            check_remote_paths=True,
            json=True,
        )

        exit_code = cli.do_validate(args)

        self.assertEqual(exit_code, 0)
        mock_test_ssh_connection.assert_called_once()
        mock_print_json.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["config_path"], str(config_path))
        self.assertEqual(payload["workspace_mode"], "per-run")
        self.assertEqual(payload["selected_jobs"], ["train"])
        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["ssh_checked"], True)
        self.assertEqual(payload["remote_checks"]["ok"], True)

    @patch("launcher.cli.console.print_json")
    def test_validate_json_failure_payload(self, mock_print_json) -> None:
        args = argparse.Namespace(
            config=None,
            workspace=None,
            only=None,
            ssh=False,
            check_remote_paths=True,
            json=True,
        )

        exit_code = cli.do_validate(args)

        self.assertEqual(exit_code, 1)
        mock_print_json.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], False)
        self.assertEqual(payload["ssh_checked"], False)
        self.assertEqual(payload["remote_checks"]["requested"], True)
        self.assertEqual(
            payload["errors"],
            ["ERROR: --check-remote-paths requires --ssh."],
        )

    @patch("launcher.cli.build_sbatch_script")
    @patch("launcher.cli.build_job_script")
    @patch("launcher.cli.format_sbatch_options")
    @patch("launcher.cli.build_predefined_sbatch_command")
    @patch("launcher.cli.resolve_remote_paths")
    @patch("launcher.cli._validate_predefined_sbatch_jobs")
    @patch("launcher.cli.prepare_jobs")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli.load_config")
    @patch("launcher.cli._resolve_config_path")
    @patch("launcher.cli.console.print_json")
    def test_render_json_payload(
        self,
        mock_print_json,
        mock_resolve_config_path,
        mock_load_config,
        mock_build_settings,
        mock_prepare_jobs,
        mock_validate_predefined_jobs,
        mock_resolve_remote_paths,
        mock_build_predefined_sbatch_command,
        mock_format_sbatch_options,
        mock_build_job_script,
        mock_build_sbatch_script,
    ) -> None:
        settings = make_settings()
        config_path = Path("/tmp/config.py")
        jobs = [
            JobSpec(name="train", command="python train.py"),
            JobSpec(name="shared", sbatch_file="slurm/shared.sbatch"),
        ]
        remote_paths = RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )
        mock_resolve_config_path.return_value = config_path
        mock_load_config.return_value = object()
        mock_build_settings.return_value = settings
        mock_prepare_jobs.return_value = jobs
        mock_resolve_remote_paths.return_value = remote_paths
        mock_build_predefined_sbatch_command.return_value = (
            "/remote/workspaces/project_001/slurm/shared.sbatch",
            "sbatch /remote/workspaces/project_001/slurm/shared.sbatch",
        )
        mock_format_sbatch_options.return_value = {"job-name": "train"}
        mock_build_job_script.return_value = "#!/bin/bash\npython train.py\n"
        mock_build_sbatch_script.return_value = (
            "#!/bin/bash\n#SBATCH --job-name=train\n"
        )
        args = argparse.Namespace(
            config=None,
            workspace=None,
            only=None,
            job_script=False,
            json=True,
        )

        exit_code = cli.do_render(args)

        self.assertEqual(exit_code, 0)
        mock_validate_predefined_jobs.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["selected_jobs"], ["train", "shared"])
        self.assertIn("train", payload["job_scripts"])
        self.assertIn("train", payload["sbatch_scripts"])
        self.assertEqual(payload["rendered_jobs"][1]["job_type"], "sbatch_file")

    @patch("launcher.cli.sync_project")
    @patch("launcher.cli.test_ssh_connection")
    @patch("launcher.cli.resolve_remote_paths")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli._load_run_config")
    @patch("launcher.cli.console.print_json")
    def test_stage_json_payload_includes_commands(
        self,
        mock_print_json,
        mock_load_run_config,
        mock_build_settings,
        mock_resolve_remote_paths,
        mock_test_ssh_connection,
        mock_sync_project,
    ) -> None:
        settings = make_settings()
        remote_paths = RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )
        mock_load_run_config.return_value = (object(), Path("/tmp/config.py"))
        mock_build_settings.return_value = settings
        mock_resolve_remote_paths.return_value = remote_paths
        mock_sync_project.return_value = [
            "ssh user@cluster mkdir -p /remote/workspaces/project_001",
            "rsync -az ./ user@cluster:/remote/workspaces/project_001/",
        ]
        args = argparse.Namespace(config=None, workspace=None, dry_run=True, json=True)

        exit_code = cli.do_stage(args)

        self.assertEqual(exit_code, 0)
        mock_test_ssh_connection.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["job_folder"], "project_001")
        self.assertEqual(len(payload["commands"]), 2)
        self.assertEqual(payload["dry_run"], True)

    @patch("launcher.cli.submit_job")
    @patch("launcher.cli.sync_project")
    @patch("launcher.cli.test_ssh_connection")
    @patch("launcher.cli.resolve_remote_paths")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli._load_run_config")
    @patch("launcher.cli.console.print_json")
    def test_sbatch_json_payload(
        self,
        mock_print_json,
        mock_load_run_config,
        mock_build_settings,
        mock_resolve_remote_paths,
        mock_test_ssh_connection,
        mock_sync_project,
        mock_submit_job,
    ) -> None:
        settings = make_settings()
        remote_paths = RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )
        submission = SubmissionResult(
            job_id="dry-run",
            sbatch_command="sbatch /remote/workspaces/project_001/slurm/train.sbatch",
            sbatch_options={},
            remote_sbatch_path="/remote/workspaces/project_001/slurm/train.sbatch",
            commands=["ssh user@cluster 'sbatch /remote/workspaces/project_001/slurm/train.sbatch'"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            sbatch_file = Path(tmpdir) / "train.sbatch"
            sbatch_file.write_text("#!/bin/bash\necho train\n", encoding="utf-8")
            settings = make_settings(project_root=Path(tmpdir))
            mock_load_run_config.return_value = (object(), Path("/tmp/config.py"))
            mock_build_settings.return_value = settings
            mock_resolve_remote_paths.return_value = remote_paths
            mock_sync_project.return_value = ["rsync dry-run"]
            mock_submit_job.return_value = submission
            args = argparse.Namespace(
                config=None,
                workspace=None,
                sbatch_file=str(sbatch_file),
                name="train",
                sbatch_arg=[],
                dry_run=True,
                json=True,
            )

            exit_code = cli.do_sbatch(args)

        self.assertEqual(exit_code, 0)
        mock_test_ssh_connection.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["selected_jobs"], ["train"])
        self.assertEqual(payload["tracking_file"], None)
        self.assertEqual(len(payload["commands"]), 2)
        self.assertIn("squeue -u $USER", payload["monitor_command"])

    @patch("launcher.cli.write_job_tracking_file")
    @patch("launcher.cli._collect_submission_results")
    @patch("launcher.cli.test_ssh_connection")
    @patch("launcher.cli.resolve_remote_paths_for_job_folder")
    @patch("launcher.cli._validate_predefined_sbatch_jobs")
    @patch("launcher.cli.prepare_jobs")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli._load_run_config")
    @patch("launcher.cli.console.print_json")
    def test_submit_json_payload(
        self,
        mock_print_json,
        mock_load_run_config,
        mock_build_settings,
        mock_prepare_jobs,
        mock_validate_predefined_jobs,
        mock_resolve_remote_paths,
        mock_test_ssh_connection,
        mock_collect_submission_results,
        mock_write_job_tracking_file,
    ) -> None:
        settings = make_settings(
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )
        job = JobSpec(name="train", command="python train.py")
        remote_paths = RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )
        submitted_jobs = [
            {
                "job_name": "train",
                "job_id": "12345",
                "stdout": "/remote/train-12345.out",
                "stderr": "/remote/train-12345.err",
            }
        ]
        mock_load_run_config.return_value = (object(), Path("/tmp/config.py"))
        mock_build_settings.return_value = settings
        mock_prepare_jobs.return_value = [job]
        mock_resolve_remote_paths.return_value = remote_paths
        mock_collect_submission_results.return_value = (
            submitted_jobs,
            submitted_jobs,
            ["ssh user@cluster 'sbatch /remote/train.sbatch'"],
        )
        mock_write_job_tracking_file.return_value = Path(
            "/tmp/slurm_output/project_001/jobs.json"
        )
        args = argparse.Namespace(
            config=None,
            workspace=None,
            only=None,
            job_folder="project_001",
            dry_run=False,
            json=True,
        )

        exit_code = cli.do_submit(args)

        self.assertEqual(exit_code, 0)
        mock_validate_predefined_jobs.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(
            payload["tracking_file"], "/tmp/slurm_output/project_001/jobs.json"
        )
        self.assertEqual(payload["submitted_jobs"][0]["job_id"], "12345")
        self.assertIn("squeue -j 12345", payload["monitor_command"])

    @patch("launcher.cli._collect_submission_results")
    @patch("launcher.cli.sync_project")
    @patch("launcher.cli.test_ssh_connection")
    @patch("launcher.cli.resolve_remote_paths")
    @patch("launcher.cli._validate_predefined_sbatch_jobs")
    @patch("launcher.cli.prepare_jobs")
    @patch("launcher.cli.build_settings")
    @patch("launcher.cli._load_run_config")
    @patch("launcher.cli.console.print_json")
    def test_run_json_dry_run_payload(
        self,
        mock_print_json,
        mock_load_run_config,
        mock_build_settings,
        mock_prepare_jobs,
        mock_validate_predefined_jobs,
        mock_resolve_remote_paths,
        mock_test_ssh_connection,
        mock_sync_project,
        mock_collect_submission_results,
    ) -> None:
        settings = make_settings()
        job = JobSpec(name="train", command="python train.py")
        remote_paths = RemotePaths(
            job_folder="project_001",
            workdir="/remote/workspaces/project_001",
            logdir="/remote/logs/project_001",
            slurm_output_dir="/remote/logs/project_001/slurm_output",
        )
        submitted_jobs = [{"job_name": "train", "job_id": "dry-run"}]
        mock_load_run_config.return_value = (object(), Path("/tmp/config.py"))
        mock_build_settings.return_value = settings
        mock_prepare_jobs.return_value = [job]
        mock_resolve_remote_paths.return_value = remote_paths
        mock_sync_project.return_value = [
            "ssh user@cluster mkdir -p /remote/workspaces/project_001",
            "rsync --dry-run -az ./ user@cluster:/remote/workspaces/project_001/",
        ]
        mock_collect_submission_results.return_value = (
            submitted_jobs,
            [],
            ["ssh user@cluster <<'EOF'\nsbatch /remote/train.sbatch\nEOF"],
        )
        args = argparse.Namespace(
            config=None,
            workspace=None,
            only=None,
            dry_run=True,
            json=True,
        )

        exit_code = cli.do_run(args)

        self.assertEqual(exit_code, 0)
        mock_validate_predefined_jobs.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertIsNone(payload["tracking_file"])
        self.assertEqual(len(payload["commands"]), 3)
        self.assertIn("squeue -u $USER", payload["monitor_command"])

    @patch("launcher.cli.console.print_json")
    def test_monitor_json_payload_includes_command_and_result(
        self,
        mock_print_json,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking_file = Path(tmpdir) / "jobs.json"
            tracking_file.write_text(
                (
                    '{"cluster_login":"user@cluster","ssh_config_file":"/dev/null",'
                    '"ssh_options":["-o","BatchMode=yes"],'
                    '"jobs":[{"job_name":"train","job_id":"12345"}]}'
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                tracking_file=str(tracking_file),
                only=None,
                dry_run=False,
                json=True,
            )
            with patch(
                "launcher.cli.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    args=[],
                    returncode=0,
                    stdout="QUEUE\n",
                    stderr="",
                ),
            ) as mock_subprocess_run:
                exit_code = cli.do_monitor(args)

        self.assertEqual(exit_code, 0)
        mock_subprocess_run.assert_called_once()
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["job_ids"], ["12345"])
        self.assertIn("squeue -j 12345", payload["command"])
        self.assertEqual(payload["stdout"], "QUEUE\n")

    @patch("launcher.cli.resolve_tracking_file")
    @patch("launcher.cli.console.print_json")
    def test_monitor_json_failure_payload(
        self,
        mock_print_json,
        mock_resolve_tracking_file,
    ) -> None:
        mock_resolve_tracking_file.return_value = None
        args = argparse.Namespace(
            tracking_file=None, only=None, dry_run=True, json=True
        )

        exit_code = cli.do_monitor(args)

        self.assertEqual(exit_code, 1)
        payload = mock_print_json.call_args.kwargs["data"]
        self.assertEqual(payload["ok"], False)
        self.assertIsNone(payload["tracking_file"])
        self.assertEqual(payload["job_ids"], [])


if __name__ == "__main__":
    unittest.main()
