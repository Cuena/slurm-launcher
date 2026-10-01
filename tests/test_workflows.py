from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher import execution
from launcher.artifacts import run_artifacts
from launcher.download_logs import run_download_logs
from launcher.job_tools import resolve_job_log_info
from tests.helpers import write_tracking_file


FULL_TRACKING_PAYLOAD = {
    "cluster_login": "user@cluster",
    "rsync_login": "user@transfer",
    "job_folder": "project_001",
    "remote_workdir": "/remote/work/project_001",
    "artifact_paths": ["outputs/model.ckpt"],
    "jobs": [
        {
            "job_name": "train",
            "job_id": "12345",
            "stdout": "/logs/train.out",
            "stderr": "/logs/train.err",
        },
        {"job_name": "eval", "job_id": "12346", "stdout": "/logs/eval.out"},
    ],
}


class DownloadWorkflowTests(unittest.TestCase):
    def test_log_download_selection_uses_transfer_host(self) -> None:
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
            with patch("builtins.print") as output:
                self.assertEqual(run_download_logs(args), 0)
            payload = json.loads(output.call_args.args[0])
        commands = "\n".join(payload["commands"])
        self.assertIn("user@transfer:/logs/train.out", commands)
        self.assertIn("user@transfer:/logs/train.err", commands)
        self.assertNotIn("eval.out", commands)

    def test_artifact_override_and_job_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            with patch("builtins.print") as output:
                self.assertEqual(
                    run_artifacts(
                        subcommand="download",
                        tracking_file=str(tracking),
                        selected_jobs=["train"],
                        artifact_paths=["custom/path"],
                        output_dir=str(Path(tmpdir) / "out"),
                        dry_run=True,
                        json_output=True,
                    ),
                    0,
                )
            payload = json.loads(output.call_args.args[0])
        commands = "\n".join(payload["commands"])
        self.assertIn("user@transfer:/remote/work/project_001/custom/path", commands)
        self.assertNotIn("outputs/model.ckpt", commands)
        self.assertEqual(
            {entry["job_name"] for entry in payload["artifacts"]}, {"train"}
        )

    def test_artifact_download_uses_saved_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json", FULL_TRACKING_PAYLOAD
            )
            with patch("builtins.print") as output:
                self.assertEqual(
                    run_artifacts(
                        subcommand="download",
                        tracking_file=str(tracking),
                        selected_jobs=["train"],
                        artifact_paths=None,
                        output_dir=str(Path(tmpdir) / "out"),
                        dry_run=True,
                        json_output=True,
                    ),
                    0,
                )
            payload = json.loads(output.call_args.args[0])
        self.assertIn("outputs/model.ckpt", payload["commands"][0])

    def test_invalid_tracking_cannot_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json",
                {**FULL_TRACKING_PAYLOAD, "remote_workdir": ""},
            )
            with patch("builtins.print") as output:
                self.assertEqual(
                    run_artifacts(
                        subcommand="download",
                        tracking_file=str(tracking),
                        artifact_paths=["x"],
                        dry_run=True,
                        json_output=True,
                    ),
                    1,
                )
            self.assertFalse(json.loads(output.call_args.args[0])["ok"])


class FrozenExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / "config.py"
        self.config.write_text(
            f"LOCAL_ROOT = {str(self.root)!r}\nCLUSTER_LOGIN = 'user@cluster'\n"
            "REMOTE_WORKSPACE_BASE = '/work'\nREMOTE_LOG_BASE_PATH = '/logs'\n"
            "RUN_JOBS = ['train', 'eval']\n"
            "JOBS = [{'name': 'train', 'command': 'echo frozen', 'requires': ['input.dat']}, {'name': 'eval', 'command': 'true'}]\n"
        )
        self.addCleanup(patch.stopall)
        patch("launcher.execution.test_ssh_connection").start()
        patch("launcher.execution.sync_project", return_value=["rsync planned"]).start()
        self.output = patch("launcher.execution.console.print_json").start()

    def args(self, **values):
        return argparse.Namespace(config=str(self.config), json=True, **values)

    def stage(self, **values) -> Path:
        self.assertEqual(execution.do_stage(self.args(**values)), 0)
        return Path(self.output.call_args.kwargs["data"]["tracking_file"])

    def payload(self):
        return self.output.call_args.kwargs["data"]

    def test_selection_is_required_before_prepare_or_sync(self) -> None:
        self.config.write_text(
            self.config.read_text().replace(
                "RUN_JOBS = ['train', 'eval']", "RUN_JOBS = []"
            )
            + "def prepare():\n    raise AssertionError('must not run')\n"
        )
        self.assertEqual(execution.do_run(self.args(dry_run=False)), 1)
        self.assertIn("Select jobs", self.payload()["error"])
        self.assertFalse((self.root / "slurm_output").exists())
        self.assertEqual(execution.do_stage(self.args(all_jobs=True, dry_run=True)), 0)

    def test_frozen_submit_never_imports_config_and_preserves_selection(self) -> None:
        tracking = self.stage(only=["train"])
        self.config.unlink()
        with (
            patch(
                "launcher.execution.load_config",
                side_effect=AssertionError("config imported"),
            ),
            patch("launcher.core.ssh_script", return_value=("12345\n", "")) as dispatch,
        ):
            self.assertEqual(
                execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 0
            )
        self.assertEqual(self.payload()["job_ids"], ["12345"])
        self.assertEqual(self.payload()["selected_jobs"], ["train"])
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(
            json.loads(tracking.read_text())["jobs"][0]["state"], "submitted"
        )

    def test_partial_acknowledgment_is_durable_before_next_dispatch(self) -> None:
        tracking = self.stage()
        calls = 0

        def dispatch(*args, **kwargs):
            nonlocal calls
            records = json.loads(tracking.read_text())["jobs"]
            calls += 1
            if calls == 1:
                self.assertEqual(records[0]["state"], "submitting")
                return "12345\n", ""
            self.assertEqual(records[0]["job_id"], "12345")
            self.assertEqual(records[0]["state"], "submitted")
            self.assertEqual(records[1]["state"], "submitting")
            raise subprocess.CalledProcessError(255, ["ssh"], stderr="disconnected")

        with patch("launcher.core.ssh_script", side_effect=dispatch):
            self.assertEqual(
                execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
            )
        self.assertEqual(self.payload()["job_ids"], ["12345"])
        records = json.loads(tracking.read_text())["jobs"]
        self.assertEqual(
            [record["state"] for record in records], ["submitted", "unknown"]
        )
        for only in (["train"], ["eval"]):
            with patch(
                "launcher.core.ssh_script",
                side_effect=AssertionError("duplicate submission"),
            ):
                self.assertEqual(
                    execution.do_submit(
                        argparse.Namespace(run=str(tracking), only=only, json=True)
                    ),
                    1,
                )
        self.assertEqual(json.loads(tracking.read_text())["jobs"], records)

    def test_confirmed_failure_can_retry_without_losing_previous_jobs(self) -> None:
        tracking = self.stage()
        with patch(
            "launcher.core.ssh_script",
            side_effect=[
                ("12345\n", ""),
                subprocess.CalledProcessError(1, ["ssh"], stderr="sbatch rejected"),
            ],
        ):
            self.assertEqual(
                execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
            )
        with patch("launcher.core.ssh_script", return_value=("12346\n", "")):
            self.assertEqual(
                execution.do_submit(
                    argparse.Namespace(run=str(tracking), only=["eval"], json=True)
                ),
                0,
            )
        records = json.loads(tracking.read_text())["jobs"]
        self.assertEqual([record["job_id"] for record in records], ["12345", "12346"])
        self.assertEqual(records[1]["attempts"][0]["state"], "failed")

    def test_ambiguous_success_output_is_unknown_not_retryable(self) -> None:
        tracking = self.stage(only=["train"])
        with patch("launcher.core.ssh_script", return_value=("warning\n12345\n", "")):
            self.assertEqual(
                execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
            )
        self.assertEqual(
            json.loads(tracking.read_text())["jobs"][0]["state"], "unknown"
        )

    def test_prepare_runs_only_for_real_staging_before_snapshot(self) -> None:
        self.config.write_text(
            self.config.read_text()
            + "\nfrom pathlib import Path\ndef prepare():\n    Path(LOCAL_ROOT, 'prepared').write_text('ready')\n"
        )
        self.assertEqual(execution.do_stage(self.args(dry_run=True)), 0)
        self.assertFalse((self.root / "prepared").exists())
        self.stage()
        self.assertEqual((self.root / "prepared").read_text(), "ready")

    def test_handwritten_snapshot_survives_local_file_removal(self) -> None:
        script = b"#!/bin/bash\r\necho 'SBATCH_SCRIPT'\r\n\r\n"
        source = self.root / "hand.sbatch"
        source.write_bytes(script)
        self.config.write_text(
            self.config.read_text()
            + "\nJOBS = [{'name': 'hand', 'sbatch_file': 'hand.sbatch'}]\nRUN_JOBS = ['hand']\n"
        )
        tracking = self.stage()
        self.assertEqual((tracking.parent / "scripts/0000.sbatch").read_bytes(), script)
        source.unlink()
        self.config.unlink()
        self.assertEqual(
            execution.do_submit(
                argparse.Namespace(run=str(tracking), dry_run=True, json=True)
            ),
            0,
        )
        command = self.payload()["commands"][0]
        transfer = next(
            line for line in command.splitlines() if line.startswith("printf %s ")
        )
        self.assertEqual(base64.b64decode(shlex.split(transfer)[2]), script)

    def test_legacy_tracking_cannot_submit(self) -> None:
        tracking = write_tracking_file(
            self.root / "legacy/jobs.json", FULL_TRACKING_PAYLOAD
        )
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
        )
        self.assertIn("no frozen plan", self.payload()["error"])

    def test_preflight_uses_frozen_requirements_without_config(self) -> None:
        tracking = self.stage(only=["train"])
        self.config.unlink()
        with patch("launcher.preflight.console.print_json") as output:
            self.assertEqual(
                execution.do_preflight(
                    argparse.Namespace(run=str(tracking), dry_run=True, json=True)
                ),
                0,
            )
        self.assertEqual(
            output.call_args.kwargs["data"]["jobs"][0]["requirements"], ["input.dat"]
        )


class JobLogProbeTests(unittest.TestCase):
    @patch("launcher.job_tools._run_ssh_capture")
    def test_failed_probes_do_not_claim_verified_logs(self, probe) -> None:
        probe.side_effect = [
            subprocess.CompletedProcess([], 1, "", "err"),
            subprocess.CompletedProcess([], 1, "", "err"),
        ]
        info = resolve_job_log_info("user@cluster", "99999", archive_dir=None)
        self.assertFalse(info.verified)
        self.assertIsNone(info.stdout)
        self.assertEqual(len(info.probe_errors), 2)


if __name__ == "__main__":
    unittest.main()
