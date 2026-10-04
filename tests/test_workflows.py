from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher import execution
from tests.helpers import LocalScheduler, write_tracking_file


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


class FrozenExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cluster = LocalScheduler(self.root)
        self.config = self.root / "config.py"
        self.config.write_text(
            f"LOCAL_ROOT = {str(self.root)!r}\nCLUSTER_LOGIN = 'user@cluster'\n"
            f"REMOTE_WORKSPACE_BASE = {str(self.root / 'remote-work')!r}\n"
            f"REMOTE_LOG_BASE_PATH = {str(self.root / 'remote-logs')!r}\n"
            "RUN_JOBS = ['train', 'eval']\n"
            "JOBS = [{'name': 'train', 'command': 'echo frozen', 'requires': ['input.dat']}, {'name': 'eval', 'command': 'true'}]\n"
        )
        self.addCleanup(patch.stopall)
        patch("launcher.execution.test_ssh_connection").start()
        patch("launcher.execution.sync_project", return_value=["rsync planned"]).start()
        self.output = patch("launcher.execution.console.print_json").start()
        patch("launcher.core.transport.run_ssh_capture", side_effect=self.cluster.capture).start()
        patch("launcher.job_tools.run_ssh_capture", side_effect=self.cluster.capture).start()

    def args(self, **values):
        return argparse.Namespace(config=str(self.config), json=True, **values)

    def stage(self, **values) -> Path:
        self.assertEqual(execution.do_stage(self.args(**values)), 0)
        Path(self.payload()["remote_workdir"]).mkdir(parents=True, exist_ok=True)
        return Path(self.output.call_args.kwargs["data"]["tracking_file"])

    def payload(self):
        return self.output.call_args.kwargs["data"]

    def test_selection_is_required_before_prepare_or_sync(self) -> None:
        self.config.write_text(
            self.config.read_text().replace(
                "RUN_JOBS = ['train', 'eval']", "RUN_JOBS = []"
            )
            + "from pathlib import Path\ndef prepare():\n    Path(LOCAL_ROOT, 'prepare-ran').touch()\n"
        )
        self.assertEqual(execution.do_run(self.args(dry_run=False)), 1)
        self.assertFalse((self.root / "prepare-ran").exists())
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
        ):
            self.assertEqual(
                execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 0
            )
        self.assertEqual(self.payload()["job_ids"], ["7001"])
        self.assertEqual(self.payload()["selected_jobs"], ["train"])
        self.assertEqual(self.cluster.job_ids(), ["7001"])
        self.assertEqual(
            json.loads(tracking.read_text())["jobs"][0]["state"], "submitted"
        )

    def test_partial_acknowledgment_is_durable_before_next_dispatch(self) -> None:
        tracking = self.stage()
        calls = 0

        def before_dispatch():
            nonlocal calls
            records = json.loads(tracking.read_text())["jobs"]
            calls += 1
            if calls == 1:
                self.assertEqual(records[0]["state"], "submitting")
                return
            self.assertEqual(records[0]["job_id"], "7001")
            self.assertEqual(records[0]["state"], "submitted")
            self.assertEqual(records[1]["state"], "submitting")
            self.cluster.disconnect_after_dispatch = True

        self.cluster.before_dispatch = before_dispatch
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
        )
        self.assertEqual(self.payload()["job_ids"], ["7001"])
        records = json.loads(tracking.read_text())["jobs"]
        self.assertEqual([record["state"] for record in records], ["submitted", "unknown"])
        self.assertIn("7002", records[1]["submission_stdout"])
        self.assertIn("connection closed", records[1]["submission_stderr"])
        self.cluster.before_dispatch = None
        for only in (["train"], ["eval"]):
            self.assertEqual(
                execution.do_submit(
                    argparse.Namespace(run=str(tracking), only=only, json=True)
                ), 1,
            )
        self.assertEqual(self.cluster.job_ids(), ["7001", "7002"])
        self.assertEqual(json.loads(tracking.read_text())["jobs"], records)

    def test_confirmed_failure_can_retry_without_losing_previous_jobs(self) -> None:
        tracking = self.stage()
        self.cluster.set_modes("accept", "reject", "accept")
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
        )
        rejected = json.loads(tracking.read_text())["jobs"][1]
        self.assertEqual(rejected["state"], "failed")
        self.assertIn("request rejected", rejected["submission_stderr"])
        self.assertEqual(
            execution.do_submit(
                argparse.Namespace(run=str(tracking), only=["eval"], json=True)
            ), 0,
        )
        records = json.loads(tracking.read_text())["jobs"]
        self.assertEqual([record["job_id"] for record in records], ["7001", "7002"])
        self.assertEqual(records[1]["attempts"][0]["state"], "failed")

    def test_acknowledgment_safety_for_scheduler_output_and_exit_trap(self) -> None:
        for mode, trap, state, rc in (
            ("ambiguous", False, "unknown", 1),
            ("ambiguous_nonzero", False, "unknown", 1),
            ("accept_nonzero", False, "submitted", 0),
            ("accept", True, "submitted", 0),
        ):
            with self.subTest(mode=mode, exit_trap=trap):
                root = self.root / f"{mode}-{trap}"
                root.mkdir()
                cluster = LocalScheduler(root)
                cluster.set_modes(mode)
                cluster.trap_exit = trap
                tracking = self.stage(only=["train"])
                args = argparse.Namespace(run=str(tracking), json=True)
                with patch("launcher.core.transport.run_ssh_capture", side_effect=cluster.capture):
                    self.assertEqual(execution.do_submit(args), rc)
                    record = json.loads(tracking.read_text())["jobs"][0]
                    self.assertEqual(record["state"], state)
                    if state == "submitted":
                        self.assertEqual(record["job_id"], "7001")
                    else:
                        self.assertIn("7001", record["submission_stdout"])
                    self.assertEqual(execution.do_submit(args), 1)
                self.assertEqual(cluster.job_ids(), ["7001"])

    def test_missing_frame_remains_unknown_for_any_transport_exit(self) -> None:
        self.cluster.disconnect_after_dispatch = True
        for returncode in (0, 1, 255):
            with self.subTest(returncode=returncode):
                tracking = self.stage(only=["train"])
                self.cluster.disconnect_returncode = returncode
                self.assertEqual(
                    execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
                )
                record = json.loads(tracking.read_text())["jobs"][0]
                self.assertEqual(record["state"], "unknown")
                self.assertEqual(record["submission_returncode"], returncode)
                self.assertEqual(
                    execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
                )
        self.assertEqual(self.cluster.job_ids(), ["7001", "7002", "7003"])

    def test_relocated_bundle_receives_acknowledgment_under_selected_lock(self) -> None:
        tracking = self.stage(only=["train"])
        latest = tracking.parent.parent / "latest_jobs.json"
        original_latest = latest.read_bytes()
        recovery = self.root / "recovery" / "recovered-bundle"
        recovery.parent.mkdir()
        tracking.parent.rename(recovery)
        relocated = recovery / "jobs.json"
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(relocated), json=True)), 0
        )
        record = json.loads(relocated.read_text())["jobs"][0]
        self.assertEqual((record["state"], record["job_id"]), ("submitted", "7001"))
        self.assertEqual(self.payload()["tracking_file"], str(relocated))
        self.assertEqual(self.payload()["run_id"], tracking.parent.name)
        self.assertFalse(tracking.exists())
        self.assertEqual(latest.read_bytes(), original_latest)
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(relocated), json=True)), 1
        )
        self.assertEqual(self.cluster.job_ids(), ["7001"])

    def test_interruption_after_acknowledgment_never_overwrites_id(self) -> None:
        source = self.root / "hand.sbatch"
        source.write_text("#!/bin/bash\ntrue\n")
        self.config.write_text(
            self.config.read_text()
            + "\nJOBS = [{'name': 'hand', 'sbatch_file': 'hand.sbatch'}]\nRUN_JOBS = ['hand']\n"
        )
        tracking = self.stage()
        with patch(
            "launcher.job_tools.resolve_job_log_info",
            side_effect=KeyboardInterrupt(),
        ):
            self.assertEqual(
                execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
            )
        record = json.loads(tracking.read_text())["jobs"][0]
        self.assertEqual((record["state"], record["job_id"]), ("submitted", "7001"))
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
        )
        self.assertEqual(self.cluster.job_ids(), ["7001"])

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
                argparse.Namespace(run=str(tracking), json=True)
            ),
            0,
        )
        self.assertEqual(self.cluster.job_ids(), ["7001"])
        submitted_path = Path(json.loads(tracking.read_text())["jobs"][0]["remote_sbatch"])
        self.assertEqual(submitted_path.read_bytes(), script)

    def test_legacy_tracking_cannot_submit(self) -> None:
        tracking = write_tracking_file(
            self.root / "legacy/jobs.json", FULL_TRACKING_PAYLOAD
        )
        self.assertEqual(
            execution.do_submit(argparse.Namespace(run=str(tracking), json=True)), 1
        )

    def test_obsolete_plan_is_readable_but_cannot_dispatch(self) -> None:
        tracking = self.stage(only=["train"])
        plan_path = tracking.parent / "plan.json"
        plan = json.loads(plan_path.read_text())
        plan["settings"]["local_artifact_root"] = "/obsolete/default"
        for version in (1, 2):
            with self.subTest(version=version):
                plan["version"] = version
                plan_path.write_text(json.dumps(plan))
                self.assertEqual(
                    execution.do_submit(argparse.Namespace(run=str(tracking), json=True)),
                    1,
                )
                self.assertFalse(self.payload()["ok"])
                self.assertEqual(self.cluster.job_ids(), [])
                self.assertEqual(
                    json.loads(tracking.read_text())["jobs"][0]["state"], "planned"
                )

    def test_invalid_job_contracts_fail_before_prepare_or_state_creation(self) -> None:
        original = self.config.read_text()
        invalid_jobs = (
            [{"name": "../train", "command": "true"}],
            [{"name": "train", "command": "true", "sbatch": {"chdir": "/outside"}}],
            [{"name": "train", "command": "true"}, {"name": "train", "command": "true"}],
        )
        for jobs in invalid_jobs:
            with self.subTest(jobs=jobs):
                self.config.write_text(
                    original + f"\nJOBS={jobs!r}\nRUN_JOBS={[job['name'] for job in jobs]!r}\n"
                    + "from pathlib import Path\ndef prepare():\n    Path(LOCAL_ROOT, 'prepare-ran').touch()\n"
                )
                self.assertEqual(execution.do_stage(self.args()), 1)
                self.assertFalse((self.root / "slurm_output").exists())
                self.assertFalse((self.root / "prepare-ran").exists())

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


class ProvenanceWorkflowTests(unittest.TestCase):
    def test_post_prepare_provenance_matches_frozen_and_transferred_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            subprocess.run(["git", "init", "-q", str(project)], check=True)
            cluster = LocalScheduler(root)
            cluster.install_rsync()
            config = project / "config.py"
            config.write_text(
                f"LOCAL_ROOT={str(project)!r}\nCLUSTER_LOGIN='fixture'\n"
                f"REMOTE_WORKSPACE_BASE={str(root / 'work')!r}\nREMOTE_LOG_BASE_PATH={str(root / 'logs')!r}\n"
                "RUN_JOBS=['train']\nJOBS=[{'name':'train','command':'true'}]\n"
                "from pathlib import Path\ndef prepare():\n    Path(LOCAL_ROOT,'prepared').write_text('ready')\n"
            )
            with (
                patch.dict(os.environ, {"PATH": cluster.environment["PATH"]}),
                patch("launcher.core.transport.run_ssh_capture", side_effect=cluster.capture),
                patch("launcher.execution.console.print_json") as output,
            ):
                self.assertEqual(
                    execution.do_stage(argparse.Namespace(config=str(config), json=True)), 0
                )
            payload = output.call_args.kwargs["data"]
            frozen = json.loads((Path(payload["tracking_file"]).parent / "plan.json").read_text())
            remote = json.loads((Path(payload["remote_workdir"]) / ".slurm_run/source.json").read_text())
            self.assertEqual(frozen["provenance"], remote)
            self.assertIn("prepared", remote["git"]["untracked_files"])
            self.assertFalse(any(path.startswith("slurm_output/") for path in remote["git"]["untracked_files"]))


if __name__ == "__main__":
    unittest.main()
