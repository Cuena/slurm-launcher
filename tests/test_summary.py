"""Summary reads tracking without executing config or rewriting run metadata."""

import json
import os
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from launcher.cli import main
from tests.test_status import scheduler_fixture


class SummaryTests(TestCase):
    def test_summary_exposes_array_evidence_without_executing_config_or_writing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracking = root / "jobs.json"
            tracking.write_text(
                json.dumps(
                    {
                        "cluster_login": "fixture",
                        "job_folder": "original-run",
                        "remote_workdir": "/original/work",
                        "jobs": [
                            {"job_name": "train", "job_id": "123", "array_spec": "0-1"}
                        ],
                    }
                )
            )
            config_dir = root / ".slurm"
            config_dir.mkdir()
            (config_dir / "remote_launcher_config.py").write_text(
                "raise RuntimeError('summary must not execute project configuration')\n"
            )
            accounting = (
                "123_0|train|COMPLETED|0:0|-|-|-|00:01|cpu\n"
                "123_1|train|FAILED|1:0|-|-|-|00:01|cpu\n"
            )
            before = tracking.read_bytes()
            with scheduler_fixture(root, accounting, ""):
                existing = {path.relative_to(root) for path in root.rglob("*")}
                previous_cwd = Path.cwd()
                try:
                    os.chdir(root)
                    with mock.patch("launcher.summary.console.print_json") as output:
                        result = main(["summary", "--run", str(tracking), "--json"])
                finally:
                    os.chdir(previous_cwd)
                self.assertEqual(
                    {path.relative_to(root) for path in root.rglob("*")}, existing
                )
            self.assertEqual(result, 0)
            summary = output.call_args.kwargs["data"]
            self.assertEqual(summary["statuses"][0]["derived_state"], "FAILED")
            self.assertTrue(summary["statuses"][0]["array_complete"])
            self.assertEqual(
                [(task["job_id"], task["derived_state"])
                 for task in summary["statuses"][0]["tasks"]],
                [("123_0", "DONE"), ("123_1", "FAILED")],
            )
            self.assertEqual(tracking.read_bytes(), before)
