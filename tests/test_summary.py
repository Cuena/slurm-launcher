"""Summary reads tracking without executing config or rewriting run metadata."""

import json
import tempfile
from pathlib import Path
from unittest import TestCase, mock

from launcher.cli import main
from launcher.status import StatusQueryResult


class SummaryTests(TestCase):
    def test_summary_preserves_tracking_without_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tracking = root / "jobs.json"
            tracking.write_text(
                json.dumps(
                    {
                        "cluster_login": "fixture",
                        "job_folder": "original-run",
                        "remote_workdir": "/original/work",
                        "jobs": [{"job_name": "train", "job_id": "123"}],
                    }
                )
            )
            before = tracking.read_bytes()
            with mock.patch(
                "launcher.summary.query_job_statuses",
                return_value=StatusQueryResult([], [], []),
            ):
                with mock.patch("launcher.summary.console.print_json") as output:
                    result = main(["summary", "--run", str(tracking), "--json"])
            self.assertEqual(result, 0)
            self.assertEqual(
                output.call_args.kwargs["data"]["remote_workdir"], "/original/work"
            )
            self.assertEqual(tracking.read_bytes(), before)
            self.assertEqual(set(root.iterdir()), {tracking})
