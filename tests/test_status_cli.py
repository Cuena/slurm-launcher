"""Reject mixed identities before accessing the cluster."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher.cli import main, parse_args
from tests.test_status import scheduler_fixture


class StatusSelectionTests(unittest.TestCase):
    def test_direct_and_tracked_targets_cannot_be_combined(self) -> None:
        with (
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            parse_args(["status", "--run", "first", "--job-id", "123"])
        self.assertEqual(raised.exception.code, 2)

    def test_tracked_status_rejects_context_override(self) -> None:
        with patch("launcher.cli.console.print_json") as output:
            result = main(
                [
                    "status",
                    "--run",
                    "first",
                    "--cluster-login",
                    "wrong-cluster",
                    "--json",
                ]
            )
        self.assertEqual(result, 1)
        self.assertFalse(output.call_args.kwargs["data"]["ok"])

    def test_direct_parent_query_json_retains_failed_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with scheduler_fixture(
                Path(directory),
                "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n"
                "123_1|array|FAILED|1:0|-|-|-|00:01|cpu\n",
                "",
            ):
                with patch("launcher.status.console.print_json") as output:
                    result = main(
                        [
                            "status", "--job-id", "123",
                            "--cluster-login", "fixture", "--json",
                        ]
                    )
        self.assertEqual(result, 0)
        parent = output.call_args.kwargs["data"]["jobs"][0]
        self.assertEqual(parent["derived_state"], "FAILED")
        self.assertFalse(parent["array_complete"])
        self.assertEqual(
            [(task["job_id"], task["derived_state"]) for task in parent["tasks"]],
            [("123_0", "DONE"), ("123_1", "FAILED")],
        )
