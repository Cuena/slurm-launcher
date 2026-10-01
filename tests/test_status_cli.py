"""Reject mixed identities before accessing the cluster."""

import io
import unittest
from unittest.mock import patch

from launcher.cli import main, parse_args


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
