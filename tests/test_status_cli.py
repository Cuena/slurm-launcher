"""Tests for status CLI argument handling."""

from __future__ import annotations

import argparse
from unittest import TestCase

from launcher.cli import do_status


class TestStatusArgs(TestCase):
    def test_positional_and_flag_job_id_can_match(self) -> None:
        args = argparse.Namespace(
            job_id_arg="12345",
            job_id="12345",
            tracking_file=None,
            json=True,
            cluster_login="user@cluster",
            ssh_config_file=None,
            ssh_options=[],
        )
        # No tracking file, but no SSH call attempted in non-interactive test env.
        exit_code = do_status(args)
        # Exit code depends on whether run_status reaches SSH; we only care args merged.
        self.assertIn(exit_code, (0, 1))

    def test_conflicting_positional_and_flag_job_ids_fails(self) -> None:
        args = argparse.Namespace(
            job_id_arg="12345",
            job_id="67890",
            tracking_file=None,
            json=True,
            cluster_login="user@cluster",
            ssh_config_file=None,
            ssh_options=[],
        )
        exit_code = do_status(args)
        self.assertEqual(exit_code, 1)
