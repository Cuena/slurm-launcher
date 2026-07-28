"""Tests for status CLI argument handling."""

from __future__ import annotations

import argparse
from unittest import TestCase
from unittest.mock import patch

from launcher.cli import do_status


class TestStatusArgs(TestCase):
    @patch("launcher.cli.run_status", return_value=0)
    def test_positional_and_flag_job_id_can_match(self, mock_run_status) -> None:
        args = argparse.Namespace(
            job_id_arg="12345",
            job_id="12345",
            tracking_file=None,
            json=True,
            cluster_login="user@cluster",
            ssh_config_file=None,
            ssh_options=[],
        )
        exit_code = do_status(args)

        self.assertEqual(exit_code, 0)
        self.assertEqual(mock_run_status.call_args.kwargs["job_id"], "12345")

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

    @patch("launcher.cli._resolve_cluster_login_from_args")
    @patch("launcher.cli.run_status", return_value=0)
    def test_tracked_status_does_not_load_unrelated_generic_config(
        self,
        mock_run_status,
        mock_resolve_cluster_login,
    ) -> None:
        args = argparse.Namespace(
            job_id_arg=None,
            job_id=None,
            tracking_file="slurm_output/run/jobs.json",
            json=True,
            config=None,
        )

        exit_code = do_status(args)

        self.assertEqual(exit_code, 0)
        mock_resolve_cluster_login.assert_not_called()
        self.assertIsNone(mock_run_status.call_args.kwargs["ssh_config_file"])
        self.assertIsNone(mock_run_status.call_args.kwargs["ssh_options"])
