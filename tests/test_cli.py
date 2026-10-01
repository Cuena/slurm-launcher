from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher import cli
from launcher.config_utils import validate_predefined_sbatch_file_job
from launcher.core import JobSpec
from tests.helpers import make_settings


class CliTests(unittest.TestCase):
    def test_bare_invocation_never_launches(self) -> None:
        with patch.dict(
            cli.COMMAND_HANDLERS,
            {"run": lambda args: self.fail("unexpected submission")},
        ):
            with (
                patch("sys.stdout", io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                cli.main([])
        self.assertEqual(raised.exception.code, 0)

    def test_ambiguous_launch_selection_is_rejected(self) -> None:
        with (
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.parse_args(["run", "--all", "--only", "train"])
        self.assertEqual(raised.exception.code, 2)

    def test_frozen_submit_requires_an_explicit_run(self) -> None:
        with (
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.parse_args(["submit", "--json"])
        self.assertEqual(raised.exception.code, 2)

    def test_sbatch_cannot_escape_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = make_settings(project_root=Path(tmpdir))
            job = JobSpec(name="shared", sbatch_file="../shared/train.sbatch")
            with self.assertRaises(SystemExit):
                validate_predefined_sbatch_file_job(settings, job)
