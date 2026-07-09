"""Tests for preflight script generation."""

from __future__ import annotations

from unittest import TestCase

from launcher.preflight import build_remote_check_script


class TestRemoteCheckScript(TestCase):
    def test_plain_path_and_glob(self) -> None:
        script = build_remote_check_script(
            "/work/project",
            ["data/processed", "models/*.pt"],
        )
        self.assertIn("cd /work/project", script)
        self.assertIn("CHECK_START|data/processed", script)
        self.assertIn("CHECK_START|models/*.pt", script)
        self.assertIn("compgen -G 'models/*.pt'", script)
        self.assertIn("[ -e data/processed ]", script)
