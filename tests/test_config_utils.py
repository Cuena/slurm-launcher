"""Tests for launcher configuration normalization."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType
from unittest import TestCase

from launcher.config_utils import build_settings, configured_run_only


class TestBuildSettings(TestCase):
    def _minimal_config(self) -> ModuleType:
        config = ModuleType("test_config")
        config.CLUSTER_LOGIN = "user@cluster"
        config.WORKSPACE_MODE = "fixed"
        config.REMOTE_WORKSPACE_DIR = "/remote/project"
        config.REMOTE_LOG_BASE_PATH = "/remote/logs"
        return config

    def test_removed_copy_spelling_is_rejected(self) -> None:
        config = self._minimal_config()
        config.SYNC_SYMLINKS = "copy"

        with self.assertRaisesRegex(
            SystemExit,
            "SYNC_SYMLINKS must be one of: copy-links, preserve",
        ):
            build_settings(config, Path("config.py"))


class TestExplicitSelection(TestCase):
    def test_selection_requires_intent_and_respects_precedence(self) -> None:
        config = ModuleType("test_config")
        for run_jobs in (None, [], [" ", ""]):
            config.RUN_JOBS = run_jobs
            with self.subTest(run_jobs=run_jobs), self.assertRaises(ValueError):
                configured_run_only(config, argparse.Namespace())
        config.RUN_JOBS = ["train"]
        self.assertEqual(configured_run_only(config, argparse.Namespace()), ["train"])
        self.assertEqual(
            configured_run_only(config, argparse.Namespace(only=["eval"])), ["eval"]
        )
        self.assertIsNone(
            configured_run_only(config, argparse.Namespace(all_jobs=True))
        )
        with self.assertRaises(ValueError):
            configured_run_only(
                config, argparse.Namespace(all_jobs=True, only=["eval"])
            )
