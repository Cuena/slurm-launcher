"""Tests for launcher configuration normalization."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest import TestCase

from launcher.config_utils import build_settings


class TestBuildSettings(TestCase):
    def _minimal_config(self) -> ModuleType:
        config = ModuleType("test_config")
        config.CLUSTER_LOGIN = "user@cluster"
        config.WORKSPACE_MODE = "fixed"
        config.REMOTE_WORKSPACE_DIR = "/remote/project"
        config.REMOTE_LOG_BASE_PATH = "/remote/logs"
        return config

    def test_omitted_sync_symlinks_defaults_to_preserve(self) -> None:
        settings = build_settings(self._minimal_config(), Path("config.py"))

        self.assertEqual(settings.sync_symlinks, "preserve")

    def test_removed_copy_spelling_is_rejected(self) -> None:
        config = self._minimal_config()
        config.SYNC_SYMLINKS = "copy"

        with self.assertRaisesRegex(
            SystemExit,
            "SYNC_SYMLINKS must be one of: copy-links, preserve",
        ):
            build_settings(config, Path("config.py"))
