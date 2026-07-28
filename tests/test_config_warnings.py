"""Tests for config validation warnings."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from launcher.config_utils import JobSpec, LauncherSettings, collect_config_warnings


class TestConfigWarnings(TestCase):
    def _settings(self, **kwargs: object) -> LauncherSettings:
        defaults = {
            "cluster_login": "user@cluster",
            "ssh_config_file": None,
            "ssh_options": [],
            "project_root": Path(".").resolve(),
            "project_prefix": "test",
            "workspace_mode": "per-run",
            "remote_workspace_base": "/work",
            "remote_workspace_dir": None,
            "remote_log_base_path": "/logs",
            "default_env": {},
            "default_sbatch": {},
            "runtime_mode": "native",
            "venv_python_executable": None,
            "singularity_image_path": None,
            "singularity_exec_flags": [],
            "artifact_paths": ["outputs"],
            "extra_rsync_excludes": ["data/"],
            "extra_rsync_args": [],
            "remote_slurm_dashboard_log_archive_dir": None,
            "remote_slurm_dashboard_log_view_dir": None,
            "require_clean_git": False,
            "sync_symlinks": "preserve",
            "local_artifact_root": None,
            "verbose": False,
        }
        defaults.update(kwargs)
        return LauncherSettings(**defaults)

    def test_gpu_job_without_requires_warns(self) -> None:
        settings = self._settings()
        jobs = [
            JobSpec(
                name="train",
                command="python train.py",
                sbatch={"gres": "gpu:1"},
            ),
        ]
        warnings = collect_config_warnings(settings, jobs)
        self.assertEqual(len(warnings), 1)
        self.assertIn("GPU", warnings[0])

    def test_required_path_excluded_warns(self) -> None:
        settings = self._settings()
        jobs = [
            JobSpec(
                name="train",
                command="python train.py",
                requires=["data/processed"],
            ),
        ]
        warnings = collect_config_warnings(settings, jobs)
        self.assertEqual(len(warnings), 1)
        self.assertIn("rsync exclude", warnings[0])

    def test_sbatch_file_without_requires_warns_that_preflight_will_fail(self) -> None:
        settings = self._settings()
        jobs = [JobSpec(name="shared", sbatch_file="slurm/shared.sbatch")]

        warnings = collect_config_warnings(settings, jobs)

        self.assertEqual(len(warnings), 1)
        self.assertIn("sbatch_file", warnings[0])
        self.assertIn("Preflight will fail", warnings[0])

    def test_sbatch_file_required_path_excluded_warns(self) -> None:
        settings = self._settings()
        jobs = [
            JobSpec(
                name="shared",
                sbatch_file="slurm/shared.sbatch",
                requires=["data/processed"],
            )
        ]

        warnings = collect_config_warnings(settings, jobs)

        self.assertEqual(len(warnings), 1)
        self.assertIn("rsync exclude", warnings[0])

    def test_output_dir_not_declared_warns(self) -> None:
        settings = self._settings(artifact_paths=[])
        jobs = [
            JobSpec(
                name="train",
                command="python train.py --output-dir results",
            ),
        ]
        warnings = collect_config_warnings(settings, jobs)
        self.assertTrue(any("output dir" in warning for warning in warnings))

    def test_no_warnings_for_clean_config(self) -> None:
        settings = self._settings(extra_rsync_excludes=[])
        jobs = [
            JobSpec(
                name="train",
                command="python train.py",
                requires=["data/processed"],
                artifacts=["outputs/train"],
            ),
        ]
        warnings = collect_config_warnings(settings, jobs)
        self.assertEqual(warnings, [])
