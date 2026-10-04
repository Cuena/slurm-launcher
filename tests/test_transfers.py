"""Local rsync regressions for advertised download destinations."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

from launcher.artifacts import run_artifacts
from launcher.download_logs import run_download_logs
from tests.helpers import write_tracking_file


@skipUnless(shutil.which("rsync"), "rsync is required for local transfer scenarios")
class TestDownloads(TestCase):
    def setUp(self) -> None:
        fixture = tempfile.TemporaryDirectory()
        self.addCleanup(fixture.cleanup)
        self.root = Path(fixture.name)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.output = self.root / "downloads"
        self.ssh = self.root / "ssh"
        self.ssh.write_text('#!/bin/bash\nshift\nexec "$@"\n')
        self.ssh.chmod(0o755)
        environment = patch.dict(
            os.environ, {"PATH": f"{self.root}:{os.environ['PATH']}"}
        )
        environment.start()
        self.addCleanup(environment.stop)

    def tracking(self, jobs: list[dict[str, object]]) -> Path:
        return write_tracking_file(
            self.root / "jobs.json",
            {"cluster_login": "fixture", "job_folder": "run",
             "remote_workdir": str(self.remote), "jobs": jobs},
        )

    def logs(self, tracking: Path, *, dry_run: bool = False) -> tuple[int, dict]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = run_download_logs(argparse.Namespace(
                tracking_file=str(tracking), job_name=[], job_id=[],
                output_dir=str(self.output), dry_run=dry_run, json=True,
            ))
        return rc, json.loads(output.getvalue())

    def artifacts(self, tracking: Path, *, dry_run: bool = False) -> tuple[int, dict]:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = run_artifacts(
                subcommand="download", tracking_file=str(tracking),
                output_dir=str(self.output), json_output=True, dry_run=dry_run,
            )
        return rc, json.loads(output.getvalue())

    def test_streams_and_job_ids_are_isolated_and_shared_paths_copied_once(self) -> None:
        stdout = self.remote / "stdout" / "train.log"
        stderr = self.remote / "stderr" / "train.log"
        stdout.parent.mkdir()
        stderr.parent.mkdir()
        stdout.write_text("stdout payload")
        stderr.write_text("stderr payload that must survive")
        tracking = self.tracking([
            {"job_name": "train", "job_id": "1", "stdout": str(stdout), "stderr": str(stderr)},
            {"job_name": "train", "job_id": "2", "stdout": str(stdout), "stderr": str(stdout)},
        ])
        rc, payload = self.logs(tracking)
        self.assertEqual(rc, 0, payload)
        self.assertEqual(
            {str(path.relative_to(self.output)): path.read_text()
             for path in self.output.rglob("*.log")},
            {"train/1/stdout/train.log": "stdout payload",
             "train/1/stderr/train.log": "stderr payload that must survive",
             "train/2/stdout/train.log": "stdout payload"},
        )
        self.assertEqual(len(payload["downloads"]), 3)
        self.assertEqual([entry["job_id"] for entry in payload["downloads"]], ["1", "1", "2"])

    def test_file_and_directory_destinations_first_and_repeat_with_trailing_slash(self) -> None:
        directory = self.remote / "outputs"
        directory.mkdir()
        metric = directory / "metric.txt"
        result = self.remote / "result|$epoch.txt"
        for declaration in ("outputs", "outputs/"):
            with self.subTest(declaration=declaration):
                if self.output.exists():
                    shutil.rmtree(self.output)
                tracking = self.tracking([{
                    "job_name": "train", "job_id": "1",
                    "artifacts": [declaration, result.name],
                }])
                for version in ("first", "second with different size"):
                    metric.write_text(version)
                    result.write_text(version)
                    rc, payload = self.artifacts(tracking)
                    self.assertEqual(rc, 0, payload)
                    advertised_dir = Path(payload["artifacts"][0]["destination"])
                    advertised_file = Path(payload["artifacts"][1]["destination"])
                    self.assertEqual(advertised_dir / "metric.txt", self.output / "train/1/outputs/metric.txt")
                    self.assertEqual((advertised_dir / "metric.txt").read_text(), version)
                    self.assertFalse((advertised_dir / "outputs").exists())
                    self.assertEqual(advertised_file.read_text(), version)

    def test_missing_remote_and_transport_failure_preserve_json_diagnostics(self) -> None:
        tracking = self.tracking([{
            "job_name": "train", "job_id": "1", "artifacts": ["missing"],
            "stdout": str(self.remote / "missing.log"),
        }])
        for invoke, key in ((self.artifacts, "artifacts"), (self.logs, "downloads")):
            with self.subTest(operation=key):
                rc, payload = invoke(tracking)
                self.assertEqual(rc, 1)
                self.assertFalse(payload["ok"])
                self.assertEqual(payload["failures"], 1)
                self.assertNotEqual(payload[key][0]["returncode"], 0)
                self.assertIn("No such file", payload[key][0]["stderr"])
                self.assertFalse(Path(payload[key][0]["destination"]).exists())
        self.ssh.write_text('#!/bin/bash\nprintf "fixture connection failed\\n" >&2\nexit 255\n')
        rc, payload = self.artifacts(tracking)
        self.assertEqual(rc, 1)
        self.assertIn("fixture connection failed", payload["artifacts"][0]["stderr"])

    def test_dry_run_never_probes_or_copies(self) -> None:
        marker = self.root / "ssh-called"
        self.ssh.write_text(f'#!/bin/bash\ntouch "{marker}"\nexit 255\n')
        tracking = self.tracking([{
            "job_name": "train", "job_id": "1", "artifacts": ["outputs/"],
            "stdout": str(self.remote / "train.log"),
        }])
        for invoke in (self.artifacts, self.logs):
            rc, payload = invoke(tracking, dry_run=True)
            self.assertEqual(rc, 0, payload)
        self.assertFalse(marker.exists())
        self.assertFalse(self.output.exists())

    def test_unsafe_destination_components_are_rejected_before_copy(self) -> None:
        for job_name, job_id in (("../escape", "1"), ("train", "../escape")):
            tracking = self.tracking([{
                "job_name": job_name, "job_id": job_id,
                "stdout": str(self.remote / "train.log"), "artifacts": ["outputs"],
            }])
            for invoke in (self.artifacts, self.logs):
                rc, payload = invoke(tracking)
                self.assertEqual(rc, 1)
                self.assertFalse(payload["ok"])
        tracking = self.tracking([{
            "job_name": "train", "job_id": "1", "artifacts": ["../escape"],
        }])
        rc, payload = self.artifacts(tracking)
        self.assertEqual(rc, 1)
        self.assertFalse(payload["ok"])
        tracking = self.tracking([{
            "job_name": "train", "job_id": "1",
            "stdout": str(self.remote / "../escape.log"),
        }])
        rc, payload = self.logs(tracking)
        self.assertEqual(rc, 1)
        self.assertFalse(payload["ok"])
        self.assertFalse(self.output.exists())
