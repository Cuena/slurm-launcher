"""Tests for the artifacts command."""

from __future__ import annotations

import json
import contextlib
import io
import os
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from launcher.artifacts import list_artifacts, run_artifacts
from launcher.tracking import load_tracking_payload
from tests.helpers import write_tracking_file


class TestArtifactDiscovery(TestCase):
    def test_list_artifacts_uses_job_specific_and_payload_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            payload = load_tracking_payload(write_tracking_file(
                Path(tmp) / "jobs.json",
                {
                    "cluster_login": "user@cluster",
                    "remote_workdir": "/work/project",
                    "job_folder": "run_20250101_120000",
                    "artifact_paths": ["outputs"],
                    "jobs": [
                        {"job_name": "train", "job_id": "12345",
                         "artifacts": ["outputs/train", "checkpoints/best.pt"]},
                        {"job_name": "eval", "job_id": "12346", "artifacts": []},
                    ],
                },
            ))
            entries = list_artifacts(payload, output_dir)

        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["path"], "outputs/train")
        self.assertEqual(entries[0]["remote_path"], "/work/project/outputs/train")
        self.assertIn("train/12345", entries[0]["destination"])

        # eval falls back to payload artifact_paths
        eval_entries = [e for e in entries if e["job_name"] == "eval"]
        self.assertEqual(len(eval_entries), 1)
        self.assertEqual(eval_entries[0]["path"], "outputs")

    def test_list_artifacts_selected_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            payload = load_tracking_payload(write_tracking_file(
                Path(tmp) / "jobs.json",
                {
                    "cluster_login": "user@cluster",
                    "remote_workdir": "/work/project",
                    "job_folder": "run",
                    "artifact_paths": ["outputs"],
                    "jobs": [
                        {"job_name": "train", "job_id": "1"},
                        {"job_name": "eval", "job_id": "2"},
                    ],
                },
            ))
            entries = list_artifacts(payload, output_dir, selected_jobs=["train"])
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["job_name"], "train")

    def test_list_json_is_explicitly_declared_and_not_remote_checked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tracking = write_tracking_file(
                Path(tmp) / "jobs.json",
                {
                    "cluster_login": "",
                    "job_folder": "run",
                    "remote_workdir": "/work/project",
                    "jobs": [
                        {
                            "job_name": "train",
                            "job_id": "1",
                            "artifacts": ["outputs/result.json"],
                        }
                    ],
                },
            )
            with patch("builtins.print") as mock_print:
                exit_code = run_artifacts(
                    subcommand="list",
                    tracking_file=str(tracking),
                    json_output=True,
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(mock_print.call_args.args[0])
        self.assertEqual(payload["operation"], "list")
        self.assertTrue(payload["declared_only"])
        self.assertFalse(payload["remote_checked"])
        self.assertFalse(payload["copy_attempted"])
        self.assertNotIn("commands", payload)

    def test_check_json_reports_actual_remote_existence_and_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "result|$epoch.json").write_text("result")
            ssh = root / "ssh"
            ssh.write_text('#!/bin/bash\nshift\nexec "$@"\n')
            ssh.chmod(0o755)
            tracking = write_tracking_file(
                root / "jobs.json",
                {
                    "cluster_login": "fixture",
                    "job_folder": "run",
                    "remote_workdir": tmp,
                    "jobs": [{
                        "job_name": "train", "job_id": "1",
                        "artifacts": ["outputs", "outputs/result|$epoch.json", "missing"],
                    }],
                },
            )
            output = io.StringIO()
            with patch.dict(os.environ, {"PATH": f"{tmp}:{os.environ['PATH']}"}):
                with contextlib.redirect_stdout(output):
                    exit_code = run_artifacts(
                        subcommand="check", tracking_file=str(tracking), json_output=True
                    )
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertTrue(payload["remote_checked"])
        self.assertEqual(payload["artifacts"][0]["kind"], "directory")
        self.assertEqual(payload["artifacts"][1]["kind"], "file")
        self.assertEqual(payload["artifacts"][1]["size_bytes"], 6)
        self.assertFalse(payload["artifacts"][2]["exists"])
