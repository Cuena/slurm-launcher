"""Tests for preflight script generation and behavior."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from launcher.preflight import (
    build_remote_check_script,
    run_preflight,
    run_preflight_for_job,
)


class TestRemoteCheckScript(TestCase):
    def test_literal_special_paths_and_existing_glob_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ["model$epoch.pt", 'model"quoted".pt', "model|version.pt"]
            for name in names:
                (root / name).write_text("weights")
            (root / "dangling.pt").symlink_to(root / "absent")
            script = build_remote_check_script(tmp, [*names, "model*.pt", "dangling.pt"])
            result = subprocess.run(
                ["bash", "-s"], input=script, text=True, capture_output=True, check=False
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stdout.splitlines(),
            ["0|true|exists", "1|true|exists", "2|true|exists",
             "3|true|matched 3", "4|false|broken symlink"],
        )
        self.assertEqual(result.stderr, "")

    def test_glob_with_only_dangling_symlink_fails_and_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dangling.pt").symlink_to(root / "absent")
            (root / "present").write_text("data")
            result = subprocess.run(
                ["bash", "-s"],
                input=build_remote_check_script(tmp, ["dang*.pt", "missing*", "present"]),
                text=True, capture_output=True, check=False,
            )
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            result.stdout.splitlines(),
            ["0|false|glob matched 0 existing targets",
             "1|false|glob matched 0 existing targets", "2|true|exists"],
        )

    def test_failed_transport_cannot_publish_passed_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input.txt").write_text("data")
            ssh = root / "ssh"
            ssh.write_text(
                '#!/bin/bash\nshift\n"$@"\n'
                'printf "connection closed before exit status\\n" >&2\nexit 255\n'
            )
            ssh.chmod(0o755)
            settings = SimpleNamespace(
                cluster_login="fixture", ssh_config_file=None, ssh_options=[]
            )
            remote_paths = SimpleNamespace(workdir=tmp)
            jobs = [SimpleNamespace(name="train", requires=["input.txt"])]
            output = io.StringIO()
            with patch.dict(os.environ, {"PATH": f"{tmp}:{os.environ['PATH']}"}):
                with contextlib.redirect_stdout(output):
                    rc = run_preflight(settings, remote_paths, jobs, json_output=True)
        payload = json.loads(output.getvalue())
        self.assertEqual(rc, 1)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["jobs"][0]["status"], "failed")
        self.assertTrue(payload["jobs"][0]["checks"][0]["ok"])
        self.assertEqual(payload["jobs"][0]["transport_returncode"], 255)
        self.assertIn("connection closed", payload["jobs"][0]["stderr"])

    def test_successful_transport_with_incomplete_records_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("first", "second"):
                (root / name).write_text("input")
            ssh = root / "ssh"
            ssh.write_text(
                '#!/bin/bash\nshift\nresponse=$("$@")\n'
                'printf "%s\\n" "${response%%$\'\\n\'*}"\n'
            )
            ssh.chmod(0o755)
            settings = SimpleNamespace(
                cluster_login="fixture", ssh_config_file=None, ssh_options=[]
            )
            with patch.dict(os.environ, {"PATH": f"{tmp}:{os.environ['PATH']}"}):
                result = run_preflight_for_job(
                    settings, SimpleNamespace(workdir=tmp), "train", ["first", "second"]
                )
        self.assertEqual(result.transport_returncode, 0)
        self.assertFalse(result.ok)
        self.assertTrue(result.checks[0].ok)
        self.assertFalse(result.checks[1].ok)


class TestPreflightDryRunJson(TestCase):
    def test_dry_run_json_emits_valid_payload(self) -> None:
        from launcher.core import JobSpec, RemotePaths
        from tests.helpers import make_settings

        settings = make_settings(
            workspace_mode="fixed", remote_workspace_dir="/work/project"
        )
        remote_paths = RemotePaths(
            "run_001", "/work/project", "/logs/run_001", "/logs/run_001/slurm_output"
        )
        jobs = [
            JobSpec(
                name="eval",
                sbatch_file="slurm/eval.sbatch",
                requires=["data/input/*.mp4", "models/model.onnx"],
            ),
        ]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_preflight(
                settings,
                remote_paths,
                jobs,
                json_output=True,
                dry_run=True,
            )
        self.assertEqual(exit_code, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["remote_workdir"], "/work/project")
        self.assertEqual(payload["checks_planned"], 2)
        self.assertEqual(payload["checks_run"], 0)
        self.assertEqual(payload["warnings"], [])
        self.assertEqual(len(payload["jobs"]), 1)
        self.assertEqual(payload["jobs"][0]["job_name"], "eval")
        self.assertEqual(payload["jobs"][0]["status"], "planned")
        self.assertIn("data/input/*.mp4", payload["jobs"][0]["requirements"])

    def test_live_json_rejects_no_selected_jobs(self) -> None:

        settings = SimpleNamespace()
        remote_paths = SimpleNamespace(workdir="/work/project")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_preflight(
                settings,
                remote_paths,
                [],
                json_output=True,
                dry_run=False,
            )

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["dry_run"])
        self.assertEqual(payload["checks_run"], 0)
        self.assertEqual(payload["jobs"], [])

    def test_live_json_rejects_job_without_requires_explicitly(self) -> None:
        from launcher.core import JobSpec

        settings = SimpleNamespace()
        remote_paths = SimpleNamespace(workdir="/work/project")
        jobs = [JobSpec(name="shared", sbatch_file="slurm/shared.sbatch")]

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exit_code = run_preflight(
                settings,
                remote_paths,
                jobs,
                json_output=True,
            )

        self.assertEqual(exit_code, 1)
        payload = json.loads(buf.getvalue())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["checks_planned"], 0)
        self.assertEqual(payload["checks_run"], 0)
        self.assertEqual(payload["jobs"][0]["job_name"], "shared")
        self.assertEqual(payload["jobs"][0]["status"], "not-configured")
        self.assertFalse(payload["jobs"][0]["ok"])

    def test_absolute_requirement_keeps_absolute_remote_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prerequisite = root / "model|$epoch.pt"
            prerequisite.write_text("weights")
            ssh = root / "ssh"
            ssh.write_text('#!/bin/bash\nshift\nexec "$@"\n')
            ssh.chmod(0o755)
            settings = SimpleNamespace(
                cluster_login="fixture", ssh_config_file=None, ssh_options=[]
            )
            with patch.dict(os.environ, {"PATH": f"{tmp}:{os.environ['PATH']}"}):
                result = run_preflight_for_job(
                    settings, SimpleNamespace(workdir=tmp), "train", [str(prerequisite)]
                )
        self.assertTrue(result.ok)
        self.assertEqual(result.checks[0].remote_path, str(prerequisite))
