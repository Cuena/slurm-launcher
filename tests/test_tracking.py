from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher.tracking import (
    TrackingError,
    TrackingPayload,
    load_tracking_payload,
    resolve_tracking_file,
    atomic_write_json,
)
from tests.helpers import write_tracking_file


MINIMAL_PAYLOAD = {
    "cluster_login": "user@cluster",
    "ssh_config_file": "/dev/null",
    "ssh_options": ["-o", "BatchMode=yes"],
    "job_folder": "project_001",
    "remote_workdir": "/remote/work/project_001",
    "artifact_paths": ["outputs/"],
    "jobs": [
        {
            "job_name": "train",
            "job_id": "12345",
            "stdout": "/logs/12345.out",
            "stderr": "/logs/12345.err",
            "sbatch_command": "sbatch train.sbatch",
            "submitted_at": "2026-04-01T12:00:00",
            "launcher": {
                "managed": True,
                "runtime_kind": "native",
                "runtime_artifact": None,
                "entry_command": "python train.py",
            },
        },
        {
            "job_name": "eval",
            "job_id": "12346",
            "stdout": "/logs/12346.out",
            "stderr": "/logs/12346.err",
        },
    ],
}


class LoadTrackingPayloadTests(unittest.TestCase):
    def test_loads_valid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tracking_file(Path(tmpdir) / "jobs.json", MINIMAL_PAYLOAD)
            payload = load_tracking_payload(path)

        self.assertEqual(payload.cluster_login, "user@cluster")
        self.assertEqual(payload.rsync_login, "user@cluster")
        self.assertEqual(payload.ssh_config_file, "/dev/null")
        self.assertEqual(payload.ssh_options, ["-o", "BatchMode=yes"])
        self.assertEqual(payload.job_folder, "project_001")
        self.assertEqual(payload.remote_workdir, "/remote/work/project_001")
        self.assertEqual(payload.artifact_paths, ["outputs/"])
        self.assertEqual(len(payload.jobs), 2)
        self.assertEqual(payload.source_path, path)
        train = payload.jobs[0]
        self.assertEqual(train.job_name, "train")
        self.assertEqual(train.job_id, "12345")
        self.assertEqual(train.stdout, "/logs/12345.out")
        self.assertEqual(train.stderr, "/logs/12345.err")
        self.assertEqual(train.sbatch_command, "sbatch train.sbatch")
        self.assertEqual(train.submitted_at, "2026-04-01T12:00:00")
        self.assertTrue(train.launcher["managed"])

    def test_loads_dedicated_rsync_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {**MINIMAL_PAYLOAD, "rsync_login": "user@transfer1"}
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            payload = load_tracking_payload(path)

        self.assertEqual(payload.cluster_login, "user@cluster")
        self.assertEqual(payload.rsync_login, "user@transfer1")

    def test_raises_on_nonexistent_file(self) -> None:
        with self.assertRaises(TrackingError):
            load_tracking_payload(Path("/nonexistent/jobs.json"))

    def test_raises_on_invalid_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            path.write_text("not json", encoding="utf-8")
            with self.assertRaises(TrackingError):
                load_tracking_payload(path)

    def test_raises_on_non_dict_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(TrackingError):
                load_tracking_payload(path)

    def test_raises_on_invalid_jobs_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"cluster_login": "user@host", "jobs": "not a list"}
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            with self.assertRaises(TrackingError):
                load_tracking_payload(path)

    def test_missing_cluster_login_defaults_to_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"jobs": [{"job_name": "train", "job_id": "1"}]}
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            payload = load_tracking_payload(path)

        self.assertEqual(payload.cluster_login, "")
        self.assertEqual(len(payload.jobs), 1)

    def test_skips_non_dict_job_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "cluster_login": "user@host",
                "jobs": [
                    {"job_name": "train", "job_id": "1"},
                    "not a dict",
                    42,
                    None,
                ],
            }
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            payload = load_tracking_payload(path)

        self.assertEqual(len(payload.jobs), 1)
        self.assertEqual(payload.jobs[0].job_name, "train")

    def test_ssh_options_sanitized_from_non_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"cluster_login": "u@h", "ssh_options": None, "jobs": []}
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            payload = load_tracking_payload(path)

        self.assertEqual(payload.ssh_options, [])


class FilterJobsTests(unittest.TestCase):
    def _payload(self) -> TrackingPayload:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tracking_file(Path(tmpdir) / "jobs.json", MINIMAL_PAYLOAD)
            return load_tracking_payload(path)

    def test_filter_selection(self) -> None:
        payload = self._payload()
        for filters, expected in (
            ({}, ["train", "eval"]),
            ({"names": {"train"}}, ["train"]),
            ({"ids": {"12346"}}, ["eval"]),
            ({"names": {"train"}, "ids": {"12346"}}, ["train", "eval"]),
            ({"names": {"nonexistent"}}, []),
        ):
            with self.subTest(filters=filters):
                self.assertEqual(
                    [job.job_name for job in payload.filter_jobs(**filters)], expected
                )


class ResolveTrackingFileTests(unittest.TestCase):
    def test_explicit_path_returned_if_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "custom.json"
            path.write_text("{}", encoding="utf-8")
            result = resolve_tracking_file(str(path))
            self.assertEqual(result, path)

    def test_explicit_path_returns_none_if_missing(self) -> None:
        result = resolve_tracking_file("/nonexistent/custom.json")
        self.assertIsNone(result)

    def test_named_latest_directory_and_tracking_path_resolve_same_run(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(temporary.name)
        self.assertIsNone(resolve_tracking_file(None))
        path = write_tracking_file(
            Path("slurm_output/run_001/jobs.json"), {"run_id": "run_001"}
        )
        for selector in (
            "run_001",
            "slurm_output/run_001",
            str(path),
            "latest",
            None,
        ):
            with self.subTest(selector=selector):
                self.assertEqual(resolve_tracking_file(selector), path)


class AtomicTrackingTests(unittest.TestCase):
    def test_failed_replace_preserves_previous_complete_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "jobs.json"
            atomic_write_json(path, {"jobs": [{"job_id": "12345"}]})
            with patch(
                "launcher.tracking.os.replace", side_effect=OSError("disk error")
            ):
                with self.assertRaises(OSError):
                    atomic_write_json(
                        path, {"jobs": [{"job_id": "12345"}, {"job_id": "12346"}]}
                    )
            self.assertEqual(
                json.loads(path.read_text()), {"jobs": [{"job_id": "12345"}]}
            )
            self.assertEqual(list(Path(tmpdir).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
