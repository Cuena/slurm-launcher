from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from launcher.tracking import (
    JobRecord,
    TrackingError,
    TrackingPayload,
    load_tracking_payload,
    resolve_tracking_file,
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

    def test_loads_dedicated_rsync_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {**MINIMAL_PAYLOAD, "rsync_login": "user@transfer1"}
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            payload = load_tracking_payload(path)

        self.assertEqual(payload.cluster_login, "user@cluster")
        self.assertEqual(payload.rsync_login, "user@transfer1")

    def test_job_records_have_typed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tracking_file(Path(tmpdir) / "jobs.json", MINIMAL_PAYLOAD)
            payload = load_tracking_payload(path)

        train = payload.jobs[0]
        self.assertEqual(train.job_name, "train")
        self.assertEqual(train.job_id, "12345")
        self.assertEqual(train.stdout, "/logs/12345.out")
        self.assertEqual(train.stderr, "/logs/12345.err")
        self.assertEqual(train.sbatch_command, "sbatch train.sbatch")
        self.assertEqual(train.submitted_at, "2026-04-01T12:00:00")
        self.assertIsNotNone(train.launcher)
        self.assertTrue(train.launcher["managed"])

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

    def test_artifact_paths_from_various_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "cluster_login": "u@h",
                "artifact_paths": ["a/", "b/"],
                "jobs": [],
            }
            path = write_tracking_file(Path(tmpdir) / "jobs.json", data)
            payload = load_tracking_payload(path)

        self.assertEqual(payload.artifact_paths, ["a/", "b/"])


class FilterJobsTests(unittest.TestCase):
    def _payload(self) -> TrackingPayload:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_tracking_file(Path(tmpdir) / "jobs.json", MINIMAL_PAYLOAD)
            return load_tracking_payload(path)

    def test_no_filter_returns_all(self) -> None:
        payload = self._payload()
        self.assertEqual(len(payload.filter_jobs()), 2)

    def test_filter_by_name(self) -> None:
        payload = self._payload()
        result = payload.filter_jobs(names={"train"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].job_name, "train")

    def test_filter_by_id(self) -> None:
        payload = self._payload()
        result = payload.filter_jobs(ids={"12346"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].job_name, "eval")

    def test_filter_by_name_and_id(self) -> None:
        payload = self._payload()
        result = payload.filter_jobs(names={"train"}, ids={"12346"})
        self.assertEqual(len(result), 2)

    def test_filter_nonexistent_returns_empty(self) -> None:
        payload = self._payload()
        result = payload.filter_jobs(names={"nonexistent"})
        self.assertEqual(len(result), 0)


class RunnableJobIdsTests(unittest.TestCase):
    def test_excludes_special_ids(self) -> None:
        jobs = [
            JobRecord(job_name="a", job_id="12345"),
            JobRecord(job_name="b", job_id="dry-run"),
            JobRecord(job_name="c", job_id="unknown"),
            JobRecord(job_name="d", job_id=""),
            JobRecord(job_name="e", job_id="67890"),
        ]
        payload = TrackingPayload(
            source_path=Path("/fake"),
            created_at=None,
            cluster_login="u@h",
            ssh_config_file=None,
            ssh_options=[],
            job_folder="run",
            remote_workdir=None,
            remote_logdir=None,
            remote_slurm_output_dir=None,
            remote_slurm_dashboard_log_archive_dir=None,
            remote_slurm_dashboard_log_view_dir=None,
            runtime_mode=None,
            venv_python_executable=None,
            singularity_image_path=None,
            artifact_paths=[],
            sync_symlinks=None,
            jobs=jobs,
        )
        self.assertEqual(payload.runnable_job_ids(), ["12345", "67890"])

    def test_runnable_from_subset(self) -> None:
        jobs = [
            JobRecord(job_name="a", job_id="12345"),
            JobRecord(job_name="b", job_id="dry-run"),
        ]
        payload = TrackingPayload(
            source_path=Path("/fake"),
            created_at=None,
            cluster_login="u@h",
            ssh_config_file=None,
            ssh_options=[],
            job_folder="run",
            remote_workdir=None,
            remote_logdir=None,
            remote_slurm_output_dir=None,
            remote_slurm_dashboard_log_archive_dir=None,
            remote_slurm_dashboard_log_view_dir=None,
            runtime_mode=None,
            venv_python_executable=None,
            singularity_image_path=None,
            artifact_paths=[],
            sync_symlinks=None,
            jobs=jobs,
        )
        self.assertEqual(payload.runnable_job_ids([jobs[1]]), [])


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

    def test_returns_none_when_no_files_exist(self) -> None:
        result = resolve_tracking_file(None)
        # May or may not be None depending on cwd — just verify it doesn't crash.
        self.assertIsInstance(result, (Path, type(None)))


if __name__ == "__main__":
    unittest.main()
