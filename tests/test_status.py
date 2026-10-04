from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from launcher.status import (
    _status_payload,
    query_job_statuses,
)
from launcher.tracking import JobRecord


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


@contextmanager
def scheduler_fixture(
    root: Path, accounting: str, queue: str, *, queue_returncode: int = 0
):
    """Run real SSH/Bash command construction against local scheduler programs."""
    bindir = root / "bin"
    bindir.mkdir()
    scripts = {
        "ssh": '#!/bin/sh\nshift\nexec "$@"\n',
        "sacct": (
            "#!/bin/sh\n"
            'case " $* " in *" --array "*) ;; *) exit 8;; esac\n'
            'case " $* " in *"JobID%128"*) ;; *) exit 9;; esac\n'
            f"printf '%s' {shlex.quote(accounting)}\n"
        ),
        "squeue": (
            "#!/bin/sh\n"
            'case " $* " in *" -r "*) ;; *) exit 8;; esac\n'
            f"printf '%s' {shlex.quote(queue)}\n"
            + ("printf '%s\\n' 'queue unavailable' >&2\n" if queue_returncode else "")
            + f"exit {queue_returncode}\n"
        ),
    }
    for name, script in scripts.items():
        path = bindir / name
        path.write_text(script)
        path.chmod(0o755)
    with patch.dict(os.environ, {"PATH": f"{bindir}:{os.environ['PATH']}"}):
        yield


class QueryJobStatusesTests(TestCase):
    @patch("launcher.status.run_ssh_capture")
    def test_queue_fallback_for_missing_failed_and_unknown_accounting(self, mock_ssh) -> None:
        for accounting, rc, state in (
            ("", 0, "RUNNING"),
            ("", 127, "PENDING"),
            ("400|train|UNKNOWN|-|-|-|-|-|gpu\n", 0, "RUNNING"),
        ):
            with self.subTest(accounting=accounting, returncode=rc):
                mock_ssh.side_effect = [
                    _completed(accounting, returncode=rc),
                    _completed(f"400|train|{state}|-|submit|start|-|00:03|gpu\n"),
                ]
                result = query_job_statuses(
                    "user@cluster", [JobRecord(job_name="train", job_id="400")]
                )
                self.assertEqual(
                    [(status.state, status.derived_state, status.partition, status.source)
                     for status in result.statuses],
                    [(state, state, "gpu", "squeue")],
                )
                self.assertTrue(result.ok)
                self.assertEqual(result.probes[0].ok, rc == 0)

    @patch("launcher.status.run_ssh_capture")
    def test_merges_sacct_history_with_squeue_live_state(self, mock_ssh) -> None:
        mock_ssh.side_effect = [
            _completed(
                "100|finished|COMPLETED|0:0|2026-07-15T09:00:00|"
                "2026-07-15T09:01:00|2026-07-15T09:02:00|00:01:00|cpu\n"
            ),
            _completed(
                "200|running|RUNNING|-|2026-07-15T10:00:00|"
                "2026-07-15T10:01:00|-|00:03|gpu\n"
            ),
        ]

        result = query_job_statuses(
            "user@cluster",
            [
                JobRecord(job_name="finished", job_id="100"),
                JobRecord(job_name="running", job_id="200"),
            ],
        )
        statuses = result.statuses

        self.assertEqual(
            [status.derived_state for status in statuses], ["DONE", "RUNNING"]
        )
        self.assertTrue(result.ok)

    @patch("launcher.status.run_ssh_capture")
    def test_unknown_only_after_both_sources_omit_job(self, mock_ssh) -> None:
        mock_ssh.side_effect = [_completed(""), _completed("")]

        result = query_job_statuses(
            "user@cluster",
            [JobRecord(job_name="missing", job_id="999")],
        )
        statuses = result.statuses

        self.assertIsNone(statuses[0].state)
        self.assertEqual(statuses[0].derived_state, "UNKNOWN")
        self.assertTrue(result.ok)
        self.assertEqual(result.unresolved_job_ids, ["999"])

    @patch("launcher.status.run_ssh_capture")
    def test_failed_probes_make_unresolved_status_an_error(self, mock_ssh) -> None:
        mock_ssh.side_effect = [
            _completed("", returncode=255),
            _completed("", returncode=255),
        ]

        result = query_job_statuses(
            "user@cluster",
            [JobRecord(job_name="missing", job_id="999")],
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.unresolved_job_ids, ["999"])
        self.assertEqual([probe.returncode for probe in result.probes], [255, 255])



class ArrayStatusTests(TestCase):
    def query(self, accounting, queue="", *, selector="123", spec=None, queue_returncode=0):
        with tempfile.TemporaryDirectory() as directory:
            with scheduler_fixture(
                Path(directory), accounting, queue, queue_returncode=queue_returncode
            ):
                return query_job_statuses(
                    "fixture",
                    [JobRecord(job_name="array", job_id=selector, array_spec=spec)],
                )

    def test_parent_keeps_failed_sibling_even_if_placeholder_completed(self):
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n"
            "123_1|array|FAILED|1:0|-|-|-|00:01|cpu\n",
            spec="0-1",
        )
        self.assertEqual(result.statuses[0].derived_state, "FAILED")
        payload = _status_payload(None, "fixture", result)["jobs"][0]
        self.assertEqual(
            [(task["job_id"], task["derived_state"]) for task in payload["tasks"]],
            [("123_0", "DONE"), ("123_1", "FAILED")],
        )

    def test_completed_accounting_and_compressed_live_siblings_are_merged(self):
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n",
            "123_[1-3:2]|array|PENDING|-|-|-|-|00:00|cpu|123|1-3:2\n",
            spec="0,1,3%2",
        )
        status = result.statuses[0]
        self.assertEqual(status.derived_state, "PENDING")
        self.assertEqual(
            [(task.job_id, task.derived_state) for task in status.tasks],
            [("123_0", "DONE"), ("123_1", "PENDING"), ("123_3", "PENDING")],
        )
        self.assertEqual(status.source, "sacct+squeue")
        self.assertTrue(status.array_complete)

    def test_queue_parent_mapping_preserves_running_array_allocation(self):
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n",
            "124|array|RUNNING|-|-|-|-|00:01|cpu|123|1\n",
            spec="0-1",
        )
        self.assertEqual(result.statuses[0].derived_state, "RUNNING")
        self.assertEqual(result.statuses[0].tasks[1].job_id, "123_1")

    def test_individual_completed_task_resolves_task_identity_not_allocation(self):
        # Slurm would print raw allocation 124 with JobIDRaw; JobID is 123_0.
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n",
            selector="123_0",
        )
        self.assertEqual(result.statuses[0].derived_state, "DONE")
        self.assertEqual(result.statuses[0].job_id, "123_0")

    def test_missing_accounting_sibling_prevents_parent_done(self):
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n", spec="0-1"
        )
        status = result.statuses[0]
        self.assertEqual(status.derived_state, "UNKNOWN")
        self.assertFalse(status.array_complete)
        self.assertEqual(status.tasks[1].derived_state, "UNKNOWN")

    def test_unknown_membership_prevents_parent_done(self):
        result = self.query("123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n")
        self.assertEqual(result.statuses[0].derived_state, "UNKNOWN")
        self.assertFalse(result.statuses[0].array_complete)

    def test_all_known_tasks_successful_and_queue_empty_is_done(self):
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n"
            "123_1|array|COMPLETED|0:0|-|-|-|00:01|cpu\n",
            spec="0-1",
        )
        self.assertEqual(result.statuses[0].derived_state, "DONE")
        self.assertTrue(result.statuses[0].array_complete)

    def test_queue_failure_prevents_parent_done_and_keeps_diagnostics(self):
        result = self.query(
            "123_0|array|COMPLETED|0:0|-|-|-|00:01|cpu\n",
            spec="0", queue_returncode=1,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.statuses[0].derived_state, "UNKNOWN")
        self.assertEqual(result.probes[1].stderr, "queue unavailable")

    def test_scalar_completed_nonzero_exit_is_failed(self):
        for code in ("2:0", "0:9"):
            with self.subTest(exit_code=code):
                result = self.query(
                    f"123|scalar|COMPLETED|{code}|-|-|-|00:01|cpu\n"
                )
                self.assertEqual(result.statuses[0].derived_state, "FAILED")

    def test_scalar_success_preserves_status_fields_and_json_shape(self):
        result = self.query(
            "123|scalar|COMPLETED|0:0|submit|start|end|00:01|cpu\n"
        )
        payload = _status_payload(None, "fixture", result)["jobs"][0]
        self.assertEqual(payload["derived_state"], "DONE")
        self.assertEqual(payload["exit_code"], "0:0")
        self.assertEqual(payload["partition"], "cpu")
        self.assertEqual(payload["elapsed"], "00:01")
        self.assertNotIn("tasks", payload)
