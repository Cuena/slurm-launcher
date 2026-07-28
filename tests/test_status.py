from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from launcher.status import (
    StatusQueryResult,
    _build_sacct_script,
    _build_squeue_script,
    query_job_statuses,
    run_status,
)
from launcher.tracking import JobRecord
from tests.helpers import write_tracking_file


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["ssh"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


class QueryJobStatusesTests(TestCase):
    @patch("launcher.status._run_ssh_capture")
    def test_falls_back_to_squeue_when_sacct_omits_running_job(self, mock_ssh) -> None:
        mock_ssh.side_effect = [
            _completed(""),
            _completed(
                "43330070|train|RUNNING|-|2026-07-15T10:00:00|2026-07-15T10:01:00|-|00:03|gpu\n"
            ),
        ]

        result = query_job_statuses(
            "user@cluster",
            [JobRecord(job_name="train", job_id="43330070")],
        )
        statuses = result.statuses

        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0].state, "RUNNING")
        self.assertEqual(statuses[0].derived_state, "RUNNING")
        self.assertEqual(statuses[0].partition, "gpu")
        self.assertEqual(statuses[0].source, "squeue")
        self.assertTrue(result.ok)
        self.assertEqual(mock_ssh.call_count, 2)
        self.assertIn("squeue", mock_ssh.call_args_list[1].args[1])

    @patch("launcher.status._run_ssh_capture")
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
        squeue_script = mock_ssh.call_args_list[1].args[1]
        self.assertIn("-j 200", squeue_script)
        self.assertNotIn("-j 100", squeue_script)
        self.assertTrue(result.ok)

    @patch("launcher.status._run_ssh_capture")
    def test_falls_back_to_squeue_when_sacct_command_fails(self, mock_ssh) -> None:
        mock_ssh.side_effect = [
            _completed("", returncode=127),
            _completed("300|queued|PENDING|-|2026-07-15T10:00:00|N/A|-|00:00|gpu\n"),
        ]

        result = query_job_statuses(
            "user@cluster",
            [JobRecord(job_name="queued", job_id="300")],
        )
        statuses = result.statuses

        self.assertEqual(statuses[0].state, "PENDING")
        self.assertEqual(statuses[0].derived_state, "PENDING")
        self.assertTrue(result.ok)
        self.assertFalse(result.probes[0].ok)

    @patch("launcher.status._run_ssh_capture")
    def test_falls_back_to_squeue_when_sacct_state_is_unknown(self, mock_ssh) -> None:
        mock_ssh.side_effect = [
            _completed("400|train|UNKNOWN|-|-|-|-|-|gpu\n"),
            _completed(
                "400|train|RUNNING|-|2026-07-15T10:00:00|"
                "2026-07-15T10:01:00|-|00:03|gpu\n"
            ),
        ]

        result = query_job_statuses(
            "user@cluster",
            [JobRecord(job_name="train", job_id="400")],
        )
        statuses = result.statuses

        self.assertEqual(statuses[0].state, "RUNNING")
        self.assertEqual(statuses[0].derived_state, "RUNNING")

    @patch("launcher.status._run_ssh_capture")
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

    @patch("launcher.status._run_ssh_capture")
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

    def test_generated_status_scripts_execute_format_options(self) -> None:
        sacct_stub = """
sacct() {
  case " $* " in
    *" --format "*) printf '%s\\n' '1|job|COMPLETED|0:0|-|-|-|00:01|gpp' ;;
    *) return 9 ;;
  esac
}
"""
        sacct = subprocess.run(
            ["bash", "-s"],
            input=sacct_stub + _build_sacct_script(["1"]) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(sacct.returncode, 0, sacct.stderr)
        self.assertIn("1|job|COMPLETED", sacct.stdout)

        squeue_stub = """
squeue() {
  case " $* " in
    *" -o "*) printf '%s\\n' '2|job|RUNNING|-|-|-|-|00:01|gpp' ;;
    *) return 9 ;;
  esac
}
"""
        squeue = subprocess.run(
            ["bash", "-s"],
            input=squeue_stub + _build_squeue_script(["2"]) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(squeue.returncode, 0, squeue.stderr)
        self.assertIn("2|job|RUNNING", squeue.stdout)

    @patch("launcher.status.console.print_json")
    @patch("launcher.status.query_job_statuses")
    def test_tracking_file_owns_complete_ssh_context(
        self,
        mock_query_job_statuses,
        _mock_print_json,
    ) -> None:
        mock_query_job_statuses.return_value = StatusQueryResult(
            statuses=[], probes=[], unresolved_job_ids=[]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            tracking = write_tracking_file(
                Path(tmpdir) / "jobs.json",
                {
                    "cluster_login": "acc",
                    "ssh_config_file": None,
                    "ssh_options": [],
                    "jobs": [{"job_name": "job", "job_id": "123"}],
                },
            )

            exit_code = run_status(
                tracking_file=str(tracking),
                job_id="123",
                cluster_login="user@cluster",
                ssh_config_file="/dev/null",
                ssh_options=["-o", "BatchMode=yes"],
                json_output=True,
            )

        self.assertEqual(exit_code, 0)
        args, kwargs = mock_query_job_statuses.call_args
        self.assertEqual(args[0], "acc")
        self.assertIsNone(kwargs["ssh_config_file"])
        self.assertEqual(kwargs["ssh_options"], [])
