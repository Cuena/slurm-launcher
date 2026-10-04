from __future__ import annotations

import io
import json
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from launcher.job_tools import list_recent_jobs, resolve_job_log_info, show_job_details


class JobToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        environment = patch.dict(os.environ, {"PATH": f"{self.bin_dir}:{os.environ['PATH']}"})
        environment.start()
        self.addCleanup(environment.stop)
        self._executable(
            "ssh",
            "while (($#)); do\n"
            '  case "$1" in\n'
            "    -F|-o|-p|-i|-J|-l) shift 2 ;;\n"
            "    -*) shift ;;\n"
            '    *) shift; exec "$@" ;;\n'
            "  esac\n"
            "done\n"
            "exit 2\n",
        )
        self._scontrol()
        self._sacct()

    def _executable(self, name: str, body: str) -> None:
        path = self.bin_dir / name
        path.write_text("#!/bin/bash\n" + body, encoding="utf-8")
        path.chmod(0o755)

    @staticmethod
    def _response(output: str = "", error: str = "", rc: int = 0) -> str:
        return (
            f"printf '%s' {shlex.quote(output)}\n"
            f"printf '%s' {shlex.quote(error)} >&2\n"
            f"exit {rc}\n"
        )

    def _scontrol(
        self,
        output: str = "",
        *,
        error: str = "",
        rc: int = 0,
        sbatch: str = "",
        sbatch_error: str = "",
        sbatch_rc: int = 0,
    ) -> None:
        self._executable(
            "scontrol",
            'case "$1 $2" in\n'
            '  "show job")\n'
            + self._response(output, error, rc)
            + '  ;;\n  "write batch_script")\n'
            + self._response(sbatch, sbatch_error, sbatch_rc)
            + "  ;;\n  *) exit 2 ;;\nesac\n",
        )

    def _sacct(
        self,
        output: str = "",
        *,
        error: str = "",
        rc: int = 0,
        narrow: str | None = None,
        recent: str = "",
    ) -> None:
        self._executable(
            "sacct",
            'case "$*" in\n'
            "  *--starttime*)\n"
            + self._response(recent)
            + "  ;;\n  *StdOut,StdErr,SubmitLine*)\n"
            + self._response(output, error, rc)
            + "  ;;\n  *)\n"
            + self._response(output if narrow is None else narrow, error, 0 if narrow is not None else rc)
            + "  ;;\nesac\n",
        )

    def _show(self, cluster: str = "cluster-b", *, include_sbatch: bool = False) -> tuple[int, dict]:
        output = io.StringIO()
        with patch("launcher.job_tools.console", Console(file=output, width=120)):
            rc = show_job_details(
                cluster,
                "123",
                json_output=True,
                include_sbatch=include_sbatch,
                ssh_config_file=str(self.root / "saved ssh config"),
                ssh_options=["-o", "BatchMode=yes"],
            )
        return rc, json.loads(output.getvalue())

    def _tracking(self, cluster: str, **identity: str) -> None:
        directory = self.root / "slurm_output" / "run_001"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "jobs.json").write_text(
            json.dumps({
                "cluster_login": cluster,
                "runtime_mode": "singularity",
                "singularity_image_path": "/cluster-a/train.sif",
                "jobs": [{
                    "job_id": "123",
                    "job_name": "train",
                    "launcher": {
                        "managed": True,
                        "runtime_kind": "singularity",
                        "runtime_artifact": "/cluster-a/train.sif",
                        "entry_command": "python train.py",
                    },
                    **identity,
                }],
            }),
            encoding="utf-8",
        )

    def test_show_preserves_spaced_fields_and_fills_missing_primary_values(self) -> None:
        self._scontrol(
            "JobId=123 JobName=my training JobState=RUNNING Partition=N/A "
            "Command=/work/my project/run.sbatch WorkDir=/work/my project "
            "StdOut=/logs/my project/%j.out StdErr=(null) NodeList=None "
            "NumNodes=1 SubmitTime=2026-03-26T12:00:00 EndTime=Unknown"
        )
        self._sacct(
            "123|my training|RUNNING|acc|2026-03-26T12:00:00|Unknown|Unknown|"
            "node01|1|gpu:1|/work/my project|/logs/my project/%j.out|"
            "/logs/my project/%J.err|sbatch '/work/my project/run.sbatch'\n"
        )

        rc, payload = self._show()

        self.assertEqual(rc, 0)
        self.assertEqual(payload["job_name"], "my training")
        self.assertEqual(payload["command"], "/work/my project/run.sbatch")
        self.assertEqual(payload["work_dir"], "/work/my project")
        self.assertEqual(payload["stdout"], "/logs/my project/123.out")
        self.assertEqual(payload["stderr"], "/logs/my project/123.err")
        self.assertEqual(payload["partition"], "acc")
        self.assertEqual(payload["node_list"], "node01")
        self.assertNotIn("end_time", payload)

    def test_log_resolution_merges_partial_streams_without_replacing_primary(self) -> None:
        self._scontrol(
            "JobId=123 JobName=my training JobState=RUNNING "
            "StdOut=/logs/my project/%j.out StdErr=N/A"
        )
        self._sacct("123|my training|RUNNING|/other/123.out|/logs/my project/%J.err\n")

        info = resolve_job_log_info("cluster-b", "123", archive_dir=None)

        self.assertEqual(info.stdout, "/logs/my project/123.out")
        self.assertEqual(info.stderr, "/logs/my project/123.err")
        self.assertEqual(info.stdout_source, "scontrol")
        self.assertEqual(info.stderr_source, "sacct")
        self.assertEqual(info.job_name, "my training")
        self.assertTrue(info.verified)

    def test_failed_probe_diagnostic_survives_successful_accounting(self) -> None:
        self._scontrol(error="slurm_load_jobs: Invalid job id specified", rc=1)
        self._sacct("123|finished training|COMPLETED|/logs/123.out|/logs/123.err\n")

        info = resolve_job_log_info("cluster-b", "123", archive_dir=None)

        self.assertEqual(info.stdout, "/logs/123.out")
        self.assertEqual(info.stderr, "/logs/123.err")
        self.assertEqual(info.stdout_source, "sacct")
        self.assertTrue(info.verified)
        self.assertEqual(len(info.probe_errors), 1)
        self.assertIn("Invalid job id specified", info.probe_errors[0])

    def test_unresolved_probes_return_bounded_stderr_evidence(self) -> None:
        self._scontrol(error="scheduler unavailable: " + "x" * 3000, rc=1)
        self._sacct(error="accounting unavailable", rc=2)

        info = resolve_job_log_info("cluster-b", "123", archive_dir=None)

        self.assertIsNone(info.stdout)
        self.assertIsNone(info.stderr)
        self.assertFalse(info.verified)
        self.assertEqual(info.source, "unresolved")
        self.assertEqual(len(info.probe_errors), 2)
        self.assertIn("scheduler unavailable", info.probe_errors[0])
        self.assertLess(len(info.probe_errors[0]), 1100)
        self.assertIn("accounting unavailable", info.probe_errors[1])

    def test_missing_sentinels_are_not_reported_as_scheduler_paths(self) -> None:
        for sentinel in ("", "(null)", "(none)", "None", "N/A", "Unknown"):
            with self.subTest(sentinel=sentinel):
                self._scontrol("JobId=123 JobName=train JobState=COMPLETED")
                self._sacct(f"123|train|COMPLETED|{sentinel}|{sentinel}\n")

                unresolved = resolve_job_log_info("cluster-b", "123", archive_dir=None)
                fallback = resolve_job_log_info("cluster-b", "123", archive_dir="/archive")

                self.assertIsNone(unresolved.stdout)
                self.assertIsNone(unresolved.stderr)
                self.assertEqual(unresolved.job_name, "train")
                self.assertEqual(fallback.stdout, "/archive/123.out")
                self.assertEqual(fallback.stderr, "/archive/123.err")
                self.assertEqual(fallback.stdout_source, "archive:config")
                self.assertFalse(fallback.verified)

    def test_archive_fills_only_missing_stream_and_remains_unverified(self) -> None:
        self._scontrol("JobId=123 StdOut=/real/123.out StdErr=Unknown")
        self._sacct("123|train|COMPLETED|N/A|None\n")

        info = resolve_job_log_info("cluster-b", "123", archive_dir="/archive/")

        self.assertEqual(info.stdout, "/real/123.out")
        self.assertEqual(info.stderr, "/archive/123.err")
        self.assertEqual(info.stdout_source, "scontrol")
        self.assertEqual(info.stderr_source, "archive:config")
        self.assertFalse(info.verified)

    def test_transport_failure_does_not_promote_archive_guesses(self) -> None:
        self._scontrol(error="connection closed", rc=255)
        self._sacct("123|train|RUNNING|/real/123.out|Unknown\n")

        info = resolve_job_log_info("cluster-b", "123", archive_dir="/archive")

        self.assertEqual(info.stdout, "/real/123.out")
        self.assertIsNone(info.stderr)
        self.assertIsNone(info.stderr_source)
        self.assertFalse(info.verified)
        self.assertIn("connection closed", info.probe_errors[0])

    def test_foreign_cluster_tracking_is_not_attributed(self) -> None:
        self._tracking("cluster-a")
        self._scontrol("JobId=123 JobName=foreign JobState=RUNNING Command=(null)")

        rc, payload = self._show("cluster-b")

        self.assertEqual(rc, 0)
        self.assertEqual(payload["job_name"], "foreign")
        self.assertNotIn("launcher", payload)

    def test_stale_same_cluster_script_identity_is_not_attributed(self) -> None:
        self._tracking("cluster-b", remote_sbatch="/old run/train.sbatch")
        self._scontrol(
            "JobId=123 JobName=foreign JobState=RUNNING Command=/new run/train.sbatch"
        )

        rc, payload = self._show()

        self.assertEqual(rc, 0)
        self.assertNotIn("launcher", payload)

    def test_stale_accounting_submission_identity_is_not_attributed(self) -> None:
        self._tracking("cluster-b", sbatch_command="sbatch --parsable '/old run/train.sbatch'")
        self._scontrol(rc=1)
        self._sacct(
            "123|foreign|COMPLETED|acc|2026-03-26T12:00:00|Unknown|Unknown|"
            "node01|1|gpu:1|/work|/logs/123.out|/logs/123.err|"
            "sbatch --parsable '/new run/train.sbatch'\n"
        )

        rc, payload = self._show()

        self.assertEqual(rc, 0)
        self.assertNotIn("launcher", payload)

    def test_same_cluster_matching_identity_reads_launcher_metadata(self) -> None:
        self._tracking("cluster-b", remote_sbatch="/old run/train.sbatch")
        self._scontrol(
            "JobId=123 JobName=train JobState=RUNNING Command=/old run/train.sbatch"
        )

        rc, payload = self._show()

        self.assertEqual(rc, 0)
        self.assertEqual(payload["launcher"]["runtime_kind"], "singularity")
        self.assertEqual(payload["launcher"]["entry_command"], "python train.py")

    def test_narrow_accounting_fallback_survives_unsupported_wide_fields(self) -> None:
        self._scontrol(error="Invalid job id specified", rc=1)
        self._sacct(
            error="Invalid field requested: SubmitLine",
            rc=1,
            narrow="123|finished training|FAILED|/logs/%j.out|/logs/%j.err\n",
        )

        rc, payload = self._show()

        self.assertEqual(rc, 0)
        self.assertEqual(payload["job_name"], "finished training")
        self.assertEqual(payload["state"], "FAILED")
        self.assertEqual(payload["stdout"], "/logs/123.out")
        self.assertEqual(payload["stderr"], "/logs/123.err")
        self.assertNotIn("partition", payload)

    def test_requested_sbatch_is_returned_and_failure_is_reported(self) -> None:
        script = "#!/bin/bash\n#SBATCH --time=00:10:00\necho train\n"
        self._scontrol("JobId=123 JobName=train JobState=RUNNING", sbatch=script)

        rc, payload = self._show(include_sbatch=True)

        self.assertEqual(rc, 0)
        self.assertEqual(payload["sbatch"], script)
        self._scontrol(
            "JobId=123 JobName=train JobState=COMPLETED",
            sbatch_rc=1,
            sbatch_error="batch script has expired",
        )
        rc, payload = self._show(include_sbatch=True)
        self.assertEqual(rc, 1)
        self.assertFalse(payload["ok"])
        self.assertNotIn("sbatch", payload)
        self.assertIn("batch script has expired", payload["error"])

    def test_recent_jobs_filters_state_modifier_and_normalizes_missing_values(self) -> None:
        self._sacct(recent=(
            "1|job-a|RUNNING|acc|2026-03-26T12:00:00|Unknown|Unknown|00:01:00\n"
            "2|job-b|CANCELLED by 4840|acc|2026-03-26T11:00:00|Unknown|Unknown|00:01:00\n"
        ))
        output = io.StringIO()
        with patch("launcher.job_tools.console", Console(file=output, width=120)):
            rc = list_recent_jobs(
                "cluster-b", user=None, hours=24, limit=5,
                states={"cancelled"}, json_output=True,
            )
        payload = json.loads(output.getvalue())

        self.assertEqual(rc, 0)
        self.assertEqual([job["job_id"] for job in payload["jobs"]], ["2"])
        self.assertEqual(payload["jobs"][0]["start"], "")
        self.assertEqual(payload["jobs"][0]["end"], "")


if __name__ == "__main__":
    unittest.main()
