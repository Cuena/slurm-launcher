from __future__ import annotations

import base64
import shlex
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from launcher.core import (
    JobSpec,
    RemotePaths,
    build_predefined_sbatch_command,
    parse_job_id,
    submit_job,
    sync_project,
)
from tests.helpers import make_settings


class CommandContractTests(unittest.TestCase):
    def _remote_paths(self) -> RemotePaths:
        return RemotePaths(
            "project_001",
            "/remote/workspaces/project_001",
            "/remote/logs/project_001",
            "/remote/logs/project_001/slurm_output",
        )

    def test_sync_uses_distinct_control_and_transfer_hosts(self) -> None:
        settings = make_settings(
            cluster_login="user@alogin",
            rsync_login="user@transfer1",
            ssh_config_file="/dev/null",
            ssh_options=["-o", "BatchMode=yes"],
        )
        commands = sync_project(
            settings, self._remote_paths(), dry_run=True, quiet=True
        )
        self.assertIn("user@alogin", commands[0])
        self.assertIn("user@transfer1:/remote/workspaces/project_001/", commands[1])
        self.assertIn("-e 'ssh -F /dev/null -o BatchMode=yes'", commands[1])

    def test_parsable_output_never_guesses_a_job_id(self) -> None:
        self.assertEqual(parse_job_id("12345;cluster-a\n"), "12345")
        for output in (
            "",
            "warning\n12345\n",
            "Submitted batch job 12345",
            "12345\n12346",
            "0",
            "12345_2",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_job_id(output)

    def test_frozen_handwritten_script_preserves_exact_bytes(self) -> None:
        script = (
            "#!/bin/bash\r\n#SBATCH --time=00:01:00\r\necho 'SBATCH_SCRIPT'\r\n\r\n"
        )
        result = submit_job(
            make_settings(),
            self._remote_paths(),
            JobSpec(name="train", sbatch_file="gone.sbatch"),
            dry_run=True,
            quiet=True,
            frozen_script=script,
        )
        transfer = next(
            line
            for line in result.commands[0].splitlines()
            if line.startswith("printf %s ")
        )
        encoded = shlex.split(transfer)[2]
        self.assertEqual(base64.b64decode(encoded), script.encode())
        self.assertIn("sbatch --parsable ", result.sbatch_command)

    def test_build_predefined_command_rejects_outside_root(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "sbatch_file must stay inside LOCAL_ROOT"
        ):
            build_predefined_sbatch_command(
                make_settings(project_root=Path("/tmp/project")),
                self._remote_paths(),
                JobSpec(name="shared", sbatch_file="../shared/train.sbatch"),
            )

    def test_acknowledgment_precedes_failing_log_enrichment(self) -> None:
        acknowledged = []

        def failed_probe(*args):
            self.assertEqual(acknowledged, ["12345"])
            raise OSError("probe unavailable")

        with (
            patch("launcher.core.ssh_script", return_value=("12345\n", "")),
            patch(
                "launcher.core.resolve_submitted_job_log_paths",
                side_effect=failed_probe,
            ),
        ):
            result = submit_job(
                make_settings(),
                self._remote_paths(),
                JobSpec(name="train", sbatch_file="gone.sbatch"),
                dry_run=False,
                quiet=True,
                frozen_script="#!/bin/bash\ntrue\n",
                on_acknowledged=lambda result: acknowledged.append(result.job_id),
            )
        self.assertEqual(result.job_id, "12345")

    def test_acknowledged_id_survives_transport_nonzero_exit(self) -> None:
        failure = subprocess.CalledProcessError(
            255, ["ssh"], output="12345\n", stderr="connection closed"
        )
        acknowledged = []
        with patch("launcher.core.ssh_script", side_effect=failure):
            result = submit_job(
                make_settings(),
                self._remote_paths(),
                JobSpec(name="train", command="true"),
                dry_run=False,
                quiet=True,
                on_acknowledged=lambda result: acknowledged.append(result.job_id),
            )
        self.assertEqual(result.job_id, "12345")
        self.assertEqual(acknowledged, ["12345"])


if __name__ == "__main__":
    unittest.main()
