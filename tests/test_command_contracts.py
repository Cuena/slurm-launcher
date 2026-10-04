from __future__ import annotations

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from unittest.mock import patch

from launcher.core import (
    JobSpec,
    RemotePaths,
    build_job_script,
    format_sbatch_options,
    parse_job_id,
    submit_job,
)
from tests.helpers import LocalScheduler, make_settings


class CommandContractTests(unittest.TestCase):
    def test_parsable_output_never_guesses_a_job_id(self) -> None:
        self.assertEqual(parse_job_id("12345;cluster-a\n"), "12345")
        for output in (
            "", "warning\n12345\n", "Submitted batch job 12345",
            "12345\n12346", "0", "12345_2",
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_job_id(output)

    def test_handwritten_submission_enriches_complete_spaced_stream_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cluster = LocalScheduler(root)
            paths = RemotePaths("run", str(root), str(root / "logs"), str(root / "logs/output"))
            scontrol = cluster.bin / "scontrol"
            scontrol.write_text(
                f"#!{sys.executable}\n"
                "print('JobId=7001 JobName=train JobState=PENDING StdOut=/logs/with spaces/train.out StdErr=(null) WorkDir=/work')\n"
            )
            scontrol.chmod(0o755)
            sacct = cluster.bin / "sacct"
            sacct.write_text(
                f"#!{sys.executable}\n"
                "print('7001|train|PENDING||/logs/with spaces/train.err|')\n"
            )
            sacct.chmod(0o755)
            with (
                patch("launcher.core.transport.run_ssh_capture", side_effect=cluster.capture),
                patch("launcher.job_tools.run_ssh_capture", side_effect=cluster.capture),
            ):
                result = submit_job(
                    make_settings(project_root=root), paths,
                    JobSpec(name="train", sbatch_file="gone.sbatch"),
                    dry_run=False, quiet=True, frozen_script="#!/bin/bash\ntrue\n",
                )
            self.assertEqual(result.sbatch_options["output"], "/logs/with spaces/train.out")
            self.assertEqual(result.sbatch_options["error"], "/logs/with spaces/train.err")

    def test_entire_shell_command_executes_inside_container(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cluster = LocalScheduler(root)
            singularity = cluster.bin / "singularity"
            singularity.write_text('#!/bin/bash\nshift 2\nexport CONTAINER_SCOPE=container\nexec "$@"\n')
            singularity.chmod(0o755)
            probe = cluster.bin / "probe"
            probe.write_text(
                f"#!{sys.executable}\nimport os, sys\n"
                "payload = sys.stdin.read() if sys.argv[1] == 'piped' else ''\n"
                "print(':'.join([sys.argv[1], os.environ.get('CONTAINER_SCOPE', 'host'), os.environ.get('MODE', ''), payload]))\n"
            )
            probe.chmod(0o755)
            settings = make_settings(
                runtime_mode="singularity", singularity_image_path=str(root / "image' name.sif")
            )
            script = build_job_script(
                JobSpec(name="train", command="MODE=train probe first && probe second && printf payload | probe piped"),
                settings, RemotePaths("run", str(root), str(root / "logs"), str(root / "logs/output")),
            )
            result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=cluster.environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["first:container:train:", "second:container::", "piped:container::payload"])

    def test_apostrophe_path_activates_real_virtualenv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment = root / "venv'odd"
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            command = "python -c " + shlex.quote("import sys; print(sys.prefix)")
            script = build_job_script(
                JobSpec(name="train", command=command),
                make_settings(runtime_mode="venv", venv_python_executable=str(environment / "bin/python")),
                RemotePaths("run", str(root), str(root / "logs"), str(root / "logs/output")),
            )
            shell_env = os.environ.copy()
            shell_env.pop("BASH_ENV", None)
            result = subprocess.run(["bash", "-s"], input=script, text=True, capture_output=True, env=shell_env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(environment))

    def test_explicit_ntasks_is_not_inflated_or_inferred(self) -> None:
        paths = RemotePaths("run", "/work", "/logs", "/logs/output")
        settings = make_settings()
        explicit = format_sbatch_options(
            JobSpec(name="train", command="true", sbatch={"nodes": 2, "ntasks-per-node": 8, "ntasks": 2}),
            settings, paths,
        )
        self.assertEqual(explicit["ntasks"], 2)
        implicit = format_sbatch_options(
            JobSpec(name="train", command="true", sbatch={"nodes": 2, "ntasks-per-node": 8}),
            settings, paths,
        )
        self.assertNotIn("ntasks", implicit)


if __name__ == "__main__":
    unittest.main()
