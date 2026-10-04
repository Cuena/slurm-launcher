from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ConfigurationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.py"
        self.bin = self.root / "bin"
        self.bin.mkdir()
        ssh = self.bin / "ssh"
        ssh.write_text(
            '#!/bin/sh\n[ "$1" = "local-test" ] || exit 99\n'
            'shift\nexec /bin/sh -c "$*"\n',
            encoding="utf-8",
        )
        ssh.chmod(0o755)
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
        }

    def configure(self, jobs: str, *, workspace: str = "/work", extra: str = "") -> None:
        self.config.write_text(
            f"LOCAL_ROOT = {str(self.root)!r}\n"
            "CLUSTER_LOGIN = 'local-test'\n"
            f"REMOTE_WORKSPACE_BASE = {workspace!r}\n"
            "REMOTE_LOG_BASE_PATH = '/logs'\n"
            "RUN_JOBS = []\n"
            f"JOBS = {jobs}\n{extra}",
            encoding="utf-8",
        )

    def invoke(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "launcher", *arguments],
            env=self.environment,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )

    def test_previews_and_invalid_input_never_prepare_or_create_run_state(self) -> None:
        marker = self.root / "prepared"
        for jobs, workspace, commands, expected in (
            ("[{'name': 'train', 'command': 'true'}]", "/work",
             ("validate", "render", "stage", "run"), 0),
            ("[{'name': '../train', 'command': 'true'}]", "/work",
             ("validate", "render", "stage", "run"), 1),
            ("[{'command': 'true'}]", "/work", ("validate", "render"), 1),
            ("[{'name': 'train', 'command': 'true'}]", "relative", ("run",), 1),
        ):
            for command in commands:
                with self.subTest(jobs=jobs, workspace=workspace, command=command):
                    self.configure(
                        jobs, workspace=workspace,
                        extra=f"def prepare():\n    open({str(marker)!r}, 'w').write('prepared')\n",
                    )
                    selection = (
                        ("--all", "--dry-run") if expected == 0 or workspace == "relative"
                        else ("--all",)
                    ) if command in {"stage", "run"} else ()
                    result = self.invoke(
                        command, "--config", str(self.config), *selection, "--json"
                    )
                    self.assertEqual(result.returncode, expected, result.stderr)
                    self.assertEqual(json.loads(result.stdout)["ok"], expected == 0)
                    self.assertFalse(marker.exists())
                    self.assertFalse((self.root / "slurm_output").exists())

    def test_doctor_ssh_success_has_one_json_result(self) -> None:
        result = self.invoke("doctor", "--cluster-login", "local-test", "--ssh", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ssh_ok"])
        self.assertEqual(set(payload["remote_tools"]), {"sacct", "scontrol", "squeue"})

    @unittest.skipUnless(shutil.which("git"), "Git is required")
    def test_init_ignores_generated_state_but_preserves_shareable_example(self) -> None:
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        result = self.invoke("init", "--non-interactive")
        self.assertEqual(result.returncode, 0, result.stderr)
        generated = [
            ".slurm/remote_launcher_config.mn5.py",
            ".slurm/__pycache__/config.pyc",
            "slurm_output/run/jobs.json",
        ]
        ignored = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input="\n".join(generated) + "\n",
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(set(ignored.stdout.splitlines()), set(generated))
        example = subprocess.run(
            ["git", "check-ignore", "--quiet", ".slurm/remote_launcher_config.mn5.example.py"],
            cwd=self.root,
            check=False,
        )
        self.assertEqual(example.returncode, 1)


if __name__ == "__main__":
    unittest.main()
