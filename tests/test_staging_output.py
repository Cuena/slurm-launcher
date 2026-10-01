from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("rsync"), "rsync is required")
class StagingOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        # Execute the remote commands locally; rsync itself remains real.
        ssh = self.bin / "ssh"
        ssh.write_text(
            '#!/bin/sh\n[ "$1" = "local-smoke" ] || exit 99\n'
            'shift\nexec /bin/sh -c "$*"\n',
            encoding="utf-8",
        )
        ssh.chmod(0o755)
        self.environment = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ.get('PATH', '')}",
        }
        self.config = self.project / "config.py"
        self.config.write_text(
            f"LOCAL_ROOT = {str(self.project)!r}\n"
            "CLUSTER_LOGIN = 'local-smoke'\n"
            f"REMOTE_WORKSPACE_BASE = {str(self.root / 'remote-work')!r}\n"
            f"REMOTE_LOG_BASE_PATH = {str(self.root / 'remote-logs')!r}\n"
            "RUNTIME_MODE = 'native'\n"
            "RUN_JOBS = ['train']\n"
            "JOBS = [{'name': 'train', 'command': 'true'}]\n",
            encoding="utf-8",
        )
        (self.project / "input.dat").write_bytes(b"staged input\n")

    def stage(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "launcher",
                "stage",
                "--config",
                str(self.config),
                "--json",
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=self.environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_success_keeps_progress_out_of_json_stdout(self) -> None:
        result = self.stage()
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["ok"])
        self.assertEqual(
            (Path(payload["remote_workdir"]) / "input.dat").read_bytes(),
            b"staged input\n",
        )

    def test_failure_preserves_json_and_rsync_diagnostics(self) -> None:
        with self.config.open("a", encoding="utf-8") as config:
            config.write("EXTRA_RSYNC_ARGS = ['--slurm-launcher-invalid-option']\n")
        result = self.stage()
        self.assertEqual(result.returncode, 1)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("--slurm-launcher-invalid-option", result.stderr)
        self.assertEqual(
            json.loads(Path(payload["tracking_file"]).read_text())["stage_state"],
            "staging",
        )


if __name__ == "__main__":
    unittest.main()
