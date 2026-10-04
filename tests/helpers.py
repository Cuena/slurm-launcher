from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from launcher.core import LauncherSettings


def write_tracking_file(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return path


def make_settings(**overrides: object) -> LauncherSettings:
    defaults: dict[str, object] = {
        "cluster_login": "user@cluster",
        "rsync_login": None,
        "ssh_config_file": None,
        "ssh_options": [],
        "remote_workspace_base": "/remote/workspaces",
        "remote_log_base_path": "/remote/logs",
        "workspace_mode": "per-run",
        "remote_workspace_dir": None,
        "project_root": Path("/tmp/project"),
        "project_prefix": "project",
        "venv_python_executable": None,
        "default_env": {},
        "default_sbatch": {},
        "extra_rsync_excludes": [],
        "extra_rsync_args": [],
        "remote_slurm_dashboard_log_archive_dir": None,
        "remote_slurm_dashboard_log_view_dir": None,
        "runtime_mode": "native",
        "singularity_image_path": None,
        "singularity_exec_flags": [],
        "artifact_paths": [],
        "require_clean_git": False,
        "sync_symlinks": "copy-links",
    }
    defaults.update(overrides)
    return LauncherSettings(**defaults)


class LocalScheduler:
    """Run generated submission Bash against a deterministic local scheduler."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "fixture-bin"
        self.bin.mkdir()
        self.modes = root / "scheduler-modes.json"
        self.modes.write_text("[]")
        self.accepted = root / "accepted-jobs"
        self.trap_exit = False
        self.disconnect_after_dispatch = False
        self.disconnect_returncode = 255
        self.before_dispatch = None
        self.environment = os.environ.copy()
        self.environment["PATH"] = f"{self.bin}:{self.environment['PATH']}"
        self.environment.pop("BASH_ENV", None)
        self.environment["FIXTURE_SCHEDULER_ROOT"] = str(root)
        scheduler = self.bin / "sbatch"
        scheduler.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "root = Path(os.environ['FIXTURE_SCHEDULER_ROOT'])\n"
            "mode_file = root / 'scheduler-modes.json'\n"
            "modes = json.loads(mode_file.read_text())\n"
            "mode = modes.pop(0) if modes else 'accept'\n"
            "mode_file.write_text(json.dumps(modes))\n"
            "if mode == 'reject':\n"
            "    print('sbatch: error: request rejected', file=sys.stderr)\n"
            "    sys.exit(1)\n"
            "accepted = root / 'accepted-jobs'\n"
            "previous = accepted.read_text().splitlines() if accepted.exists() else []\n"
            "job_id = str(7001 + len(previous))\n"
            "with accepted.open('a') as handle:\n"
            "    handle.write(job_id + '\\n')\n"
            "if mode in {'ambiguous', 'ambiguous_nonzero'}:\n"
            "    print('scheduler warning')\n"
            "print(job_id)\n"
            "if mode in {'accept_nonzero', 'ambiguous_nonzero'}:\n"
            "    sys.exit(1)\n"
        )
        scheduler.chmod(0o755)

    def install_rsync(self) -> None:
        transfer = self.bin / "rsync"
        transfer.write_text(
            f"#!{sys.executable}\n"
            "import shutil, sys\n"
            "from pathlib import Path\n"
            "source = Path(sys.argv[-2].rstrip('/'))\n"
            "destination = Path(sys.argv[-1].split(':', 1)[1])\n"
            "if source.is_dir():\n"
            "    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git', 'slurm_output', '__pycache__'))\n"
            "else:\n"
            "    destination.parent.mkdir(parents=True, exist_ok=True)\n"
            "    shutil.copy2(source, destination)\n"
        )
        transfer.chmod(0o755)

    def set_modes(self, *modes: str) -> None:
        self.modes.write_text(json.dumps(modes))

    def job_ids(self) -> list[str]:
        return self.accepted.read_text().splitlines() if self.accepted.exists() else []

    def capture(self, login: str, script: str, **kwargs) -> subprocess.CompletedProcess:
        if self.before_dispatch is not None:
            self.before_dispatch()
        environment = self.environment.copy()
        if self.trap_exit:
            trap = self.root / "bash-env"
            trap.write_text("trap 'echo site_cleanup_warning; exit 1' EXIT\n")
            environment["BASH_ENV"] = str(trap)
        result = subprocess.run(
            ["bash", "-s"], input=script, text=True, capture_output=True,
            env=environment,
        )
        if self.disconnect_after_dispatch:
            result.stdout = "".join(
                line for line in result.stdout.splitlines(keepends=True)
                if not line.startswith("__SLURM_LAUNCHER_")
            )
            result.returncode = self.disconnect_returncode
            result.stderr += "connection closed\n"
        return result
