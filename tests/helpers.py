from __future__ import annotations

import json
from pathlib import Path

from launcher.core import LauncherSettings


def write_tracking_file(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")
    return path


def make_settings(**overrides: object) -> LauncherSettings:
    defaults: dict[str, object] = {
        "cluster_login": "user@cluster",
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
        "verbose": False,
    }
    defaults.update(overrides)
    return LauncherSettings(**defaults)
