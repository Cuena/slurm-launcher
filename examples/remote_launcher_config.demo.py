"""Minimal demo launcher config.

Use this file with dry-run:
    uv run slurm-launcher --config examples/remote_launcher_config.demo.py --dry-run
"""

from __future__ import annotations

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = CONFIG_DIR.parent

CLUSTER_LOGIN = "your_user@mn5.bsc.es"
WORKSPACE_MODE = "per-run"  # per-run | fixed
REMOTE_WORKSPACE_BASE = "/absolute/path/on/cluster/slurm-launcher-demo/work"
REMOTE_WORKSPACE_DIR: str | None = None
REMOTE_LOG_BASE_PATH = "/absolute/path/on/cluster/slurm-launcher-demo/logs"
REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR: str | None = None
REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR: str | None = None

PROJECT_NAME = "slurm-launcher-demo"

RUNTIME_MODE = "native"  # native | venv | singularity
VENV_PYTHON_EXECUTABLE: str | None = None
SINGULARITY_IMAGE_PATH: str | None = None
SINGULARITY_EXEC_FLAGS: list[str] = []

DEFAULT_ENV = {
    "PYTHONUNBUFFERED": "1",
}
DEFAULT_SBATCH = {
    "account": "your_mn5_account",
    "qos": "gp_debug",
    "time": "00-00:20:00",
    "ntasks": 1,
    "cpus-per-task": 2,
}

EXTRA_RSYNC_EXCLUDES = [".git/", "slurm_output/", ".venv/"]
EXTRA_RSYNC_ARGS: list[str] = []
VERBOSE = True

RUN_JOBS: list[str] = []

JOBS = [
    {
        "name": "train",
        "command": "python3 examples/demo_train.py --config configs/train.yaml --epochs 2 --lr 1e-4",
        "env": {"EXPERIMENT_NAME": "train"},
        "sbatch": {
            "qos": "acc_debug",
            "gres": "gpu:1",
            "cpus-per-task": 20,
        },
    },
    {
        "name": "eval",
        "command": "python3 examples/demo_eval.py --config configs/eval.yaml --ckpt checkpoints/latest.pt",
        "env": {"EXPERIMENT_NAME": "eval"},
    },
    {
        "name": "shell_example",
        "command": "srun bash scripts/run_eval.sh --config-name eval",
    },
]
