# examples/remote_launcher_config.mn5.example.py
# What: MN5-oriented launcher config with reusable sbatch helper functions.
# Why: Keeps a shareable MN5 example while personal credentials stay local-only.
# RELEVANT FILES: examples/remote_launcher_config.mn5.py, launcher/templates/config.py.template, launcher/cli.py, README.md

from __future__ import annotations

from pathlib import Path
from typing import Any

CONFIG_DIR = Path(__file__).resolve().parent
LOCAL_ROOT = CONFIG_DIR.parent

CLUSTER_LOGIN = "your_user@alogin1.bsc.es"
RSYNC_LOGIN = "your_user@transfer1.bsc.es"
SSH_CONFIG_FILE: str | None = None
SSH_OPTIONS: list[str] = []
WORKSPACE_MODE = "per-run"  # per-run | fixed
REMOTE_WORKSPACE_BASE = "/absolute/path/on/cluster/slurm-launcher/work"
REMOTE_WORKSPACE_DIR: str | None = None
REMOTE_LOG_BASE_PATH = "/absolute/path/on/cluster/slurm-launcher/logs"
REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR = (
    "/absolute/path/on/cluster/slurm-dashboard/logs"
)
REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR = (
    "/absolute/path/on/cluster/slurm-dashboard/projects"
)

PROJECT_NAME = "slurm-launcher"
ARTIFACT_PATHS = ["outputs"]

RUNTIME_MODE = "native"  # native | venv | singularity
VENV_PYTHON_EXECUTABLE: str | None = None
SINGULARITY_IMAGE_PATH: str | None = None
SINGULARITY_EXEC_FLAGS: list[str] = []

MN5_ACCOUNT = "your_mn5_account"


def mn5_accel_sbatch(
    *,
    gpus: int = 1,
    qos: str = "acc_debug",
    job_time: str = "00-00:30:00",
) -> dict[str, Any]:
    cpus_per_task = max(1, 20 * max(1, gpus))
    return {
        "account": MN5_ACCOUNT,
        "qos": qos,
        "time": job_time,
        "ntasks": 1,
        "cpus-per-task": cpus_per_task,
        "gres": f"gpu:{max(1, gpus)}",
    }


def mn5_multinode_accel_sbatch(
    *,
    nodes: int,
    gpus_per_node: int = 4,
    cpus_per_task: int = 80,
    qos: str = "acc_debug",
    job_time: str = "02:00:00",
) -> dict[str, Any]:
    return {
        "account": MN5_ACCOUNT,
        "partition": "acc",
        "qos": qos,
        "time": job_time,
        "nodes": max(1, nodes),
        "ntasks-per-node": 1,
        "cpus-per-task": max(1, cpus_per_task),
        "gres": f"gpu:{max(1, gpus_per_node)}",
    }


def mn5_cpu_sbatch(
    *,
    qos: str = "gp_debug",
    job_time: str = "00-01:00:00",
) -> dict[str, Any]:
    return {
        "account": MN5_ACCOUNT,
        "qos": qos,
        "time": job_time,
        "ntasks": 1,
        "cpus-per-task": 1,
        "nodes": 1,
        "ntasks-per-node": 1,
    }


DEFAULT_ENV = {
    # "PYTHONUNBUFFERED": "1",
}
DEFAULT_SBATCH = {
    "account": MN5_ACCOUNT,
}
EXTRA_RSYNC_EXCLUDES = [".git/", ".venv/", "slurm_output/"]
EXTRA_RSYNC_ARGS: list[str] = []
VERBOSE = True

RUN_JOBS: list[str] = ["train_gpu"]

JOBS = [
    {
        "name": "train_gpu",
        "command": "python3 examples/demo_train.py --config configs/train_small.yaml --epochs 2 --lr 1e-4",
        "env": {
            "EXPERIMENT_NAME": "train_gpu",
        },
        "sbatch": {
            **mn5_accel_sbatch(gpus=1, qos="acc_debug", job_time="00:30:00"),
            "job-name": "mn5_train_gpu",
        },
    },
    {
        "name": "eval_cpu",
        "command": "python3 examples/demo_eval.py --config configs/eval.yaml --ckpt checkpoints/latest.pt",
        "env": {
            "EXPERIMENT_NAME": "eval_cpu",
        },
        "sbatch": {
            **mn5_cpu_sbatch(qos="gp_debug", job_time="00-00:20:00"),
            "job-name": "mn5_eval_cpu",
        },
    },
    {
        "name": "custom_shell_command",
        "command": "srun ./my_application_executable arg1 arg2",
        "setup": [
            "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}",
        ],
        "sbatch": {
            **mn5_cpu_sbatch(qos="gp_debug", job_time="00-00:15:00"),
            "job-name": "mn5_shell_command",
        },
    },
    {
        "name": "multinode_torchrun",
        "command": (
            "srun torchrun "
            "--nnodes=$SLURM_NNODES "
            "--nproc-per-node=$GPUS_PER_NODE "
            "--node-rank=$SLURM_NODEID "
            "--rdzv-backend=c10d "
            "--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT "
            "scripts/train_distributed.py --config configs/train.yaml"
        ),
        "setup": [
            "export GPUS_PER_NODE=4",
            "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)",
            "export MASTER_PORT=6000",
        ],
        "sbatch": {
            **mn5_multinode_accel_sbatch(
                nodes=4,
                gpus_per_node=4,
                cpus_per_task=80,
                qos="acc_debug",
                job_time="2:00:00",
            ),
            "job-name": "mn5_multinode_torchrun",
        },
    },
]
