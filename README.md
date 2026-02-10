# slurm-launcher

Reusable remote SLURM launcher with a small Python CLI.

## What this is for

This repo is infrastructure for running another codebase (for example, an AI training repo) on a remote SLURM cluster.

It automates the repetitive workflow:

- sync project files to the cluster (`rsync`)
- create per-run remote work/log folders
- generate and submit `sbatch` scripts over SSH
- track submitted job IDs and log paths locally
- inspect logs, monitor queue state, and download `.out/.err` files

The launcher is SLURM-cluster agnostic, but the bundled examples are MN5-oriented.

## Repository layout

- `launcher/`: launcher package (`uv run slurm-launcher ...` or `uv run -m launcher ...`)
- `launcher/templates/config.py.template`: starter config copied by `init`
- `examples/remote_launcher_config.demo.py`: minimal dry-run example
- `examples/remote_launcher_config.mn5.example.py`: MN5-oriented example

## Requirements

- `uv`
- `ssh` access to a SLURM cluster
- `rsync` available locally
- `git` optional (used only to include commit hash in job folder names)

## Quick start

1. Create a config:
   - `uv run slurm-launcher init`
2. Edit config:
   - repo-local mode: `remote_launcher_config.py`
   - wrapper mode: `.slurm/remote_launcher_config.mn5.py`
   - set `CLUSTER_LOGIN`
   - set `REMOTE_BASE_PATH`
   - if you are not on MN5/BSC, replace MN5-specific account/QoS/path defaults
   - optional (for slurm-dashboard): set `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR`
   - optional (for slurm-dashboard organization): set `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR`
   - define `JOBS`
3. Validate commands without submission:
   - `uv run slurm-launcher --dry-run`
4. Submit jobs:
   - `uv run slurm-launcher`

## Use from another repo (wrapper mode)

If your project code lives in a different repo, keep launcher code separate and keep
config local to each project repo.

1. Clone launcher once (outside your project repo):
   - `git clone https://github.com/<you-or-org>/slurm-launcher.git`
2. In the project repo, install launcher once (editable for fast iteration):
   - `uv add --editable /path/to/slurm-launcher`
3. Bootstrap `.slurm` config files and ignore rules:
   - `bash /path/to/slurm-launcher/scripts/init_wrapper_repo.sh .`
4. Edit your private config:
   - `.slurm/remote_launcher_config.mn5.py`
5. Run dry-run:
   - `uv run slurm-launcher --dry-run`
6. Submit:
   - `uv run slurm-launcher`

No-install fallback (longer command each time):

- `uv run --with-editable /path/to/slurm-launcher slurm-launcher --dry-run`

What the init script creates in the project repo:

- `.slurm/remote_launcher_config.mn5.py` (private, not committed)
- `.slurm/remote_launcher_config.mn5.example.py` (sanitized, commit this one)
- `.gitignore` entries:
  - `.slurm/*.py`
  - `!.slurm/*.example.py`

## Common commands

- `uv run slurm-launcher init --force`: overwrite existing config
- `uv run slurm-launcher --only train eval`: run a subset of jobs
- `uv run slurm-launcher --no-sync`: skip rsync
- `uv run slurm-launcher --config path/to/config.py`: custom config path
- default config lookup for `run`: `.slurm/remote_launcher_config.mn5.py`, then `remote_launcher_config.py`
- `uv run slurm-launcher logs`: show tracked `.out/.err` paths from latest run
- `uv run slurm-launcher logs --json`: print full tracking payload
- `uv run slurm-launcher download-logs`: download tracked `.out/.err` files from latest run
- `uv run slurm-launcher monitor`: run `squeue` for tracked job IDs from the latest run
- `uv run slurm-launcher monitor --dry-run`: print the monitoring command only

## Download logs locally

Use `download-logs` to fetch remote `.out/.err` files recorded in a tracking file.

- Download logs for all jobs in latest project run:
  - `uv run slurm-launcher download-logs`
- Download only one job by name:
  - `uv run slurm-launcher download-logs --job-name train_gpu`
- Download only one job by id:
  - `uv run slurm-launcher download-logs --job-id 36114735`
- Use a specific tracking file:
  - `uv run slurm-launcher download-logs --tracking-file slurm_output/<job_folder>/jobs.json`
- Preview rsync commands without downloading:
  - `uv run slurm-launcher download-logs --dry-run`

Script compatibility:

- `uv run python scripts/download_logs.py ...` still works from the launcher repository root.

## Runtime modes

- `native` (default): run each job `command` as written
- `venv`: source the environment from `VENV_PYTHON_EXECUTABLE`
- `singularity`: run jobs with `singularity exec`
  - set `SINGULARITY_IMAGE_PATH`
  - optional `SINGULARITY_EXEC_FLAGS` (for example `["--nv"]`)

## Config contract

Required top-level settings:

- `CLUSTER_LOGIN`: remote SSH login (`user@host`)
- `REMOTE_BASE_PATH`: remote directory where each run workdir is created
- `JOBS`: list of job dictionaries

Optional top-level settings:

- `LOCAL_ROOT`, `PROJECT_NAME`
- `REMOTE_LOG_BASE_PATH`
- `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR`
- `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR`
- `RUNTIME_MODE`, `VENV_PYTHON_EXECUTABLE`, `SINGULARITY_IMAGE_PATH`, `SINGULARITY_EXEC_FLAGS`
- `DEFAULT_ENV`, `DEFAULT_SBATCH`, `RUN_JOBS`
- `EXTRA_RSYNC_EXCLUDES`, `EXTRA_RSYNC_ARGS`, `VERBOSE`

## Job config model

Each job in `JOBS` is a dictionary with:

- required: `name`, `command`
- optional: `setup`, `env`, `sbatch`

Examples:

- `{"name": "train", "command": "python3 scripts/train.py --config-name=train"}`
- `{"name": "prep", "command": "bash scripts/prep.sh"}`
- `{"name": "eval", "command": "srun python3 scripts/eval.py"}`

Global defaults:

- `DEFAULT_ENV`: merged into each job `env`
- `DEFAULT_SBATCH`: merged into each job `sbatch`
- `RUN_JOBS`: optional list of job names to run by default

## MN5 defaults

`examples/remote_launcher_config.mn5.example.py` includes helper builders:

- `mn5_accel_sbatch(...)`
- `mn5_cpu_sbatch(...)`

Use `examples/remote_launcher_config.mn5.py` for personal MN5 credentials; it is gitignored.

Dry-run examples:

- `uv run slurm-launcher --config examples/remote_launcher_config.mn5.example.py --dry-run`
- `uv run slurm-launcher --config examples/remote_launcher_config.demo.py --dry-run`

## Outputs

Each run creates a unique job folder (timestamp + git hash when available).

- local artifacts: `slurm_output/<job_folder>/`
- tracking file: `slurm_output/<job_folder>/jobs.json`
- latest run index: `slurm_output/latest_jobs.json`
- default remote logs (no archive dir configured): `<job-name>-<job-id>.out` and `<job-name>-<job-id>.err`

## slurm-dashboard compatibility

To make this launcher write logs in the convention used by `slurm-dashboard-static`,
set an absolute shared archive directory in your config:

```python
REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR = "/home/bsc/<user>/.slurm-dashboard/logs"
REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR = "/home/bsc/<user>/.slurm-dashboard/projects"  # optional
```

With this set, launcher defaults become:

- `--output=/home/bsc/<user>/.slurm-dashboard/logs/%j.out`
- `--error=/home/bsc/<user>/.slurm-dashboard/logs/%j.err`

The launcher creates the archive directory automatically before submission.

If `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR` is also set, launcher creates
human-friendly symlinks after each submit:

- `<view_dir>/<project>/<YYYY-MM-DD>/<job-name>-<job-id>.out -> <archive_dir>/<job-id>.out`
- `<view_dir>/<project>/<YYYY-MM-DD>/<job-name>-<job-id>.err -> <archive_dir>/<job-id>.err`

This keeps job-id based recovery for the TUI and adds browsable per-project views.
