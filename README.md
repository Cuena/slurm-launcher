# slurm-launcher

Reusable remote SLURM launcher with a small Python CLI.

## What this is for

This repo is infrastructure for running another codebase (for example, an AI training repo) on a remote SLURM cluster.

It automates the repetitive workflow:

- stage project files to the cluster (`rsync`)
- choose remote workspace strategy: per-run folder or fixed remote project folder
- generate and submit `sbatch` scripts over SSH
- track submitted job IDs and log paths locally
- inspect logs, monitor queue state, and download `.out/.err` files

The launcher is SLURM-cluster agnostic, but the bundled examples are MN5-oriented.

## Repository layout

- `launcher/`: launcher package (`slurm-launcher ...` after tool install, or `uv run slurm-launcher ...`)
- `launcher/templates/config.py.template`: starter config copied by `init`
- `examples/remote_launcher_config.demo.py`: minimal dry-run example
- `examples/remote_launcher_config.mn5.example.py`: MN5-oriented example

## Requirements

- `uv`
- `ssh` access to a SLURM cluster
- `rsync` available locally (required for `stage` and `run`)
- `git` optional (used only to include commit hash in job folder names)

## Install once (recommended)

For day-to-day use across multiple repos, install launcher once as a uv tool:

1. Clone launcher once:
   - `git clone https://github.com/<you-or-org>/slurm-launcher.git`
2. Install the CLI in editable mode:
   - `uv tool install --editable /path/to/slurm-launcher`
3. Ensure your shell PATH is updated for uv tools:
   - `uv tool update-shell`
4. Verify:
   - `slurm-launcher --help`

If you prefer not to install as a tool, you can still run commands as
`uv run slurm-launcher ...` from an environment that has launcher available.

## Quick start

1. Create a config:
   - `slurm-launcher init` (creates `.slurm/remote_launcher_config.mn5.py` + `.slurm/remote_launcher_config.mn5.example.py`)
2. Edit config:
   - `.slurm/remote_launcher_config.mn5.py` (private, gitignored)
   - optional: commit `.slurm/remote_launcher_config.mn5.example.py` as a sanitized reference
   - set `CLUSTER_LOGIN`
   - set `WORKSPACE_MODE` (`per-run` or `fixed`)
   - set `REMOTE_WORKSPACE_BASE` for `WORKSPACE_MODE=per-run`
   - set `REMOTE_WORKSPACE_DIR` for `WORKSPACE_MODE=fixed`
   - if you are not on MN5/BSC, replace MN5-specific account/QoS/path defaults
   - optional (for slurm-dashboard): set `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR`
   - optional (for slurm-dashboard organization): set `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR`
   - define `JOBS`
3. Validate/preview without submission:
   - `slurm-launcher validate` (local config sanity checks)
   - `slurm-launcher render` (prints generated sbatch scripts)
   - `slurm-launcher stage --dry-run` (prints SSH/rsync commands)
   - `slurm-launcher submit --dry-run --job-folder <existing_folder>` (prints SSH/sbatch commands)
   - `slurm-launcher run --dry-run` (prints full stage + submit flow)
4. Run:
   - `slurm-launcher run`

Project-managed alternative (reproducible dependency per repo):

- `uv add --editable /path/to/slurm-launcher`
- run via `uv run slurm-launcher ...`

No-install fallback (longer command each time):

- `uv run --with-editable /path/to/slurm-launcher slurm-launcher --dry-run`

What the init script creates in the project repo:

- `.slurm/remote_launcher_config.mn5.py` (private, not committed)
- `.slurm/remote_launcher_config.mn5.example.py` (sanitized, commit this one)
- `.gitignore` entries:
  - `.slurm/*.py`
  - `!.slurm/*.example.py`

## Common commands

- `slurm-launcher init --force`: overwrite existing config
- `slurm-launcher run --only train eval`: run a subset of jobs
- `slurm-launcher run --workspace per-run`: run from a new per-run folder (stage + submit)
- `slurm-launcher run --workspace fixed`: run from `REMOTE_WORKSPACE_DIR` (stage + submit)
- `slurm-launcher --config path/to/config.py`: custom config path
- default config lookup for commands using config: `.slurm/remote_launcher_config.mn5.py`, then `remote_launcher_config.py`
- `slurm-launcher validate`: validate config without submission
- `slurm-launcher validate --ssh`: validate config and test SSH connectivity
- `slurm-launcher validate --ssh --check-remote-paths`: also check remote venv/singularity prerequisites (no writes)
- `slurm-launcher render`: print generated sbatch scripts without submission
- `slurm-launcher render --only train`: render only a subset of jobs
- `slurm-launcher stage`: run only the SSH + rsync stage phase (no job submission)
- `slurm-launcher stage --workspace fixed --dry-run`: print stage commands only
- `slurm-launcher submit --workspace per-run --job-folder <folder>`: submit only (no rsync)
- `slurm-launcher submit --workspace fixed`: submit only against fixed workspace
- `slurm-launcher logs`: show tracked `.out/.err` paths from latest run
- `slurm-launcher logs --json`: print full tracking payload
- `slurm-launcher download-logs`: download tracked `.out/.err` files from latest run
- `slurm-launcher monitor`: run `squeue` for tracked job IDs from the latest run
- `slurm-launcher monitor --dry-run`: print the monitoring command only

## Download logs locally

Use `download-logs` to fetch remote `.out/.err` files recorded in a tracking file.

- Download logs for all jobs in latest project run:
  - `slurm-launcher download-logs`
- Download only one job by name:
  - `slurm-launcher download-logs --job-name train_gpu`
- Download only one job by id:
  - `slurm-launcher download-logs --job-id 36114735`
- Use a specific tracking file:
  - `slurm-launcher download-logs --tracking-file slurm_output/<job_folder>/jobs.json`
- Preview rsync commands without downloading:
  - `slurm-launcher download-logs --dry-run`

Script compatibility:

- `uv run python scripts/download_logs.py ...` still works from the launcher repository root.

## Runtime modes

- `native` (default): run each job `command` as written
- `venv`: source the environment from `VENV_PYTHON_EXECUTABLE`
- `singularity`: run jobs with `singularity exec`
  - set `SINGULARITY_IMAGE_PATH`
  - optional `SINGULARITY_EXEC_FLAGS` (for example `["--nv"]`)

## Workspace modes

- `WORKSPACE_MODE=per-run` (default):
  - launcher creates a unique remote workdir under `REMOTE_WORKSPACE_BASE`
  - launcher rsyncs `LOCAL_ROOT` to that folder before submission
- `WORKSPACE_MODE=fixed`:
  - launcher runs jobs from `REMOTE_WORKSPACE_DIR` (fixed folder)
  - launcher rsyncs `LOCAL_ROOT` into that fixed folder before submission

CLI `--workspace` overrides `WORKSPACE_MODE` for one command.

## Config contract

Required top-level settings:

- `CLUSTER_LOGIN`: remote SSH login (`user@host`)
- `JOBS`: list of job dictionaries

Required settings by workspace mode:

- `WORKSPACE_MODE=per-run`: `REMOTE_WORKSPACE_BASE`
- `WORKSPACE_MODE=fixed`: `REMOTE_WORKSPACE_DIR`
- `REMOTE_LOG_BASE_PATH` is optional; when omitted it defaults to the selected workspace path.

Optional top-level settings:

- `LOCAL_ROOT`, `PROJECT_NAME`, `WORKSPACE_MODE`, `REMOTE_WORKSPACE_DIR`
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

- `slurm-launcher --config examples/remote_launcher_config.mn5.example.py --dry-run`
- `slurm-launcher --config examples/remote_launcher_config.demo.py --dry-run`

## Outputs

Each run creates a unique job folder (timestamp + git hash when available).

- local artifacts: `slurm_output/<job_folder>/`
- tracking file: `slurm_output/<job_folder>/jobs.json`
- latest run index: `slurm_output/latest_jobs.json`
- default remote logs (no archive dir configured): `REMOTE_LOG_BASE_PATH/<job_folder>/slurm_output/<job-name>-<job-id>.out|err`

## slurm-dashboard compatibility

To make this launcher write logs in the convention used by `slurm-dashboard`, set an
absolute archive directory in your config. This can be private (default dashboard
location under your home) or shared (if you want others to be able to read logs or
to avoid home quota).

```python
# Private (matches slurm-dashboard default if SLURM_DASHBOARD_LOG_ARCHIVE_DIR is unset):
REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR = "/home/<user>/.slurm-dashboard/logs"
REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR = "/home/<user>/.slurm-dashboard/projects"  # optional

# Or shared:
# REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR = "/absolute/shared/path/slurm-dashboard/logs"
# REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR = "/absolute/shared/path/slurm-dashboard/projects"  # optional
```

With this set, launcher defaults become:

- `--output=<archive_dir>/%j.out`
- `--error=<archive_dir>/%j.err`

The launcher creates the archive directory automatically before submission.

If `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR` is also set, launcher creates
human-friendly symlinks after each submit:

- `<view_dir>/<project>/<YYYY-MM-DD>/<job-name>-<job-id>.out -> <archive_dir>/<job-id>.out`
- `<view_dir>/<project>/<YYYY-MM-DD>/<job-name>-<job-id>.err -> <archive_dir>/<job-id>.err`

This keeps job-id based recovery for the TUI and adds browsable per-project views.

Important: slurm-dashboard reads the archive dir from the environment variable
`SLURM_DASHBOARD_LOG_ARCHIVE_DIR` (default: `~/.slurm-dashboard/logs`). To keep
the dashboard fallback aligned with your launcher submissions, set it to the same
directory as `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR` when running slurm-dashboard.
(If you use the default `~/.slurm-dashboard/logs`, you don't need to set it.)

In short:

- `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR` (launcher config): where SLURM writes `.out/.err`
- `SLURM_DASHBOARD_LOG_ARCHIVE_DIR` (dashboard env var): where the dashboard looks for archived logs

Example setup on the cluster:

```bash
export SLURM_DASHBOARD_LOG_ARCHIVE_DIR="/absolute/shared/path/slurm-dashboard/logs"
mkdir -p "$SLURM_DASHBOARD_LOG_ARCHIVE_DIR"
slurm-dashboard
```
