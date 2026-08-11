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
- `launcher/command_specs.py`: shared public command metadata used by parser help, tests, and agent docs
- `launcher/payloads.py`: shared JSON payload builders for machine-readable command output
- `launcher/templates/config.py.template`: starter config copied by `init`
- `examples/remote_launcher_config.demo.py`: minimal dry-run example
- `examples/remote_launcher_config.mn5.example.py`: MN5-oriented example

`launcher/command_specs.py` is the source of truth for command summaries, JSON-capable examples, and the expected machine-readable fields. This README keeps the human workflow shorter on purpose.

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
   - `slurm-launcher --version`
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

## Command groups

Use these command groups consistently:

- Global cluster inspection:
  - `slurm-launcher doctor`
  - `slurm-launcher jobs`
  - `slurm-launcher job-show <job_id>`
  - `slurm-launcher job-log <job_id>`
- Project execution:
  - `slurm-launcher init`
  - `slurm-launcher validate`
  - `slurm-launcher preflight`
  - `slurm-launcher render`
  - `slurm-launcher stage`
  - `slurm-launcher submit`
  - `slurm-launcher sbatch`
  - `slurm-launcher run`
- Project tracking and retrieval:
  - `slurm-launcher status`
  - `slurm-launcher logs`
  - `slurm-launcher monitor`
  - `slurm-launcher download-logs`
  - `slurm-launcher download-artifacts`
  - `slurm-launcher artifacts`
  - `slurm-launcher summary`

Only `doctor`, `jobs`, `job-show`, and `job-log` are intended to work as generic cluster tools from any directory.
The execution commands above assume you are in the target project repo or passed the intended repo config.

## Identifier model

Commands do not share one universal run identifier.

| Operation | Identifier |
| --- | --- |
| `stage` | Creates `job_folder` and `tracking_file` |
| `preflight`, per-run `submit` | `--job-folder <folder>` |
| `status` | Direct job ID, `--tracking-file`, or `--latest` |
| `logs` | Tracking file/latest; `--job` selects a tracked job |
| `monitor` | Tracking file/latest |
| `artifacts`, downloads, `summary` | Tracking file/latest |
| `job-show`, `job-log` | Direct SLURM job ID |

A SLURM job ID identifies one scheduler record. A tracking file preserves the launcher context that the scheduler ID does not: the cluster and SSH settings, the multi-job run, remote workspace, exact log paths, and declared artifacts. Use a direct ID for one-off `status`, `job-show`, or `job-log`; use tracking for run-scoped retrieval.

Inspect `<command> --help` after upgrades rather than carrying an argument from a related command. In particular, `status`, `logs`, and `monitor` do not accept `--job-folder`.

## Recommended workflow

Recommended execution loop for humans and coding agents:

1. Validate:
   - `slurm-launcher validate`
   - optional machine-readable contract: `slurm-launcher validate --json`
2. Preview:
   - `slurm-launcher render`
   - `slurm-launcher render --json`
   - `slurm-launcher stage --dry-run`
   - `slurm-launcher stage --dry-run --json`
   - for per-run preflight: `slurm-launcher stage --json`, then use the returned `job_folder`
   - `slurm-launcher preflight --job-folder <job_folder> --dry-run --json`
   - `slurm-launcher submit --dry-run --job-folder <existing_folder> --json`
   - `slurm-launcher run --dry-run --json`
3. Execute:
   - `slurm-launcher run`
   - or split flow: `slurm-launcher stage --json`, `slurm-launcher preflight --job-folder <job_folder>`, then `slurm-launcher submit --job-folder <job_folder>`
4. Inspect and retrieve:
   - `slurm-launcher status`
   - `slurm-launcher logs`
   - `slurm-launcher monitor`
   - `slurm-launcher job-show <job_id>`
   - `slurm-launcher job-log <job_id>`
   - `slurm-launcher download-logs`
   - `slurm-launcher download-artifacts`
   - `slurm-launcher artifacts list` / `slurm-launcher artifacts check` / `slurm-launcher artifacts download`
   - `slurm-launcher summary`

## Common commands

- `slurm-launcher init --force`: overwrite existing config
- `slurm-launcher run --only train eval`: run a subset of jobs
- `slurm-launcher run --workspace per-run`: run from a new per-run folder (stage + submit)
- `slurm-launcher run --workspace fixed`: run from `REMOTE_WORKSPACE_DIR` (stage + submit)
- `slurm-launcher --config path/to/config.py`: custom config path
- default config lookup for commands using config: `.slurm/remote_launcher_config.mn5.py`, then `remote_launcher_config.py`
- `slurm-launcher validate`: validate config without submission
- `slurm-launcher validate --json`: print validation results as one JSON object
- `slurm-launcher validate --ssh`: validate config and test SSH connectivity
- `slurm-launcher validate --ssh --check-remote-paths`: also check remote venv/singularity prerequisites (no writes)
- `slurm-launcher validate`: also prints non-fatal warnings about missing artifacts, excluded requirements, GPU jobs without prerequisites, etc.
- `slurm-launcher preflight --job-folder <folder> --only <job>`: run remote prerequisite checks (`requires` paths and globs) against an already staged per-run workspace
- `slurm-launcher preflight --job-folder <folder> --dry-run --only <job>`: print the preflight script without executing it
- `slurm-launcher preflight --workspace fixed --only <job>`: check a fixed workspace without a job folder
- preflight fails when any selected job has no `requires`; JSON reports that job with `status: "not-configured"` instead of treating an empty check set as success
- `slurm-launcher render`: print generated sbatch scripts without submission
- `slurm-launcher render --json`: print rendered job metadata and scripts as JSON
- `slurm-launcher render --only train`: render only a subset of jobs
- `slurm-launcher stage`: run only the SSH + rsync stage phase (no job submission)
- `slurm-launcher stage --json`: print the stage result payload as JSON
- `slurm-launcher stage --workspace fixed --dry-run`: print stage commands only
- `slurm-launcher submit --workspace per-run --job-folder <folder>`: submit only (no rsync)
- `slurm-launcher submit --json --workspace per-run --job-folder <folder>`: print submission result payload as JSON
- `slurm-launcher submit --workspace fixed`: submit only against fixed workspace
- `slurm-launcher sbatch slurm/train.sbatch`: stage + submit one existing sbatch file
- `slurm-launcher sbatch slurm/train.sbatch --sbatch-arg --export=ALL,SEED=1`: pass extra sbatch args
- `slurm-launcher sbatch slurm/train.sbatch --dry-run --json`: machine-readable dry-run for one sbatch file
- `slurm-launcher run --json`: print run result payload as JSON
- `--json` only changes output format; it does not imply `--dry-run`
- `slurm-launcher status`: show SLURM status for tracked jobs from latest run
- `slurm-launcher status --json`: print job status as JSON
- `slurm-launcher status <job_id>`: query one job by id
- `slurm-launcher status <job_id> --cluster-login user@cluster --json`: query one job without a tracking file or repo config
- `status` merges `sacct` history with `squeue` live state; JSON reports each probe and the source used for each job
- if a failed probe leaves jobs unresolved, `status` exits nonzero with `ok=false`; otherwise treat `UNKNOWN` as inconclusive and verify with `job-show <job_id>`
- `slurm-launcher logs`: show tracked `.out/.err` paths from latest run
- `slurm-launcher logs --json`: print tracking metadata and resolved paths, not log text
- `slurm-launcher logs --job <job_id> --lines 100`: tail a tracked job remotely
- `slurm-launcher logs --job <job_id> --follow`: follow tracked stdout remotely
- `slurm-launcher logs --job <job_id> --stderr --lines 100`: tail tracked stderr remotely
- `slurm-launcher download-logs`: download tracked `.out/.err` files from latest run
- `slurm-launcher download-logs --dry-run --json`: inspect log download plan for agents
- `slurm-launcher download-artifacts`: download configured artifact paths from latest run
- `slurm-launcher download-artifacts --dry-run --json`: inspect artifact download plan for agents
- `slurm-launcher download-artifacts --path outputs --path checkpoints/best.pt`: override configured artifact paths
- `slurm-launcher artifacts list`: list paths declared by tracked jobs without contacting the cluster
- `slurm-launcher artifacts list --json`: returns `declared_only=true` and `remote_checked=false`
- `slurm-launcher artifacts check --json`: check remote existence, type, and size over SSH without downloading
- `slurm-launcher artifacts download`: download declared artifacts with rsync
- `slurm-launcher artifacts download --dry-run --json`: preview download plan
- `slurm-launcher artifacts download --only train eval`: download only selected jobs
- `slurm-launcher summary`: write/update local summary.json with current job states
- `slurm-launcher monitor`: run `squeue` for tracked job IDs from the latest run
- `slurm-launcher monitor --dry-run`: print the monitoring command only
- `slurm-launcher monitor --json`: print the monitor command contract as JSON

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
- Preview machine-readable download plan:
  - `slurm-launcher download-logs --dry-run --json`

Script compatibility:

- `uv run python scripts/download_logs.py ...` still works from the launcher repository root.

## Download artifacts locally

Use `download-artifacts` to fetch outputs or checkpoints from the tracked remote workdir.

- Set `ARTIFACT_PATHS` in your config to make common downloads one-command:
  - `ARTIFACT_PATHS = ["outputs", "checkpoints/best.ckpt"]`
- Download configured artifact paths for the latest project run:
  - `slurm-launcher download-artifacts`
- Override configured paths for one download:
  - `slurm-launcher download-artifacts --path outputs --path reports/metrics.json`
- Use a specific tracking file:
  - `slurm-launcher download-artifacts --tracking-file slurm_output/<job_folder>/jobs.json`
- Preview rsync commands without downloading:
  - `slurm-launcher download-artifacts --dry-run`
- Preview machine-readable download plan:
  - `slurm-launcher download-artifacts --dry-run --json`

Path rules:

- Relative paths are resolved from the tracked `remote_workdir`.
- Absolute paths are allowed, but relative paths are the intended default.

Script compatibility:

- `uv run python scripts/download_artifacts.py ...` still works from the launcher repository root.

## General SLURM utilities

These commands do not use `slurm_output/.../jobs.json`. They query the cluster
directly over SSH, which makes them useful even for jobs not launched by this repo.

- List recent jobs on the cluster:
  - `slurm-launcher jobs`
- Use a specific config or SSH target:
  - `slurm-launcher jobs --config .slurm/remote_launcher_config.mn5.py`
  - `slurm-launcher jobs --cluster-login user@cluster`
- Change query window or output shape:
  - `slurm-launcher jobs --hours 72 --limit 50`
  - `slurm-launcher jobs --json`
- Show generic details for one job id:
  - `slurm-launcher job-show 36114735`
  - `slurm-launcher job-show 36114735 --json`
  - `slurm-launcher job-show 36114735 --sbatch` also retrieves the exact batch script stored by SLURM
  - combine `--sbatch --json` to return the script in the `sbatch` field
  - successful JSON includes `ok=true`, `job_id`, `resolved_via`, and `detail_level`; failures return `ok=false`
  - when only log resolution is available, unresolved fields are omitted instead of emitted as `null`
- Read stdout for a job id:
  - `slurm-launcher job-log 36114735`
- Print structured log resolution without reading the file:
  - `slurm-launcher job-log 36114735 --json`
  - check `ok`, `path_verified`, and `content_included` before using the returned path
- Read stderr instead:
  - `slurm-launcher job-log 36114735 --stream stderr`
- Print the resolved remote log path only:
  - `slurm-launcher job-log 36114735 --path-only`
- Follow the live log:
  - `slurm-launcher job-log 36114735 --follow`
- Check the effective generic config:
  - `slurm-launcher doctor`
- Also test SSH plus remote tool availability:
  - `slurm-launcher doctor --ssh`

Resolution rules:

- `jobs`, `job-show`, `job-log`, and `doctor` use `--cluster-login` when provided. Without an explicit `--config`, this uses the caller's normal SSH configuration and does not merge in the generic config's SSH options.
- Otherwise they try `--config`, then `.slurm/remote_launcher_config.mn5.py`, then
  `remote_launcher_config.py`, then `~/.config/slurm-launcher/config.py`.
- `job-log` first asks SLURM for `StdOut`/`StdErr` paths. It uses an archive fallback only when `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR` is explicitly configured.
- SSH transport failures are reported as unresolved; they never produce a guessed archive path.

Recommended user-level default config for generic commands:

```python
CLUSTER_LOGIN = "your_user@cluster"
SSH_CONFIG_FILE = "/dev/null"  # optional
SSH_OPTIONS = ["-o", "BatchMode=yes"]  # optional
REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR = "/home/your_user/.slurm-dashboard/logs"  # optional
```

This file is only used by `jobs`, `job-show`, `job-log`, and `doctor`. Project execution commands still
use the repo-local config lookup.

SSH notes:

- `CLUSTER_LOGIN` is required.
- Prefer a canonical `user@host` in committed examples and portable tracking data. A nickname such as `acc` is an alias from one user's `~/.ssh/config`; it may not exist for another user or machine.
- `SSH_CONFIG_FILE` is optional and maps to `ssh -F <path>`.
- `SSH_OPTIONS` is optional and is appended to every launcher-managed `ssh`
  invocation. Example: `["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]`
- Use these only when you need to override machine-specific SSH behavior; most
  users should rely on their normal `ssh` setup.
- Tracking-backed commands use the complete SSH context recorded in `jobs.json`; an unrelated user-level generic config is not mixed into that run.

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
- `SSH_CONFIG_FILE`, `SSH_OPTIONS`
- `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR`
- `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR`
- `ARTIFACT_PATHS`
- `RUNTIME_MODE`, `VENV_PYTHON_EXECUTABLE`, `SINGULARITY_IMAGE_PATH`, `SINGULARITY_EXEC_FLAGS`
- `DEFAULT_ENV`, `DEFAULT_SBATCH`, `RUN_JOBS`
- `SYNC_SYMLINKS` (`"preserve"` by default, or `"copy-links"`), `EXTRA_RSYNC_EXCLUDES`, `EXTRA_RSYNC_ARGS`, `VERBOSE`

## Job config model

Each job in `JOBS` is a dictionary with:

- required: `name` and exactly one of:
  - `command` (launcher-managed sbatch generation)
  - `sbatch_file` (submit an existing/shared sbatch file)
- optional for all jobs: `artifacts`, `requires`
- optional for `command` jobs: `setup`, `env`, `sbatch`
- optional for `sbatch_file` jobs: `sbatch_args` (extra args forwarded to `sbatch`)

`requires` accepts workspace-relative or absolute remote paths and globs. Preflight
checks them before submission for both launcher-generated and hand-written sbatch jobs.

Examples:

- `{"name": "train", "command": "python3 scripts/train.py --config-name=train"}`
- `{"name": "prep", "command": "bash scripts/prep.sh"}`
- `{"name": "eval", "command": "srun python3 scripts/eval.py"}`
- `{"name": "shared_train", "sbatch_file": "slurm/train_shared.sbatch", "requires": ["/models/base.pt"]}`
- `{"name": "shared_train_seed1", "sbatch_file": "slurm/train_shared.sbatch", "sbatch_args": ["--export=ALL,SEED=1"]}`

Minimal multi-node pattern (maps to `#SBATCH --nodes=...` + `srun torchrun`):

```python
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
        "account": "bsc70",
        "partition": "acc",
        "qos": "acc_debug",
        "time": "2:00:00",
        "nodes": 4,
        "ntasks-per-node": 1,
        "cpus-per-task": 80,
        "gres": "gpu:4",
    },
}
```

Notes:

- For shell-variable expansion in exports (for example `${SLURM_NNODES}`), use `setup` lines instead of `env`.
- Keep `RUNTIME_MODE="native"` when your `command` already includes `srun torchrun ...`.

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
- downloaded artifact destination: `slurm_output/downloaded_artifacts/<job_folder>/`
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
