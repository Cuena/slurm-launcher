# slurm-launcher

Stage a research project over rsync, submit selected SLURM jobs, and keep enough
local context to inspect or recover the experiment later. Supports native commands,
virtualenvs, Singularity, and existing sbatch scripts. Project scripts own application
and distributed-runtime logic; the launcher is not an SSH security sandbox.

## Install

Requires Python 3.10+, SSH and rsync locally; Bash, rsync and SLURM remotely.
Log inspection also needs remote `python3`; Git is optional for source provenance.
Install once with uv for use across projects:

```bash
uv tool install --editable /path/to/slurm-launcher
slurm-launcher --version
slurm-launcher --help
```

For an agent, install the repository's `SKILL.md` as the `slurm-launcher-operator`
skill. Use a symlink to this file for local editable installations; update copied
skills with the CLI release. Do not maintain a second command catalog.

## Launch an experiment

In the project to launch:

```bash
slurm-launcher init --non-interactive
# Edit .slurm/remote_launcher_config.mn5.py for your cluster and project.
slurm-launcher validate --json
slurm-launcher run --only train --dry-run --json
slurm-launcher run --only train --json
```

Bare invocation shows help. Execution requires selected job names, a nonempty
`RUN_JOBS`, or explicit `--all`; an empty default never submits the entire matrix.
Use validation after config changes and previews for unfamiliar or costly launches.
These are available checks, not a mandatory sequence before every established run.
`--json` selects output format; it never implies `--dry-run`.
Operational `--json` results use one stdout document without progress chatter.
Probe/transfer diagnostics are included in JSON; help and usage errors remain text.

Initialization ignores private configs, `__pycache__/`, and `slurm_output/`, while
keeping `.slurm/*.example.py` shareable. Existing projects must add these exclusions
before enabling `REQUIRE_CLEAN_GIT`. Source provenance is captured once after
preparation and before creating launcher state, then reused for staging and tracking.

For an existing project-owned script:

```bash
slurm-launcher sbatch scripts/train.sbatch --dry-run --json
slurm-launcher sbatch scripts/train.sbatch --json
```

Split staging from execution when you need to check remote prerequisites:

```bash
slurm-launcher stage --only train --json
slurm-launcher preflight --run <returned-run-id> --json
slurm-launcher submit --run <returned-run-id> --json
```

Each run saves a version-3 `plan.json`, one `.sbatch` snapshot per job, and
`jobs.json`. Submit/preflight use this frozen plan, not the current config.
Restage changed job definitions or version-1/2 plans. `submit --only ...` can narrow
the saved selection, not add jobs. Relocated or renamed bundles save progress in
the selected directory while retaining their original run ID.

## Inspect state and logs

```bash
slurm-launcher jobs --state running --json
slurm-launcher status --job-id 123 --json
slurm-launcher logs --job-id 123
slurm-launcher logs --job-id 123 --stream stderr --json
```

One log call resolves paths and reads content: **both streams**, at most **100
lines per initial file tail and 64 KiB total content per response**. JSON and text
read the same content. Per-file status distinguishes missing, empty, unreadable,
and successful reads; a scheduler-derived path does not prove existence.

```bash
slurm-launcher logs --run <run-id> --job train --lines 200 --json
slurm-launcher logs --run <run-id> --job train --search Traceback --context 20 --json
slurm-launcher logs --cursor <returned-cursor> --json
slurm-launcher logs --run <run-id> --path logs/application.err --json
```

Search is literal, bounded, and resumable. Continue the returned cursor and check
completion before concluding that no match exists. Cursors report replacement,
truncation, and rewrites touching sampled prefix/pre-offset regions—not every
possible rewrite. Malformed UTF-8 is represented rather than silently skipped.
Keep cursors private: they contain paths and SSH context, not authorization.

`--path` reads inside the tracked workspace; use SSH for application logs elsewhere.
`--path-only` stats files without reading content. Missing tracked paths are
resolved initially; saved non-null paths remain authoritative. Per-stream provenance
and `resolution_errors` persist in cursors.

`--follow` is an interactive text operation; agents should normally use bounded
reads. Use `job-show 123 --sbatch --json` for detailed scheduler metadata or the
original submitted batch script. Launcher attribution is restricted to the queried
cluster and rejects mismatching available submission identities.

Array status includes task IDs. `array_complete` requires the exact saved task set,
known task states, and a successful queue probe; it does not mean all tasks succeeded.
`DONE` additionally requires every task to succeed. A failed task takes precedence
over running/pending tasks. Direct/legacy tracking without a saved task expression
cannot prove completion. Scalar `COMPLETED` with a nonzero exit code or signal is `FAILED`.

### Run identity

Run-scoped commands accept `--run` with a folder ID, a path to `jobs.json`, or
`latest`. Inspection defaults to latest; submit/preflight require an explicit
selector. A returned run ID is local to the project; use the tracking path from
another directory. Older tracking files remain inspectable, but submission
requires a new staged plan.

`status --job-id` and `logs --job-id` inspect a scheduler job directly. Add
`--cluster-login <ssh-alias-or-user@host>` if needed. An explicit login uses normal
SSH configuration unless `--config` is also explicit. Tracked inspection uses the
saved complete SSH context and does not import project configuration.

## Recover from submission failure

Tracking saves acknowledged IDs atomically before optional log enrichment. Failure
output retains earlier IDs and the failed/unknown attempt. Uncertain attempts save
`submission_stdout`, `submission_stderr`, and `submission_returncode` for recovery.

- `submitted`: an acknowledged scheduler ID; never automatically resubmitted.
- `failed`: an explicit unsuccessful scheduler attempt; inspect the error before retrying.
- `submitting` or `unknown`: the scheduler may have accepted it. Reconcile through
  scheduler inspection before launching anything again; no blind retry.

A disconnected SSH session is not evidence that `sbatch` did nothing. A process
interrupted while dispatching likewise requires reconciliation. This tool does
not promise exactly-once execution across a network failure. Do not edit away an
unknown outcome merely to bypass the guard.

Scheduler `UNKNOWN` is inconclusive, not a terminal state. Use `job-show` or SSH
to investigate. Inspection failure is distinct from experiment failure.

## Retrieve outputs only when requested

```bash
slurm-launcher artifacts list --run <run-id> --json
slurm-launcher artifacts check --run <run-id> --json
slurm-launcher artifacts download --run <run-id> --dry-run --json
slurm-launcher artifacts download --run <run-id> --path outputs/metrics.json --json
slurm-launcher download-logs --run <run-id> --json
slurm-launcher summary --run <run-id> --json
```

`list` reports declarations; `check` probes existence/type and filesystem metadata
size, not recursive directory usage. `download` copies to the reported destination;
repeating a directory download does not add another nested copy. Dry runs create no
directories or remote probes. Transfer failures retain return codes and bounded stderr.

Downloaded logs are separated under
`<destination>/<job-name>/<job-id>/<stdout|stderr>/<basename>`, so repeated job names
and identical stdout/stderr basenames do not overwrite each other. If both streams
refer to the same remote file, it is downloaded once.

A request to read logs does not authorize a download. `summary` reads tracking and
current scheduler state; it writes neither local nor remote files.

## Configuration

Config lookup: `.slurm/remote_launcher_config.mn5.py`, then
`remote_launcher_config.py`; override with `--config`. Direct cluster commands also
use `~/.config/slurm-launcher/config.py`. MN5 filenames are conventions, not a
cluster restriction. Adapt the template's hosts, account, paths, and scripts before use.

```python
from pathlib import Path

LOCAL_ROOT = Path(__file__).resolve().parent.parent
CLUSTER_LOGIN = "user@cluster"
WORKSPACE_MODE = "per-run"
REMOTE_WORKSPACE_BASE = "/scratch/user/project/runs"
REMOTE_LOG_BASE_PATH = "/scratch/user/project/logs"
RUNTIME_MODE = "native"
DEFAULT_ENV = {"PYTHONUNBUFFERED": "1"}
DEFAULT_SBATCH = {"time": "00:30:00", "cpus-per-task": 4}
RUN_JOBS = ["train"]
JOBS = [{
    "name": "train",
    "command": "python3 train.py",
    "requires": ["train.py", "/shared/datasets/training"],
    "artifacts": ["outputs/metrics.json"],
}]
```

Each job supplies exactly one of `command` or `sbatch_file`. Command jobs support
`setup`, `env`, and `sbatch` overrides. Existing-script jobs support `sbatch_args`
and own their runtime/directives. Both support `requires` and `artifacts`.
Prerequisites are paths/globs checked in the staged workspace; a job with no
prerequisites is reported as not configured, not as a passed check. Literal paths
can contain spaces, quotes, dollar signs, and pipes. Globs must match existing
targets, not merely dangling symlinks. An incomplete response or failed SSH process
cannot pass preflight; JSON retains transport status and bounded diagnostics.

Relevant settings:

- **Workspace:** `LOCAL_ROOT`, `PROJECT_NAME`, `WORKSPACE_MODE`,
  `REMOTE_WORKSPACE_BASE` (per-run) or `REMOTE_WORKSPACE_DIR` (fixed),
  `REMOTE_LOG_BASE_PATH`.
- **Runtime:** `RUNTIME_MODE` native/venv/singularity;
  `VENV_PYTHON_EXECUTABLE` for venv; `SINGULARITY_IMAGE_PATH` and
  `SINGULARITY_EXEC_FLAGS` for Singularity.
- **Transport:** `CLUSTER_LOGIN`, optional `RSYNC_LOGIN`, `SSH_CONFIG_FILE`,
  `SSH_OPTIONS`. Both SSH endpoints must access the same remote directories.
- **Staging:** `EXTRA_RSYNC_EXCLUDES`, `EXTRA_RSYNC_ARGS`, `SYNC_SYMLINKS`,
  `REQUIRE_CLEAN_GIT`. Keep datasets, checkpoints, secrets and caches out of code sync.
- **Outputs:** global `ARTIFACT_PATHS`, per-job `artifacts`, optional dashboard
  archive/view settings. Choose local download roots with `--output-dir`.

Explicit `ntasks` wins; otherwise Slurm determines task count from the other directives.
Singularity executes the whole `command` via `bash -euo pipefail -c` inside the image,
which must provide Bash. Job `setup` runs first in the host batch shell.
`VERBOSE` and the unused `LOCAL_ARTIFACT_ROOT` are no longer settings.

### Preparation and trust

Python config import executes trusted code, including during validate/render/dry-run.
Keep import-time job generation pure. Define a module-level `prepare()` to write
resolved generated inputs: it runs only for real stage/run/sbatch, before sync.
Previews and frozen-plan submission skip it. Do not alter job selection inside the hook.

### Fixed versus per-run

Per-run creates an isolated code directory. Models, datasets and caches can stay
shared outside it. Fixed mode rsyncs into the configured mutable directory; queued
or running jobs may see restaged code, and stale files are not automatically deleted.
Neither a Git hash nor a saved plan makes a fixed code tree immutable. Use fixed
mode deliberately for iteration, not as an isolation guarantee.

## 0.2.0 interface changes

This release intentionally removes overlapping spellings:

| Previous interface | Current interface |
| --- | --- |
| `job-log 123` | `logs --job-id 123` |
| `logs --stderr` | `logs --stream stderr` |
| `logs --job 123` (scheduler ID) | `logs --job-id 123`, or `--run ID --job NAME` |
| `monitor` | `status` |
| `status 123` / `status --job 123` | `status --job-id 123` |
| `download-artifacts` | `artifacts download` |
| `--tracking-file PATH`, `--job-folder ID`, `--latest` | `--run PATH`, `--run ID`, `--run latest` |
| Bare command launches; empty `RUN_JOBS` launches everything | Explicit `run`; select jobs or pass `--all` |
| JSON log discovery only | JSON log content and metadata; optional `--path-only` |
| `summary` writes files and imports config | Read-only tracked summary |
| `scripts/download_logs.py` | `slurm-launcher download-logs` |
| `scripts/init_wrapper_repo.sh` | `slurm-launcher init` |
| `latest_run.txt`, duplicated `.sh` snapshots | `latest_run.json`, one `.sbatch` snapshot per job |

Consult `<command> --help` for exact options. The skill teaches decisions; help
owns the command reference. Use SSH for investigations outside these workflows,
with explicit intent for destructive actions and expensive submissions.

## Development

Runtime dependencies contain only Rich. Ruff and Vulture belong to the development
dependency group:

```bash
uv sync --locked --group dev
uv run python -m unittest discover -s tests
uv run ruff check .
uv run vulture launcher --min-confidence 100
uv build
git diff --check
```

Regression matrices reuse local SSH/scheduler fixtures; Bash and rsync execute locally.
No tests submit to a live cluster. Rsync-dependent cases skip if it is unavailable.
