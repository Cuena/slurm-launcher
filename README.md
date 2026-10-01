# slurm-launcher

Stage a research project over rsync, submit selected SLURM jobs, and keep enough
local context to inspect or recover the experiment later. Supports native commands,
virtualenvs, Singularity, and existing sbatch scripts. Project scripts own application
and distributed-runtime logic; the launcher is not an SSH security sandbox.

## Install

Requires Python 3.10+, SSH and rsync locally, and SLURM on the remote cluster.
Bounded log inspection additionally requires `python3` on the remote login node.
Git is optional for source provenance. Install once for use across projects:

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
In JSON mode, stdout contains a single JSON result; rsync progress and diagnostics
go to stderr.

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

The run directory contains a resolved `plan.json`, script snapshots, and `jobs.json`.
Submission/preflight read that frozen plan, not the current Python config. Stage a
new run to change job definitions. `submit --run ... --only ...` can narrow the
saved selection; it cannot add jobs absent from the plan.

## Inspect state and logs

```bash
slurm-launcher jobs --state running --json
slurm-launcher status --job-id 123 --json
slurm-launcher logs --job-id 123
slurm-launcher logs --job-id 123 --stream stderr --json
```

One log call resolves paths and reads content. The default is **both streams**, at
most **100 lines per initial file tail and 64 KiB total content per response**, with separate labels and filesystem status.
JSON contains the same content as text. Scheduler-derived paths are not proof of
existence. Missing, empty, unreadable, and successful reads are distinguished.

```bash
slurm-launcher logs --run <run-id> --job train --lines 200 --json
slurm-launcher logs --run <run-id> --job train --search Traceback --context 20 --json
slurm-launcher logs --cursor <returned-cursor> --json
slurm-launcher logs --run <run-id> --path logs/application.err --json
```

Search is literal, bounded, and resumable; check completion metadata before
concluding that a pattern does not occur. The cursor continues the read without
repeating previous bytes and reports file replacement/truncation explicitly.
Keep cursors private: they contain paths and SSH context, not an authorization token.
`--path` is a file inside the tracked workspace. For application logs elsewhere,
use SSH rather than recursively searching GPFS through this tool.

`--path-only` is optional discovery. `--follow` is an interactive text operation;
agents should normally use bounded reads. Use `job-show 123 --sbatch --json` for
detailed scheduler metadata or the original submitted batch script.

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

Tracking is updated atomically after each acknowledged submission. Failure output
preserves earlier successful IDs and identifies the failed or unknown attempt.

- `submitted`: an acknowledged scheduler ID; never automatically resubmitted.
- `failed`: a known unsuccessful attempt; inspect the error before retrying.
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

`list` reports declarations, `check` probes existence/type/size, and `download`
copies files. A request to read logs does not authorize a download. `summary` reads
tracking and current scheduler state; it writes neither local nor remote files.

## Configuration

Config lookup: `.slurm/remote_launcher_config.mn5.py`, then
`remote_launcher_config.py`; override with `--config`. Direct cluster commands can
also use `~/.config/slurm-launcher/config.py`. MN5 filenames are conventions, not a
restriction to that cluster. The template and examples show all supported settings.

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
prerequisites is reported as not configured, not as a passed check.

Relevant settings:

- **Workspace:** `LOCAL_ROOT`, `PROJECT_NAME`, `WORKSPACE_MODE`,
  `REMOTE_WORKSPACE_BASE` (per-run) or `REMOTE_WORKSPACE_DIR` (fixed),
  `REMOTE_LOG_BASE_PATH`.
- **Runtime:** `RUNTIME_MODE` native/venv/singularity;
  `VENV_PYTHON_EXECUTABLE` for venv; `SINGULARITY_IMAGE_PATH` and
  `SINGULARITY_EXEC_FLAGS` for Singularity.
- **Transport:** `CLUSTER_LOGIN`, optional `RSYNC_LOGIN` for a transfer endpoint,
  `SSH_CONFIG_FILE`, `SSH_OPTIONS`.
- **Staging:** `EXTRA_RSYNC_EXCLUDES`, `EXTRA_RSYNC_ARGS`, `SYNC_SYMLINKS`,
  `REQUIRE_CLEAN_GIT`. Keep datasets, checkpoints, secrets and caches out of code sync.
- **Outputs:** global `ARTIFACT_PATHS`, per-job `artifacts`, optional
  `LOCAL_ARTIFACT_ROOT`; dashboard archive/view settings remain optional.

### Preparation and trust

Python config import is executable trusted code, even for validate/render/dry-run.
Keep import-time job generation pure. If generated inputs must be written, expose:

```python
def prepare():
    # Write already-resolved project inputs here.
    generated_input.write_text(serialized_input, encoding="utf-8")
```

The launcher calls this hook only for real stage/run/sbatch operations, before
syncing. It is skipped during previews and frozen-plan submission. The hook should
prepare inputs, not silently alter resolved job selection. No environment manager
or plugin framework is needed.

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

Consult `<command> --help` for exact options. The skill teaches decisions; help
owns the command reference. Use SSH for investigations outside these workflows,
with explicit intent for destructive actions and expensive submissions.
