---
name: slurm-launcher-operator
hide: true
description: "Launch selected experiments on a remote SLURM cluster, inspect job state and bounded logs, and retrieve requested outputs. Use SSH for investigations outside the launcher's scope."
---

# SLURM launcher operator

Use `slurm-launcher` for consistent staging, submission, tracking, and routine inspection.
Project configs and job scripts own experiment logic. The launcher is not a security sandbox
or a general remote shell.

## Inspect a job

```bash
slurm-launcher status --job-id 123 --json
slurm-launcher logs --job-id 123 --json
slurm-launcher logs --job-id 123 --stream stderr --json
```

One log call returns content and metadata: both streams, at most 100 lines per
initial file tail and 64 KiB total content per response. JSON reads the same content;
change `--lines` or `--max-bytes` only when needed.

For launcher-managed experiments, prefer the returned run ID:

```bash
slurm-launcher status --run <run-id> --json
slurm-launcher logs --run <run-id> --job train --json
```

`--run` accepts a run ID, a path to `jobs.json`, or `latest`.
Inspection defaults to the latest local run. Keep an explicit ID when working on
multiple experiments; do not let a later submission silently change your target.
`--job-id` is a scheduler ID; `logs --job` selects a tracked job by name.
Tracked inspection uses saved SSH context and does not import the project config.
For direct inspection, use `--cluster-login <alias-or-user@host>` when needed.

Arrays expose task states; `array_complete` requires exact saved membership, known
states, and a successful queue probe. Only all-successful complete arrays are `DONE`.
Direct/legacy arrays may remain `UNKNOWN` despite completed visible tasks.
Check task failures rather than trusting a parent accounting row.

## Navigate logs without repeated tails

```bash
slurm-launcher logs --cursor <returned-cursor> --json
slurm-launcher logs --run <run-id> --job train --search Traceback --context 20 --json
slurm-launcher logs --run <run-id> --path logs/application.err --json
```

- Continue from the returned cursor for additional or newly appended output.
- Search is literal and bounded. Check completion/continuation fields before concluding no match exists.
- Check each file's status: missing, empty, and unreadable are different outcomes.
- Path provenance does not prove a file exists; use the returned filesystem metadata.
- Replacement/truncation resets are explicit. Do not treat them as uninterrupted output.
- Rewrite detection fingerprints only sampled prefix/pre-offset regions, not the whole file.
- Retain `resolution_errors`: failed path resolution is not the same as a missing log file.
- `--path` reads an application log inside the tracked workspace. For logs elsewhere,
  use SSH with the actual path reported by the application rather than repeatedly reading SLURM output.
- `--path-only` is optional discovery and does not read log content. `--follow` is for
  interactive text sessions, not a default agent operation. Prefer finite reads.

## Launch a selected experiment

Work in the intended project or supply its trusted `--config`.
After config changes, validate. Render when script construction or resource settings
are uncertain. Preview unfamiliar or costly submissions; do not repeat every preview
command for an established unchanged workflow.

```bash
slurm-launcher validate --json
slurm-launcher run --only train --dry-run --json
slurm-launcher run --only train --json
```

Execution requires `--only`, a nonempty configured `RUN_JOBS`, or explicit `--all`.
Bare `slurm-launcher` shows help and never submits. Save the returned run ID and tracking path.
A project with an existing sbatch script can use `sbatch scripts/train.sbatch`.

Split staging from submission only when useful:

```bash
slurm-launcher stage --only train --json
slurm-launcher preflight --run <run-id> --json
slurm-launcher submit --run <run-id> --json
```

Submit/preflight use the frozen plan, not current config. Restage changed definitions
or version-1/2 plans. Relocated/renamed bundles save in the selected directory.
Fixed workspaces are mutable; use per-run directories for overlapping experiments
and keep large data/caches shared.

## Recover safely

- Distinguish a failed experiment from a failed inspection or submission command.
- Nonzero submission exit does not mean nothing launched. Read partial results and tracking.
- Submitted jobs are not resubmitted within the same run. Unknown/submitting outcomes
  require scheduler reconciliation before any new submission; never retry blindly.
- Retryable failure requires an explicit scheduler rejection, not an arbitrary SSH exit.
  Preserve raw submission stdout/stderr/returncode for uncertain attempts.
- `UNKNOWN` scheduler state is inconclusive. Use `job-show <id> --json` or SSH if needed.
- A missing prerequisite declaration is not a passed preflight check.
- Do not replace a failed launcher operation with a new `sbatch` until its outcome is known.

## Permissions and boundaries

Read-only inspection does not authorize submission, downloads, cancellation, or filesystem changes.
`--json` is not `--dry-run`. Live `run`, `submit`, and `sbatch` consume cluster resources.
Live staging writes remote files; `prepare()` can write project-local generated inputs.
Python configs are trusted executable code, including during validate/render/dry-run;
those commands skip the preparation hook, not arbitrary import-time side effects.

Use `artifacts list` for declarations, `artifacts check` for existence, and
`artifacts download` or `download-logs` only when the user requested a local copy.
`summary --run <id> --json` is read-only and needs no project config.
Downloaded logs are isolated by job name, scheduler ID, and stream. Directory
artifact downloads use the reported destination without repeated nesting.

Use SSH directly when investigating application-specific paths, modules, environments,
or containers outside the CLI's capabilities. State the relevant reason briefly;
do not exhaust unsuitable launcher commands first. Destructive actions need explicit intent.

## Reference

Use `<command> --help` for exact flags, defaults, and examples, especially after an upgrade.
The README documents configuration and migration. Do not invent wrappers or preserve
removed command spellings. This file is the canonical operator skill; install it by
symlink or copy from this repository rather than maintaining a separate command catalog.
