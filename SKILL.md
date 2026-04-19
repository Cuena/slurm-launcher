---
name: slurm-launcher-operator
description: "Use when the user needs an assistant to inspect or operate jobs on a remote SLURM cluster. `slurm-launcher` (also available as `sl` in some environments) is the command-line tool used both for generic cluster inspection, such as listing jobs or reading job logs, and for project-scoped workflows that turn a local project configuration into remote SLURM jobs, submit them over SSH, monitor them, and retrieve logs or artifacts. The goal may be either to answer cluster-level questions without any project context or to take a project from local configuration through validation, staging, submission, monitoring, failure diagnosis, and reruns until the remote job succeeds. This skill interacts with the remote scheduler environment and, when relevant, the local project files. Do not use for modifying slurm-launcher's source repository unless explicitly requested."
---

Use this workflow when the task is to inspect or operate jobs on a cluster that uses SLURM, a common batch scheduler for HPC and GPU systems.

`slurm-launcher` is the CLI that bridges the user environment and that remote cluster. Some commands are global cluster tools that work without any project checkout, such as listing jobs, showing job details, or reading logs from recent jobs. Other commands are project-scoped: they read local job configuration, render or stage scripts and assets, submit jobs to SLURM through SSH, track the resulting job metadata locally, and later fetch logs or artifacts.

Use this file as an execution guide for the full `slurm-launcher` feature set.

Command model to keep straight:

- Global cluster inspection:
  - `slurm-launcher doctor`
  - `slurm-launcher jobs`
  - `slurm-launcher job-show`
  - `slurm-launcher job-log`
- Project execution:
  - `slurm-launcher init`
  - `slurm-launcher validate`
  - `slurm-launcher render`
  - `slurm-launcher stage`
  - `slurm-launcher submit`
  - `slurm-launcher sbatch`
  - `slurm-launcher run`
- Project tracking and retrieval:
  - `slurm-launcher logs`
  - `slurm-launcher monitor`
  - `slurm-launcher download-logs`
  - `slurm-launcher download-artifacts`

1. Resolve tool and execution context
- Prefer calling `slurm-launcher` directly.
- If unavailable, use `sl` only if that alias is installed in the user environment.
- `slurm-launcher` project execution commands still assume the current repo contains the intended project config.
- `slurm-launcher jobs`, `slurm-launcher job-show`, and `slurm-launcher job-log` should also be treated as global commands: run them directly from any directory when the user only wants generic cluster inspection.
- Do not require `cd` into the project repo for `slurm-launcher jobs`, `slurm-launcher job-show`, or `slurm-launcher job-log`.
- `init`, `run`, `stage`, `submit`, `render`, and `validate` remain project-scoped commands and should be run in the intended repo.
- Do not introduce wrappers, PATH shims, or repo code changes unless the current execution environment is the thing preventing normal tool use.
- If the current runtime already documents a repo-local launcher checkout fallback, use that runtime's documented invocation rather than inventing a new wrapper.
- If an SSH-backed `slurm-launcher` command fails inside the agent sandbox, retry it with escalated permissions before diagnosing the failure as a cluster or project issue.
- If `slurm-launcher` fails because the local `ssh` command is broken in the current agent environment even after escalation, explain that this is an environment issue, not a launcher usage issue.
- Only use an `ssh` wrapper or override as a temporary execution workaround when needed to unblock the task; prefer fixes to machine `ssh` config or explicit launcher support for custom SSH options as the long-term solution.
- If network or DNS access is blocked by the sandbox, request escalation rather than replacing tool usage with ad hoc non-tool flows.
- When a repo already contains `.slurm/remote_launcher_config*.py`, inspect and extend that config before proposing a parallel launcher structure.
- If the needed tool is unavailable, stop and report missing installation.

2. Cover config lifecycle features
- Default config lookup is:
  `.slurm/remote_launcher_config.mn5.py`, then `remote_launcher_config.py`.
- `init` features:
  - `slurm-launcher init`
  - `slurm-launcher init --force`
  - `slurm-launcher init --non-interactive`
- Init creates:
  - `.slurm/remote_launcher_config.mn5.py` (private)
  - `.slurm/remote_launcher_config.mn5.example.py` (shareable)
  - `.gitignore` entries for `.slurm/*.py` and `!.slurm/*.example.py`

3. Cover run/stage/submit/sbatch execution features
- `run` is the default command when no subcommand is given.
- Shared knobs:
  - `--config <path>`
  - `--workspace per-run|fixed`
  - `--only <job...>` (where supported)
  - `--dry-run` (where supported)
- Full run:
  - `slurm-launcher run`
  - `slurm-launcher run --dry-run`
  - `slurm-launcher run --only train eval`
- Split flow:
  - `slurm-launcher stage [--workspace ...] [--dry-run]`
  - `slurm-launcher submit [--workspace ...] [--only ...] [--dry-run]`
  - For `submit` in `per-run` mode, require `--job-folder <existing_folder>`.
- Existing sbatch file flow:
  - `slurm-launcher sbatch <sbatch_file>`
  - Optional: `--name <tracking_name>`
  - Optional repeatable: `--sbatch-arg <arg>`
  - Supports `--workspace`, `--config`, and `--dry-run`.

4. Cover validation and rendering features
- Validate local config:
  - `slurm-launcher validate`
  - `slurm-launcher validate --only <job...>`
  - Prefer `slurm-launcher validate --json` for agent use
- Validate connectivity/prereqs:
  - `slurm-launcher validate --ssh`
  - `slurm-launcher validate --ssh --check-remote-paths`
  - `--check-remote-paths` is valid only with `--ssh`.
- Render scripts without submit:
  - `slurm-launcher render`
  - `slurm-launcher render --only <job...>`
  - `slurm-launcher render --job-script`
  - Prefer `slurm-launcher render --json` for agent use

5. Cover tracking, monitoring, and retrieval features
- Tracking files:
  - per-run `slurm_output/<job_folder>/jobs.json`
  - pointer `slurm_output/latest_jobs.json`
  - commands may also auto-pick latest `slurm_output/*/jobs.json`
- Inspect tracking:
  - `slurm-launcher logs`
  - `slurm-launcher logs --only <job...>`
  - Prefer `slurm-launcher logs --json` for agent use
  - `slurm-launcher logs --tracking-file <jobs.json>`
- Monitor queue:
  - `slurm-launcher monitor`
  - `slurm-launcher monitor --only <job...>`
  - `slurm-launcher monitor --tracking-file <jobs.json>`
  - `slurm-launcher monitor --dry-run`
  - Prefer `slurm-launcher monitor --json` for agent use
- Download `.out/.err`:
  - `slurm-launcher download-logs`
  - `--job-name <name>` repeatable
  - `--job-id <id>` repeatable
  - `--tracking-file <jobs.json>`
  - `--output-dir <local_dir>`
  - `--dry-run`
- Download artifacts from tracked `remote_workdir`:
  - `slurm-launcher download-artifacts`
  - `--path <remote_relative_or_absolute_path>` repeatable
  - `--tracking-file <jobs.json>`
  - `--output-dir <local_dir>`
  - `--dry-run`
- Generic cluster inspection without tracking files:
  - `slurm-launcher jobs`
  - Prefer `slurm-launcher jobs --json` for agent use
  - `slurm-launcher doctor`
  - `slurm-launcher doctor --ssh`
  - `slurm-launcher jobs --cluster-login <user@host>`
  - `slurm-launcher jobs --config <path>`
  - default generic config lookup also checks `~/.config/slurm-launcher/config.py`
  - these generic commands are intended to work from any directory, not only from inside a project repo
  - `slurm-launcher jobs --hours <n> --limit <n>`
  - keep `jobs --json` list-shaped; if more detail is needed for one job, use `job-show`
  - `slurm-launcher job-show <job_id>`
  - Prefer `slurm-launcher job-show <job_id> --json` for agent use
  - `job-show` should resolve generic SLURM fields from `scontrol show job -o` and optionally `sacct`
  - `job-show` may add launcher enrichment when available, but that enrichment is additive and must not be required for generic inspection
  - `slurm-launcher job-log <job_id>`
  - `slurm-launcher job-log <job_id> --json`
  - `slurm-launcher job-log <job_id> --stream stdout|stderr`
  - `slurm-launcher job-log <job_id> --lines <n>`
  - `slurm-launcher job-log <job_id> --follow`
  - `slurm-launcher job-log <job_id> --full`
  - `slurm-launcher job-log <job_id> --path-only`
  - `job-log --json` should return structured log resolution only, not log content parsing
  - `job-log` resolves `StdOut`/`StdErr` from SLURM first, then falls back to the default archive convention for old jobs.
  - archive fallback should be explained explicitly: use `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR` when configured, otherwise default to `~/.slurm-dashboard/logs`.
  - user-level generic config should be treated as minimal: `CLUSTER_LOGIN` is required; `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR` is optional.
  - when machine-specific SSH behavior needs overrides, prefer config keys like `SSH_CONFIG_FILE` and `SSH_OPTIONS` instead of agent-only wrappers.

6. Cover workspace/runtime/config model features
- Workspace modes:
  - `per-run`: requires `REMOTE_WORKSPACE_BASE`; creates unique run folder.
  - `fixed`: requires `REMOTE_WORKSPACE_DIR`; reuses fixed remote directory.
- Runtime modes:
  - `native` (default)
  - `venv` (requires `VENV_PYTHON_EXECUTABLE`)
  - `singularity` (requires `SINGULARITY_IMAGE_PATH`, optional `SINGULARITY_EXEC_FLAGS`)
- Core required config:
  - `CLUSTER_LOGIN`
  - `JOBS`
  - workspace path requirement by mode above
- Job model:
  - each job must define exactly one of `command` or `sbatch_file`
  - `sbatch_file` jobs may use `sbatch_args`
  - `command` jobs may use `setup`, `env`, `sbatch`
  - `RUN_JOBS` can define default subset; `--only` overrides it
  - `DEFAULT_ENV` and `DEFAULT_SBATCH` apply globally
- Optional archive integration:
  - `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR`
  - `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR`
- Optional SSH behavior:
  - `SSH_CONFIG_FILE` to map to `ssh -F <path>`
  - `SSH_OPTIONS` as a list of extra args appended to launcher-managed `ssh` calls

7. Standard operation loop
- Start with `validate`, preferably `validate --json` when another agent step needs a stable payload.
- If a command will cross SSH boundaries (`validate --ssh`, `jobs`, `job-show`, `job-log`, `run`, `stage`, `submit`), be ready to retry with escalation if the sandbox interferes.
- Then use `render --json` and/or dry-runs before submission, especially after edits to quoting, runtime mode, or staged paths.
- Submit via `run` or split `stage` + `submit`/`sbatch` as requested.
- Prefer `stage --dry-run --json`, `submit --dry-run --json`, or `run --dry-run --json` before costly remote actions.
- Use `logs`, `monitor`, `download-logs`, `download-artifacts`, `doctor`, `jobs`, `job-show`, and `job-log` to inspect failures and recover outputs.
- When the task is generic cluster inspection rather than project execution, prefer the generic `doctor` / `jobs` / `job-show` / `job-log` commands over repo-scoped launcher workflows.
- For `doctor` / `jobs` / `job-show` / `job-log`, assume the normal invocation is the installed `slurm-launcher` binary from any directory, backed by `~/.config/slurm-launcher/config.py` when needed.
- Apply minimal fixes in the target project and rerun until success criteria are met.

8. Guardrails
- Prefer minimal, reversible edits.
- Avoid dependency/lockfile changes unless explicitly requested.
- Use dry-run before costly or risky remote actions.
- Distinguish tool behavior from environment behavior. If `slurm-launcher` would work in a normal shell but fails in the current agent sandbox because of `ssh` config or blocked network access, state that clearly.
- When a temporary wrapper or environment override is required to run the installed CLI in this agent session, keep it outside the repo when possible and avoid turning an execution-environment workaround into a product change unless the user asks for that change.
- Report: what changed, exact commands run, and current job state.
