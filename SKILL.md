---
name: slurm-launcher-operator
description: "Use when the user needs an assistant to inspect or operate jobs on a remote SLURM cluster. `slurm-launcher` is the CLI for both generic cluster inspection and project-scoped stage/submit/track/retrieve workflows. Use it to validate configs, render scripts, stage code, submit jobs, inspect logs, monitor tracked runs, and download artifacts. Do not use for modifying slurm-launcher's source repository unless explicitly requested."
---

Use this skill when the task is about operating a SLURM cluster or a project that submits SLURM jobs through `slurm-launcher`.

The CLI serves three command groups. Keep them separate.

- Global cluster inspection:
  `doctor`, `jobs`, `job-show`, `job-log`
- Project execution:
  `init`, `validate`, `preflight`, `render`, `stage`, `submit`, `sbatch`, `run`
- Project tracking and retrieval:
  `status`, `logs`, `monitor`, `download-logs`, `download-artifacts`, `artifacts`, `summary`

The canonical public command metadata now lives in:

- `launcher/command_specs.py`
- `launcher/payloads.py`

Keep this skill aligned with those files when the source repo changes.

## Identifier Contracts

Do not assume one identifier works across project commands.

- `stage` creates a `job_folder` and returns its local `tracking_file`.
- `preflight` and per-run `submit` accept `--job-folder <folder>`.
- `status` accepts a direct SLURM job ID, `--tracking-file <jobs.json>`, or `--latest`. It does not accept `--job-folder`.
- `logs` resolves a tracking file or the latest tracked run; `--job <job_id>` selects a job from that tracking file.
- `monitor` resolves a tracking file or the latest tracked run.
- `artifacts list|check|download`, `download-logs`, `download-artifacts`, and `summary` resolve a tracking file or the latest tracked run.
- `job-show` and `job-log` accept a direct SLURM job ID and do not require launcher tracking.

A job ID identifies one scheduler record. A tracking file additionally preserves the complete SSH context, multi-job membership, remote workspace, exact log paths, and declared artifacts. Prefer direct IDs for one-off inspection and tracking files for run-scoped retrieval.

After a launcher upgrade, run `slurm-launcher --version` and inspect `<command> --help` before first using a command whose arguments or effects matter. Treat installed help as authoritative when it differs from remembered guidance.

## Operating Rules

1. Prefer `slurm-launcher` directly.
- Use `sl` only if that alias is already installed in the user environment.
- Treat SSH nicknames such as `acc` as user-and-machine-local. Prefer canonical `user@host` destinations in portable configs and tracking data.

2. Distinguish global commands from project commands.
- `doctor`, `jobs`, `job-show`, and `job-log` are intended to work from any directory.
- `init`, `validate`, `preflight`, `render`, `stage`, `submit`, `sbatch`, and `run` are project-scoped.

3. Prefer JSON-capable commands for agent workflows.
- For agents, default to `--json` whenever the command supports it.
- Read content only after a discovery step. Example: use `job-log --json` before `job-log --follow`.
- Do not omit `--json` because the plain text looks easier; the JSON fields are the stable contract for agents.
- `--json` changes output format only. It does not imply `--dry-run`.

4. Use dry-run before costly or risky remote actions.
- Prefer `stage --dry-run --json`, `submit --dry-run --json`, `sbatch --dry-run --json`, or `run --dry-run --json` before live submission.
- For per-run preflight, run `stage --json` first, read the returned `job_folder`, then run `preflight --job-folder <job_folder> --json`.
- Prefer `download-logs --dry-run --json` and `download-artifacts --dry-run --json` before copying files.

5. Treat SSH or sandbox failures as environment issues first.
- If an SSH-backed command fails in the sandbox, retry with escalation before assuming the cluster or repo is broken.
- For generic commands, explicit `--cluster-login` without explicit `--config` uses the caller's normal SSH configuration; it must not inherit `SSH_CONFIG_FILE` or `SSH_OPTIONS` from the generic config.
- For tracking-backed commands, use the complete cluster login and SSH context stored in `jobs.json`; do not merge in a user-level generic config.

6. Avoid raw SSH and rsync when a launcher command exists.
- Do not run manual `ssh ... squeue`, `ssh ... sacct`, `ssh ... find`, `ssh ... du`, or `rsync` for launcher-managed jobs before trying the matching `slurm-launcher` command.
- Use `jobs`, `job-show`, `job-log`, `status`, `logs`, `monitor`, `artifacts`, `download-logs`, `download-artifacts`, and `summary` to resolve locations, status, logs, artifacts, downloads, and run summaries.
- Use raw SSH only when the launcher command cannot answer the request, and state why.

## Decision Guide

Use this quick mapping before choosing a command.

- “Is my config valid?”:
  `slurm-launcher validate --json`
- “What exactly will be submitted?”:
  `slurm-launcher render --json`
- “Stage files but do not submit”:
  `slurm-launcher stage --dry-run --json` or `slurm-launcher stage --json`
- “Check remote prerequisites in a per-run workspace”:
  `slurm-launcher stage --json`
  then `slurm-launcher preflight --job-folder <job_folder> --json`
- “Submit from already staged code”:
  `slurm-launcher submit --job-folder <folder> --json`
- “Stage and submit end to end”:
  `slurm-launcher run --json`
- “Submit one existing sbatch file from the project”:
  `slurm-launcher sbatch <sbatch_file> --json`
- “What jobs exist on the cluster?”:
  `slurm-launcher jobs --json`
- “What jobs are currently running?”:
  `slurm-launcher jobs --state running --json`
- “What happened to one job?”:
  `slurm-launcher job-show <job_id> --json`
- “Where is the log for one job?”:
  `slurm-launcher job-log <job_id> --json`
- “Read the latest lines for one job without downloading them”:
  `slurm-launcher job-log <job_id> --json`
  then `slurm-launcher job-log <job_id> --lines 50`
- “What logs were tracked for the last launcher run?”:
  `slurm-launcher logs --json`
- “What is the current state of tracked jobs?”:
  `slurm-launcher status --json`
- “Are tracked jobs still queued/running?”:
  `slurm-launcher monitor --json`
- “List declarations, verify existence, or download tracked artifacts”:
  `slurm-launcher artifacts list --json`
  `slurm-launcher artifacts check --json`
  `slurm-launcher artifacts download --dry-run --json`
- “Download tracked logs/artifacts locally”:
  `slurm-launcher download-logs --dry-run --json`
  `slurm-launcher download-artifacts --dry-run --json`
- “Write a run summary for future sessions”:
  `slurm-launcher summary --json`

## Action Effects

Know what a command will do before running it.

- Read-only local/config inspection:
  `validate --json`, `preflight --dry-run --json`, `render --json`, `logs --json`, `artifacts list --json`
- Read-only remote inspection over SSH:
  `doctor --ssh --json`, `jobs --json`, `job-show --json`, `job-log --json`, `status --json`, `monitor --json`, `artifacts check --json`, and live `preflight --json`
- Remote write without job submission:
  `stage --json` syncs project files to the remote workspace and writes local tracking artifacts.
- Remote job submission:
  `submit --json`, `sbatch --json`, and `run --json` call `sbatch` and can consume queue/GPU time.
- Local downloads:
  `download-logs`, `download-artifacts`, and `artifacts download` copy files to local `slurm_output/...` or the chosen `--output-dir`.
  Run them only when the user requested a local copy; a request to inspect, read, or tail remote logs does not authorize a download.
- Summary writes:
  `summary --json` updates local `slurm_output/<job_folder>/summary.json` and writes a remote `.slurm_run/summary.json` for the tracked run.

## Per-Run Preflight Flow

In `WORKSPACE_MODE="per-run"`, `preflight` must target an existing staged folder. Do not run plain `preflight --json`; it will fail because there is no stable per-run workspace to check.

Agent-safe sequence:

1. `slurm-launcher stage --json`
2. Read `job_folder` and `remote_workdir` from the JSON response.
3. `slurm-launcher preflight --job-folder <job_folder> --json`
4. If preflight passes, `slurm-launcher submit --job-folder <job_folder> --json`

For `WORKSPACE_MODE="fixed"`, `preflight --workspace fixed --json` can run without `--job-folder` because the configured remote workspace is stable.

Preflight is generic. It checks only each job's `requires` entries:

- plain paths must exist
- globs must match at least one path
- broken symlinks fail

Every selected job must declare at least one `requires` entry. An undeclared job
returns `ok=false`, `status="not-configured"`, and a nonzero exit instead of a
vacuous successful result. This applies equally to `command` and `sbatch_file`
jobs.

It does not know about project-specific model packages. Express those expectations as explicit `requires` paths or project-local checks outside launcher core.

## Config Model

These are the config concepts an agent should keep straight.

- Required for most project commands:
  `CLUSTER_LOGIN`
  `JOBS`
- Workspace mode:
  `WORKSPACE_MODE = "per-run" | "fixed"`
- Path requirement for `per-run`:
  `REMOTE_WORKSPACE_BASE`
- Path requirement for `fixed`:
  `REMOTE_WORKSPACE_DIR`
- Optional local project identity:
  `LOCAL_ROOT` controls the synced local tree; defaults to the config directory.
  `PROJECT_NAME` controls generated job-folder prefixes; defaults to the local root name.
- Optional log base:
  `REMOTE_LOG_BASE_PATH`
  If omitted, the launcher falls back to the configured workspace path.
- Runtime mode:
  `RUNTIME_MODE = "native" | "venv" | "singularity"`
- Required for `venv`:
  `VENV_PYTHON_EXECUTABLE`
- Required for `singularity`:
  `SINGULARITY_IMAGE_PATH`
- Optional for `singularity`:
  `SINGULARITY_EXEC_FLAGS`
  `SINGULARITY_EXTRA_ARGS` is removed; rename old configs to `SINGULARITY_EXEC_FLAGS`.
- Optional SSH behavior:
  `SSH_CONFIG_FILE`
  `SSH_OPTIONS`
- Optional sync/download behavior:
  `EXTRA_RSYNC_EXCLUDES`
  `EXTRA_RSYNC_ARGS`
  `ARTIFACT_PATHS`
  `SYNC_SYMLINKS`
  `LOCAL_ARTIFACT_ROOT`
  `VERBOSE`
- Optional job prerequisites:
  per-job `requires`
- Optional dashboard/archive integration:
  `REMOTE_SLURM_DASHBOARD_LOG_ARCHIVE_DIR`
  `REMOTE_SLURM_DASHBOARD_LOG_VIEW_DIR`

## Job Model

- Each job must define exactly one of:
  `command`
  `sbatch_file`
- `command` jobs may also define:
  `setup`
  `env`
  `sbatch`
- `sbatch_file` jobs may also define:
  `sbatch_args`
- Both job types may also define:
  `artifacts`
  `requires`
- `RUN_JOBS` defines a default subset.
- `--only` overrides `RUN_JOBS`.
- `DEFAULT_ENV` and `DEFAULT_SBATCH` apply globally before per-job overrides.
- `requires` entries are remote-workspace paths or globs used by `preflight`.

## Command Contracts

Each command below lists the purpose, when to use it, recommended agent invocation, and the most relevant arguments.

### `doctor`

- Purpose:
  Resolve cluster login, SSH settings, and archive-dir behavior. Optionally test SSH and remote SLURM tools.
- Use when:
  The task is generic cluster inspection or diagnosing config/SSH problems before other commands.
- Recommended for agents:
  `slurm-launcher doctor --json`
  `slurm-launcher doctor --ssh --json`
- Key arguments:
  `--cluster-login` to bypass config lookup and use the caller's normal SSH resolution, unless `--config` is also explicit.
  `--config` to point at a repo or user-level config.
  `--ssh` to test connectivity and `sacct`/`scontrol`/`squeue`.
  `--json` for machine-readable output.
- JSON fields:
  `cluster_login`, `config_path`, `ssh_config_file`, `ssh_options`, `archive_dir`, `archive_dir_source`, optional `ssh_ok`, optional `remote_tools`

### `jobs`

- Purpose:
  List recent jobs directly from the cluster.
- Use when:
  You need a broad view of job history or queue state, including jobs not launched by this repo.
- Recommended for agents:
  `slurm-launcher jobs --json`
  `slurm-launcher jobs --state running --json`
  `slurm-launcher jobs --hours 72 --limit 50 --json`
- Key arguments:
  `--cluster-login`
  `--config`
  `--user`
  `--hours`
  `--limit`
  repeatable `--state`
  `--json`
- JSON fields:
  `cluster_login`, `user`, `hours`, `limit`, `states`, `source`, `jobs`

### `job-show`

- Purpose:
  Show generic job details for one SLURM job ID.
- Use when:
  `jobs` already identified the interesting job and you need state, paths, command, or node details.
- Recommended for agents:
  `slurm-launcher job-show <job_id> --json`
- Key arguments:
  positional `job_id`
  `--cluster-login`
  `--config`
  `--json`
- JSON fields:
  `ok`, `job_id`, `detail_level`, `resolved_via`, plus any resolved fields among `job_name`, `state`, `partition`, `command`, `work_dir`, `stdout`, `stderr`, `node_list`, `num_nodes`, `gres`, `submit_time`, `start_time`, `end_time`, optional `launcher`; failures return `ok=false` and `error`
- Important behavior:
  Check `detail_level` before assuming all metadata is present.
  `detail_level=log-resolution` means the command could resolve log paths and core state, but omitted unavailable fields instead of returning a large set of `null` values.

### `job-log`

- Purpose:
  Resolve and optionally read stdout or stderr for a SLURM job ID.
- Use when:
  A specific job ID is already known.
- Recommended for agents:
  First: `slurm-launcher job-log <job_id> --json`
  Then, if needed: `slurm-launcher job-log <job_id> --stream stderr --follow`
- Key arguments:
  positional `job_id`
  `--stream stdout|stderr`
  `--lines <n>`
  `--follow`
  `--full`
  `--path-only`
  `--cluster-login`
  `--config`
  `--json`
- JSON fields:
  On success: `ok`, `job_id`, `job_name`, `state`, `stream`, `path`, `resolved_via`, `path_verified`, `content_included`, optional `probe_errors`; failures return `ok=false`, `error`, and `probe_errors`
- Important behavior:
  Path resolution tries SLURM metadata first. It uses an archive convention only when an archive directory is explicitly configured, and marks that fallback `path_verified=false`.
  SSH transport failures return an unresolved error instead of a guessed archive path.
  JSON discovery never includes log content; `content_included=false`.

### `init`

- Purpose:
  Create `.slurm/remote_launcher_config.mn5.py` and `.slurm/remote_launcher_config.mn5.example.py`.
- Use when:
  A project has not been configured for launcher use yet.
- Recommended for agents:
  `slurm-launcher init --non-interactive`
- Key arguments:
  `--force`
  `--non-interactive`

### `validate`

- Purpose:
  Validate config structure, selected jobs, path requirements, and optional remote runtime prerequisites.
- Use when:
  Before render, stage, submit, or run, especially after config edits.
- Recommended for agents:
  `slurm-launcher validate --json`
  `slurm-launcher validate --ssh --check-remote-paths --json`
- Key arguments:
  `--config`
  `--workspace per-run|fixed`
  `--only <job...>`
  `--ssh`
  `--check-remote-paths`
  `--json`
- JSON fields:
  `ok`, `config_path`, `workspace_mode`, `selected_jobs`, `warnings`, `errors`, `ssh_checked`, `remote_checks`

### `preflight`

- Purpose:
  Check generic remote prerequisites declared in each selected job's `requires`.
- Use when:
  Code has been staged, but before submitting jobs that would waste queue/GPU time if inputs, models, or symlink targets are missing.
- Recommended for agents:
  In per-run mode:
  `slurm-launcher stage --json`
  then `slurm-launcher preflight --job-folder <job_folder> --json`
  In fixed mode:
  `slurm-launcher preflight --workspace fixed --json`
- Key arguments:
  `--config`
  `--workspace`
  `--only <job...>`
  `--job-folder <folder>`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `dry_run`, `remote_workdir`, `checks_planned`, `checks_run`, `warnings`, `jobs`
- Important behavior:
  In `per-run` mode, `--job-folder` is required because the launcher must know which staged workspace to inspect.
  `--dry-run --json` returns the generated remote check script and does not SSH into the cluster.
  Live preflight runs remote shell checks only; it does not submit jobs.
  Checks are generic: plain paths, globs, and broken symlinks.
  If any selected job has no `requires`, preflight reports it as
  `status="not-configured"`, returns `ok=false`, and exits nonzero.

### `render`

- Purpose:
  Render launcher-generated scripts without submitting them.
- Use when:
  Inspecting final sbatch directives, shell quoting, runtime wrapping, or the exact command body.
- Recommended for agents:
  `slurm-launcher render --json`
  `slurm-launcher render --job-script --json`
- Key arguments:
  `--config`
  `--workspace`
  `--only <job...>`
  `--job-script`
  `--json`
- JSON fields:
  `ok`, `config_path`, `workspace_mode`, `selected_jobs`, `rendered_jobs`, `job_scripts`, `sbatch_scripts`

### `stage`

- Purpose:
  Sync project files to the remote workdir without submitting jobs.
- Use when:
  The workflow should be split into `stage` then `submit`, or when rsync/path debugging is needed.
- Recommended for agents:
  `slurm-launcher stage --dry-run --json`
- Key arguments:
  `--config`
  `--workspace`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `config_path`, `workspace_mode`, `remote_workdir`, `job_folder`, `commands`, `dry_run`

### `submit`

- Purpose:
  Submit jobs using an already prepared remote workdir.
- Use when:
  Stage already happened, or a fixed workspace is already current.
- Recommended for agents:
  `slurm-launcher submit --workspace per-run --job-folder <folder> --dry-run --json`
  `slurm-launcher submit --workspace fixed --json`
- Key arguments:
  `--config`
  `--workspace`
  `--only <job...>`
  `--job-folder <folder>`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `config_path`, `workspace_mode`, `remote_workdir`, `job_folder`, `selected_jobs`, `submitted_jobs`, `tracking_file`, `commands`, `monitor_command`, `dry_run`

### `sbatch`

- Purpose:
  Stage project files and submit one existing sbatch file from the project tree.
- Use when:
  The project already contains a hand-written sbatch script and you still want launcher staging/tracking behavior.
- Recommended for agents:
  `slurm-launcher sbatch slurm/train.sbatch --dry-run --json`
- Key arguments:
  positional `sbatch_file`
  `--name`
  repeatable `--sbatch-arg`
  `--config`
  `--workspace`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `config_path`, `workspace_mode`, `remote_workdir`, `job_folder`, `selected_jobs`, `submitted_jobs`, `tracking_file`, `commands`, `monitor_command`, `dry_run`

### `run`

- Purpose:
  Stage the project and submit the selected jobs in one command.
- Use when:
  Validation and render checks are already complete and the user wants the normal end-to-end flow.
- Recommended for agents:
  `slurm-launcher run --dry-run --json`
  `slurm-launcher run --only train eval --json`
- Key arguments:
  `--config`
  `--workspace`
  `--only <job...>`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `config_path`, `workspace_mode`, `remote_workdir`, `job_folder`, `selected_jobs`, `submitted_jobs`, `tracking_file`, `commands`, `monitor_command`, `dry_run`

### `status`

- Purpose:
  Query current SLURM state for tracked jobs or one explicit job ID.
- Use when:
  You need current `sacct`/`squeue` state after a launcher run, or a one-off job status without manually SSHing.
- Recommended for agents:
  `slurm-launcher status --json`
  `slurm-launcher status <job_id> --json`
- Key arguments:
  optional positional `job_id`
  `--job <job_id>`
  `--cluster-login <user@host>` for a direct job query
  `--tracking-file`
  `--latest`
  `--config`
  `--json`
- JSON fields:
  `ok`, `tracking_file`, `cluster_login`, `probes`, `unresolved_job_ids`, `jobs`; each job includes `source`
- Important behavior:
  With no job ID, status resolves the latest tracking file.
  With a job ID, status can use config cluster settings and may enrich from a tracking file if one exists.
  Returned job entries include `state`, `derived_state`, `exit_code`, timestamps, `elapsed`, and `partition` when SLURM reports them.
  Status merges `sacct` history with `squeue` state for jobs missing from accounting.
  Probe failures are explicit. If they leave any job unresolved, JSON returns `ok=false` and the command exits nonzero.
  Treat `derived_state=UNKNOWN` as inconclusive, not as evidence that a job stopped or failed. Verify with `job-show <job_id> --json`.
  Tracking-backed status uses the complete SSH context recorded in `jobs.json`; do not mix in an unrelated generic config.

### `logs`

- Purpose:
  Resolve stdout/stderr paths from local tracking and optionally read their remote content.
- Use when:
  You want paths or content for jobs from the latest or a specified launcher submission.
- Recommended for agents:
  `slurm-launcher logs --json`
  Then, if content was requested:
  `slurm-launcher logs --job <job_id> --lines 50`
- Key arguments:
  `--tracking-file`
  `--latest`
  `--job <job_id>`
  `--only <job...>`
  `--stderr`
  `--lines <n>`
  `--follow`
  `--full`
  `--json`
- JSON fields:
  `ok`, `source`, `content_included`, `remote_checked`, `created_at`, `cluster_login`, `ssh_config_file`, `ssh_options`, `job_folder`, `remote_workdir`, `remote_logdir`, `remote_slurm_output_dir`, `jobs`
- Important behavior:
  JSON mode returns tracking metadata and resolved paths; it does not include log text.
  Plain `logs` shows paths. Non-JSON mode reads remote content when selecting a job or passing `--stderr`, `--lines`, `--follow`, or `--full`.

### `artifacts`

- Purpose:
  List declared artifact paths, verify them remotely, or download them.
- Use when:
  You want the launcher to resolve declared outputs from the latest/tracked run instead of manually building `rsync` commands.
- Recommended for agents:
  `slurm-launcher artifacts list --json`
  `slurm-launcher artifacts check --json`
  `slurm-launcher artifacts download --dry-run --json`
- Key arguments:
  subcommand `list|check|download`
  `--tracking-file`
  `--only <job...>`
  `--output-dir`
  `--dry-run` for download
  `--json`
- JSON fields:
  Always: `ok`, `operation`, `source`, `tracking_file`, `output_dir`, `remote_checked`, `declared_only`, `copy_attempted`, `artifacts`; downloads also include `dry_run`, `commands`, and `failures`; checked artifacts include `exists`, `kind`, and `size_bytes`
- Important behavior:
  `list` does not SSH; it derives remote/local paths from the tracking file and returns `declared_only=true`, `remote_checked=false`.
  `check` performs one read-only SSH query and reports actual remote existence, type, and size without downloading.
  `download --dry-run --json` returns the `rsync` commands without copying.
  `download` copies declared artifacts under `slurm_output/downloaded_artifacts/<job_folder>/...` unless `--output-dir` is set.
  Do not turn an inspect/read request into a download; require the user to request a local copy.

### `monitor`

- Purpose:
  Run `squeue` for tracked job IDs.
- Use when:
  The tracking file already exists and current queue state is needed.
- Recommended for agents:
  `slurm-launcher monitor --json`
  `slurm-launcher monitor --dry-run --json`
- Key arguments:
  `--tracking-file`
  `--only <job...>`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `tracking_file`, `job_ids`, `command`, `dry_run`, optional `returncode`, optional `stdout`, optional `stderr`

### `download-logs`

- Purpose:
  Download tracked `.out` and `.err` files locally.
- Use when:
  Remote log paths are already known from tracking and a local copy is needed.
- Recommended for agents:
  `slurm-launcher download-logs --dry-run --json`
  `slurm-launcher download-logs --json`
- Key arguments:
  `--tracking-file`
  repeatable `--job-name`
  repeatable `--job-id`
  `--output-dir`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `tracking_file`, `cluster_login`, `selected_jobs`, `downloads`, `commands`, `output_dir`, `dry_run`, `failures`
- Important behavior:
  Use the returned `commands` and `downloads` instead of constructing `rsync` manually.

### `download-artifacts`

- Purpose:
  Download configured or explicit artifact paths from the tracked remote workdir.
- Use when:
  Outputs, checkpoints, or reports need to be copied back locally after a run.
- Recommended for agents:
  `slurm-launcher download-artifacts --dry-run --json`
  `slurm-launcher download-artifacts --path outputs/metrics.json --json`
- Key arguments:
  `--tracking-file`
  repeatable `--path`
  `--output-dir`
  `--dry-run`
  `--json`
- JSON fields:
  `ok`, `tracking_file`, `cluster_login`, `remote_workdir`, `artifact_paths`, `artifacts`, `commands`, `output_dir`, `dry_run`, `failures`
- Important behavior:
  Use `--path` for targeted downloads and the returned `artifacts`/`commands` fields instead of remote `find`, `du`, or manual `rsync`.

### `summary`

- Purpose:
  Write/update a compact run summary for the latest or specified tracking file.
- Use when:
  A future assistant/session should quickly know what was launched, what state jobs are in, and where outputs/logs are expected.
- Recommended for agents:
  `slurm-launcher summary --json`
  `slurm-launcher summary --tracking-file slurm_output/<job_folder>/jobs.json --json`
- Key arguments:
  `--tracking-file`
  `--config`
  `--json`
- JSON fields:
  `ok`, `summary_path`, `config_path`
- Important behavior:
  Summary is anchored to the tracking payload's `job_folder` and `remote_workdir`; it should not create a fresh per-run folder.
  It queries current SLURM status, writes local `slurm_output/<job_folder>/summary.json`, and writes remote `.slurm_run/summary.json` in the tracked workspace.

## Recommended Agent Loop

1. Resolve the mode.
- Generic cluster inspection: start with `doctor`, `jobs`, `job-show`, or `job-log`.
- Project execution: start with `validate`.

2. Validate before submitting.
- `slurm-launcher validate --json`
- Add `--ssh --check-remote-paths` before remote execution when runtime paths matter.

3. Preview before writing or submitting.
- `slurm-launcher render --json`
- `slurm-launcher stage --dry-run --json`
- `slurm-launcher submit --dry-run --json`
- `slurm-launcher sbatch --dry-run --json`
- `slurm-launcher run --dry-run --json`

4. For per-run preflight, stage before checking.
- `slurm-launcher stage --json`
- Read `job_folder`.
- `slurm-launcher preflight --job-folder <job_folder> --json`

5. Execute the smallest safe step.
- Prefer `--only` when iterating on one job.
- Prefer `submit --job-folder <job_folder>` over `run` when the stage step already succeeded.

6. Inspect or recover.
- `status --json`
- `logs --json`
- `monitor --json`
- `job-show <job_id> --json`
- `job-log <job_id> --json`
- `artifacts list --json`
- `artifacts check --json` when remote existence matters
- `download-logs --dry-run --json`
- `download-artifacts --dry-run --json`
- `summary --json`

## Guardrails

- Do not invent wrappers when the installed CLI should work.
- Do not assume project commands work outside the intended repo.
- Do not read full logs before first resolving the path and stream intentionally.
- Do not use raw SSH or rsync for status, log lookup, or artifact downloads until the launcher command path has been tried.
- Do not infer command arguments from related commands; consult `<command> --help` after upgrades or when a command rejects an argument.
- Do not interpret `UNKNOWN` as a terminal job state; verify the job with `job-show`.
- Do not interpret `artifacts list` as proof that a path exists; use `artifacts check`.
- Do not download logs or artifacts when the user asked only to inspect, read, or tail them remotely.
- Do not treat launcher enrichment as required for generic cluster inspection.
- Do not modify this source repository unless the user explicitly asks for source changes.
