"""Shared command contract metadata for docs and agent-facing guidance."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandSpec:
    """Public-facing contract metadata for one CLI command."""

    summary: str
    agent_recommendation: str
    supports_json: bool
    json_fields: tuple[str, ...] = field(default_factory=tuple)
    examples: tuple[str, ...] = field(default_factory=tuple)


COMMAND_SPECS: dict[str, CommandSpec] = {
    "doctor": CommandSpec(
        summary="Resolve generic cluster config and optionally check SSH plus SLURM tools.",
        agent_recommendation="Prefer `doctor --json`; add `--ssh` when an agent needs to separate config issues from remote environment issues.",
        supports_json=True,
        json_fields=(
            "cluster_login",
            "config_path",
            "ssh_config_file",
            "ssh_options",
            "archive_dir",
            "archive_dir_source",
            "ssh_ok",
            "remote_tools",
        ),
        examples=(
            "slurm-launcher doctor --json",
            "slurm-launcher doctor --ssh --json",
        ),
    ),
    "jobs": CommandSpec(
        summary="List recent jobs directly from the cluster without using tracking files.",
        agent_recommendation="Prefer `jobs --json` for automation; narrow the query with `--user`, `--hours`, `--limit`, and `--state` before calling `job-show` on one job.",
        supports_json=True,
        json_fields=(
            "cluster_login",
            "user",
            "hours",
            "limit",
            "states",
            "source",
            "jobs",
        ),
        examples=(
            "slurm-launcher jobs --json",
            "slurm-launcher jobs --state running --json",
            "slurm-launcher jobs --hours 72 --limit 50 --json",
        ),
    ),
    "job-show": CommandSpec(
        summary="Show generic details for one SLURM job ID.",
        agent_recommendation="Prefer `job-show <job_id> --json`; check `detail_level` before assuming every SLURM field is available, and treat launcher enrichment as optional metadata.",
        supports_json=True,
        json_fields=(
            "job_id",
            "detail_level",
            "job_name",
            "state",
            "partition",
            "command",
            "work_dir",
            "stdout",
            "stderr",
            "node_list",
            "num_nodes",
            "gres",
            "submit_time",
            "start_time",
            "end_time",
            "resolved_via",
            "launcher",
        ),
        examples=("slurm-launcher job-show 12345 --json",),
    ),
    "job-log": CommandSpec(
        summary="Resolve and optionally read stdout or stderr for one SLURM job ID.",
        agent_recommendation="Use `job-log <job_id> --json` to resolve paths first; only read content after checking the returned `path` and `resolved_via` fields.",
        supports_json=True,
        json_fields=("job_id", "job_name", "state", "stream", "path", "resolved_via"),
        examples=(
            "slurm-launcher job-log 12345 --json",
            "slurm-launcher job-log 12345 --stream stderr --follow",
        ),
    ),
    "init": CommandSpec(
        summary="Create the local project config scaffold in `.slurm/`.",
        agent_recommendation="Use plain text output here; agents should prefer `init --non-interactive` unless they are explicitly collecting answers from the user.",
        supports_json=False,
        examples=(
            "slurm-launcher init",
            "slurm-launcher init --non-interactive",
        ),
    ),
    "validate": CommandSpec(
        summary="Validate the config, selected jobs, and optional remote runtime prerequisites.",
        agent_recommendation="Prefer `validate --json`; add `--ssh --check-remote-paths` before expensive remote actions.",
        supports_json=True,
        json_fields=(
            "ok",
            "config_path",
            "workspace_mode",
            "selected_jobs",
            "warnings",
            "errors",
            "ssh_checked",
            "remote_checks",
        ),
        examples=(
            "slurm-launcher validate --json",
            "slurm-launcher validate --ssh --check-remote-paths --json",
        ),
    ),
    "preflight": CommandSpec(
        summary="Check remote prerequisites for selected jobs before submitting.",
        agent_recommendation="Prefer `preflight --dry-run --json` first; fail loud on missing globs or broken symlinks.",
        supports_json=True,
        json_fields=(
            "ok",
            "remote_workdir",
            "jobs",
        ),
        examples=(
            "slurm-launcher preflight --only sam3_batch_quality_all_clips --dry-run --json",
            "slurm-launcher preflight --json",
        ),
    ),
    "render": CommandSpec(
        summary="Render generated job and sbatch scripts without submitting anything.",
        agent_recommendation="Prefer `render --json`; add `--job-script` only when the job-body script itself matters.",
        supports_json=True,
        json_fields=(
            "ok",
            "config_path",
            "workspace_mode",
            "selected_jobs",
            "rendered_jobs",
            "job_scripts",
            "sbatch_scripts",
        ),
        examples=(
            "slurm-launcher render --json",
            "slurm-launcher render --only train --job-script --json",
        ),
    ),
    "stage": CommandSpec(
        summary="Sync project files to the remote workdir without submitting jobs.",
        agent_recommendation="Prefer `stage --dry-run --json` before real staging when an agent is still validating path resolution.",
        supports_json=True,
        json_fields=(
            "ok",
            "config_path",
            "workspace_mode",
            "remote_workdir",
            "job_folder",
            "commands",
            "dry_run",
        ),
        examples=(
            "slurm-launcher stage --dry-run --json",
            "slurm-launcher stage --workspace fixed --json",
        ),
    ),
    "artifacts": CommandSpec(
        summary="List or download job artifacts from a tracked submission.",
        agent_recommendation="Prefer `artifacts list --json` before `artifacts download`; use `--only` to limit scope.",
        supports_json=True,
        json_fields=(
            "ok",
            "tracking_file",
            "output_dir",
            "dry_run",
            "artifacts",
            "commands",
            "failures",
        ),
        examples=(
            "slurm-launcher artifacts list --json",
            "slurm-launcher artifacts download --dry-run --json",
            "slurm-launcher artifacts download --only train eval --json",
        ),
    ),
    "submit": CommandSpec(
        summary="Submit jobs without syncing code first.",
        agent_recommendation="Prefer `submit --dry-run --json`; in `per-run` mode pass `--job-folder` from a previous stage result.",
        supports_json=True,
        json_fields=(
            "ok",
            "config_path",
            "workspace_mode",
            "remote_workdir",
            "job_folder",
            "selected_jobs",
            "submitted_jobs",
            "tracking_file",
            "commands",
            "monitor_command",
            "dry_run",
        ),
        examples=(
            "slurm-launcher submit --workspace per-run --job-folder run_001 --dry-run --json",
            "slurm-launcher submit --workspace fixed --json",
        ),
    ),
    "sbatch": CommandSpec(
        summary="Stage code and submit one existing sbatch file from the project.",
        agent_recommendation="Prefer `sbatch --dry-run --json` first, especially when extra `--sbatch-arg` values change export or partition behavior.",
        supports_json=True,
        json_fields=(
            "ok",
            "config_path",
            "workspace_mode",
            "remote_workdir",
            "job_folder",
            "selected_jobs",
            "submitted_jobs",
            "tracking_file",
            "commands",
            "monitor_command",
            "dry_run",
        ),
        examples=(
            "slurm-launcher sbatch slurm/train.sbatch --dry-run --json",
            "slurm-launcher sbatch slurm/train.sbatch --sbatch-arg --export=ALL,SEED=1 --json",
        ),
    ),
    "run": CommandSpec(
        summary="Stage the project and submit the selected jobs in one command.",
        agent_recommendation="Prefer `run --dry-run --json` before a live submission; use `--only` to limit blast radius while iterating.",
        supports_json=True,
        json_fields=(
            "ok",
            "config_path",
            "workspace_mode",
            "remote_workdir",
            "job_folder",
            "selected_jobs",
            "submitted_jobs",
            "tracking_file",
            "commands",
            "monitor_command",
            "dry_run",
        ),
        examples=(
            "slurm-launcher run --dry-run --json",
            "slurm-launcher run --only train eval --json",
        ),
    ),
    "logs": CommandSpec(
        summary="Show tracked stdout and stderr paths from a previous submission.",
        agent_recommendation="Prefer `logs --json` before `monitor`, `download-logs`, or targeted manual inspection.",
        supports_json=True,
        json_fields=(
            "created_at",
            "cluster_login",
            "ssh_config_file",
            "ssh_options",
            "job_folder",
            "remote_workdir",
            "remote_logdir",
            "remote_slurm_output_dir",
            "jobs",
        ),
        examples=("slurm-launcher logs --json",),
    ),
    "status": CommandSpec(
        summary="Show current SLURM state for tracked jobs or a single job id.",
        agent_recommendation="Prefer `status --json` after a run; use `--job` for one-off checks without a tracking file.",
        supports_json=True,
        json_fields=(
            "ok",
            "tracking_file",
            "cluster_login",
            "jobs",
        ),
        examples=(
            "slurm-launcher status --latest --json",
            "slurm-launcher status --job 43054508 --json",
        ),
    ),
    "monitor": CommandSpec(
        summary="Run `squeue` for the tracked job IDs from a prior submission.",
        agent_recommendation="Prefer `monitor --json`; use `--dry-run` first when you only need the exact SSH command.",
        supports_json=True,
        json_fields=(
            "ok",
            "tracking_file",
            "job_ids",
            "command",
            "dry_run",
            "returncode",
            "stdout",
            "stderr",
        ),
        examples=(
            "slurm-launcher monitor --json",
            "slurm-launcher monitor --dry-run --json",
        ),
    ),
    "download-logs": CommandSpec(
        summary="Download tracked stdout and stderr files to the local machine.",
        agent_recommendation="Prefer `download-logs --dry-run --json` before a live download; use the returned commands instead of constructing rsync manually.",
        supports_json=True,
        json_fields=(
            "ok",
            "tracking_file",
            "cluster_login",
            "selected_jobs",
            "downloads",
            "commands",
            "output_dir",
            "dry_run",
            "failures",
        ),
        examples=(
            "slurm-launcher download-logs --dry-run --json",
            "slurm-launcher download-logs --job-name train --json",
        ),
    ),
    "download-artifacts": CommandSpec(
        summary="Download configured or explicit artifact paths from the tracked remote workdir.",
        agent_recommendation="Prefer `download-artifacts --dry-run --json` before a live download; use `--path` for targeted artifacts instead of raw rsync.",
        supports_json=True,
        json_fields=(
            "ok",
            "tracking_file",
            "cluster_login",
            "remote_workdir",
            "artifact_paths",
            "artifacts",
            "commands",
            "output_dir",
            "dry_run",
            "failures",
        ),
        examples=(
            "slurm-launcher download-artifacts --dry-run --json",
            "slurm-launcher download-artifacts --path outputs/metrics.json --json",
        ),
    ),
}


COMMAND_NAMES = tuple(COMMAND_SPECS)
DEFAULT_COMMAND = "run"
