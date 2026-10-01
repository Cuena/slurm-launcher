"""Short command descriptions and examples used by CLI help."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    summary: str
    examples: tuple[str, ...] = ()


COMMAND_SPECS = {
    "init": CommandSpec("Create a project configuration.", ("init --non-interactive",)),
    "doctor": CommandSpec(
        "Inspect cluster configuration and optional SSH connectivity.",
        ("doctor --ssh --json",),
    ),
    "validate": CommandSpec(
        "Validate trusted project config without preparing or submitting jobs.",
        ("validate --json",),
    ),
    "render": CommandSpec(
        "Preview configured scripts without staging or submitting.",
        ("render --only train --json",),
    ),
    "stage": CommandSpec(
        "Prepare and stage selected jobs; save their resolved launch plan.",
        ("stage --only train --json",),
    ),
    "run": CommandSpec(
        "Stage and submit selected experiments, preserving partial results.",
        ("run --only train --dry-run --json", "run --only train --json"),
    ),
    "submit": CommandSpec(
        "Submit eligible jobs from a frozen staged plan; no config import.",
        ("submit --run latest --json",),
    ),
    "sbatch": CommandSpec(
        "Stage a project and submit one existing sbatch script.",
        ("sbatch scripts/train.sbatch --dry-run --json",),
    ),
    "preflight": CommandSpec(
        "Check the prerequisites saved in a staged run.",
        ("preflight --run latest --json",),
    ),
    "jobs": CommandSpec(
        "List recent scheduler jobs, including jobs outside the launcher.",
        ("jobs --state running --json",),
    ),
    "status": CommandSpec(
        "Read scheduler state and exit codes for a run or direct job ID.",
        ("status --run latest --json", "status --job-id 123 --json"),
    ),
    "job-show": CommandSpec(
        "Read detailed scheduler metadata or the submitted batch script.",
        ("job-show 123 --sbatch --json",),
    ),
    "logs": CommandSpec(
        "Read bounded log content; JSON includes content and continuation.",
        (
            "logs --job-id 123 --json",
            "logs --run latest --job train --stream stderr --json",
            "logs --cursor TOKEN --json",
        ),
    ),
    "artifacts": CommandSpec(
        "List declarations, check remote outputs, or explicitly download them.",
        (
            "artifacts check --run latest --json",
            "artifacts download --run latest --path outputs/metrics.json --dry-run --json",
        ),
    ),
    "download-logs": CommandSpec(
        "Copy tracked scheduler logs locally when a download is requested.",
        ("download-logs --run latest --dry-run --json",),
    ),
    "summary": CommandSpec(
        "Read run metadata and current scheduler state without writing files.",
        ("summary --run latest --json",),
    ),
}

COMMAND_NAMES = tuple(COMMAND_SPECS)
