"""Read-only run inspection, independent of the current project configuration."""

from argparse import Namespace
from dataclasses import asdict

from rich.console import Console

from .status import query_job_statuses, status_payload
from .tracking import TrackingError, load_tracking_payload, resolve_tracking_file

console = Console()


def run_summary(args: Namespace) -> int:
    try:
        tracking_file = resolve_tracking_file(args.run)
        if tracking_file is None:
            raise TrackingError("Run not found. Pass --run ID or a tracking path.")
        tracked = load_tracking_payload(tracking_file)
        if not tracked.cluster_login:
            raise TrackingError("Run has no saved cluster login.")
        status = query_job_statuses(
            tracked.cluster_login,
            tracked.jobs,
            ssh_config_file=tracked.ssh_config_file,
            ssh_options=tracked.ssh_options,
        )
        result = {
            "ok": status.ok,
            "run_id": tracked.job_folder,
            "tracking_file": str(tracking_file),
            "cluster_login": tracked.cluster_login,
            "remote_workdir": tracked.remote_workdir,
            "jobs": [asdict(job) for job in tracked.jobs],
            "statuses": [status_payload(job) for job in status.statuses],
            "probes": [asdict(probe) for probe in status.probes],
            "unresolved_job_ids": status.unresolved_job_ids,
        }
    except (TrackingError, OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    console.print_json(data=result)
    return 0 if result["ok"] else 1
