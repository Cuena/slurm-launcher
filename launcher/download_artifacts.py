from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .core import build_rsync_ssh_command
from .tracking import (
    TrackingError,
    TrackingPayload,
    load_tracking_payload,
    resolve_tracking_file,
)


def add_download_artifacts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tracking-file",
        help=(
            "Path to a jobs.json file. Defaults to slurm_output/latest_jobs.json, "
            "or the most recent slurm_output/*/jobs.json."
        ),
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help=(
            "Artifact path to download. Relative paths are resolved from the tracked "
            "remote workdir. Repeatable. Overrides tracked ARTIFACT_PATHS defaults."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Local destination directory. "
            "Default: slurm_output/downloaded_artifacts/<job_folder>/"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rsync commands without executing them.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )


def _selected_artifact_paths(
    args: argparse.Namespace, payload: TrackingPayload
) -> list[str]:
    requested_paths = [str(item).strip() for item in args.path if str(item).strip()]
    paths = requested_paths or payload.artifact_paths
    seen: set[str] = set()
    unique_paths: list[str] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _resolve_remote_artifact_path(remote_workdir: str, artifact_path: str) -> str:
    if artifact_path.startswith("/"):
        return artifact_path
    return f"{remote_workdir.rstrip('/')}/{artifact_path.lstrip('/')}"


def _local_artifact_path(artifact_path: str) -> Path:
    path = Path(artifact_path)
    if path.is_absolute():
        return Path("absolute") / path.relative_to("/")
    return path


def _artifact_entry(
    cluster_login: str,
    remote_workdir: str,
    artifact_path: str,
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> dict[str, object]:
    remote_path = _resolve_remote_artifact_path(remote_workdir, artifact_path)
    local_path = _local_artifact_path(artifact_path)
    destination_parent = output_dir / local_path.parent
    source = f"{cluster_login}:{remote_path}"
    cmd = [
        "rsync",
        "-az",
        "-e",
        build_rsync_ssh_command(ssh_config_file, ssh_options),
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([source, str(destination_parent)])
    return {
        "path": artifact_path,
        "remote_path": remote_path,
        "destination": str(destination_parent),
        "command": shlex.join(cmd),
        "argv": cmd,
    }


def _artifact_entries(
    cluster_login: str,
    remote_workdir: str,
    artifact_paths: list[str],
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[dict[str, object]]:
    return [
        _artifact_entry(
            cluster_login,
            remote_workdir,
            artifact_path,
            output_dir,
            dry_run=dry_run,
            ssh_config_file=ssh_config_file,
            ssh_options=ssh_options,
        )
        for artifact_path in artifact_paths
    ]


def _run_downloads(
    cluster_login: str,
    remote_workdir: str,
    artifact_paths: list[str],
    output_dir: Path,
    *,
    dry_run: bool,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    quiet: bool = False,
) -> int:
    failures = 0
    entries = _artifact_entries(
        cluster_login,
        remote_workdir,
        artifact_paths,
        output_dir,
        dry_run=dry_run,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    for entry in entries:
        artifact_path = str(entry["path"])
        remote_path = str(entry["remote_path"])
        destination_parent = Path(str(entry["destination"]))
        cmd = list(entry["argv"])

        if not quiet:
            print(f"{artifact_path} -> {remote_path}")
            print(f"  -> {destination_parent}")
            print(f"  $ {entry['command']}")

        if dry_run:
            continue

        destination_parent.mkdir(parents=True, exist_ok=True)
        if quiet:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            if not quiet:
                print(
                    f"ERROR: rsync failed ({result.returncode}) for {artifact_path}: {remote_path}",
                    file=sys.stderr,
                )
    return failures


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _emit_json_error(message: str, **extra: object) -> int:
    payload: dict[str, object] = {"ok": False, "error": message}
    payload.update(extra)
    _print_json(payload)
    return 1


def run_download_artifacts(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False))
    tracking_path = resolve_tracking_file(args.tracking_file)
    if tracking_path is None:
        message = (
            "No tracking file found. "
            "Run a non-dry submission first or pass --tracking-file."
        )
        if json_output:
            return _emit_json_error(message, tracking_file=args.tracking_file)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    try:
        payload = load_tracking_payload(tracking_path)
    except TrackingError as exc:
        if json_output:
            return _emit_json_error(str(exc), tracking_file=str(tracking_path))
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not payload.cluster_login:
        if json_output:
            return _emit_json_error(
                f"Missing cluster_login in {tracking_path}",
                tracking_file=str(tracking_path),
            )
        print(f"ERROR: Missing cluster_login in {tracking_path}", file=sys.stderr)
        return 1

    if not payload.remote_workdir:
        if json_output:
            return _emit_json_error(
                f"Missing remote_workdir in {tracking_path}",
                tracking_file=str(tracking_path),
            )
        print(f"ERROR: Missing remote_workdir in {tracking_path}", file=sys.stderr)
        return 1

    artifact_paths = _selected_artifact_paths(args, payload)
    if not artifact_paths:
        message = (
            "No artifact paths configured. "
            "Set ARTIFACT_PATHS in the config used for submission or pass --path."
        )
        if json_output:
            return _emit_json_error(
                message,
                tracking_file=str(tracking_path),
                cluster_login=payload.cluster_login,
                remote_workdir=payload.remote_workdir,
            )
        print(f"ERROR: {message}", file=sys.stderr)
        return 1

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("slurm_output") / "downloaded_artifacts" / payload.job_folder
    )

    entries = _artifact_entries(
        payload.rsync_login or payload.cluster_login,
        payload.remote_workdir,
        artifact_paths,
        output_dir,
        dry_run=args.dry_run,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
    )

    if json_output:
        failures = _run_downloads(
            payload.rsync_login or payload.cluster_login,
            payload.remote_workdir,
            artifact_paths,
            output_dir,
            dry_run=args.dry_run,
            ssh_config_file=payload.ssh_config_file,
            ssh_options=payload.ssh_options,
            quiet=True,
        )
        _print_json(
            {
                "ok": failures == 0,
                "tracking_file": str(tracking_path),
                "cluster_login": payload.cluster_login,
                "remote_workdir": payload.remote_workdir,
                "artifact_paths": artifact_paths,
                "artifacts": [
                    {
                        "path": entry["path"],
                        "remote_path": entry["remote_path"],
                        "destination": entry["destination"],
                    }
                    for entry in entries
                ],
                "commands": [str(entry["command"]) for entry in entries],
                "output_dir": str(output_dir),
                "dry_run": bool(args.dry_run),
                "failures": failures,
            }
        )
        return 0 if failures == 0 else 1

    print(f"Tracking file: {tracking_path}")
    print(f"Cluster: {payload.cluster_login}")
    print(f"Remote workdir: {payload.remote_workdir}")
    print(f"Artifact paths to download: {len(artifact_paths)}")
    print(f"Local destination: {output_dir}")
    if args.dry_run:
        print("Dry-run mode: commands will not be executed.")

    failures = _run_downloads(
        payload.rsync_login or payload.cluster_login,
        payload.remote_workdir,
        artifact_paths,
        output_dir,
        dry_run=args.dry_run,
        ssh_config_file=payload.ssh_config_file,
        ssh_options=payload.ssh_options,
    )
    if failures:
        print(f"Completed with {failures} failed download(s).", file=sys.stderr)
        return 1

    print("Download complete.")
    return 0


def parse_download_artifacts_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download remote artifacts tracked by slurm-launcher. "
            "Defaults to ARTIFACT_PATHS from the latest tracking file."
        )
    )
    add_download_artifacts_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_download_artifacts(parse_download_artifacts_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
