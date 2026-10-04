from __future__ import annotations

import argparse
import base64
import binascii
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from .core import resolve_log_path
from .transport import build_ssh_command
from .job_tools import resolve_job_log_info
from .tracking import TrackingError, load_tracking_payload, resolve_tracking_file

MAX_FILES = 256
MAX_CURSOR = 2 * 1024 * 1024


def add_logs_args(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--run", help="Run folder ID, tracking path, or latest (default)."
    )
    target.add_argument("--job-id", help="Read logs for a direct SLURM job ID.")
    parser.add_argument("--job", help="Select a tracked job by name.")
    parser.add_argument(
        "--stream",
        choices=("both", "stdout", "stderr"),
        default="both",
        help="Stream to read; default: both, separately labeled.",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=100,
        help="Initial tail line limit per file (default: 100).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=65536,
        help="Total content byte limit per response (default: 65536; maximum: 1048576).",
    )
    parser.add_argument(
        "--search", help="Search for a literal string from the beginning."
    )
    parser.add_argument(
        "--context",
        type=int,
        default=20,
        help="Lines around a search match (default: 20).",
    )
    parser.add_argument(
        "--cursor", help="Resume an opaque cursor without repeating returned bytes."
    )
    parser.add_argument(
        "--path-only", action="store_true", help="Stat paths without reading content."
    )
    parser.add_argument(
        "--follow", action="store_true", help="Follow interactively until interrupted."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print content and navigation metadata as JSON.",
    )
    parser.add_argument(
        "--config", type=Path, help="Configuration for a direct job ID."
    )
    parser.add_argument("--cluster-login", help="SSH login for a direct job ID.")
    parser.add_argument(
        "--path", help="Application log path relative to the tracked workdir."
    )


def _encode_cursor(state: dict[str, Any]) -> str:
    token = (
        base64.urlsafe_b64encode(json.dumps(state, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )
    if len(token) > MAX_CURSOR:
        raise ValueError("Cursor is too large.")
    return token


def _validate_options(options: dict[str, Any]) -> None:
    for key, low, high in (
        ("lines", 1, 100000),
        ("max_bytes", 4, 1048576),
        ("context", 0, 10000),
        ("scan_bytes", 65536, 8388608),
    ):
        value = options.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(
                f"{key.replace('_', '-')} must be between {low} and {high}."
            )
    search = options.get("search")
    if search is not None and (
        not isinstance(search, str) or not search or len(search.encode()) > 4096
    ):
        raise ValueError("--search must contain between 1 and 4096 UTF-8 bytes.")
    if type(options.get("path_only")) is not bool:
        raise ValueError("Invalid path-only setting.")


def _decode_cursor(token: str) -> dict[str, Any]:
    try:
        if len(token) > MAX_CURSOR:
            raise ValueError("Cursor is too large.")
        state = json.loads(
            base64.b64decode(
                token + "=" * (-len(token) % 4), altchars=b"-_", validate=True
            )
        )
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Unsupported cursor version.")
        _validate_options(state["options"])
        context = state["context"]
        if not isinstance(context, dict):
            raise ValueError("Invalid SSH context.")
        login = context["cluster_login"]
        if (
            not isinstance(login, str)
            or not login
            or login.startswith("-")
            or any(c.isspace() for c in login)
        ):
            raise ValueError("Invalid cluster login.")
        for key in ("ssh_config_file", "archive_dir", "config_path"):
            if context.get(key) is not None and not isinstance(context[key], str):
                raise ValueError("Invalid SSH context.")
        if not isinstance(context["ssh_options"], list) or not all(
            isinstance(v, str) and "\0" not in v for v in context["ssh_options"]
        ):
            raise ValueError("Invalid SSH options.")
        if state.get("tracking_file") is not None and not isinstance(
            state["tracking_file"], str
        ):
            raise ValueError("Invalid tracking path.")
        files = state["files"]
        if not isinstance(files, list) or not 1 <= len(files) <= MAX_FILES:
            raise ValueError("Invalid cursor file count.")
        for spec in files:
            for key in ("job_id", "job_name", "stream", "source", "path", "root"):
                value = spec.get(key)
                if value is not None and (
                    not isinstance(value, str) or "\0" in value or len(value) > 4096
                ):
                    raise ValueError("Invalid cursor file identity.")
            if spec.get("stream") not in ("stdout", "stderr", "application"):
                raise ValueError("Invalid cursor stream.")
            errors = spec.get("resolution_errors", [])
            if not isinstance(errors, list) or len(errors) > 32 or not all(
                isinstance(error, str) and len(error) <= 4096 for error in errors
            ):
                raise ValueError("Invalid cursor resolution errors.")
            position = spec.get("position", {})
            if not isinstance(position, dict):
                raise ValueError("Invalid cursor position.")
            for key in (
                "offset", "emitted", "pending_end", "anchor_length",
                "boundary_start", "boundary_length",
            ):
                value = position.get(key, 0)
                if type(value) is not int or not 0 <= value <= (
                    64 if key in ("anchor_length", "boundary_length") else 2**63 - 1
                ):
                    raise ValueError("Invalid cursor offset.")
            if "identity" in position and (
                not isinstance(position["identity"], str) or len(position["identity"]) > 128
            ):
                raise ValueError("Invalid cursor identity.")
            for key in ("anchor", "boundary"):
                if key in position and (
                    not isinstance(position[key], str)
                    or len(position[key]) != 64
                    or any(c not in "0123456789abcdef" for c in position[key])
                ):
                    raise ValueError("Invalid cursor fingerprint.")
            if "boundary" in position and (
                "boundary_start" not in position or "boundary_length" not in position
                or position["boundary_start"] + position["boundary_length"]
                != position.get("offset", 0)
            ):
                raise ValueError("Invalid cursor boundary.")
        return state
    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        binascii.Error,
        UnicodeError,
    ) as exc:
        raise ValueError(f"Invalid log cursor: {exc}") from exc


def _initial_state(args: argparse.Namespace, cluster_context) -> dict[str, Any]:
    if args.job_id and (args.run or args.job or args.path is not None):
        raise ValueError("--job and --path select tracked logs, not a direct job ID.")
    options = {
        "lines": args.lines,
        "max_bytes": args.max_bytes,
        "search": args.search,
        "context": args.context,
        "path_only": args.path_only,
        "scan_bytes": min(8388608, max(65536, args.max_bytes * 16)),
    }
    _validate_options(options)
    streams = ("stdout", "stderr") if args.stream == "both" else (args.stream,)
    files = []
    tracking_file = None
    if args.job_id:
        if cluster_context is None:
            if not args.cluster_login or args.config:
                raise ValueError(
                    "Direct logs require a resolved cluster context or --cluster-login."
                )
            cluster_context = (args.cluster_login, None, None, [], None)
        login, archive_dir, ssh_config, ssh_options, config_path = cluster_context
        context = {
            "cluster_login": login,
            "archive_dir": archive_dir,
            "ssh_config_file": ssh_config,
            "ssh_options": list(ssh_options or []),
            "config_path": str(config_path) if config_path else None,
        }
        info = resolve_job_log_info(
            login,
            args.job_id,
            archive_dir=archive_dir,
            ssh_config_file=ssh_config,
            ssh_options=ssh_options,
        )
        for stream in streams:
            files.append(
                {
                    "job_id": args.job_id,
                    "job_name": info.job_name,
                    "stream": stream,
                    "path": getattr(info, stream),
                    "source": getattr(info, stream + "_source") or "unresolved",
                    "resolution_errors": list(info.probe_errors),
                }
            )
    else:
        if args.config or args.cluster_login:
            raise ValueError(
                "Tracked logs use the SSH settings saved in the run; omit --config and --cluster-login."
            )
        tracking_path = resolve_tracking_file(args.run)
        if tracking_path is None:
            raise ValueError("No tracking file found for the requested run.")
        tracking = load_tracking_payload(tracking_path)
        tracking_file = str(tracking_path.resolve())
        context = {
            "cluster_login": tracking.cluster_login,
            "archive_dir": tracking.remote_slurm_dashboard_log_archive_dir,
            "ssh_config_file": tracking.ssh_config_file,
            "ssh_options": tracking.ssh_options,
            "config_path": None,
        }
        jobs = [
            job for job in tracking.jobs if not args.job or job.job_name == args.job
        ]
        if not jobs:
            raise ValueError("No tracked jobs match the selection.")
        if args.path is not None:
            relative = PurePosixPath(args.path)
            if relative.is_absolute() or ".." in relative.parts or str(relative) == ".":
                raise ValueError(
                    "--path must be a contained relative application log path."
                )
            if not tracking.remote_workdir:
                raise ValueError("The tracked run has no remote workdir.")
            files.append(
                {
                    "job_id": jobs[0].job_id if args.job else None,
                    "job_name": args.job,
                    "stream": "application",
                    "source": "tracking:workdir",
                    "path": str(PurePosixPath(tracking.remote_workdir) / relative),
                    "root": tracking.remote_workdir,
                }
            )
        else:
            for job in jobs:
                saved_paths = {
                    stream: resolve_log_path(getattr(job, stream), job.job_id)
                    for stream in streams
                }
                info = None
                if any(path is None for path in saved_paths.values()):
                    info = resolve_job_log_info(
                        context["cluster_login"], job.job_id,
                        archive_dir=context["archive_dir"],
                        ssh_config_file=context["ssh_config_file"],
                        ssh_options=context["ssh_options"],
                    )
                for stream in streams:
                    saved_path = saved_paths[stream]
                    path = saved_path if saved_path is not None else getattr(info, stream)
                    source = (
                        "tracking" if saved_path is not None
                        else getattr(info, stream + "_source") or "unresolved"
                    )
                    files.append(
                        {
                            "job_id": job.job_id,
                            "job_name": job.job_name,
                            "stream": stream,
                            "path": path,
                            "source": source,
                            "resolution_errors": list(info.probe_errors) if info else [],
                        }
                    )
    state = {
        "version": 1,
        "context": context,
        "tracking_file": tracking_file,
        "options": options,
        "files": files,
    }
    # Initial and resumed requests obey the same bounded, typed contract.
    return _decode_cursor(_encode_cursor(state))


def _read_remote(state: dict[str, Any]) -> dict[str, Any]:
    context = state["context"]
    source = Path(__file__).with_name("_remote_logs.py").read_text(encoding="utf-8")
    command = [
        *build_ssh_command(
            context["cluster_login"],
            ssh_config_file=context["ssh_config_file"],
            ssh_options=context["ssh_options"],
        ),
        "python3 -c " + shlex.quote(source),
    ]
    result = subprocess.run(
        command,
        input=json.dumps({"options": state["options"], "files": state["files"]}),
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"Remote log read failed (exit {result.returncode}): {result.stderr.strip()[:4096]}"
        )
    try:
        response = json.loads(result.stdout)
        if len(response["files"]) != len(state["files"]) or len(
            response["positions"]
        ) != len(state["files"]):
            raise ValueError("Incorrect remote response file count.")
        for spec, position in zip(state["files"], response["positions"]):
            spec["position"] = position
        return {
            "ok": all(file["status"] in ("ok", "empty") for file in response["files"]),
            "cluster_login": context["cluster_login"],
            "tracking_file": state["tracking_file"],
            "files": response["files"],
            "cursor": _encode_cursor(state),
            "limits": state["options"],
        }
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid remote log response: {exc}") from exc


def _emit(payload: dict[str, Any], json_output: bool, path_only: bool) -> None:
    if json_output:
        print(json.dumps(payload, ensure_ascii=True))
        return
    for file in payload["files"]:
        if path_only:
            print(file["path"] or "(unresolved)")
        else:
            print(
                f"== {file['job_id'] or '-'} {file['stream']}: {file['path'] or '(unresolved)'} [{file['status']}] =="
            )
            sys.stdout.write(file["content"])
            if file["content"] and not file["content"].endswith("\n"):
                sys.stdout.write("\n")
            print(f"Source: {file['source'] or 'unresolved'}")
        for error in file.get("resolution_errors") or []:
            print(f"Resolution: {error}", file=sys.stderr)
        if file["error"]:
            print(file["error"], file=sys.stderr)
        if file["reset"]:
            print("Log replaced or truncated; navigation restarted.", file=sys.stderr)
        if file["search_complete"] is False:
            print("Search incomplete; resume with the cursor.", file=sys.stderr)
        if file["context_limited"]:
            print("Search context limited by the scan boundary.", file=sys.stderr)
    if any(
        file["has_more"] or file["search_complete"] is False
        for file in payload["files"]
    ):
        print(
            "More content is available; use --json for a continuation cursor.",
            file=sys.stderr,
        )


def run_logs(args: argparse.Namespace, *, cluster_context=None) -> int:
    try:
        if args.follow and (args.json or args.path_only or not sys.stdout.isatty()):
            raise ValueError(
                "--follow requires an interactive terminal and cannot be combined with --json or --path-only."
            )
        if args.cursor:
            if any(
                (
                    args.run,
                    args.job_id,
                    args.job,
                    args.path,
                    args.config,
                    args.cluster_login,
                    args.search is not None,
                    args.stream != "both",
                    args.lines != 100,
                    args.max_bytes != 65536,
                    args.context != 20,
                    args.path_only,
                )
            ):
                raise ValueError(
                    "--cursor resumes its saved selection and limits; omit selection and content options."
                )
            state = _decode_cursor(args.cursor)
        else:
            state = _initial_state(args, cluster_context)
        if args.follow and state["options"]["path_only"]:
            raise ValueError("--follow cannot resume a path-only cursor.")
        while True:
            payload = _read_remote(state)
            _emit(payload, args.json, state["options"]["path_only"])
            if not args.follow or not payload["ok"]:
                return 0 if payload["ok"] else 1
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, TrackingError) as exc:
        if args.json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "cluster_login": None,
                        "tracking_file": None,
                        "files": [],
                        "cursor": None,
                        "limits": {},
                        "error": str(exc),
                    }
                )
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
