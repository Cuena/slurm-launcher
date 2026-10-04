"""SSH command construction and captured remote script execution."""

from __future__ import annotations

import shlex
import subprocess


def build_ssh_transport_args(
    ssh_config_file: str | None,
    ssh_options: list[str] | None = None,
) -> list[str]:
    args: list[str] = []
    if ssh_config_file:
        args.extend(["-F", ssh_config_file])
    if ssh_options:
        args.extend(ssh_options)
    return args


def build_ssh_command(
    cluster_login: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> list[str]:
    return [
        "ssh",
        *build_ssh_transport_args(ssh_config_file, ssh_options),
        cluster_login,
    ]


def format_ssh_command(
    cluster_login: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
    remote_command: str | None = None,
) -> str:
    command = build_ssh_command(
        cluster_login,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
    )
    if remote_command is not None:
        command.append(remote_command)
    return shlex.join(command)


def format_ssh_script_command(
    cluster_login: str,
    script: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> str:
    command = format_ssh_command(
        cluster_login,
        ssh_config_file=ssh_config_file,
        ssh_options=ssh_options,
        remote_command="bash -s",
    )
    return "\n".join([f"{command} <<'EOF'", script.rstrip(), "EOF"])


def build_rsync_ssh_command(
    ssh_config_file: str | None,
    ssh_options: list[str] | None = None,
) -> str:
    return shlex.join(["ssh", *build_ssh_transport_args(ssh_config_file, ssh_options)])


def run_ssh_capture(
    cluster_login: str,
    script: str,
    *,
    ssh_config_file: str | None = None,
    ssh_options: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            *build_ssh_command(
                cluster_login,
                ssh_config_file=ssh_config_file,
                ssh_options=ssh_options,
            ),
            "bash",
            "-s",
        ],
        input=script.rstrip() + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
