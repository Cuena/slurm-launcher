# launcher/init_wizard.py
# What: Interactive initializer that writes a repo-local launcher config with inferred path defaults.
# Why: Makes adopting slurm-launcher in new repos quick while keeping config private and gitignored.
# RELEVANT FILES: launcher/cli.py, launcher/templates/config.py.template, README.md, scripts/init_wrapper_repo.sh

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InitAnswers:
    project_name: str
    cluster_login: str
    code_source_mode: str
    remote_base_path: str | None
    remote_code_dir: str | None
    remote_log_base_path: str
    runtime_mode: str
    venv_python_executable: str | None
    singularity_image_path: str | None
    singularity_exec_flags: list[str]
    mn5_account: str


def _prompt(text: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{text}{suffix}: ").strip()
    if not value and default is not None:
        return default
    return value


def _prompt_choice(
    text: str,
    *,
    choices: list[str],
    default: str,
) -> str:
    choice_set = {c.lower() for c in choices}
    while True:
        value = _prompt(text, default=default).strip().lower()
        if value in choice_set:
            return value
        print(f"Please enter one of: {', '.join(choices)}", file=sys.stderr)


def _normalize_optional(value: str) -> str | None:
    value = value.strip()
    return value or None


def _infer_project_name(cwd: Path) -> str:
    pyproject = cwd / "pyproject.toml"
    if pyproject.exists():
        name = _infer_project_name_from_pyproject(pyproject.read_text(encoding="utf-8"))
        if name:
            return _normalize_project_name(name)
    return _normalize_project_name(cwd.name or "project")


def _infer_project_name_from_pyproject(content: str) -> str | None:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        try:
            data = tomllib.loads(content)
        except Exception:
            data = {}
        project = data.get("project", {})
        if isinstance(project, dict):
            name = project.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return None

    in_project = False
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if not in_project:
            continue
        match = re.match(r'^name\s*=\s*"([^"]+)"\s*$', line)
        if match:
            return match.group(1).strip()
    return None


def _normalize_project_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name or "project"


def _ensure_gitignore_line(repo_root: Path, pattern: str) -> None:
    gitignore_path = repo_root / ".gitignore"
    existing = (
        gitignore_path.read_text(encoding="utf-8").splitlines()
        if gitignore_path.exists()
        else []
    )
    if pattern in existing:
        return
    existing.append(pattern)
    gitignore_path.write_text("\n".join(existing).rstrip() + "\n", encoding="utf-8")


def _replace_assignment(source: str, name: str, value: str) -> str:
    pattern = re.compile(
        rf"^(?P<indent>\s*){re.escape(name)}(?P<ann>\s*:\s*[^=]+)?\s*=\s*(?P<rhs>.*?)(?P<comment>\s+#.*)?$",
        re.MULTILINE,
    )

    def repl(match: re.Match[str]) -> str:
        indent = match.group("indent") or ""
        ann = (match.group("ann") or "").rstrip()
        comment = match.group("comment") or ""
        return f"{indent}{name}{ann} = {value}{comment}"

    updated, count = pattern.subn(repl, source, count=1)
    if count != 1:
        raise RuntimeError(f"Unable to update assignment for {name}.")
    return updated


def _apply_answers_to_template(template: str, answers: InitAnswers) -> str:
    updated = template
    updated = _replace_assignment(updated, "PROJECT_NAME", repr(answers.project_name))
    updated = _replace_assignment(updated, "CLUSTER_LOGIN", repr(answers.cluster_login))
    updated = _replace_assignment(
        updated, "CODE_SOURCE_MODE", repr(answers.code_source_mode)
    )
    updated = _replace_assignment(
        updated,
        "REMOTE_BASE_PATH",
        repr(answers.remote_base_path or "/absolute/path/on/cluster/my_project"),
    )
    updated = _replace_assignment(
        updated,
        "REMOTE_CODE_DIR",
        "None" if answers.remote_code_dir is None else repr(answers.remote_code_dir),
    )
    updated = _replace_assignment(
        updated, "REMOTE_LOG_BASE_PATH", repr(answers.remote_log_base_path)
    )
    updated = _replace_assignment(updated, "RUNTIME_MODE", repr(answers.runtime_mode))
    updated = _replace_assignment(
        updated,
        "VENV_PYTHON_EXECUTABLE",
        "None"
        if answers.venv_python_executable is None
        else repr(answers.venv_python_executable),
    )
    updated = _replace_assignment(
        updated,
        "SINGULARITY_IMAGE_PATH",
        "None"
        if answers.singularity_image_path is None
        else repr(answers.singularity_image_path),
    )
    updated = _replace_assignment(
        updated,
        "SINGULARITY_EXEC_FLAGS",
        repr(answers.singularity_exec_flags),
    )
    updated = _replace_assignment(updated, "MN5_ACCOUNT", repr(answers.mn5_account))
    return updated


def run_init_wizard(*, cwd: Path, template_path: Path) -> tuple[InitAnswers, str]:
    inferred_project_name = _infer_project_name(cwd)
    project_name = _prompt("Project name", default=inferred_project_name)
    project_name = _normalize_project_name(project_name)

    cluster_login = _prompt("Cluster login (user@host)", default="your_user@mn5.bsc.es")
    code_source_mode = _prompt_choice(
        "Code source mode (sync|remote)",
        choices=["sync", "remote"],
        default="sync",
    )

    runtime_mode = _prompt_choice(
        "Runtime mode (native|venv|singularity)",
        choices=["native", "venv", "singularity"],
        default="native",
    )

    remote_home_dir = _prompt(
        "Remote home dir (absolute; used only for suggestions)",
        default="/home/bsc/<bsc_user>",
    )
    scratch_pipelines_dir = _prompt(
        "Scratch pipelines root (absolute; used only for suggestions)",
        default="/gpfs/scratch/<group>/users/<scratch_user>/pipelines",
    )

    base = f"{scratch_pipelines_dir.rstrip('/')}/{project_name}"
    suggested_log_base = f"{base}/logs"

    if code_source_mode == "sync":
        suggested_remote_base = f"{base}/work"
        remote_base_path = _prompt("REMOTE_BASE_PATH", default=suggested_remote_base)
        remote_code_dir = None
    else:
        suggested_code_dir = f"{remote_home_dir.rstrip('/')}/code/{project_name}"
        remote_code_dir = _prompt("REMOTE_CODE_DIR", default=suggested_code_dir)
        remote_base_path = None

    remote_log_base_path = _prompt("REMOTE_LOG_BASE_PATH", default=suggested_log_base)

    venv_python_executable: str | None = None
    singularity_image_path: str | None = None
    singularity_exec_flags: list[str] = []
    if runtime_mode == "venv":
        while venv_python_executable is None:
            venv_python_executable = _normalize_optional(
                _prompt("VENV_PYTHON_EXECUTABLE (absolute)", default=None)
            )
    elif runtime_mode == "singularity":
        while singularity_image_path is None:
            singularity_image_path = _normalize_optional(
                _prompt("SINGULARITY_IMAGE_PATH (absolute)", default=None)
            )
        raw_flags = _prompt(
            "SINGULARITY_EXEC_FLAGS (optional; shell-style, e.g. --nv --bind /a:/b)",
            default="",
        )
        if raw_flags.strip():
            singularity_exec_flags = shlex.split(raw_flags)

    mn5_account = _prompt("MN5_ACCOUNT (slurm account)", default="your_mn5_account")

    answers = InitAnswers(
        project_name=project_name,
        cluster_login=cluster_login,
        code_source_mode=code_source_mode,
        remote_base_path=remote_base_path,
        remote_code_dir=remote_code_dir,
        remote_log_base_path=remote_log_base_path,
        runtime_mode=runtime_mode,
        venv_python_executable=venv_python_executable,
        singularity_image_path=singularity_image_path,
        singularity_exec_flags=singularity_exec_flags,
        mn5_account=mn5_account,
    )
    template = template_path.read_text(encoding="utf-8")
    return answers, _apply_answers_to_template(template, answers)


def init_config(
    *,
    cwd: Path,
    template_path: Path,
    dest_path: Path,
    force: bool,
    interactive: bool,
) -> tuple[Path, InitAnswers | None]:
    if dest_path.exists() and not force:
        raise FileExistsError(dest_path)
    if not template_path.exists():
        raise FileNotFoundError(template_path)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    answers: InitAnswers | None = None
    if interactive:
        answers, rendered = run_init_wizard(cwd=cwd, template_path=template_path)
        dest_path.write_text(rendered.rstrip() + "\n", encoding="utf-8")
    else:
        dest_path.write_text(
            template_path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8"
        )

    if dest_path.parent.name == ".slurm":
        _ensure_gitignore_line(cwd, ".slurm/*.py")
        _ensure_gitignore_line(cwd, "!.slurm/*.example.py")
    else:
        _ensure_gitignore_line(cwd, dest_path.name)
    return dest_path, answers
