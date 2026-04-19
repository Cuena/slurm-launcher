from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from launcher.init_wizard import (
    InitAnswers,
    _apply_answers_to_template,
    _infer_project_name,
    _infer_project_name_from_pyproject,
    _normalize_project_name,
    _replace_assignment,
    init_config,
)


TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "launcher"
    / "templates"
    / "config.py.template"
)


class NormalizeProjectNameTests(unittest.TestCase):
    def test_spaces_replaced(self) -> None:
        self.assertEqual(_normalize_project_name("my project"), "my_project")

    def test_special_chars_stripped(self) -> None:
        self.assertEqual(_normalize_project_name("my@project!"), "my_project_")

    def test_empty_falls_back(self) -> None:
        self.assertEqual(_normalize_project_name(""), "project")


class InferProjectNameTests(unittest.TestCase):
    def test_infers_from_directory_name(self) -> None:
        with tempfile.TemporaryDirectory(suffix="_test-proj") as tmpdir:
            name = _infer_project_name(Path(tmpdir))
        self.assertIn("test-proj", name)

    def test_infers_from_pyproject_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text(
                '[project]\nname = "my-cool-project"\nversion = "1.0"\n',
                encoding="utf-8",
            )
            name = _infer_project_name(Path(tmpdir))
        self.assertEqual(name, "my-cool-project")

    def test_pyproject_without_name_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pyproject = Path(tmpdir) / "pyproject.toml"
            pyproject.write_text("[build-system]\n", encoding="utf-8")
            name = _infer_project_name(Path(tmpdir))
        self.assertTrue(len(name) > 0)


class InferProjectNameFromPyprojectTests(unittest.TestCase):
    def test_parses_name(self) -> None:
        self.assertEqual(
            _infer_project_name_from_pyproject('[project]\nname = "foo"\n'),
            "foo",
        )

    def test_returns_none_without_project_section(self) -> None:
        self.assertIsNone(
            _infer_project_name_from_pyproject('[build-system]\nrequires = ["uv"]\n')
        )


class ReplaceAssignmentTests(unittest.TestCase):
    def test_replaces_simple_assignment(self) -> None:
        source = 'PROJECT_NAME = "old"\n'
        result = _replace_assignment(source, "PROJECT_NAME", '"new"')
        self.assertIn('PROJECT_NAME = "new"', result)

    def test_preserves_comment(self) -> None:
        source = 'RUNTIME_MODE = "native"  # native | venv | singularity\n'
        result = _replace_assignment(source, "RUNTIME_MODE", '"venv"')
        self.assertIn('"venv"', result)
        self.assertIn("# native | venv | singularity", result)

    def test_raises_on_missing_assignment(self) -> None:
        with self.assertRaises(RuntimeError):
            _replace_assignment("FOO = 1\n", "BAR", "2")


class ApplyAnswersToTemplateTests(unittest.TestCase):
    def test_applies_answers_to_real_template(self) -> None:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
        answers = InitAnswers(
            project_name="test_proj",
            cluster_login="user@cluster.example",
            workspace_mode="per-run",
            remote_workspace_base="/scratch/work",
            remote_workspace_dir=None,
            remote_log_base_path="/scratch/logs",
            runtime_mode="native",
            venv_python_executable=None,
            singularity_image_path=None,
            singularity_exec_flags=[],
            mn5_account="test_acct",
        )
        result = _apply_answers_to_template(template, answers)

        self.assertIn("'test_proj'", result)
        self.assertIn("'user@cluster.example'", result)
        self.assertIn("'per-run'", result)
        self.assertIn("'/scratch/work'", result)
        self.assertIn("'/scratch/logs'", result)
        self.assertIn("'test_acct'", result)


class InitConfigTests(unittest.TestCase):
    def test_non_interactive_creates_config_from_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            dest = cwd / ".slurm" / "remote_launcher_config.mn5.py"

            created_path, answers = init_config(
                cwd=cwd,
                template_path=TEMPLATE_PATH,
                dest_path=dest,
                force=False,
                interactive=False,
            )

            self.assertEqual(created_path, dest)
            self.assertIsNone(answers)
            self.assertTrue(dest.exists())
            content = dest.read_text(encoding="utf-8")
            self.assertIn("CLUSTER_LOGIN", content)
            self.assertIn("JOBS", content)

    def test_raises_on_existing_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            dest = cwd / ".slurm" / "remote_launcher_config.mn5.py"
            dest.parent.mkdir(parents=True)
            dest.write_text("existing", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                init_config(
                    cwd=cwd,
                    template_path=TEMPLATE_PATH,
                    dest_path=dest,
                    force=False,
                    interactive=False,
                )

    def test_force_overwrites_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            dest = cwd / ".slurm" / "remote_launcher_config.mn5.py"
            dest.parent.mkdir(parents=True)
            dest.write_text("old content", encoding="utf-8")

            created_path, _ = init_config(
                cwd=cwd,
                template_path=TEMPLATE_PATH,
                dest_path=dest,
                force=True,
                interactive=False,
            )

            self.assertEqual(created_path, dest)
            self.assertNotEqual(dest.read_text(encoding="utf-8"), "old content")

    def test_creates_gitignore_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            dest = cwd / ".slurm" / "remote_launcher_config.mn5.py"

            init_config(
                cwd=cwd,
                template_path=TEMPLATE_PATH,
                dest_path=dest,
                force=False,
                interactive=False,
            )

            gitignore = (cwd / ".gitignore").read_text(encoding="utf-8")
            self.assertIn(".slurm/*.py", gitignore)
            self.assertIn("!.slurm/*.example.py", gitignore)

    def test_raises_on_missing_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            dest = cwd / ".slurm" / "remote_launcher_config.mn5.py"

            with self.assertRaises(FileNotFoundError):
                init_config(
                    cwd=cwd,
                    template_path=Path("/nonexistent/template.py"),
                    dest_path=dest,
                    force=False,
                    interactive=False,
                )


if __name__ == "__main__":
    unittest.main()
