from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from launcher.init_wizard import (
    _infer_project_name,
    _infer_project_name_from_pyproject,
    _normalize_project_name,
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



class InitConfigTests(unittest.TestCase):

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
