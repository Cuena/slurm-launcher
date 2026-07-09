"""Tests for named model package validators."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from launcher.validators import validate_rfdetr, validate_sam3


class TestRFDetrValidator(TestCase):
    def test_missing_everything(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            issues = validate_rfdetr(tmp)
        self.assertEqual(len(issues), 3)

    def test_valid_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "class_names.txt").write_text("foo")
            (root / "inference_config.json").write_text("{}")
            (root / "model.onnx").write_text("model")
            issues = validate_rfdetr(tmp)
        self.assertEqual(issues, [])


class TestSam3Validator(TestCase):
    def test_missing_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.yaml").write_text("a: b")
            issues = validate_sam3(tmp)
        self.assertEqual(len(issues), 1)

    def test_valid_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.yaml").write_text("a: b")
            (root / "model.pt").write_text("model")
            issues = validate_sam3(tmp)
        self.assertEqual(issues, [])
