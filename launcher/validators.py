"""Named model package validators for common ML artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationIssue:
    """One validation failure."""

    validator: str
    path: str
    message: str


def _exists(base: Path, relative: str) -> bool:
    return (base / relative).exists()


def _any_exists(base: Path, candidates: list[str]) -> bool:
    return any(_exists(base, candidate) for candidate in candidates)


def validate_rfdetr(base_path: str) -> list[ValidationIssue]:
    """Validate an RF-DETR package shape.

    Expected files:
      - class_names.txt
      - inference_config.json
      - model.onnx or equivalent model file
    """
    base = Path(base_path)
    issues: list[ValidationIssue] = []
    if not base.exists():
        return [
            ValidationIssue("rfdetr", base_path, "RF-DETR package path does not exist")
        ]
    if not _exists(base, "class_names.txt"):
        issues.append(
            ValidationIssue(
                "rfdetr",
                str(base / "class_names.txt"),
                "Missing class_names.txt",
            )
        )
    if not _exists(base, "inference_config.json"):
        issues.append(
            ValidationIssue(
                "rfdetr",
                str(base / "inference_config.json"),
                "Missing inference_config.json",
            )
        )
    if not _any_exists(base, ["model.onnx", "model.pt", "model.pth", "model.ckpt"]):
        issues.append(
            ValidationIssue(
                "rfdetr",
                base_path,
                "Missing model file (expected model.onnx, model.pt, model.pth, or model.ckpt)",
            )
        )
    return issues


def validate_sam3(base_path: str) -> list[ValidationIssue]:
    """Validate a SAM3 checkpoint/package shape.

    Expected files:
      - config files (yaml or json)
      - checkpoint/model files (pt, pth, ckpt, safetensors)
    """
    base = Path(base_path)
    issues: list[ValidationIssue] = []
    if not base.exists():
        return [
            ValidationIssue("sam3", base_path, "SAM3 package path does not exist")
        ]

    config_exts = {".yaml", ".yml", ".json"}
    has_config = any(
        path.suffix.lower() in config_exts for path in base.iterdir() if path.is_file()
    )
    if not has_config:
        issues.append(
            ValidationIssue(
                "sam3",
                base_path,
                "Missing config file (expected .yaml, .yml, or .json)",
            )
        )

    model_exts = {".pt", ".pth", ".ckpt", ".safetensors"}
    has_model = any(
        path.suffix.lower() in model_exts for path in base.rglob("*") if path.is_file()
    )
    if not has_model:
        issues.append(
            ValidationIssue(
                "sam3",
                base_path,
                "Missing checkpoint/model file (expected .pt, .pth, .ckpt, or .safetensors)",
            )
        )
    return issues


VALIDATORS: dict[str, callable[[str], list[ValidationIssue]]] = {
    "rfdetr": validate_rfdetr,
    "sam3": validate_sam3,
}


def run_validator(name: str, base_path: str) -> list[ValidationIssue]:
    """Run a named validator if it exists."""
    normalized = name.strip().lower()
    validator = VALIDATORS.get(normalized)
    if validator is None:
        return [
            ValidationIssue(
                normalized,
                base_path,
                f"Unknown validator '{name}'. Known: {', '.join(sorted(VALIDATORS))}",
            )
        ]
    return validator(base_path)
