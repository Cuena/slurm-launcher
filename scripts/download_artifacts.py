# scripts/download_artifacts.py
# What: Downloads tracked remote artifact paths for one launcher run from the repository root.
# Why: Preserves script-style compatibility for users who prefer invoking utilities directly.
# RELEVANT FILES: launcher/download_artifacts.py, launcher/cli.py, launcher/core.py, README.md

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launcher.download_artifacts import main


if __name__ == "__main__":
    raise SystemExit(main())
