# scripts/download_logs.py
# What: Downloads tracked remote SLURM log files for one or many jobs from a launcher tracking file.
# Why: Provides a simple way to fetch logs locally after submission without manually copying remote paths.
# RELEVANT FILES: launcher/cli.py, launcher/core.py, README.md, scripts/init_wrapper_repo.sh

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launcher.download_logs import main


if __name__ == "__main__":
    raise SystemExit(main())
