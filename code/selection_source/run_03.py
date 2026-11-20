#!/usr/bin/env python
"""VSCode entry point for RP project step 03: leakage-aware feature selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RP project 03: ICC, variance, MI, LASSO and Boruta feature selection.")
    parser.add_argument("--stage", choices=("preflight", "select", "all"), default="all")
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.json")
    parser.add_argument("--modality", action="append", default=[], help="Optional modality name; may be supplied more than once.")
    return parser.parse_args()


def main() -> int:
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or newer is required.")
    from rp_selection.pipeline import run_pipeline

    args = parse_args()
    return run_pipeline(args.config.resolve(), args.stage, args.modality, PROJECT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
