#!/usr/bin/env python3
"""Run APB prediction, judge, and scoring for any split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from apb_eval.runner import run_cli


if __name__ == "__main__":
    run_cli("all")
