#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apb_eval.runner import run_cli

if __name__ == "__main__":
    run_cli("holistic_planning", "judge_score")
