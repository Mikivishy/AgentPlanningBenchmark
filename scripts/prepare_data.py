#!/usr/bin/env python3
"""Download the APB dataset from Hugging Face and restore asset directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

from apb_eval.io import ensure_assets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="Mikivis/AgentPlanningbBenchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("data/AgentPlanningbBenchmark"))
    parser.add_argument("--revision")
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.output_dir),
        local_dir_use_symlinks=False,
    )
    if not args.skip_assets:
        ensure_assets(args.output_dir)
    print(f"APB data prepared at {args.output_dir}")


if __name__ == "__main__":
    main()
