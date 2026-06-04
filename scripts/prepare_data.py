#!/usr/bin/env python3
"""Download the APB Hugging Face dataset and restore asset directories."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "Mikivis/AgentPlanningbBenchmark"


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target = target_dir.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"Unsafe tar member in {archive_path}: {member.name}")
        archive.extractall(target_dir)


def restore_assets(data_dir: Path) -> None:
    for name in ("Holistic", "Step-wise"):
        target = data_dir / "assets" / name
        if target.exists():
            continue
        archive = data_dir / "assets_archives" / f"{name}_assets.tar"
        if not archive.exists():
            raise FileNotFoundError(f"Missing asset archive: {archive}")
        print(f"extracting {archive}")
        safe_extract_tar(archive, data_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("data/AgentPlanningbBenchmark"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--skip-assets", action="store_true")
    args = parser.parse_args()

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    if not args.skip_assets:
        restore_assets(args.output_dir)
    print(f"data ready: {args.output_dir}")


if __name__ == "__main__":
    main()
