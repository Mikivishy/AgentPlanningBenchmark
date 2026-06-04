from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any


SPLITS = [
    "holistic_planning",
    "holistic_tool_extraneous",
    "holistic_unsolvable",
    "step_wise_planning",
    "step_wise_tool_extraneous",
    "step_wise_tool_broken",
]

HOLISTIC_SPLITS = [
    "holistic_planning",
    "holistic_tool_extraneous",
    "holistic_unsolvable",
]

STEP_WISE_SPLITS = [
    "step_wise_planning",
    "step_wise_tool_extraneous",
    "step_wise_tool_broken",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_json_field(record: dict[str, Any], key: str, default: Any) -> Any:
    value = record.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target = target_dir.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise ValueError(f"Unsafe tar member in {archive_path}: {member.name}")
        archive.extractall(target_dir)


def ensure_assets(data_dir: Path) -> None:
    for name in ("Holistic", "Step-wise"):
        target = data_dir / "assets" / name
        if target.exists():
            continue
        archive = data_dir / "assets_archives" / f"{name}_assets.tar"
        if not archive.exists():
            raise FileNotFoundError(f"Missing {target} and {archive}")
        print(f"extracting {archive}")
        safe_extract_tar(archive, data_dir)


def resolve_splits(value: str) -> list[str]:
    if value == "all":
        return SPLITS
    if value in {"holistic", "offline"}:
        return HOLISTIC_SPLITS
    if value in {"step_wise", "online"}:
        return STEP_WISE_SPLITS
    splits = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise SystemExit(f"Unknown split(s): {unknown}. Valid: {SPLITS}")
    return splits


def load_records(data_dir: Path, splits: list[str], max_samples: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        path = data_dir / "data" / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(read_jsonl(path))
    if max_samples is not None:
        records = records[:max_samples]
    return records

