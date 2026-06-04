#!/usr/bin/env python3
"""Run APB evaluation from a downloaded Hugging Face dataset folder."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import tarfile
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI


SPLITS = [
    "holistic_planning",
    "holistic_tool_extraneous",
    "holistic_unsolvable",
    "step_wise_planning",
    "step_wise_tool_extraneous",
    "step_wise_tool_broken",
]

DEFAULT_JUDGE_MODEL = "gemini-3-pro-preview"
DEFAULT_DATA_DIR = Path("data/AgentPlanningbBenchmark")

HOLISTIC_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

Judge whether the model response can solve the user task using only the
available tools and respecting constraints. Use the reference plan/tool chain as
guidance, not exact string matching.

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [goal, completeness, constraints, logic, tool_use, hallucination],
  "reasoning": "concise explanation"
}
"""

STEP_WISE_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

Judge whether the predicted next step(s) are semantically equivalent to the
reference next step(s), given the task, trajectory prefix, and available tools.
For Tool-Broken cases, also check whether the response recognizes or recovers
from the broken tool condition when that is required.

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [goal, completeness, constraints, logic, tool_use, hallucination],
  "reasoning": "concise explanation"
}
"""

UNSOLVABLE_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

This is an Unsolvable task. PASS only if the model refuses, asks for the missing
information/capability, or explicitly states why the task cannot be completed.
FAIL if it hallucinates a successful plan, invents unavailable facts, or calls
unavailable tools.

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [goal, completeness, constraints, logic, tool_use, hallucination],
  "reasoning": "concise explanation"
}
"""


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


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def load_records(data_dir: Path, splits: list[str], max_samples: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        path = data_dir / "data" / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        records.extend(read_jsonl(path))
    if max_samples is not None:
        records = records[:max_samples]
    return records


def processed_ids(path: Path, model_name: str) -> set[str]:
    if not path.exists():
        return set()
    seen: set[str] = set()
    for record in read_jsonl(path):
        if record.get("tested_model") == model_name:
            seen.add(str(record.get("id")))
    return seen


def judged_keys(path: Path, judge_model: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    seen: set[tuple[str, str]] = set()
    for record in read_jsonl(path):
        if record.get("judge_model") == judge_model and "judgement" in record:
            seen.add((str(record.get("id")), str(record.get("tested_model", ""))))
    return seen


def safe_extract_tar(archive_path: Path, target_dir: Path) -> None:
    target = target_dir.resolve()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            destination = (target_dir / member.name).resolve()
            if target not in destination.parents and destination != target:
                raise ValueError(f"Unsafe tar member in {archive_path}: {member.name}")
        archive.extractall(target_dir)


def ensure_assets(data_dir: Path, auto_extract: bool) -> None:
    required = {
        "Holistic": data_dir / "assets" / "Holistic",
        "Step-wise": data_dir / "assets" / "Step-wise",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if not missing:
        return
    if not auto_extract:
        raise FileNotFoundError(f"Missing asset directories: {missing}")
    for name in missing:
        archive = data_dir / "assets_archives" / f"{name}_assets.tar"
        if not archive.exists():
            raise FileNotFoundError(f"Missing {required[name]} and archive {archive}")
        print(f"extracting assets from {archive}")
        safe_extract_tar(archive, data_dir)


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }


def file_context_parts(data_dir: Path, files: Any, max_text_chars: int) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(files, list):
        return [], ""
    parts: list[dict[str, Any]] = []
    text_blocks: list[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        rel_path = str(item.get("path") or "")
        if not rel_path:
            continue
        full_path = data_dir / rel_path
        if not full_path.exists():
            text_blocks.append(f"[Missing referenced file: {rel_path}]")
            continue
        file_type = str(item.get("type") or "").lower()
        suffix = full_path.suffix.lower()
        if file_type == "image" or suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            parts.append(image_part(full_path))
        elif suffix in {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".xml", ".html"}:
            text = full_path.read_text(encoding="utf-8", errors="replace")[:max_text_chars]
            text_blocks.append(f"[File: {rel_path}]\n{text}")
        else:
            text_blocks.append(f"[Referenced file available at: {rel_path}]")
    return parts, "\n\n".join(text_blocks)


def build_prediction_prompt(record: dict[str, Any], data_dir: Path, max_text_chars: int) -> list[dict[str, Any]]:
    image_parts, file_text = file_context_parts(data_dir, record.get("files", []), max_text_chars)
    tools = parse_json_field(record, "tools_json", [])
    base = {
        "id": record.get("id"),
        "task": record.get("task_category"),
        "source": record.get("source"),
        "query": record.get("query"),
        "available_tools": tools,
        "file_context": file_text,
    }

    if record.get("planning_regime") == "Holistic":
        base["instruction"] = (
            "Return a JSON object with keys plan and tool_chain. "
            "For Unsolvable tasks, return is_solvable=false and explain the missing requirement."
        )
        base["unsolvable_type"] = record.get("unsolvable_type")
    else:
        base["instruction"] = "Return a JSON object with key predicted_steps for the next required step(s)."
        base["prediction_horizon"] = record.get("prediction_horizon")
        base["trajectory_prefix"] = parse_json_field(record, "trajectory_prefix_json", [])
        base["broken_tool_name"] = record.get("broken_tool_name")
        base["replacement_tool"] = parse_json_field(record, "replacement_tool_json", None)

    text_part = {
        "type": "text",
        "text": json.dumps(base, ensure_ascii=False, indent=2),
    }
    return [{"role": "user", "content": [text_part, *image_parts]}]


def predict(
    *,
    data_dir: Path,
    splits: list[str],
    output: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    max_samples: int | None,
    max_tokens: int,
    temperature: float,
    response_format: bool,
    max_text_chars: int,
) -> None:
    client = OpenAI(api_key=api_key, base_url=base_url)
    records = load_records(data_dir, splits, max_samples)
    skip = processed_ids(output, model_name)

    for idx, record in enumerate(records, start=1):
        if record["id"] in skip:
            continue
        request: dict[str, Any] = {
            "model": model_name,
            "messages": build_prediction_prompt(record, data_dir, max_text_chars),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        result = dict(record)
        result["tested_model"] = model_name
        result["model_response"] = response.choices[0].message.content or ""
        append_jsonl(output, [result])
        print(f"predicted {idx}/{len(records)} {record['id']}")


def build_judge_user_content(record: dict[str, Any]) -> str:
    if record.get("planning_regime") == "Holistic":
        reference = {
            "plan": record.get("plan"),
            "tool_chain": parse_json_field(record, "tool_chain_json", []),
            "unsolvable_type": record.get("unsolvable_type"),
        }
    else:
        reference = {
            "prediction_horizon": record.get("prediction_horizon"),
            "ground_truth_steps": parse_json_field(record, "ground_truth_steps_json", []),
            "reference_remaining_steps": parse_json_field(record, "reference_remaining_steps_json", []),
            "broken_tool_name": record.get("broken_tool_name"),
            "replacement_tool": parse_json_field(record, "replacement_tool_json", None),
        }

    payload = {
        "id": record.get("id"),
        "task": record.get("task_category"),
        "query": record.get("query"),
        "available_tools": parse_json_field(record, "tools_json", []),
        "trajectory_prefix": parse_json_field(record, "trajectory_prefix_json", []),
        "reference": reference,
        "model_response": record.get("model_response", ""),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def judge_prompt_for(record: dict[str, Any]) -> str:
    if record.get("task_family") == "Unsolvable":
        return UNSOLVABLE_JUDGE_PROMPT
    if record.get("planning_regime") == "Step-wise":
        return STEP_WISE_JUDGE_PROMPT
    return HOLISTIC_JUDGE_PROMPT


def judge(
    *,
    predictions: Path,
    output: Path,
    model_name: str,
    base_url: str,
    api_key: str,
    max_samples: int | None,
    max_tokens: int,
    response_format: bool,
) -> None:
    client = OpenAI(api_key=api_key, base_url=base_url)
    records = read_jsonl(predictions)
    if max_samples is not None:
        records = records[:max_samples]
    skip = judged_keys(output, model_name)

    for idx, record in enumerate(records, start=1):
        if (str(record["id"]), str(record.get("tested_model", ""))) in skip:
            continue
        request: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": judge_prompt_for(record)},
                {"role": "user", "content": build_judge_user_content(record)},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if response_format:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        judged = dict(record)
        judged["judge_model"] = model_name
        judged["judgement"] = parse_json_object(response.choices[0].message.content or "{}")
        append_jsonl(output, [judged])
        print(f"judged {idx}/{len(records)} {record['id']}")


def score(judgements: Path, output: Path) -> dict[str, Any]:
    records = read_jsonl(judgements)
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for key in ("overall", record.get("split", ""), record.get("planning_regime", ""), record.get("task_family", "")):
            if not key:
                continue
            groups.setdefault(str(key), []).append(record)

    summary: dict[str, Any] = {"total_records": len(records), "groups": {}}
    for group, group_records in groups.items():
        correctness = []
        grades = []
        for record in group_records:
            judgement = record.get("judgement", {})
            if isinstance(judgement, dict):
                if "is_correct" in judgement:
                    correctness.append(bool(judgement["is_correct"]))
                if isinstance(judgement.get("grade"), (int, float)):
                    grades.append(float(judgement["grade"]))
        summary["groups"][group] = {
            "records": len(group_records),
            "accuracy": mean(correctness) if correctness else None,
            "average_grade": mean(grades) if grades else None,
        }
    write_json(output, summary)
    return summary


def resolve_splits(value: str) -> list[str]:
    if value == "all":
        return SPLITS
    splits = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise SystemExit(f"Unknown split(s): {unknown}. Valid: {SPLITS}")
    return splits


def required_env(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"Missing {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", default="all", help="Comma-separated split stems or all")
    parser.add_argument("--stage", choices=["predict", "judge", "score", "all"], default="all")
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions.jsonl"))
    parser.add_argument("--judgements", type=Path, default=Path("outputs/judgements.jsonl"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores.json"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--no-auto-extract-assets", action="store_true")
    parser.add_argument("--test-model", default=os.environ.get("APB_TEST_MODEL"))
    parser.add_argument("--test-base-url", default=os.environ.get("APB_TEST_BASE_URL"))
    parser.add_argument("--test-api-key", default=os.environ.get("APB_TEST_API_KEY"))
    parser.add_argument("--judge-model", default=os.environ.get("APB_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--judge-base-url", default=os.environ.get("APB_JUDGE_BASE_URL"))
    parser.add_argument("--judge-api-key", default=os.environ.get("APB_JUDGE_API_KEY"))
    args = parser.parse_args()

    splits = resolve_splits(args.splits)
    use_response_format = not args.no_response_format

    if args.stage in {"predict", "all"}:
        ensure_assets(args.data_dir, auto_extract=not args.no_auto_extract_assets)
        predict(
            data_dir=args.data_dir,
            splits=splits,
            output=args.predictions,
            model_name=required_env(args.test_model, "--test-model or APB_TEST_MODEL"),
            base_url=required_env(args.test_base_url, "--test-base-url or APB_TEST_BASE_URL"),
            api_key=required_env(args.test_api_key, "--test-api-key or APB_TEST_API_KEY"),
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            response_format=use_response_format,
            max_text_chars=args.max_text_chars,
        )

    if args.stage in {"judge", "all"}:
        judge(
            predictions=args.predictions,
            output=args.judgements,
            model_name=args.judge_model,
            base_url=required_env(args.judge_base_url, "--judge-base-url or APB_JUDGE_BASE_URL"),
            api_key=required_env(args.judge_api_key, "--judge-api-key or APB_JUDGE_API_KEY"),
            max_samples=args.max_samples,
            max_tokens=args.judge_max_tokens,
            response_format=use_response_format,
        )

    if args.stage in {"score", "all"}:
        summary = score(args.judgements, args.scores)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
