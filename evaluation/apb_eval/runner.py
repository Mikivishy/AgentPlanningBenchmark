from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI

from .io import append_jsonl, ensure_assets, load_records, parse_json_field, read_jsonl, resolve_splits, write_json
from .prompts import PREDICT_SYSTEM, dumps, judge_payload, judge_system_prompt, prediction_payload


DEFAULT_DATA_DIR = Path("data/AgentPlanningbBenchmark")
DEFAULT_JUDGE_MODEL = "gemini-3-pro-preview"


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else {"value": value}


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


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


def prediction_messages(record: dict[str, Any], data_dir: Path, max_text_chars: int) -> list[dict[str, Any]]:
    image_parts, file_text = file_context_parts(data_dir, record.get("files", []), max_text_chars)
    payload = prediction_payload(record)
    if file_text:
        payload["file_context"] = file_text
    return [
        {"role": "system", "content": PREDICT_SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": dumps(payload)}, *image_parts]},
    ]


def completed_prediction_keys(path: Path, model_name: str) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(record.get("id"))
        for record in read_jsonl(path)
        if record.get("tested_model") == model_name and "model_response" in record
    }


def completed_judgement_keys(path: Path, judge_model: str) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {
        (str(record.get("id")), str(record.get("tested_model", "")))
        for record in read_jsonl(path)
        if record.get("judge_model") == judge_model and "judgement" in record
    }


def make_client(api_key: str, base_url: str | None) -> OpenAI:
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def predict(args: argparse.Namespace, splits: list[str]) -> None:
    ensure_assets(args.data_dir)
    client = make_client(args.test_api_key, args.test_base_url)
    records = load_records(args.data_dir, splits, args.max_samples)
    done = completed_prediction_keys(args.predictions, args.test_model)
    for index, record in enumerate(records, start=1):
        if str(record["id"]) in done and args.resume:
            continue
        request: dict[str, Any] = {
            "model": args.test_model,
            "messages": prediction_messages(record, args.data_dir, args.max_text_chars),
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.response_format:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        output = dict(record)
        output["tested_model"] = args.test_model
        output["model_response"] = response.choices[0].message.content or ""
        append_jsonl(args.predictions, [output])
        print(f"predicted {index}/{len(records)} {record['id']}")


def judge(args: argparse.Namespace, splits: list[str]) -> None:
    client = make_client(args.judge_api_key, args.judge_base_url)
    records = [record for record in read_jsonl(args.predictions) if record.get("split") in splits]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    done = completed_judgement_keys(args.judgements, args.judge_model)
    for index, record in enumerate(records, start=1):
        key = (str(record["id"]), str(record.get("tested_model", "")))
        if key in done and args.resume:
            continue
        request: dict[str, Any] = {
            "model": args.judge_model,
            "messages": [
                {"role": "system", "content": judge_system_prompt(record)},
                {"role": "user", "content": dumps(judge_payload(record))},
            ],
            "temperature": 0.0,
            "max_tokens": args.judge_max_tokens,
        }
        if args.response_format:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        judged = dict(record)
        judged["judge_model"] = args.judge_model
        judged["judgement"] = parse_json_object(response.choices[0].message.content or "{}")
        append_jsonl(args.judgements, [judged])
        print(f"judged {index}/{len(records)} {record['id']}")


def score(args: argparse.Namespace, splits: list[str]) -> dict[str, Any]:
    records = [record for record in read_jsonl(args.judgements) if record.get("split") in splits]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        keys = [
            "overall",
            record.get("split", ""),
            record.get("planning_regime", ""),
            record.get("task_family", ""),
            record.get("task_category", ""),
        ]
        for key in keys:
            if key:
                groups.setdefault(str(key), []).append(record)
    summary: dict[str, Any] = {"total_records": len(records), "groups": {}}
    for group, group_records in groups.items():
        correctness = []
        grades = []
        for record in group_records:
            judgement = record.get("judgement", {})
            if not isinstance(judgement, dict):
                continue
            if "is_correct" in judgement:
                correctness.append(bool(judgement["is_correct"]))
            if isinstance(judgement.get("grade"), (int, float)):
                grades.append(float(judgement["grade"]))
        summary["groups"][group] = {
            "records": len(group_records),
            "accuracy": mean(correctness) if correctness else None,
            "average_grade": mean(grades) if grades else None,
        }
    write_json(args.scores, summary)
    return summary


def add_common_args(parser: argparse.ArgumentParser, default_splits: str = "all", default_stage: str = "all") -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--splits", default=default_splits)
    parser.add_argument("--stage", choices=["predict", "judge", "score", "judge_score", "all"], default=default_stage)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions.jsonl"))
    parser.add_argument("--judgements", type=Path, default=Path("outputs/judgements.jsonl"))
    parser.add_argument("--scores", type=Path, default=Path("outputs/scores.json"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--no-response-format", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--test-model", default=os.environ.get("APB_TEST_MODEL"))
    parser.add_argument("--test-base-url", default=os.environ.get("APB_TEST_BASE_URL"))
    parser.add_argument("--test-api-key", default=os.environ.get("APB_TEST_API_KEY"))
    parser.add_argument("--judge-model", default=os.environ.get("APB_JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--judge-base-url", default=os.environ.get("APB_JUDGE_BASE_URL"))
    parser.add_argument("--judge-api-key", default=os.environ.get("APB_JUDGE_API_KEY"))


def require(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"Missing {name}")
    return value


def run_cli(default_splits: str = "all", default_stage: str = "all") -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_splits, default_stage)
    args = parser.parse_args()
    args.response_format = not args.no_response_format
    args.resume = not args.no_resume
    splits = resolve_splits(args.splits)
    if args.stage in {"predict", "all"}:
        args.test_model = require(args.test_model, "--test-model or APB_TEST_MODEL")
        args.test_api_key = require(args.test_api_key, "--test-api-key or APB_TEST_API_KEY")
        predict(args, splits)
    if args.stage in {"judge", "judge_score", "all"}:
        args.judge_api_key = require(args.judge_api_key, "--judge-api-key or APB_JUDGE_API_KEY")
        judge(args, splits)
    if args.stage in {"score", "judge_score", "all"}:
        summary = score(args, splits)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
