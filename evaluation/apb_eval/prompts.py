from __future__ import annotations

import json
from typing import Any

from .io import parse_json_field


PREDICT_SYSTEM = """You are an AI agent planning benchmark participant.

Return only valid JSON. Do not include markdown.
Use only the available tools from the prompt. Do not execute tools; produce the
plan or next step that should be executed by an agent.
"""


HOLISTIC_INSTRUCTION = """Generate a complete high-level plan and a linear tool chain.

Return:
{
  "plan": "concise plan",
  "tool_chain": [
    {"name": "tool_name", "parameter_description": {...}, "reason": "..."}
  ]
}
"""


HOLISTIC_TOOL_EXTRANEOUS_INSTRUCTION = """Generate a complete high-level plan and a linear tool chain.
Some available tools are intentionally irrelevant distractors. Avoid using tools
that are not necessary for solving the user task.

Return:
{
  "plan": "concise plan",
  "tool_chain": [
    {"name": "tool_name", "parameter_description": {...}, "reason": "..."}
  ]
}
"""


UNSOLVABLE_INSTRUCTION = """This task may be impossible under the provided context or tools.
If it is impossible, explicitly reject the task and explain the missing
information, missing capability, missing visual input, or constraint conflict.

Return either:
{"status": "REJECT", "reason": "why the task cannot be completed"}

or, only if truly solvable:
{"plan": "...", "tool_chain": [...]}
"""


STEP_WISE_INSTRUCTION = """Predict the next required step(s) after the provided trajectory prefix.

Return:
{
  "predicted_steps": [
    {"thought": "...", "action": "...", "tool": "optional tool name", "arguments": {}}
  ]
}
"""


STEP_WISE_TOOL_EXTRANEOUS_INSTRUCTION = """Predict the next required step after the provided trajectory prefix.
Some tools are intentionally irrelevant distractors. Avoid using any distractor
tool when it is not necessary.

Return:
{
  "predicted_steps": [
    {"thought": "...", "action": "...", "tool": "optional tool name", "arguments": {}}
  ]
}
"""


STEP_WISE_TOOL_BROKEN_INSTRUCTION = """Predict the next required step after the provided trajectory prefix.
The trajectory includes a broken or failed tool condition. Recover by choosing
the appropriate replacement/tool strategy when needed.

Return:
{
  "predicted_steps": [
    {"thought": "...", "action": "...", "tool": "optional tool name", "arguments": {}}
  ]
}
"""


HOLISTIC_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

Judge whether the model response can solve the user task using only the
available tools and respecting constraints. Use the reference plan/tool chain as
guidance, not exact string matching.

Error categories:
E1 goal misunderstanding, E2 incomplete task, E3 constraint violation,
E4 logical defect, E5 tool-use error, E6 hallucination.

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [E1,E2,E3,E4,E5,E6],
  "reasoning": "concise explanation"
}
"""


TOOL_EXTRANEOUS_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

The task includes intentionally irrelevant extra tools. Judge whether the model
solves the task and avoids using irrelevant tools. If it uses an extraneous
tool when a normal tool path is sufficient, mark E5 tool-use error.

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [E1,E2,E3,E4,E5,E6],
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
  "error_list": six integers [E1,E2,E3,E4,E5,E6],
  "reasoning": "concise explanation"
}
"""


STEP_WISE_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

Judge whether the predicted next step(s) are semantically equivalent to the
reference next step(s), given the task, trajectory prefix, and available tools.
Different wording is acceptable if the intended action and tool use are correct.

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [E1,E2,E3,E4,E5,E6],
  "reasoning": "concise explanation"
}
"""


TOOL_BROKEN_JUDGE_PROMPT = """You are an expert, impartial AI Agent Planning evaluator.

This is a Step-wise Tool-Broken case. The prediction should recognize or
recover from a broken tool condition and use the replacement/tool strategy when
required. Judge semantic correctness against the reference next step(s).

Return only JSON:
{
  "is_correct": boolean,
  "grade": one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "error_list": six integers [E1,E2,E3,E4,E5,E6],
  "reasoning": "concise explanation"
}
"""


def prediction_instruction(record: dict[str, Any]) -> str:
    split = record.get("split")
    if split == "holistic_planning":
        return HOLISTIC_INSTRUCTION
    if split == "holistic_tool_extraneous":
        return HOLISTIC_TOOL_EXTRANEOUS_INSTRUCTION
    if split == "holistic_unsolvable":
        return UNSOLVABLE_INSTRUCTION
    if split == "step_wise_planning":
        return STEP_WISE_INSTRUCTION
    if split == "step_wise_tool_extraneous":
        return STEP_WISE_TOOL_EXTRANEOUS_INSTRUCTION
    if split == "step_wise_tool_broken":
        return STEP_WISE_TOOL_BROKEN_INSTRUCTION
    return STEP_WISE_INSTRUCTION


def judge_system_prompt(record: dict[str, Any]) -> str:
    split = record.get("split")
    if split == "holistic_unsolvable":
        return UNSOLVABLE_JUDGE_PROMPT
    if split in {"holistic_tool_extraneous", "step_wise_tool_extraneous"}:
        return TOOL_EXTRANEOUS_JUDGE_PROMPT
    if split == "step_wise_tool_broken":
        return TOOL_BROKEN_JUDGE_PROMPT
    if record.get("planning_regime") == "Step-wise":
        return STEP_WISE_JUDGE_PROMPT
    return HOLISTIC_JUDGE_PROMPT


def prediction_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.get("id"),
        "task_category": record.get("task_category"),
        "source": record.get("source"),
        "source_subset": record.get("source_subset"),
        "query": record.get("query"),
        "instruction": prediction_instruction(record),
        "available_tools": parse_json_field(record, "tools_json", []),
        "tool_names": parse_json_field(record, "tool_names_json", []),
    }
    if record.get("planning_regime") == "Holistic":
        payload.update(
            {
                "unsolvable_type": record.get("unsolvable_type"),
                "extraneous_tool_names": parse_json_field(record, "extraneous_tool_names_json", []),
            }
        )
    else:
        payload.update(
            {
                "prediction_horizon": record.get("prediction_horizon"),
                "trajectory_prefix": parse_json_field(record, "trajectory_prefix_json", []),
                "extraneous_tool_names": parse_json_field(record, "extraneous_tool_names_json", []),
                "broken_tool_name": record.get("broken_tool_name"),
                "replacement_tool": parse_json_field(record, "replacement_tool_json", None),
            }
        )
    return payload


def judge_payload(record: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "id": record.get("id"),
        "task_category": record.get("task_category"),
        "query": record.get("query"),
        "available_tools": parse_json_field(record, "tools_json", []),
        "extraneous_tool_names": parse_json_field(record, "extraneous_tool_names_json", []),
        "trajectory_prefix": parse_json_field(record, "trajectory_prefix_json", []),
        "reference": reference,
        "model_response": record.get("model_response", ""),
    }


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)

