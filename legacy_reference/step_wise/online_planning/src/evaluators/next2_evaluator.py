"""
Next2 Strategy Evaluator.

This module contains the Next2 evaluator that predicts the next two steps.
"""

import os
import json
from json_repair import repair_json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

from .base_evaluator import BaseOnlinePlanningEvaluator
from ..models.data_models import PredictionResultNext2, EvalResultNext2
from ..utils.dataset_loader import DatasetLoader, get_base_dataset_type
from ..prompts.evaluation_prompts import EvaluationPrompts

from ..prompts.evaluation_prompts_loose import EvaluationPromptsLoose


class Next2Evaluator(BaseOnlinePlanningEvaluator):
    """Evaluator for next2 prediction strategy (predict next two steps)."""

    def __init__(self, *args, **kwargs):
        # Force prediction_mode to be next2
        kwargs["prediction_mode"] = "next2"
        super().__init__(*args, **kwargs)

    def truncate_trajectory(
        self, trajectory: str, system_prompt: Optional[str] = None
    ) -> tuple[str, str, str]:
        """
        Truncate trajectory at middle position for next_2 prediction.
        Predict the next TWO assistant tool call steps (without tool execution results).

        Args:
            trajectory: The trajectory to truncate
            system_prompt: Optional system prompt to prepend (for framethinker)

        Returns:
            tuple: (prefix, ground_truth_steps, remaining_steps)
                - prefix: trajectory up to truncation point
                - ground_truth_steps: the next TWO steps as a complete unit (not split)
                - remaining_steps: all remaining steps after the two steps (for reference)

        Note: Both ground truth steps must be assistant steps (role="assistant"),
              NOT user steps or tool result steps.
        """
        # Parse trajectory - handle both list and string formats
        if isinstance(trajectory, list):
            steps = trajectory
        elif trajectory.startswith("["):
            try:
                steps = json.loads(trajectory)
            except json.JSONDecodeError:
                # Fallback to string split
                steps = [s for s in trajectory.split("\n") if s.strip()]
        else:
            steps = [s for s in trajectory.split("\n") if s.strip()]

        if len(steps) <= 2:
            # Not enough steps for next2 prediction
            return "", "", ""

        # Determine truncation point based on dataset format
        # Use base dataset type for classification
        base_type = get_base_dataset_type(self.dataset_name)

        # Helper to prepend system prompt if needed
        def prepend_system_prompt(steps_list, sys_prompt):
            if not sys_prompt:
                return steps_list

            # Check if first step is already system
            if (
                steps_list
                and isinstance(steps_list[0], dict)
                and steps_list[0].get("role") == "system"
            ):
                return steps_list

            # Prepend system prompt
            return [{"role": "system", "content": sys_prompt}] + steps_list

        if base_type == "framethinker":
            # FrameThinker: dialog format like gaia/gta
            # [{"role": "user/assistant", "content": ...}, ...]
            # We want to predict the next TWO assistant steps, NOT user steps

            # Find all indices where role is "assistant"
            assistant_indices = []
            for i, step in enumerate(steps):
                if isinstance(step, dict) and step.get("role") == "assistant":
                    assistant_indices.append(i)

            if len(assistant_indices) < 2:
                # Not enough assistant steps for next2 prediction
                return "", "", ""

            # Find two consecutive assistant steps (allowing tool/user steps in between)
            # Start from middle and look for valid pairs
            mid_idx = len(assistant_indices) // 2

            # Search for a pair of assistant steps where:
            # 1. No user steps between them
            # 2. They are reasonably close together
            valid_pair_found = False
            first_assistant_idx = None
            second_assistant_idx = None

            # Try from middle onwards
            for i in range(mid_idx, len(assistant_indices) - 1):
                idx1 = assistant_indices[i]
                idx2 = assistant_indices[i + 1]

                # Check if there's any user step between idx1 and idx2
                has_user_step = False
                for j in range(idx1 + 1, idx2):
                    if isinstance(steps[j], dict) and steps[j].get("role") == "user":
                        has_user_step = True
                        break

                if not has_user_step:
                    first_assistant_idx = idx1
                    second_assistant_idx = idx2
                    valid_pair_found = True
                    break

            # If no valid pair found in latter half, try earlier pairs
            if not valid_pair_found:
                for i in range(mid_idx - 1, -1, -1):
                    if i + 1 >= len(assistant_indices):
                        continue
                    idx1 = assistant_indices[i]
                    idx2 = assistant_indices[i + 1]

                    # Check if there's any user step between idx1 and idx2
                    has_user_step = False
                    for j in range(idx1 + 1, idx2):
                        if (
                            isinstance(steps[j], dict)
                            and steps[j].get("role") == "user"
                        ):
                            has_user_step = True
                            break

                    if not has_user_step:
                        first_assistant_idx = idx1
                        second_assistant_idx = idx2
                        valid_pair_found = True
                        break

            if (
                not valid_pair_found
                or first_assistant_idx is None
                or second_assistant_idx is None
            ):
                # No valid pair found
                return "", "", ""

            # Truncate up to but NOT including the first assistant response
            truncate_idx = first_assistant_idx
            prefix_steps = steps[:truncate_idx]

            # Prepend system prompt
            prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

            prefix = json.dumps(prefix_steps, ensure_ascii=False)
            # Combine the two ground truth steps into a single JSON array
            ground_truth_steps = json.dumps(
                [steps[first_assistant_idx], steps[second_assistant_idx]],
                ensure_ascii=False,
            )
            remaining_steps = (
                json.dumps(steps[second_assistant_idx + 1 :], ensure_ascii=False)
                if (second_assistant_idx + 1) < len(steps)
                else ""
            )

        elif base_type == "opencua":
            # OpenCUA: list of dicts with {"role": "assistant", "content": "...", "action": "..."}
            # Simple: take two consecutive steps from middle
            mid = len(steps) // 2
            if mid + 1 >= len(steps):
                return "", "", ""

            prefix_steps = steps[:mid]
            # Prepend system prompt
            prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

            prefix = json.dumps(prefix_steps, ensure_ascii=False)
            # Combine the two ground truth steps into a single JSON array
            ground_truth_steps = json.dumps(
                [steps[mid], steps[mid + 1]], ensure_ascii=False
            )
            remaining_steps = (
                json.dumps(steps[mid + 2 :], ensure_ascii=False)
                if mid + 2 < len(steps)
                else ""
            )

        elif base_type in ["gaia", "gta", "toolbench", "skywork"]:
            # Dialog format: [{"role": "user/assistant/tool", "content": ...}, ...]
            # Find assistant steps and predict next TWO assistant steps
            # Both steps must be assistant (not tool, not user)

            # Find all indices where role is "assistant"
            assistant_indices = []
            for i, step in enumerate(steps):
                if isinstance(step, dict) and step.get("role") == "assistant":
                    assistant_indices.append(i)

            if len(assistant_indices) < 2:
                # Not enough assistant steps for next2 prediction
                return "", "", ""

            # Find two consecutive assistant steps where:
            # 1. No user steps between them
            # 2. They are reasonably close together
            mid_idx = len(assistant_indices) // 2

            valid_pair_found = False
            first_assistant_idx = None
            second_assistant_idx = None

            # Try from middle onwards
            for i in range(mid_idx, len(assistant_indices) - 1):
                idx1 = assistant_indices[i]
                idx2 = assistant_indices[i + 1]

                # Check if there's any user step between idx1 and idx2
                has_user_step = False
                for j in range(idx1 + 1, idx2):
                    if isinstance(steps[j], dict) and steps[j].get("role") == "user":
                        has_user_step = True
                        break

                if not has_user_step:
                    first_assistant_idx = idx1
                    second_assistant_idx = idx2
                    valid_pair_found = True
                    break

            # If no valid pair found in latter half, try earlier pairs
            if not valid_pair_found:
                for i in range(mid_idx - 1, -1, -1):
                    if i + 1 >= len(assistant_indices):
                        continue
                    idx1 = assistant_indices[i]
                    idx2 = assistant_indices[i + 1]

                    # Check if there's any user step between idx1 and idx2
                    has_user_step = False
                    for j in range(idx1 + 1, idx2):
                        if (
                            isinstance(steps[j], dict)
                            and steps[j].get("role") == "user"
                        ):
                            has_user_step = True
                            break

                    if not has_user_step:
                        first_assistant_idx = idx1
                        second_assistant_idx = idx2
                        valid_pair_found = True
                        break

            if (
                not valid_pair_found
                or first_assistant_idx is None
                or second_assistant_idx is None
            ):
                # No valid pair found
                return "", "", ""

            # Truncate up to but NOT including the first assistant response
            # This could be after a tool result
            truncate_idx = first_assistant_idx

            prefix_steps = steps[:truncate_idx]
            # Prepend system prompt
            prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

            prefix = json.dumps(prefix_steps, ensure_ascii=False)
            # Combine the two ground truth steps into a single JSON array
            ground_truth_steps = json.dumps(
                [steps[first_assistant_idx], steps[second_assistant_idx]],
                ensure_ascii=False,
            )
            remaining_steps = (
                json.dumps(steps[second_assistant_idx + 1 :], ensure_ascii=False)
                if (second_assistant_idx + 1) < len(steps)
                else ""
            )

        else:
            # Default: simple consecutive steps from middle
            mid = len(steps) // 2
            if mid + 1 >= len(steps):
                return "", "", ""

            prefix_steps = steps[:mid]
            # Prepend system prompt
            prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

            prefix = json.dumps(prefix_steps, ensure_ascii=False)
            # Combine the two ground truth steps into a single JSON array
            ground_truth_steps = json.dumps(
                [steps[mid], steps[mid + 1]], ensure_ascii=False
            )
            remaining_steps = (
                json.dumps(steps[mid + 2 :], ensure_ascii=False)
                if mid + 2 < len(steps)
                else ""
            )

        return prefix, ground_truth_steps, remaining_steps

    def generate_prediction_prompt(self, sample: Dict, prefix: str) -> str:
        """Generate prompt for test model to predict next 2 steps."""
        query = sample["query"]
        tools = sample.get("tools", [])

        # Get base dataset type for format requirements
        base_type = get_base_dataset_type(self.dataset_name)

        # Build tools information
        tools_info = ""
        if tools:
            tools_info = "\n**Available Tools:**\n"
            for tool in tools:
                if isinstance(tool, dict):
                    if "function" in tool and isinstance(tool["function"], dict):
                        func = tool["function"]
                        name = func.get("name", "unknown")
                        desc_dict = {
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                        desc = json.dumps(desc_dict, ensure_ascii=False)
                        tools_info += f"- {name}: {desc}\n"
                    elif "name" in tool and "example" in tool:
                        name = tool.get("name", "unknown")
                        desc = tool.get("description", "")
                        example = tool.get("example", "")
                        combined_desc = (
                            f"{desc} Example: {example}"
                            if desc
                            else f"Example: {example}"
                        )
                        tools_info += f"- {name}: {combined_desc}\n"
                    else:
                        name = tool.get("tool_name", "unknown")
                        desc = tool.get("description", "No description")
                        tools_info += f"- {name}: {desc}\n"
                else:
                    tools_info += f"- {tool}\n"
        else:
            tools_info = (
                "\n**Note:** Available tools are defined in the system prompt.\n"
            )

        # Build dataset-specific output format instructions for TWO steps
        format_instruction = ""
        if base_type in ["gaia", "gta"]:
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST be a JSON array containing exactly TWO assistant steps:
```json
[
    {
        "role": "assistant",
        "thought": "Your reasoning for the first step",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "arguments": {
                        "param1": "value1"
                    }
                }
            }
        ]
    },
    {
        "role": "assistant",
        "thought": "Your reasoning for the second step",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "arguments": {
                        "param1": "value1"
                    }
                }
            }
        ]
    }
]
```

- MUST be a JSON array with exactly 2 objects
- Each object MUST have `role: "assistant"`
- Each object MUST have a `thought` field
- Each object MUST have a `tool_calls` array
- Do NOT include tool execution results between the two steps
"""
        elif base_type in ["toolbench", "skywork"]:
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST be a JSON array containing exactly TWO assistant steps:
```json
[
    {
        "role": "assistant",
        "content": "Your reasoning for the first step",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "arguments": {
                        "param1": "value1"
                    }
                }
            }
        ]
    },
    {
        "role": "assistant",
        "content": "Your reasoning for the second step",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "arguments": {
                        "param1": "value1"
                    }
                }
            }
        ]
    }
]
```

- MUST be a JSON array with exactly 2 objects
- Each object MUST have `role: "assistant"`
- Each object MUST have a `content` field
- Each object MUST have a `tool_calls` array
- Do NOT include tool execution results between the two steps
"""
        elif base_type == "framethinker":
            # FrameThinker format (dialog format with think: and action: prefixes)
            format_instruction = """
IMPORTANT: You must output your prediction in the following JSON format:
{
    "step1": {
        "role": "assistant",
        "content": "think: your reasoning process for step 1\\n\\naction: your actual action/tool call for step 1"
    },
    "step2": {
        "role": "assistant",
        "content": "think: your reasoning process for step 2\\n\\naction: your actual action/tool call for step 2"
    }
}
Note: The content field MUST contain both `think:` and `action:` prefixes.
"""
        elif base_type == "opencua":
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST be a JSON array containing exactly TWO step objects:
```json
[
  {
    "role": "assistant",
    "content": "Observation: [describe screen state]\n\nThought: [reasoning for next action]",
    "action": "Click on the [element description] to [purpose]."
  },
  {
    "role": "assistant",
    "content": "Observation: [describe screen state]\n\nThought: [reasoning for next action]",
    "action": "Type '[text]' into the [input field description]."
  }
]
```

**OpenCUA Format Requirements:**
- MUST be a JSON array with exactly 2 step objects
- Each object MUST have `role: "assistant"`, `content`, and `action` fields
- `content` contains observation of screen state and thought process
- `action` contains a detailed natural language description (NOT function call syntax)
- Each action should specify: WHAT to do + WHERE (UI element details) + WHY (purpose)
- Example action: "Click on the 'Personal info' link in the left navigation pane of the Google Account page."
- Do NOT use function call format like "click(element)" - use complete descriptive sentences
"""
        else:
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST be a JSON array containing exactly TWO steps.
Each step should follow the EXACT SAME FORMAT as the previous steps.
"""

        json_only_instruction = self._json_only_instruction("array")

        prompt = f"""You are solving a task step by step only using the available tools.

**Task:** {query}
{tools_info}

**Steps Completed So Far:**
{prefix}

**Your Task:**
Predict the NEXT TWO STEPS to progress toward solving this task.
You should predict TWO consecutive assistant actions/tool calls, WITHOUT including the tool execution results in between.
Think about what needs to be done next and plan the following two actions.
{format_instruction}
{json_only_instruction}
"""
        return prompt

    def predict_sample(self, sample: Dict) -> Optional[PredictionResultNext2]:
        """Generate next2 prediction for a single sample (predict next 2 assistant steps)."""
        try:
            task_id = sample["index"]
            query = sample["query"]
            trajectory = sample["trajectory"]
            tools = sample.get("tools", [])
            meta_data = sample.get("meta_data", {})
            system_prompt = sample.get("system_prompt", None)

            print(f"\n{'=' * 80}")
            print(f"Predicting Next2: {task_id}")
            print(f"{'=' * 80}")

            # Get files with types
            base_type = get_base_dataset_type(self.dataset_name)
            files = None
            if base_type in ["framethinker", "gaia", "gta", "opencua"]:
                all_files = self._get_file_paths_with_types(sample)
                if all_files:
                    print(f"  📁 Found {len(all_files)} file(s) for this sample")
                    if base_type == "opencua":
                        files, _ = self._truncate_files_for_opencua(
                            all_files, trajectory
                        )
                    else:
                        files = all_files

            # Truncate for next2
            prefix, reference_steps, reference_remaining = self.truncate_trajectory(
                trajectory, system_prompt
            )

            if not reference_steps:
                print("⊘ Skipped: not enough steps for next2 prediction")
                return None

            # Generate prediction
            print("→ Generating next2 prediction...")
            pred_prompt = self.generate_prediction_prompt(sample, prefix)
            predicted_next = self.test_client.generate(
                pred_prompt,
                max_tokens=self.test_max_tokens,
                files=files,
                temperature=0.0,
            )

            print(f"✓ Predicted: {predicted_next[:100]}...")

            # Store the complete prediction output without parsing
            # The evaluation model will handle the interpretation
            result = PredictionResultNext2(
                task_id=task_id,
                dataset=self.dataset_name,
                query=query,
                trajectory_prefix=prefix,
                predicted_steps=predicted_next,  # Store complete output as-is
                ground_truth_steps=reference_steps,  # Store complete ground truth as-is
                reference_remaining_steps=reference_remaining,
                tools=tools,
                meta_data=meta_data,
                timestamp=datetime.now().isoformat(),
            )

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def evaluate_prediction(
        self, prediction: PredictionResultNext2
    ) -> Optional[EvalResultNext2]:
        """Evaluate a next2 prediction (overall assessment of both steps)."""
        try:
            print(f"\n{'=' * 80}")
            print(f"Evaluating Next2: {prediction.task_id}")
            print(f"{'=' * 80}")

            # Get files with types
            base_type = get_base_dataset_type(self.dataset_name)
            files = None
            prefix_file_count = 0

            if base_type in ["framethinker", "gaia", "gta", "opencua"]:
                sample = {
                    "index": prediction.task_id,
                    "meta_data": prediction.meta_data,
                    "trajectory": prediction.trajectory_prefix,
                }
                all_files = self._get_file_paths_with_types(sample)

                if all_files:
                    print(f"  📁 Found {len(all_files)} file(s) for this sample")

                    if base_type == "opencua":
                        try:
                            prefix_steps = (
                                json.loads(prediction.trajectory_prefix)
                                if prediction.trajectory_prefix.startswith("[")
                                else [prediction.trajectory_prefix]
                            )
                            prefix_file_count = len(prefix_steps)
                            print(
                                f"  ℹ️  For opencua: test model saw first {prefix_file_count} images (out of {len(all_files)} total)"
                            )
                        except Exception:
                            prefix_file_count = len(all_files) // 2

                    files = all_files

            # Evaluate both steps together as a whole
            print("→ Evaluating both steps together as a whole...")

            # Select prompt class based on strict/loose mode
            if self.use_loose_prompt:
                prompt_class = EvaluationPromptsLoose
                print("  🔓 Using LOOSE evaluation prompts")
            else:
                prompt_class = EvaluationPrompts
                print("  🔒 Using DEFAULT (STRICT) evaluation prompts")

            # Truncate predicted_steps to 30000 chars for evaluation prompt
            eval_predicted_steps = prediction.predicted_steps
            if eval_predicted_steps and len(eval_predicted_steps) > 30000:
                print(
                    f"✂️ Truncating prediction for evaluation from {len(eval_predicted_steps)} to 30000 chars"
                )
                eval_predicted_steps = eval_predicted_steps[:30000]

            eval_prompt = prompt_class.build_evaluation_prompt_next2(
                prediction.dataset,
                prediction.query,
                prediction.trajectory_prefix,
                eval_predicted_steps,  # Use truncated version
                prediction.ground_truth_steps,
                prediction.reference_remaining_steps,
                prediction.tools,
                prefix_file_count if base_type == "opencua" else 0,
                test_model=self.test_model,  # Pass test_model for framethinker format selection
            )
            eval_response = self.eval_client.generate(
                eval_prompt,
                max_tokens=self.eval_max_tokens,
                files=files,
                temperature=0.0,
            )

            # Parse evaluation response - expecting a single JSON object
            eval_data = self._parse_evaluation_next2(eval_response)
            if not eval_data:
                print("✗ Failed to parse next2 evaluation")
                return None

            result = EvalResultNext2(
                task_id=prediction.task_id,
                dataset=prediction.dataset,
                query=prediction.query,
                predicted_steps=prediction.predicted_steps,  # Store complete prediction output
                ground_truth_steps=prediction.ground_truth_steps,  # Store complete ground truth
                score=eval_data["score"],
                is_correct=eval_data["is_correct"],
                error_types={
                    "E1": eval_data.get("E1_GOAL_MISALIGNMENT", False),
                    "E2": eval_data.get("E2_PREMATURE_CONCLUSION", False),
                    "E3": eval_data.get("E3_CONSTRAINT_VIOLATION", False),
                    "E4": eval_data.get("E4_LOGIC_ERROR", False),
                    "E5": eval_data.get("E5_TOOL_USE_ERROR", False),
                    "E6": eval_data.get("E6_HALLUCINATION_ERROR", False),
                },
                reasoning=eval_data["reasoning"],
                timestamp=datetime.now().isoformat(),
            )

            print(f"✓ Overall Score: {result.score}")
            print(f"✓ Correct: {result.is_correct}")
            print(f"✓ Errors: {[k for k, v in result.error_types.items() if v]}")

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _parse_evaluation_next2(self, response: str) -> Optional[Dict]:
        """Parse next2 evaluation response with json-repair fallback."""
        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            # Try standard JSON parsing first
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Use json-repair to fix broken JSON
                print(f"  ⚠️  JSON decode error: {e}")
                print(f"  🔧 Attempting to repair JSON...")
                try:
                    repaired_json = repair_json(json_str)
                    data = json.loads(repaired_json)
                    print(f"  ✓ JSON repair successful")
                except Exception as repair_error:
                    print(f"  ✗ JSON repair failed: {repair_error}")
                    print(f"  Raw response (first 500 chars): {response[:500]}")
                    return None

            # Ensure it's a dict (single evaluation)
            if not isinstance(data, dict):
                print(f"⚠️  Expected a JSON object, got {type(data)}")
                return None

            # Determine is_correct based on whether ANY error exists
            has_any_error = any(
                [
                    data.get("E1_GOAL_MISALIGNMENT", False),
                    data.get("E2_PREMATURE_CONCLUSION", False),
                    data.get("E3_CONSTRAINT_VIOLATION", False),
                    data.get("E4_LOGIC_ERROR", False),
                    data.get("E5_TOOL_USE_ERROR", False),
                    data.get("E6_HALLUCINATION_ERROR", False),
                ]
            )

            # Override is_correct: YES only if no errors exist
            data["is_correct"] = not has_any_error

            return data

        except Exception as e:
            print(f"Parse error: {e}")
            print(f"  Raw response (first 500 chars): {response[:500]}")
            import traceback

            traceback.print_exc()
            return None

    def run_inference(
        self, dataset: Optional[List[Dict]] = None
    ) -> List[PredictionResultNext2]:
        """Run inference mode for next2: generate next2 predictions only."""
        print(f"\n{'=' * 80}")
        print(f"Starting Next2 Inference: {self.dataset_name}")
        print(f"{'=' * 80}")

        # Load dataset if not provided
        if dataset is None:
            dataset = DatasetLoader.load_dataset(self.dataset_name)

        # Apply max_samples limit
        if self.max_samples:
            dataset = dataset[: self.max_samples]

        print(f"→ Processing {len(dataset)} samples (next2 inference)")

        # Try to resume from existing predictions
        existing_predictions = []
        self.completed_task_ids = set()

        if self.resume:
            existing_predictions = self._load_latest_predictions_next2()
            if existing_predictions:
                self.completed_task_ids = {p.task_id for p in existing_predictions}
                print(
                    f"  📂 Resume: loaded {len(existing_predictions)} existing predictions"
                )

        # Generate predictions (skip already completed ones)
        predictions = existing_predictions.copy()
        skipped = 0
        generated = 0

        for i, sample in enumerate(dataset, 1):
            task_id = sample["index"]

            if task_id in self.completed_task_ids:
                skipped += 1
                print(f"[{i}/{len(dataset)}] ⏭️  Skipped: {task_id} (already completed)")
                continue

            print(f"[{i}/{len(dataset)}] Processing: {task_id}")
            prediction = self.predict_sample(sample)
            if prediction:
                predictions.append(prediction)
                self.completed_task_ids.add(task_id)
                generated += 1
                # Auto-save
                self._save_predictions_incremental_next2(predictions)

        if generated > 0:
            self.save_predictions(predictions)

        print(f"\n{'=' * 80}")
        print(f"Completed: {len(predictions)} total next2 predictions")
        print(f"  - Existing: {len(existing_predictions)}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Generated: {generated}")
        print(f"{'=' * 80}")

        return predictions

    def run_evaluation(
        self, predictions: List[PredictionResultNext2]
    ) -> List[EvalResultNext2]:
        """Run evaluation mode for next2: evaluate existing next2 predictions."""
        print(f"\n{'=' * 80}")
        print(f"Starting Next2 Evaluation: {self.dataset_name}")
        print(f"{'=' * 80}")

        print(f"→ Evaluating {len(predictions)} next2 predictions")

        # Try to resume
        existing_results = []
        evaluated_task_ids = set()

        if self.resume:
            existing_results = self._load_latest_eval_results_next2()
            if existing_results:
                evaluated_task_ids = {r.task_id for r in existing_results}
                print(
                    f"  📂 Resume: loaded {len(existing_results)} existing evaluations"
                )

        results = existing_results.copy()
        skipped = 0
        evaluated = 0

        for i, prediction in enumerate(predictions, 1):
            task_id = prediction.task_id

            if task_id in evaluated_task_ids:
                skipped += 1
                print(
                    f"[{i}/{len(predictions)}] ⏭️  Skipped: {task_id} (already evaluated)"
                )
                continue

            print(f"[{i}/{len(predictions)}] Processing: {task_id}")
            result = self.evaluate_prediction(prediction)
            if result:
                results.append(result)
                evaluated_task_ids.add(task_id)
                evaluated += 1
                # Auto-save
                self._save_results_incremental_next2(results)

        if evaluated > 0:
            self.save_results(results)

        print(f"\n{'=' * 80}")
        print(f"Completed: {len(results)} total next2 results")
        print(f"  - Existing: {len(existing_results)}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Evaluated: {evaluated}")
        print(f"{'=' * 80}")

        return results

    def run_pipeline(
        self, dataset: Optional[List[Dict]] = None
    ) -> List[EvalResultNext2]:
        """Run pipeline mode for next2: inference + evaluation."""
        print(f"\n{'=' * 80}")
        print(f"Starting Next2 Pipeline: {self.dataset_name}")
        print(f"{'=' * 80}")

        # Load dataset if not provided
        if dataset is None:
            dataset = DatasetLoader.load_dataset(self.dataset_name)

        # Apply max_samples limit
        if self.max_samples:
            dataset = dataset[: self.max_samples]

        print(f"→ Processing {len(dataset)} samples (next2 inference + evaluation)")

        # Try to resume
        existing_results = []
        evaluated_task_ids = set()

        if self.resume:
            existing_results = self._load_latest_eval_results_next2()
            if existing_results:
                evaluated_task_ids = {r.task_id for r in existing_results}
                print(f"  📂 Resume: loaded {len(existing_results)} existing results")

        results = existing_results.copy()
        predictions = []  # Collect predictions for saving
        skipped = 0
        processed = 0

        for i, sample in enumerate(dataset, 1):
            task_id = sample["index"]

            if task_id in evaluated_task_ids:
                skipped += 1
                print(f"[{i}/{len(dataset)}] ⏭️  Skipped: {task_id} (already completed)")
                continue

            print(f"[{i}/{len(dataset)}] Processing: {task_id}")

            # Predict and evaluate
            prediction = self.predict_sample(sample)
            if prediction:
                predictions.append(prediction)
                result = self.evaluate_prediction(prediction)
                if result:
                    results.append(result)
                    evaluated_task_ids.add(task_id)
                    processed += 1
                    # Auto-save results
                    self._save_results_incremental_next2(results)

        # Final save for both predictions and results
        if processed > 0:
            self.save_predictions(predictions)
            self.save_results(results)

        print(f"\n{'=' * 80}")
        print(f"Completed: {len(results)} total next2 results")
        print(f"  - Existing: {len(existing_results)}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Processed: {processed}")
        print(f"{'=' * 80}")

        return results

    def save_predictions(self, predictions: List[PredictionResultNext2]) -> str:
        """Save next2 predictions to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_predictions_next2_{timestamp}.json"
        filepath = os.path.join(self.predictions_dir, filename)

        output = {
            "dataset": self.dataset_name,
            "test_model": self.test_model,
            "timestamp": timestamp,
            "total_predictions": len(predictions),
            "predictions": [asdict(p) for p in predictions],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Next2 predictions saved to: {filepath}")

        return filepath

    def _save_predictions_incremental_next2(
        self, predictions: List[PredictionResultNext2]
    ) -> str:
        """Save next2 predictions incrementally (for resume mode)."""
        filename = f"{self.dataset_name}_predictions_next2_resume.json"
        filepath = os.path.join(self.prediction_resume_dir, filename)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "dataset": self.dataset_name,
            "test_model": self.test_model,
            "timestamp": timestamp,
            "total_predictions": len(predictions),
            "predictions": [asdict(p) for p in predictions],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        self.resume_filepath = filepath
        print(f"  💾 Auto-saved {len(predictions)} next2 predictions to: {filepath}")

        return filepath

    def _load_latest_predictions_next2(self) -> List[PredictionResultNext2]:
        """Load latest next2 predictions for resume mode."""
        resume_file = os.path.join(
            self.prediction_resume_dir,
            f"{self.dataset_name}_predictions_next2_resume.json",
        )
        if os.path.exists(resume_file):
            try:
                return self._load_predictions_from_file_next2(resume_file)
            except Exception as e:
                print(f"  ⚠️  Failed to load resume file: {e}")

        return []

    def _load_predictions_from_file_next2(
        self, filepath: str
    ) -> List[PredictionResultNext2]:
        """Load next2 predictions from a specific file."""
        print(f"  📂 Loading next2 predictions from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        predictions = []
        for pred_dict in data.get("predictions", []):
            try:
                prediction = PredictionResultNext2(**pred_dict)
                predictions.append(prediction)
            except Exception as e:
                print(f"  ⚠️  Failed to parse prediction: {e}")
                continue

        return predictions

    def load_predictions(self, filepath: str) -> List[PredictionResultNext2]:
        """Load next2 predictions from file."""
        predictions = self._load_predictions_from_file_next2(filepath)

        # Apply max_samples limit if specified
        if self.max_samples and len(predictions) > self.max_samples:
            original_count = len(predictions)
            predictions = predictions[: self.max_samples]
            print(
                f"✓ Limited to {len(predictions)} predictions by max_samples={self.max_samples} (from {original_count} total)"
            )

        return predictions

    def _save_results_incremental_next2(self, results: List[EvalResultNext2]) -> str:
        """Save next2 evaluation results incrementally (for resume mode)."""
        filename = f"{self.dataset_name}_results_next2_resume.json"
        filepath = os.path.join(self.resume_dir, filename)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate statistics
        stats = self.calculate_statistics(results)

        output = {
            "dataset": self.dataset_name,
            "test_model": self.test_model,
            "eval_model": self.eval_model,
            "timestamp": timestamp,
            "total_samples": len(results),
            "statistics": asdict(stats),
            "results": [asdict(r) for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"  💾 Auto-saved {len(results)} next2 evaluations to: {filepath}")

        return filepath

    def _load_latest_eval_results_next2(self) -> List[EvalResultNext2]:
        """Load latest next2 evaluation results for resume mode."""
        resume_file = os.path.join(
            self.resume_dir, f"{self.dataset_name}_results_next2_resume.json"
        )
        if os.path.exists(resume_file):
            try:
                return self._load_eval_results_from_file_next2(resume_file)
            except Exception as e:
                print(f"  ⚠️  Failed to load resume file: {e}")

        return []

    def _load_eval_results_from_file_next2(
        self, filepath: str
    ) -> List[EvalResultNext2]:
        """Load next2 evaluation results from a specific file."""
        print(f"  📂 Loading next2 evaluation results from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for result_dict in data.get("results", []):
            try:
                result = EvalResultNext2(**result_dict)
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  Failed to parse result: {e}")
                continue

        return results

    def save_results(self, results: List[EvalResultNext2]) -> str:
        """Save next2 results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_results_next2_{timestamp}.json"
        filepath = os.path.join(self.evaluations_dir, filename)

        # Calculate statistics
        stats = self.calculate_statistics(results)

        output = {
            "dataset": self.dataset_name,
            "test_model": self.test_model,
            "eval_model": self.eval_model,
            "timestamp": timestamp,
            "total_samples": len(results),
            "statistics": asdict(stats),
            "results": [asdict(r) for r in results],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Next2 results saved to: {filepath}")

        return filepath
