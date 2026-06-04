"""
Next1 Strategy Evaluator.

This module contains the Next1 evaluator that predicts the next single step.
"""

import os
import json
from json_repair import repair_json
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import asdict

from .base_evaluator import BaseOnlinePlanningEvaluator
from ..models.data_models import PredictionResult, EvalResult
from ..utils.dataset_loader import DatasetLoader, get_base_dataset_type
from ..prompts.evaluation_prompts import EvaluationPrompts

from ..prompts.evaluation_prompts_loose import EvaluationPromptsLoose
from ..prompts.evaluation_prompts_blind import EvaluationPromptsBlind


class Next1Evaluator(BaseOnlinePlanningEvaluator):
    """Evaluator for next1 prediction strategy (predict next single step)."""

    def __init__(self, *args, **kwargs):
        # Determine prediction_mode based on truncation_position
        truncation_position = kwargs.get("truncation_position", "middle")

        if truncation_position == "early":
            kwargs["prediction_mode"] = "next1_early"
        elif truncation_position == "late":
            kwargs["prediction_mode"] = "next1_late"
        else:
            kwargs["prediction_mode"] = "next1"

        super().__init__(*args, **kwargs)

    def _get_truncation_ratio(self) -> float:
        """
        Get truncation ratio based on truncation_position setting.

        Returns:
            float: Ratio indicating where to truncate (0.0 = start, 1.0 = end)
                - early: 0.25 (first quarter)
                - middle: 0.5 (middle)
                - late: 0.75 (third quarter)
        """
        if self.truncation_position == "early":
            return 0.25
        elif self.truncation_position == "late":
            return 0.75
        else:  # middle (default)
            return 0.5

    def _get_index_at_ratio(self, indices: list, ratio: float) -> int:
        """
        Get the index from a list based on the ratio.

        Args:
            indices: List of indices to choose from
            ratio: Ratio indicating position (0.0 = first, 1.0 = last)

        Returns:
            The index at the specified ratio position
        """
        if not indices:
            return 0
        target_pos = int(len(indices) * ratio)
        # Clamp to valid range
        target_pos = max(0, min(target_pos, len(indices) - 1))
        return indices[target_pos]

    def truncate_trajectory(
        self, trajectory: str, system_prompt: Optional[str] = None
    ) -> tuple[str, str, str]:
        """
        Truncate trajectory at specified position for next_1 prediction.

        The truncation position is determined by self.truncation_position:
        - "early": Truncate at ~25% of the trajectory (less context)
        - "middle": Truncate at ~50% of the trajectory (default)
        - "late": Truncate at ~75% of the trajectory (more context)

        Args:
            trajectory: The trajectory to truncate
            system_prompt: Optional system prompt to prepend (for framethinker)

        Returns:
            tuple: (prefix, next_step, remaining_steps)
                - prefix: trajectory up to truncation point
                - next_step: the immediate next step
                - remaining_steps: all remaining steps after next_step (for reference)
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

        if len(steps) <= 1:
            return (
                trajectory if isinstance(trajectory, str) else json.dumps(trajectory),
                "",
                "",
            )

        # Get truncation ratio based on position setting
        ratio = self._get_truncation_ratio()

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
            # [{\"role\": \"user/assistant\", \"content\": ...}, ...]
            # We want to predict the next assistant step, NOT a user step

            # Find all indices where role is \"assistant\"
            assistant_indices = []
            for i, step in enumerate(steps):
                if isinstance(step, dict) and step.get("role") == "assistant":
                    assistant_indices.append(i)

            if not assistant_indices:
                # Fallback: no assistant steps found, use simple truncation based on ratio
                truncate_pos = int(len(steps) * ratio)
                truncate_pos = max(1, min(truncate_pos, len(steps) - 1))
                prefix_steps = steps[:truncate_pos]

                # Prepend system prompt
                prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

                prefix = json.dumps(prefix_steps, ensure_ascii=False)
                next_step = json.dumps(steps[truncate_pos], ensure_ascii=False)
                remaining_steps = (
                    json.dumps(steps[truncate_pos + 1 :], ensure_ascii=False)
                    if truncate_pos + 1 < len(steps)
                    else ""
                )
            else:
                # Find the assistant step at the specified ratio
                target_assistant_idx = self._get_index_at_ratio(
                    assistant_indices, ratio
                )

                # Truncate up to but NOT including the assistant response
                # The next step should be the assistant's response
                truncate_idx = target_assistant_idx

                prefix_steps = steps[:truncate_idx]

                # Prepend system prompt
                prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

                prefix = json.dumps(prefix_steps, ensure_ascii=False)

                # The next step to predict is the assistant's response
                next_step = json.dumps(steps[target_assistant_idx], ensure_ascii=False)
                remaining_steps = (
                    json.dumps(steps[target_assistant_idx + 1 :], ensure_ascii=False)
                    if target_assistant_idx + 1 < len(steps)
                    else ""
                )

        elif base_type == "opencua":
            # OpenCUA: list of dicts with {"role": "assistant", "content": "...", "action": "..."}
            # Truncation based on ratio
            truncate_pos = int(len(steps) * ratio)
            truncate_pos = max(1, min(truncate_pos, len(steps) - 1))

            prefix_steps = steps[:truncate_pos]
            # Prepend system prompt
            prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

            prefix = json.dumps(prefix_steps, ensure_ascii=False)
            next_step = json.dumps(steps[truncate_pos], ensure_ascii=False)
            remaining_steps = (
                json.dumps(steps[truncate_pos + 1 :], ensure_ascii=False)
                if truncate_pos + 1 < len(steps)
                else ""
            )

        elif base_type in ["gaia", "gta", "toolbench", "skywork"]:
            # Dialog format: [{"role": "user/assistant/tool", "content": ...}, ...]
            # Find tool calls and truncate based on ratio
            # For these datasets, we want to predict the next assistant step after a tool call

            # Find all indices where role is "tool"
            tool_indices = []
            for i, step in enumerate(steps):
                if isinstance(step, dict) and step.get("role") == "tool":
                    tool_indices.append(i)

            if not tool_indices:
                # Fallback: no tool calls found, use simple truncation based on ratio
                truncate_pos = int(len(steps) * ratio)
                truncate_pos = max(1, min(truncate_pos, len(steps) - 1))

                prefix_steps = steps[:truncate_pos]
                # Prepend system prompt
                prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

                prefix = json.dumps(prefix_steps, ensure_ascii=False)
                next_step = json.dumps(steps[truncate_pos], ensure_ascii=False)
                remaining_steps = (
                    json.dumps(steps[truncate_pos + 1 :], ensure_ascii=False)
                    if truncate_pos + 1 < len(steps)
                    else ""
                )
            else:
                # Find the tool call at the specified ratio
                target_tool_pos = int(len(tool_indices) * ratio)
                target_tool_pos = max(0, min(target_tool_pos, len(tool_indices) - 1))
                target_tool_idx = tool_indices[target_tool_pos]

                # CRITICAL: Find a tool call where the next step is an assistant response
                # The next step to predict should NEVER be a "tool" role (tool result)
                # It should also not be a "user" role for skywork
                valid_tool_idx = None

                # First, try tool calls starting from the target position onwards
                for tool_idx in tool_indices[target_tool_pos:]:
                    if tool_idx + 1 < len(steps):
                        next_step_role = (
                            steps[tool_idx + 1].get("role")
                            if isinstance(steps[tool_idx + 1], dict)
                            else None
                        )
                        # Next step must be assistant (not tool, not user)
                        if next_step_role == "assistant":
                            valid_tool_idx = tool_idx
                            break

                # If no valid index found, try earlier tool calls
                if valid_tool_idx is None:
                    for tool_idx in reversed(tool_indices[:target_tool_pos]):
                        if tool_idx + 1 < len(steps):
                            next_step_role = (
                                steps[tool_idx + 1].get("role")
                                if isinstance(steps[tool_idx + 1], dict)
                                else None
                            )
                            # Next step must be assistant (not tool, not user)
                            if next_step_role == "assistant":
                                valid_tool_idx = tool_idx
                                break

                # Use valid index if found, otherwise fall back to target_tool_idx
                final_tool_idx = (
                    valid_tool_idx if valid_tool_idx is not None else target_tool_idx
                )

                # Truncate up to and including the tool call
                # The next step should be the assistant's response
                truncate_idx = final_tool_idx + 1

                if truncate_idx >= len(steps):
                    # Edge case: tool call is the last step
                    truncate_idx = final_tool_idx

                prefix_steps = steps[:truncate_idx]
                # Prepend system prompt
                prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

                prefix = json.dumps(prefix_steps, ensure_ascii=False)

                # The next step to predict is the assistant's response after the tool call
                if truncate_idx < len(steps):
                    next_step_role = (
                        steps[truncate_idx].get("role")
                        if isinstance(steps[truncate_idx], dict)
                        else None
                    )
                    # Double-check: if next_step is still not assistant, skip this sample
                    if next_step_role != "assistant":
                        # Return empty to indicate this sample should be skipped
                        return "", "", ""

                    next_step = json.dumps(steps[truncate_idx], ensure_ascii=False)
                    # All remaining steps after next_step
                    remaining_steps = (
                        json.dumps(steps[truncate_idx + 1 :], ensure_ascii=False)
                        if truncate_idx + 1 < len(steps)
                        else ""
                    )
                else:
                    next_step = ""
                    remaining_steps = ""

        else:
            # Default: simple truncation based on ratio
            truncate_pos = int(len(steps) * ratio)
            truncate_pos = max(1, min(truncate_pos, len(steps) - 1))

            prefix_steps = steps[:truncate_pos]
            # Prepend system prompt
            prefix_steps = prepend_system_prompt(prefix_steps, system_prompt)

            prefix = json.dumps(prefix_steps, ensure_ascii=False)
            next_step = json.dumps(steps[truncate_pos], ensure_ascii=False)
            remaining_steps = (
                json.dumps(steps[truncate_pos + 1 :], ensure_ascii=False)
                if truncate_pos + 1 < len(steps)
                else ""
            )

        return prefix, next_step, remaining_steps

    def generate_prediction_prompt(self, sample: Dict, prefix: str) -> str:
        """Generate prompt for test model."""
        query = sample["query"]
        tools = sample.get("tools", [])

        # Get base dataset type for format requirements
        base_type = get_base_dataset_type(self.dataset_name)

        # Build tools information
        # Note: If tools is an empty list, tool information is in the system_prompt
        tools_info = ""
        if tools:  # Only add tools section if tools list is not empty
            tools_info = "\n**Available Tools:**\n"
            for tool in tools:
                # Handle both dict format (with name/description) and string format
                if isinstance(tool, dict):
                    # Check for skywork nested format: {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
                    if "function" in tool and isinstance(tool["function"], dict):
                        func = tool["function"]
                        name = func.get("name", "unknown")
                        description = func.get("description", "")
                        params = func.get("parameters", {})
                        tools_info += f"- **{name}**: {description}\n"
                        if params:
                            tools_info += f"  Parameters: {json.dumps(params, ensure_ascii=False)}\n"
                    # Check for toolbench/gaia format: {"name": ..., "description": ..., "parameters": ...}
                    elif "name" in tool and "example" in tool:
                        name = tool.get("name", "unknown")
                        description = tool.get("description", "")
                        example = tool.get("example", "")
                        tools_info += f"- **{name}**: {description}\n"
                        if example:
                            tools_info += f"  Example: {example}\n"
                    else:
                        # Generic dict format
                        tools_info += f"- {json.dumps(tool, ensure_ascii=False)}\n"

                else:
                    # For string format (e.g., opencua)
                    tools_info += f"- {tool}\n"
        else:
            # If tools list is empty, tool information is already in system_prompt
            tools_info = (
                "\n**Note:** Available tools are defined in the system prompt.\n"
            )

        # Build dataset-specific output format instructions
        format_instruction = ""
        if base_type in ["gaia", "gta"]:
            # Dialog format datasets
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST follow this exact JSON structure:
```json
{
    "role": "assistant",
    "thought": "Your reasoning about what to do next",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
        }
    ]
}
```

- The `role` field MUST be "assistant"
- The `thought` field contains your reasoning
- The `tool_calls` field is an array of tool call objects
- Each tool call MUST have `type: "function"` and a `function` object with `name` and `arguments`
- If no tool call is needed, use an empty array: `"tool_calls": []`
"""
        elif base_type in ["toolbench", "skywork"]:
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST follow this exact JSON structure:
```json
{
    "role": "assistant",
    "content": "Your reasoning about what to do next",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": {
                    "param1": "value1",
                    "param2": "value2"
                }
            }
        }
    ]
}
```

- The `role` field MUST be "assistant"
- The `content` field contains your reasoning
- The `tool_calls` field is an array of tool call objects
- Each tool call MUST have `type: "function"` and a `function` object with `name` and `arguments`
- If no tool call is needed, use an empty array: `"tool_calls": []`
"""
        elif base_type == "framethinker":
            # FrameThinker format (dialog format with role and content, using think: and action: prefixes)
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST follow this exact JSON structure:
```json
{
    "role": "assistant",
    "content": "think: your reasoning process here\\n\\naction: your actual action/tool call here"
}
```

**CRITICAL FORMAT REQUIREMENTS:**
- The `role` field MUST be "assistant"
- The `content` field MUST contain BOTH `think:` and `action:` prefixes
- The `think:` section contains your reasoning, analysis, and thought process
- The `action:` section contains the actual tool call or action to execute
- Both sections are REQUIRED - missing either prefix will result in format error
- Use `\\n\\n` to separate the think and action sections for readability

**Example:**
```json
{
    "role": "assistant",
    "content": "think: I need to analyze the frames to understand the environment. Based on the query, I should first classify the environmental settings across all frames.\\n\\naction: classify_environment(frames=[0, 620, 1240], categories=['indoor', 'outdoor', 'underground'])"
}
```
"""
        elif base_type == "opencua":
            # OpenCUA format (with separate action field)
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST follow this exact JSON structure:
```json
{
    "role": "assistant",
    "content": "Observation: [describe what you see on screen]\n\nThought: [your reasoning about the next action]",
    "action": "detailed action description"
}
```

**OpenCUA Format Requirements:**
- The `role` field MUST be "assistant"
- The `content` field contains your observation of the current screen state and thought process
- The `action` field contains a detailed natural language description of ONE specific UI operation to perform
- Actions describe WHAT to do, WHERE on the screen, and WHY (the purpose)

**Content Field Structure:**
- Start with "Observation: " followed by a detailed description of what's visible on screen
- Then add "\n\nThought: " followed by your reasoning about what action to take next and why
- Be specific about UI elements, their locations, and the current state

**Action Field Guidelines:**
1. Write ONE complete sentence describing the specific action to perform
2. Be specific about UI elements (buttons, text boxes, icons, menus, etc.)
3. Include visual details (location, appearance, labels) to identify the target element
4. State the purpose or outcome of the action
5. Do NOT use function call syntax like "click(element)" - use natural language descriptions
6. Match the detail level and style of the previous steps

**Common action patterns:**
- Click actions: "Click on the [element description] to [purpose]"
- Text input: "Type '[text]' into the [input field description]"
- Selection: "Click on '[option]' in the [menu/list] to select it"
- Navigation: "Scroll [direction] on the [page/area] to [reveal something]"

**Example OpenCUA Response:**
```json
{
    "role": "assistant",
    "content": "Observation: The active application is Google Chrome, displaying the Google Account page. The layout consists of a navigation pane on the left side with items such as 'Home', 'Personal info', 'Data & privacy', 'Security', etc.\n\nThought: The task requires accessing personal information to set the work address. The 'Personal info' link in the left navigation is the logical next step to access address settings.",
    "action": "Click on the 'Personal info' link in the left navigation pane of the Google Account page."
}
```
"""
        else:
            # Default/unknown format - use the format from previous steps
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST follow the EXACT SAME FORMAT as the previous steps.
Carefully observe the structure and formatting of the steps above and replicate it precisely.
"""

        json_only_instruction = self._json_only_instruction("object")

        prompt = f"""You are solving a task step by step only using the available tools.

**Task:** {query}
{tools_info}

**Steps Completed So Far:**
{prefix}

**Your Task:**
Predict the NEXT SINGLE STEP to progress toward solving this task.
{format_instruction}
{json_only_instruction}
"""
        return prompt

    def predict_sample(self, sample: Dict) -> Optional[PredictionResult]:
        """Generate prediction for a single sample (inference mode)."""
        try:
            task_id = sample["index"]
            query = sample["query"]
            trajectory = sample["trajectory"]
            tools = sample.get("tools", [])
            meta_data = sample.get("meta_data", {})
            system_prompt = sample.get(
                "system_prompt", None
            )  # Extract system_prompt for framethinker

            print(f"\n{'=' * 80}")
            print(f"Predicting: {task_id}")
            print(f"{'=' * 80}")

            # Get files with types (for datasets that need them)
            # Use base dataset type to determine which datasets need files
            base_type = get_base_dataset_type(self.dataset_name)
            files = None
            if base_type in ["framethinker", "gaia", "gta", "opencua"]:
                all_files = self._get_file_paths_with_types(sample)
                if all_files:
                    print(f"  📁 Found {len(all_files)} file(s) for this sample")

                    # For opencua, truncate files to match trajectory truncation
                    if base_type == "opencua":
                        files, _ = self._truncate_files_for_opencua(
                            all_files, trajectory
                        )
                    else:
                        files = all_files

            # Truncate at middle (pass system_prompt for framethinker)
            prefix, reference_next, reference_remaining = self.truncate_trajectory(
                trajectory, system_prompt
            )

            if not reference_next:
                print("⊘ Skipped: trajectory too short")
                return None

            # Generate prediction (with files)
            print("→ Generating prediction...")
            pred_prompt = self.generate_prediction_prompt(sample, prefix)
            predicted_next = self.test_client.generate(
                pred_prompt,
                max_tokens=self.test_max_tokens,
                files=files,
                temperature=0.0,  # Use deterministic temperature for reproducible predictions
            )

            print(f"✓ Predicted: {predicted_next[:100]}...")

            result = PredictionResult(
                task_id=task_id,
                dataset=self.dataset_name,
                query=query,
                trajectory_prefix=prefix,
                predicted_step=predicted_next,
                ground_truth_step=reference_next,
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

    def evaluate_prediction(self, prediction: PredictionResult) -> Optional[EvalResult]:
        """Evaluate a single prediction (evaluation mode)."""
        try:
            print(f"\n{'=' * 80}")
            print(f"Evaluating: {prediction.task_id}")
            print(f"{'=' * 80}")

            # Get files with types (for datasets that need them)
            # Use base dataset type to determine which datasets need files
            base_type = get_base_dataset_type(self.dataset_name)
            files = None
            prefix_file_count = 0  # For opencua: how many files the test model saw

            if base_type in ["framethinker", "gaia", "gta", "opencua"]:
                # Reconstruct sample dict for file extraction
                sample = {
                    "index": prediction.task_id,
                    "meta_data": prediction.meta_data,
                    "trajectory": prediction.trajectory_prefix,  # Need trajectory for opencua truncation
                }
                all_files = self._get_file_paths_with_types(sample)

                if all_files:
                    print(f"  📁 Found {len(all_files)} file(s) for this sample")

                    # For opencua, calculate how many files were shown to test model
                    if base_type == "opencua":
                        # Get the trajectory to calculate truncation point
                        # The trajectory_prefix in prediction is already truncated
                        # We need to calculate how many files correspond to the prefix
                        try:
                            # Parse the prefix to count steps
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

                    files = all_files  # Eval model sees all files

            # Evaluate prediction (with files for context)
            print("→ Evaluating prediction...")

            # Select prompt class based on strict/loose mode
            # Select prompt class based on strict/loose/blind mode
            if self.use_blind_prompt:
                prompt_class = EvaluationPromptsBlind
                print("  🙈 Using BLIND evaluation prompts (no reference)")
            elif self.use_loose_prompt:
                prompt_class = EvaluationPromptsLoose
                print("  🔓 Using LOOSE evaluation prompts")
            else:
                prompt_class = EvaluationPrompts
                print("  🔒 Using DEFAULT (STRICT) evaluation prompts")

            # Truncate predicted_step to 30000 chars for evaluation prompt
            eval_predicted_step = prediction.predicted_step
            if eval_predicted_step and len(eval_predicted_step) > 30000:
                print(
                    f"✂️ Truncating prediction for evaluation from {len(eval_predicted_step)} to 30000 chars"
                )
                eval_predicted_step = eval_predicted_step[:30000]

            eval_prompt = prompt_class.build_evaluation_prompt(
                prediction.dataset,
                prediction.query,
                prediction.trajectory_prefix,
                eval_predicted_step,
                prediction.ground_truth_step,
                prediction.reference_remaining_steps,
                prediction.tools,
                prefix_file_count
                if base_type == "opencua"
                else 0,  # Pass file count for opencua
                test_model=self.test_model,  # Pass test_model for framethinker format selection
            )
            eval_response = self.eval_client.generate(
                eval_prompt,
                max_tokens=self.eval_max_tokens,
                files=files,
                temperature=0.0,  # Use deterministic temperature for stable evaluation
            )

            # Parse evaluation
            eval_data = self._parse_evaluation(eval_response)

            if not eval_data:
                print("✗ Failed to parse evaluation")
                return None

            result = EvalResult(
                task_id=prediction.task_id,
                dataset=prediction.dataset,
                query=prediction.query,
                predicted_step=prediction.predicted_step,
                ground_truth_step=prediction.ground_truth_step,
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

            print(f"✓ Score: {result.score}, Correct: {result.is_correct}")
            print(f"  Errors: {[k for k, v in result.error_types.items() if v]}")

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _parse_evaluation(self, response: str) -> Optional[Dict]:
        """Parse evaluation response with json-repair fallback."""
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
                    # save response to debug file
                    with open("debug_response.json", "w") as f:
                        f.write(response)
                    return None

            # Determine is_correct based on whether ANY error exists
            # If any error flag (E1-E6) is True, then is_correct must be False
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
            return None

    def run_inference(
        self, dataset: Optional[List[Dict]] = None
    ) -> List[PredictionResult]:
        """Run inference mode: generate predictions only."""
        print(f"\n{'=' * 80}")
        print(f"Starting Inference: {self.dataset_name}")
        print(f"{'=' * 80}")

        # Load dataset if not provided
        if dataset is None:
            dataset = DatasetLoader.load_dataset(
                self.dataset_name, data_dir=self.data_dir
            )

        if self.max_samples:
            dataset = dataset[: self.max_samples]

        # Try to resume from existing predictions
        existing_predictions = []
        if self.resume:
            existing_predictions = self._load_latest_predictions()
            if existing_predictions:
                # Only mark as completed if predicted_step is not empty
                # This allows retrying failed predictions (which are saved with empty string to persist reflection)
                self.completed_task_ids = {
                    p.task_id for p in existing_predictions if p.predicted_step
                }
                print(
                    f"  📂 Resume: loaded {len(existing_predictions)} existing predictions"
                )

        print(f"→ Generating predictions for {len(dataset)} samples")

        # Generate predictions (skip completed ones)
        predictions = existing_predictions.copy()  # Start with existing predictions
        skipped = 0
        generated = 0

        # Create a map of existing predictions for easy lookup
        existing_predictions_map = {str(p.task_id): p for p in existing_predictions}

        for i, sample in enumerate(dataset, 1):
            task_id = sample["index"]

            if task_id in self.completed_task_ids:
                skipped += 1
                print(f"[{i}/{len(dataset)}] ⏭️  Skipped: {task_id} (already completed)")
                continue

            print(f"[{i}/{len(dataset)}] Processing: {task_id}")

            # Check if we have an existing (failed) prediction with metadata
            # This is crucial for Self-Refinement to load saved reflections
            if str(task_id) in existing_predictions_map:
                existing_pred = existing_predictions_map[str(task_id)]
                if existing_pred.meta_data:
                    if "meta_data" not in sample:
                        sample["meta_data"] = {}
                    # Update sample metadata with saved metadata
                    # This allows accessing saved 'reflection' or 'initial_prediction'
                    sample["meta_data"].update(existing_pred.meta_data)
                    print(
                        f"  ↻ Loaded metadata from previous attempt for task {task_id}"
                    )

            result = self.predict_sample(sample)

            if result:
                predictions.append(result)
                self.completed_task_ids.add(task_id)
                generated += 1
                # Auto-save every prediction
                self._save_predictions_incremental(predictions)

        # Final save
        if generated > 0:
            self.save_predictions(predictions)

        print(f"\n{'=' * 80}")
        print(f"Completed: {len(predictions)} total predictions")
        print(f"  - Existing: {len(existing_predictions)}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Generated: {generated}")
        print(f"{'=' * 80}")

        return predictions

    def run_evaluation(self, predictions: List[PredictionResult]) -> List[EvalResult]:
        """Run evaluation mode: evaluate existing predictions."""
        print(f"\n{'=' * 80}")
        print(f"Starting Evaluation: {self.dataset_name}")
        print(f"{'=' * 80}")

        print(f"→ Evaluating {len(predictions)} predictions")

        # Try to resume from existing evaluation results
        existing_results = []
        evaluated_task_ids = set()

        if self.resume:
            existing_results = self._load_latest_eval_results()
            if existing_results:
                evaluated_task_ids = {r.task_id for r in existing_results}
                print(
                    f"  📂 Resume: loaded {len(existing_results)} existing evaluations"
                )

        # Evaluate predictions (skip already evaluated ones)
        results = existing_results.copy()  # Start with existing results
        skipped = 0
        evaluated = 0

        for i, prediction in enumerate(predictions, 1):
            if prediction.task_id in evaluated_task_ids:
                skipped += 1
                print(
                    f"[{i}/{len(predictions)}] ⏭️  Skipped: {prediction.task_id} (already evaluated)"
                )
                continue

            if not prediction.predicted_step:
                skipped += 1
                print(
                    f"[{i}/{len(predictions)}] ⏭️  Skipped: {prediction.task_id} (empty prediction)"
                )
                continue

            print(f"[{i}/{len(predictions)}] Processing: {prediction.task_id}")
            result = self.evaluate_prediction(prediction)

            if result:
                results.append(result)
                evaluated_task_ids.add(result.task_id)
                evaluated += 1
                # Auto-save
                self._save_results_incremental(results)

        # Final save
        if evaluated > 0:
            self.save_results(results)

        print(f"\n{'=' * 80}")
        print(f"Completed: {len(results)} total evaluations")
        print(f"  - Existing: {len(existing_results)}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Evaluated: {evaluated}")
        print(f"{'=' * 80}")

        return results

    def run_pipeline(self, dataset: Optional[List[Dict]] = None) -> List[EvalResult]:
        """Run pipeline mode: inference + evaluation."""
        print(f"\n{'=' * 80}")
        print(f"Starting Pipeline: {self.dataset_name}")
        print(f"{'=' * 80}")

        # Load dataset if not provided
        if dataset is None:
            dataset = DatasetLoader.load_dataset(
                self.dataset_name, data_dir=self.data_dir
            )

        if self.max_samples:
            dataset = dataset[: self.max_samples]

        print(f"→ Processing {len(dataset)} samples (inference + evaluation)")

        # Try to resume from existing results
        existing_results = []
        evaluated_task_ids = set()

        if self.resume:
            existing_results = self._load_latest_eval_results()
            if existing_results:
                evaluated_task_ids = {r.task_id for r in existing_results}
                print(f"  📂 Resume: loaded {len(existing_results)} existing results")

        # Process samples (skip already evaluated ones)
        results = existing_results.copy()  # Start with existing results
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

            # First predict
            prediction = self.predict_sample(sample)
            if prediction:
                predictions.append(prediction)
                # Then evaluate
                result = self.evaluate_prediction(prediction)
                if result:
                    results.append(result)
                    evaluated_task_ids.add(result.task_id)
                    processed += 1
                    # Auto-save results
                    self._save_results_incremental(results)

        # Final save for both predictions and results
        if processed > 0:
            self.save_predictions(predictions)
            self.save_results(results)

        print(f"\n{'=' * 80}")
        print(f"Completed: {len(results)} total results")
        print(f"  - Existing: {len(existing_results)}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Processed: {processed}")
        print(f"{'=' * 80}")

        return results

    def evaluate_sample(self, sample: Dict) -> Optional[EvalResult]:
        """Evaluate a single sample (pipeline mode: predict + evaluate)."""
        # First predict
        prediction = self.predict_sample(sample)
        if not prediction:
            return None

        # Then evaluate
        return self.evaluate_prediction(prediction)

    def save_predictions(self, predictions: List[PredictionResult]) -> str:
        """Save predictions to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_predictions_{timestamp}.json"
        filepath = os.path.join(self.predictions_dir, filename)

        # Filter out empty predictions - only save those with actual content
        valid_predictions = [p for p in predictions if p.predicted_step]

        output = {
            "dataset": self.dataset_name,
            "test_model": self.test_model,
            "timestamp": timestamp,
            "total_predictions": len(valid_predictions),
            "predictions": [asdict(p) for p in valid_predictions],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(
            f"\n✓ Predictions saved to: {filepath} ({len(valid_predictions)} valid predictions)"
        )

        return filepath

    def _save_predictions_incremental(self, predictions: List[PredictionResult]) -> str:
        """Save predictions incrementally (for resume mode)."""
        # Use a consistent filename for resume - saved in resume/predictions/{test_model}/ folder
        filename = f"{self.dataset_name}_predictions_resume.json"
        filepath = os.path.join(self.prediction_resume_dir, filename)

        # Filter out empty predictions - only save those with actual content
        valid_predictions = [p for p in predictions if p.predicted_step]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "dataset": self.dataset_name,
            "test_model": self.test_model,
            "timestamp": timestamp,
            "total_predictions": len(valid_predictions),
            "predictions": [asdict(p) for p in valid_predictions],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        self.resume_filepath = filepath
        print(f"  💾 Auto-saved {len(valid_predictions)} predictions to: {filepath}")

        return filepath

    def _load_latest_predictions(self) -> List[PredictionResult]:
        """Load latest predictions for resume mode."""
        # Look for resume file in resume/predictions/{test_model}/ folder
        resume_file = os.path.join(
            self.prediction_resume_dir, f"{self.dataset_name}_predictions_resume.json"
        )
        if os.path.exists(resume_file):
            try:
                return self._load_predictions_from_file(resume_file)
            except Exception as e:
                print(f"  ⚠️  Error loading resume file: {e}")

        # Fall back to finding the latest prediction file
        prediction_files = []
        for file in os.listdir(self.predictions_dir):
            if file.startswith(self.dataset_name) and file.endswith(".json"):
                filepath = os.path.join(self.predictions_dir, file)
                prediction_files.append((filepath, os.path.getmtime(filepath)))

        if not prediction_files:
            return []

        # Get most recent file
        latest_file = max(prediction_files, key=lambda x: x[1])[0]

        try:
            return self._load_predictions_from_file(latest_file)
        except Exception as e:
            print(f"  ⚠️  Error loading predictions: {e}")
            return []

    def _load_predictions_from_file(self, filepath: str) -> List[PredictionResult]:
        """Load predictions from a specific file."""
        print(f"  📂 Loading predictions from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Update test_model from the prediction file metadata
        if "test_model" in data:
            if data["test_model"] != self.test_model:
                print(
                    f"  ⚠️  Warning: test_model mismatch. File: {data['test_model']}, Current: {self.test_model}"
                )

        # Deduplicate: keep FIRST entry for each task_id
        # This preserves empty predictions with saved reflections for retry
        seen_task_ids = set()
        predictions = []
        for pred_dict in data.get("predictions", []):
            try:
                pred = PredictionResult(**pred_dict)
                task_id_str = str(pred.task_id)
                if task_id_str not in seen_task_ids:
                    seen_task_ids.add(task_id_str)
                    predictions.append(pred)
            except Exception as e:
                print(f"  ⚠️  Error parsing prediction: {e}")

        return predictions

    def load_predictions(self, filepath: str) -> List[PredictionResult]:
        """Load predictions from file."""
        print(f"→ Loading predictions from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Deduplicate: keep LAST entry for each task_id (for evaluation, we want successful retries)
        predictions_map = {}
        for pred_dict in data["predictions"]:
            pred = PredictionResult(**pred_dict)
            # Keep the last entry for each task_id
            predictions_map[str(pred.task_id)] = pred

        predictions = list(predictions_map.values())

        # Update dataset name and test_model from loaded predictions file
        if predictions:
            self.dataset_name = predictions[0].dataset

        # Update test_model from the prediction file metadata
        if "test_model" in data:
            if data["test_model"] != self.test_model:
                print(
                    f"  ⚠️  Warning: test_model mismatch. File: {data['test_model']}, Current: {self.test_model}"
                )

        # Apply max_samples limit if specified
        if self.max_samples and len(predictions) > self.max_samples:
            original_count = len(predictions)
            predictions = predictions[: self.max_samples]
            print(
                f"✓ Loaded {original_count} predictions, limited to {len(predictions)} by max_samples={self.max_samples}"
            )
        else:
            print(f"✓ Loaded {len(predictions)} predictions")

        return predictions

    def save_results(self, results: List[EvalResult]) -> str:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.dataset_name}_results_{timestamp}.json"
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
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Results saved to: {filepath}")

        return filepath

    def _save_results_incremental(self, results: List[EvalResult]) -> str:
        """Save evaluation results incrementally (for resume mode)."""
        # Use a consistent filename for resume - saved in resume/{test_model}/ folder
        filename = f"{self.dataset_name}_results_resume.json"
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
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"  💾 Auto-saved {len(results)} evaluations to: {filepath}")

        return filepath

    def _load_latest_eval_results(self) -> List[EvalResult]:
        """Load latest evaluation results for resume mode."""
        # Look for resume file in resume/{test_model}/ folder
        resume_file = os.path.join(
            self.resume_dir, f"{self.dataset_name}_results_resume.json"
        )
        if os.path.exists(resume_file):
            try:
                return self._load_eval_results_from_file(resume_file)
            except Exception as e:
                print(f"  ⚠️  Error loading resume file: {e}")

        # Fall back to finding the latest result file
        result_files = []
        for file in os.listdir(self.evaluations_dir):
            if file.startswith(self.dataset_name) and file.endswith(".json"):
                filepath = os.path.join(self.evaluations_dir, file)
                result_files.append((filepath, os.path.getmtime(filepath)))

        if not result_files:
            return []

        # Get most recent file
        latest_file = max(result_files, key=lambda x: x[1])[0]

        try:
            return self._load_eval_results_from_file(latest_file)
        except Exception as e:
            print(f"  ⚠️  Error loading eval results: {e}")
            return []

    def _load_eval_results_from_file(self, filepath: str) -> List[EvalResult]:
        """Load evaluation results from a specific file."""
        print(f"  📂 Loading evaluation results from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        for result_dict in data.get("results", []):
            try:
                result = EvalResult(**result_dict)
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  Error parsing eval result: {e}")

        return results
