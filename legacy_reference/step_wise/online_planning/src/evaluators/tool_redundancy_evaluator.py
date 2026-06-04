"""
Tool Redundancy Evaluator.

This module contains the ToolRedundancy evaluator that injects redundant (distractor) tools
and evaluates if the model correctly avoids using them.
"""

import json
import random
import json_repair
from typing import Dict, List, Optional
from datetime import datetime

from .next1_evaluator import Next1Evaluator
from ..models.data_models import PredictionResult, EvalResult
from ..prompts.evaluation_prompts import EvaluationPrompts


class ToolRedundancyEvaluator(Next1Evaluator):
    """
    Evaluator for Tool-Redundancy subtask.

    This evaluator:
    1. Injects redundant tools (from metadata) into the available tools list during prediction.
    2. Evaluates if the model uses any of these redundant tools (which is an error).
    """

    def __init__(self, *args, **kwargs):
        # Force prediction_mode to be next1 (base behavior)
        kwargs["prediction_mode"] = "next1"

        # Configuration
        self.num_redundant_tools = int(kwargs.pop("num_redundant_tools", 4))

        super().__init__(*args, **kwargs)

        # Set prediction_mode based on num_redundant_tools
        # Default (4) uses 'tool_redundancy', others use 'tool_redundancy_N'
        if self.num_redundant_tools == 4:
            self.prediction_mode = "tool_redundancy"
        else:
            self.prediction_mode = f"tool_redundancy_{self.num_redundant_tools}"

        # Override output directories to use tool_redundancy
        self._setup_directories()

        # Override data directory
        self.data_dir = "data/AgentPlanningBench/online_tool_redundancy"

    def _extract_tool_names(self, prediction_text: str) -> List[str]:
        """
        Extract tool names from prediction text.
        Supports JSON format (Gaia, GTA, ToolBench, Skywork) and FrameThinker format.
        Uses json_repair to handle malformed JSON.
        """
        tool_names = []
        try:
            # Try to parse as JSON first
            # Handle markdown code blocks if present
            clean_text = prediction_text.strip()
            if "```json" in clean_text:
                clean_text = clean_text.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_text:
                clean_text = clean_text.split("```")[1].split("```")[0].strip()

            # Use json_repair for robust parsing
            data = json_repair.loads(clean_text)

            # Check for standard tool_calls format
            if isinstance(data, dict):
                # Standard format: {"tool_calls": [{"function": {"name": ...}}]}
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    for tc in data["tool_calls"]:
                        if isinstance(tc, dict):
                            if "function" in tc and isinstance(tc["function"], dict):
                                name = tc["function"].get("name")
                                if name:
                                    tool_names.append(name)
                            # Direct name in tool call (some formats)
                            elif "name" in tc:
                                tool_names.append(tc["name"])

                # FrameThinker format: {"content": "<action>call(param)</action>"}
                # or OpenCUA: {"action": "description"} - OpenCUA usually doesn't use function names directly in action string easily
                # But let's check for FrameThinker style function calls in content or action

                # Check for direct function call in 'action' field (if it looks like code)
                if "action" in data and isinstance(data["action"], str):
                    action_str = data["action"]
                    # Simple heuristic: name(
                    if "(" in action_str:
                        possible_name = action_str.split("(")[0].strip()
                        if possible_name.isidentifier():
                            tool_names.append(possible_name)

                # Check content for <action> tags
                if "content" in data and isinstance(data["content"], str):
                    content = data["content"]
                    if "<action>" in content and "</action>" in content:
                        action_part = (
                            content.split("<action>")[1].split("</action>")[0].strip()
                        )
                        if "(" in action_part:
                            possible_name = action_part.split("(")[0].strip()
                            if possible_name.isidentifier():
                                tool_names.append(possible_name)

        except json.JSONDecodeError:
            # Fallback: regex or string search if needed, but for now rely on JSON
            pass
        except Exception as e:
            print(f"  Warning: Error extracting tool names: {e}")

        return tool_names

    def predict_sample(self, sample: Dict) -> Optional[PredictionResult]:
        """Generate prediction for a single sample with redundant tool injection."""
        try:
            task_id = sample["index"]
            query = sample["query"]
            trajectory = sample["trajectory"]
            tools = sample.get("tools", [])
            meta_data = sample.get("meta_data", {})
            system_prompt = sample.get("system_prompt", None)

            # Extract redundant tools from sample
            # They should have been pre-generated by scripts/construct_tool_redundancy_data.py
            # Check meta_data first as per user report, then fallback to top-level
            all_redundant_tools = meta_data.get("redundant_tools", [])
            if not all_redundant_tools:
                all_redundant_tools = sample.get("redundant_tools", [])

            if not all_redundant_tools:
                print(f"  ⚠️  No redundant tools found in sample {task_id}")
                # Fallback: proceed without redundant tools (or could skip)
                selected_redundant_tools = []
            else:
                # Select specified number of redundant tools
                # If we have fewer than requested, take all
                count = min(len(all_redundant_tools), self.num_redundant_tools)
                selected_redundant_tools = all_redundant_tools[:count]
                print(f"  🤡 Injected {len(selected_redundant_tools)} redundant tools")

            # Combine original tools with redundant tools
            # Make a copy to avoid modifying original sample
            modified_tools = tools.copy()
            modified_tools.extend(selected_redundant_tools)

            # Shuffle tools to avoid position bias
            # (Optional: keep original order and append? Or shuffle all? Shuffle all is more realistic)
            random.shuffle(modified_tools)

            # Create a temporary sample for prediction generation
            temp_sample = sample.copy()
            temp_sample["tools"] = modified_tools

            # Truncate trajectory (standard Next1 logic)
            prefix, reference_next, reference_remaining = self.truncate_trajectory(
                trajectory, system_prompt
            )

            if not reference_next:
                print("⊘ Skipped: trajectory too short")
                return None

            print(f"\n{'=' * 80}")
            print(f"Predicting (Tool-Redundancy): {task_id}")
            print(f"{'=' * 80}")

            # Generate prediction
            print("→ Generating prediction with redundant tools...")
            pred_prompt = self.generate_prediction_prompt(temp_sample, prefix)

            # Get files (standard logic)
            files = None
            # (Assuming standard file handling from base class is sufficient,
            # but need to call _get_file_paths_with_types if needed.
            # Next1Evaluator.predict_sample does this logic, but we are overriding it.
            # So we need to copy the file handling logic.)

            from ..utils.dataset_loader import get_base_dataset_type

            base_type = get_base_dataset_type(self.dataset_name)
            if base_type in ["framethinker", "gaia", "gta", "opencua"]:
                all_files = self._get_file_paths_with_types(sample)
                if all_files:
                    if base_type == "opencua":
                        files, _ = self._truncate_files_for_opencua(
                            all_files, trajectory
                        )
                    else:
                        files = all_files

            predicted_next = self.test_client.generate(
                pred_prompt,
                max_tokens=self.test_max_tokens,
                files=files,
                temperature=0.0,
            )

            print(f"✓ Predicted: {predicted_next[:100]}...")

            # Store metadata about redundant tools for evaluation
            # We store the NAMES of the redundant tools to easily check if they were used
            redundant_tool_names = []
            for t in selected_redundant_tools:
                name = t.get("name")
                if not name:
                    name = t.get("tool_name")
                if not name and "function" in t:
                    name = t["function"].get("name")
                if name:
                    redundant_tool_names.append(name)

            meta_data["redundant_tool_names"] = redundant_tool_names
            meta_data["redundant_tools"] = (
                selected_redundant_tools  # Store full defs too just in case
            )

            result = PredictionResult(
                task_id=task_id,
                dataset=self.dataset_name,
                query=query,
                trajectory_prefix=prefix,
                predicted_step=predicted_next,
                ground_truth_step=reference_next,
                reference_remaining_steps=reference_remaining,
                tools=modified_tools,  # Save the combined tool list
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
        """Evaluate prediction checking for redundant tool usage."""
        try:
            print(f"\n{'=' * 80}")
            print(f"Evaluating (Tool-Redundancy): {prediction.task_id}")
            print(f"{'=' * 80}")

            redundant_tool_names = prediction.meta_data.get("redundant_tool_names", [])

            if not redundant_tool_names:
                print("⚠️ No redundant tools recorded in metadata")

            # Build evaluation prompt
            # We need to pass the list of redundant tools to the prompt
            # so the evaluator knows which ones are forbidden.

            # Truncate predicted_step to 30000 chars for evaluation prompt
            eval_predicted_step = prediction.predicted_step
            if eval_predicted_step and len(eval_predicted_step) > 30000:
                print(
                    f"✂️ Truncating prediction for evaluation from {len(eval_predicted_step)} to 30000 chars"
                )
                eval_predicted_step = eval_predicted_step[:30000]

            # Get files for context if needed and compute prefix_file_count for opencua
            files = None
            prefix_file_count = 0
            from ..utils.dataset_loader import get_base_dataset_type

            base_type = get_base_dataset_type(self.dataset_name)
            if base_type in ["framethinker", "gaia", "gta", "opencua"]:
                # Reconstruct sample for file loading
                sample = {
                    "index": prediction.task_id,
                    "meta_data": prediction.meta_data,
                    "trajectory": prediction.trajectory_prefix,
                }
                all_files = self._get_file_paths_with_types(sample)
                if all_files:
                    if base_type == "opencua":
                        files, prefix_file_count = self._truncate_files_for_opencua(
                            all_files, prediction.trajectory_prefix
                        )
                    else:
                        files = all_files

            prompt = EvaluationPrompts.build_evaluation_prompt_tool_redundancy(
                dataset_name=self.dataset_name,
                query=prediction.query,
                trajectory_prefix=prediction.trajectory_prefix,
                predicted_step=eval_predicted_step,  # Use truncated version
                ground_truth_step=prediction.ground_truth_step,
                reference_remaining_steps=prediction.reference_remaining_steps,
                tools=prediction.tools,
                redundant_tool_names=redundant_tool_names,
                prefix_file_count=prefix_file_count,
                test_model=self.test_model,  # Pass test_model for framethinker format selection
            )

            # Generate evaluation
            print("→ Generating evaluation with LLM judge...")
            eval_response = self.eval_client.generate(
                prompt, max_tokens=self.eval_max_tokens, files=files, temperature=0.0
            )

            # Parse response
            # We use the standard parsing from Next1Evaluator since the output format should be similar (JSON)
            eval_data = self._parse_evaluation(eval_response)

            if not eval_data:
                print("✗ Failed to parse evaluation")
                return None

            # Check if E5 (Tool Use Error) is triggered
            # The prompt should be designed to trigger E5 if a redundant tool is used

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
                    "E5": eval_data.get(
                        "E5_TOOL_USE_ERROR", False
                    ),  # This is the key one for redundancy
                    "E6": eval_data.get("E6_HALLUCINATION_ERROR", False),
                },
                reasoning=eval_data["reasoning"],
                timestamp=datetime.now().isoformat(),
                used_redundant_tools=False,  # Default, updated below
            )

            # Rule-based check for redundant tool usage
            extracted_tools = self._extract_tool_names(prediction.predicted_step)
            used_redundant = False
            used_redundant_list = []

            for tool_name in extracted_tools:
                if tool_name in redundant_tool_names:
                    used_redundant = True
                    used_redundant_list.append(tool_name)

            result.used_redundant_tools = used_redundant

            print(f"✓ Score: {result.score}, Correct: {result.is_correct}")
            if result.error_types["E5"]:
                print("  🚨 TOOL USE ERROR DETECTED (Likely used redundant tool)")

            if used_redundant:
                print(
                    f"  🤡 RULE-BASED DETECTED: Used redundant tool(s): {used_redundant_list}"
                )
                # Optional: Force E5 to be true if rule-based check catches it?
                # For now, just logging it as a separate field as requested.

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return None
