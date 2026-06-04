"""
Tool Broken Evaluator.

This module contains the ToolBroken evaluator that simulates a tool failure
and evaluates if the model can use a replacement tool.
"""

import json
import copy
from json_repair import repair_json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import random

from .next1_evaluator import Next1Evaluator
from ..models.data_models import PredictionResult, EvalResult
from ..utils.dataset_loader import get_base_dataset_type
from ..clients.proxy_client import ProxyClient


class ToolBrokenEvaluator(Next1Evaluator):
    """
    Evaluator for Tool-Broken subtask.

    This evaluator:
    1. Reads pre-processed data where the trajectory is already truncated and the last step is modified.
    2. Uses the replacement tool provided in the data.
    3. Evaluates if the model uses the replacement tool.
    """

    def __init__(self, *args, **kwargs):
        # Force prediction_mode to be tool_broken
        kwargs["prediction_mode"] = "next1"

        # Distractor configuration
        self.add_distractors = kwargs.pop("add_distractors", False)
        self.num_distractors = kwargs.pop("num_distractors", 3)

        super().__init__(*args, **kwargs)
        self.prediction_mode = "tool_broken"  # Override after init

        # Override data directory for tool broken mode
        self.data_dir = "data/AgentPlanningBench/online_tool_broken"

        # Override output directories to use tool_broken
        self._setup_directories()

        # Initialize tool generation client (Claude 4.5 as requested)
        # Still needed for distractors if enabled
        self.tool_gen_client = ProxyClient(model="anthropic/claude-sonnet-4.5")

    def _generate_distractor_tools(self, query: str, count: int = 3) -> List[Dict]:
        """
        Generate useless/distractor tools using the LLM.

        Args:
            query: The user query (to ensure distractors are unrelated)
            count: Number of distractors to generate

        Returns:
            List[Dict]: List of distractor tool definitions
        """
        prompt = f"""You are a helpful assistant that generates USELESS distractor tools for API testing.
        
User Query: "{query}"

Please generate {count} "distractor" tools that are:
1. COMPLETELY UNRELATED to the user query.
2. Plausible-looking (valid JSON structure with name, description, parameters).
3. Distinct from each other.

For example, if the query is about "weather", generate tools about "managing library books" or "baking recipes".
If the query is about "finance", generate tools about "identifying birds" or "playing music".

Return ONLY a JSON array of tool definitions.
"""
        try:
            response = self.tool_gen_client.generate(
                prompt, temperature=0.8, max_tokens=12800
            )

            # Extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()

            distractors = json.loads(json_str)
            if isinstance(distractors, list):
                return distractors[:count]
            return []
        except Exception as e:
            print(f"  ⚠️  Failed to generate distractor tools: {e}")
            return []

    def predict_sample(self, sample: Dict) -> Optional[PredictionResult]:
        """Generate prediction for a single sample with broken tool simulation."""
        try:
            task_id = sample["index"]
            query = sample["query"]

            # Use pre-processed fields
            trajectory = sample["trajectory"]  # This is already the modified prefix
            original_trajectory = sample.get("original_trajectory", "")
            tools = sample.get(
                "tools", []
            )  # This already contains the replacement tool
            meta_data = sample.get("meta_data", {})

            broken_tool_name = sample.get("broken_tool_name")
            replacement_tool = sample.get("replacement_tool")

            print(f"\n{'=' * 80}")
            print(f"Predicting (Tool-Broken): {task_id}")
            print(f"{'=' * 80}")

            if not broken_tool_name or not replacement_tool:
                print(
                    "⊘ Skipped: missing broken_tool_name or replacement_tool in sample"
                )
                return None

            replacement_tool_name = replacement_tool.get("name")
            if not replacement_tool_name:
                replacement_tool_name = replacement_tool.get("tool_name")
            if not replacement_tool_name and "function" in replacement_tool:
                replacement_tool_name = replacement_tool["function"].get("name")

            print(f"  Broken Tool: {broken_tool_name}")
            print(f"  Replacement Tool: {replacement_tool_name}")

            # Calculate ground truth and remaining steps from original_trajectory
            # We need to find where the prefix ends in the original trajectory
            # The prefix in 'trajectory' has a modified last step (error message).
            # The original trajectory has the actual tool output.

            # Logic:
            # 1. Parse original trajectory
            # 2. The prefix length (in steps) should be the same as the modified trajectory
            # 3. The next step in original trajectory is the ground truth

            try:
                # Parse modified trajectory to get length
                if isinstance(trajectory, str):
                    prefix_steps = json.loads(trajectory)
                else:
                    prefix_steps = trajectory
                prefix_len = len(prefix_steps)

                # Parse original trajectory
                if isinstance(original_trajectory, str):
                    original_steps = json.loads(original_trajectory)
                else:
                    original_steps = original_trajectory

                if len(original_steps) <= prefix_len:
                    print(
                        "⊘ Skipped: original trajectory shorter than or equal to prefix"
                    )
                    return None

                reference_next = json.dumps(
                    original_steps[prefix_len], ensure_ascii=False
                )
                reference_remaining = json.dumps(
                    original_steps[prefix_len + 1 :], ensure_ascii=False
                )

            except Exception as e:
                print(f"⊘ Skipped: error parsing trajectories: {e}")
                return None

            # Add Distractor Tools (if enabled)
            modified_tools = tools.copy()
            if self.add_distractors:
                print(f"  🤡 Generating {self.num_distractors} distractor tools...")
                distractors = self._generate_distractor_tools(
                    query, self.num_distractors
                )
                if distractors:
                    print(
                        f"  ✨ Added {len(distractors)} distractors: {[t.get('name', t.get('tool_name', 'unknown')) for t in distractors]}"
                    )
                    modified_tools.extend(distractors)
                    # Shuffle tools to avoid position bias
                    random.shuffle(modified_tools)

            # 5. Generate prediction
            print("→ Generating prediction with replacement tool...")
            pred_prompt = self.generate_prediction_prompt(
                sample, trajectory
            )  # Use modified trajectory

            # Get files (standard logic)
            files = None
            base_type = get_base_dataset_type(self.dataset_name)
            if base_type in ["gaia", "gta"]:
                all_files = self._get_file_paths_with_types(sample)
                if all_files:
                    files = all_files

            predicted_next = self.test_client.generate(
                pred_prompt,
                max_tokens=self.test_max_tokens,
                files=files,
                temperature=0.0,
            )

            print(f"✓ Predicted: {predicted_next[:100]}...")

            # Store metadata for evaluation
            meta_data["broken_tool_name"] = broken_tool_name
            meta_data["replacement_tool_name"] = replacement_tool_name

            result = PredictionResult(
                task_id=task_id,
                dataset=self.dataset_name,
                query=query,
                trajectory_prefix=trajectory,
                predicted_step=predicted_next,
                ground_truth_step=reference_next,
                reference_remaining_steps=reference_remaining,
                tools=modified_tools,
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
        """Evaluate prediction using LLM-based judge for tool substitution with 5-classification."""
        try:
            print(f"\n{'=' * 80}")
            print(f"Evaluating (Tool-Broken): {prediction.task_id}")
            print(f"{'=' * 80}")

            replacement_tool_name = prediction.meta_data.get("replacement_tool_name")
            broken_tool_name = prediction.meta_data.get("broken_tool_name")

            if not replacement_tool_name:
                print("✗ Missing replacement tool name in metadata")
                return None

            # Find replacement tool definition
            replacement_tool_def = None
            for tool in prediction.tools:
                t_name = tool.get("name")
                if not t_name:
                    t_name = tool.get("tool_name")
                if not t_name and "function" in tool:
                    t_name = tool["function"].get("name")

                if t_name == replacement_tool_name:
                    replacement_tool_def = tool
                    break

            if not replacement_tool_def:
                print("✗ Missing replacement tool definition")
                return None

            # Build evaluation prompt
            from ..prompts.evaluation_prompts import EvaluationPrompts

            # Truncate predicted_step to 30000 chars for evaluation prompt
            eval_predicted_step = prediction.predicted_step
            if eval_predicted_step and len(eval_predicted_step) > 30000:
                print(
                    f"✂️ Truncating prediction for evaluation from {len(eval_predicted_step)} to 30000 chars"
                )
                eval_predicted_step = eval_predicted_step[:30000]

            prompt = EvaluationPrompts.build_evaluation_prompt_tool_broken(
                dataset_name=self.dataset_name,
                query=prediction.query,
                trajectory_prefix=prediction.trajectory_prefix,
                broken_tool_name=broken_tool_name,
                replacement_tool=replacement_tool_def,
                predicted_step=eval_predicted_step,  # Use truncated version
                tools=prediction.tools,
            )

            # Generate evaluation
            print("→ Generating evaluation with LLM judge (5-classification)...")
            eval_response = self.eval_client.generate(
                prompt, temperature=0.0, max_tokens=1024
            )

            # Parse response
            try:
                if "```json" in eval_response:
                    json_str = eval_response.split("```json")[1].split("```")[0].strip()
                elif "```" in eval_response:
                    json_str = eval_response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = eval_response.strip()

                # Try standard JSON parsing first
                try:
                    eval_json = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Use json-repair to fix broken JSON
                    print(f"  ⚠️  JSON decode error: {e}")
                    print(f"  🔧 Attempting to repair JSON...")
                    try:
                        repaired_json = repair_json(json_str)
                        eval_json = json.loads(repaired_json)
                        print(f"  ✓ JSON repair successful")
                    except Exception as repair_error:
                        print(f"  ✗ JSON repair failed: {repair_error}")
                        # save the failed result
                        with open("failed_eval_results.json", "a") as f:
                            f.write(json_str + "\n")
                        raise e

                # Parse 5-classification result
                category = int(eval_json.get("category", 5))
                category_name = eval_json.get("category_name", "other")
                is_correct = bool(eval_json.get("is_correct", False))
                reason = eval_json.get("reason", "No reason provided")

                # Category name mapping for consistency
                category_names = {
                    1: "replacement",
                    2: "alternative",
                    3: "retry",
                    4: "refusal",
                    5: "other",
                }

                # Validate and normalize category
                if category not in category_names:
                    category = 5
                category_name = category_names[category]

                # Ensure is_correct is consistent with category
                if category in [1, 2]:
                    is_correct = True
                else:
                    is_correct = False

            except Exception as e:
                print(f"✗ Failed to parse evaluation response: {e}")
                print(f"Response (first 500 chars): {eval_response[:500]}")
                # Skip this sample if parsing fails (return None)
                print("  ⊘ Skipping this sample due to parsing failure")
                return None

            print(f"✓ Category: {category} ({category_name})")
            print(f"  Correct: {is_correct}")
            print(f"  Reason: {reason[:200]}...")

            # Store classification info in error_types
            error_types = {
                "category": category,
                "category_name": category_name,
                "replacement": category == 1,
                "alternative": category == 2,
                "retry": category == 3,
                "refusal": category == 4,
                "other": category == 5,
            }

            return EvalResult(
                task_id=prediction.task_id,
                dataset=prediction.dataset,
                query=prediction.query,
                predicted_step=prediction.predicted_step,
                ground_truth_step=prediction.ground_truth_step,
                score=1.0 if is_correct else 0.0,
                is_correct=is_correct,
                error_types=error_types,
                reasoning=reason,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _load_eval_results_from_file(self, filepath: str) -> List[EvalResult]:
        """
        Load evaluation results from a specific file.

        Override parent method to filter out failed parsing results
        (reasoning contains 'Failed to parse LLM evaluation').
        These will be re-evaluated when running evaluation.
        """
        print(f"  📂 Loading evaluation results from: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []
        filtered_count = 0

        for result_dict in data.get("results", []):
            try:
                # Check if this result has a parsing failure that needs re-evaluation
                reasoning = result_dict.get("reasoning", "")
                if "Failed to parse LLM evaluation" in reasoning:
                    filtered_count += 1
                    print(
                        f"  🔄 Filtering failed result for re-evaluation: {result_dict.get('task_id', 'unknown')}"
                    )
                    continue

                result = EvalResult(**result_dict)
                results.append(result)
            except Exception as e:
                print(f"  ⚠️  Error parsing eval result: {e}")

        if filtered_count > 0:
            print(
                f"  ℹ️  Filtered {filtered_count} failed results - will be re-evaluated"
            )

        return results
