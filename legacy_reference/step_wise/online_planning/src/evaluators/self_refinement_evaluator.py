from math import trunc
import json
import os
import glob
from typing import Dict, Optional, List, Any
from datetime import datetime

from .next1_evaluator import Next1Evaluator
from .base_evaluator import BaseOnlinePlanningEvaluator
from ..models.data_models import PredictionResult
from ..utils.dataset_loader import get_base_dataset_type


class SelfRefinementEvaluator(Next1Evaluator):
    """Evaluator for self-refinement strategy (reflect then predict)."""

    def __init__(self, reflection_model: Optional[str] = None, *args, **kwargs):
        # Default to self_refinement if not provided, but allow override (e.g. self_refinement_claude)
        if "prediction_mode" not in kwargs:
            kwargs["prediction_mode"] = "self_refinement"
        # Bypass Next1Evaluator.__init__ because it forces prediction_mode to next1
        BaseOnlinePlanningEvaluator.__init__(self, *args, **kwargs)

        # Cache for next1 predictions
        self.next1_predictions: Optional[Dict[str, str]] = None

        # Initialize reflection client
        if reflection_model:
            print(f"✨ Using separate reflection model: {reflection_model}")
            from ..clients.proxy_client import ProxyClient

            self.reflection_client = ProxyClient(model=reflection_model)
        else:
            self.reflection_client = self.test_client

    def _load_next1_predictions(self):
        """Load next1 predictions for the current dataset and model."""
        if self.next1_predictions is not None:
            return

        # Construct path to next1 predictions
        # Structure: eval_results/predictions/{model}/next1/{dataset}_predictions_*.json
        test_model_name = (
            self.test_model.replace("/", "_").replace(":", "_").replace(".", "_")
        )
        next1_dir = os.path.join(
            self.base_output_dir, "predictions", test_model_name, "next1"
        )

        pattern = os.path.join(next1_dir, f"{self.dataset_name}_predictions_*.json")
        files = glob.glob(pattern)

        if not files:
            print(f"⚠️  No next1 predictions found in {next1_dir}")
            self.next1_predictions = {}
            return

        # Sort by timestamp (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = files[0]
        print(f"Loading next1 predictions from: {latest_file}")

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("predictions", data.get("results", []))

            self.next1_predictions = {}
            for item in items:
                # Store by task_id (as string for consistency)
                task_id = str(item.get("task_id", item.get("index", "")))
                if task_id:
                    self.next1_predictions[task_id] = item.get("predicted_step", "")

            print(f"✓ Loaded {len(self.next1_predictions)} next1 predictions")

        except Exception as e:
            print(f"✗ Error loading next1 predictions: {e}")
            self.next1_predictions = {}

    def _format_tools(self, sample: Dict) -> str:
        """Format tools for prompt."""
        tools = sample.get("tools", [])
        tools_info = ""
        if tools:
            tools_info = "\n**Available Tools:**\n"
            for tool in tools:
                if isinstance(tool, dict):
                    if "function" in tool and isinstance(tool["function"], dict):
                        func = tool["function"]
                        name = func.get("name", "unknown")
                        description = func.get("description", "")
                        params = func.get("parameters", {})
                        tools_info += f"- **{name}**: {description}\n"
                        if params:
                            tools_info += f"  Parameters: {json.dumps(params, ensure_ascii=False)}\n"
                    elif "name" in tool and "example" in tool:
                        name = tool.get("name", "unknown")
                        description = tool.get("description", "")
                        example = tool.get("example", "")
                        tools_info += f"- **{name}**: {description}\n"
                        if example:
                            tools_info += f"  Example: {example}\n"
                    else:
                        tools_info += f"- {json.dumps(tool, ensure_ascii=False)}\n"
                else:
                    tools_info += f"- {tool}\n"
        else:
            tools_info = (
                "\n**Note:** Available tools are defined in the system prompt.\n"
            )
        return tools_info

    def generate_initial_prediction_prompt(self, sample: Dict, prefix: str) -> str:
        """Generate prompt for initial prediction step."""
        # Use the standard Next1 prompt logic
        return super().generate_prediction_prompt(sample, prefix)

    def generate_reflection_prompt(
        self, sample: Dict, prefix: str, initial_prediction: str
    ) -> str:
        """Generate prompt for reflection step."""
        query = sample["query"]
        tools_info = self._format_tools(sample)

        if (
            self.prediction_mode == "refinement_without_judge"
            or self.prediction_mode == "refinement_without_judge_claude"
        ):
            prompt = f"""You are an expert planning agent.

**Task:** {query}
{tools_info}

**Steps Completed So Far:**
{prefix}

**Your Initial Prediction:**
{initial_prediction}

**Your Task:**
Perform a self-reflection on your initial prediction.
Think carefully about whether this step is the best possible action to take next.
Consider the user's goal, the current state, and the available tools.
Analyze if there are any potential issues or better alternatives.

Provide a detailed critique and reasoning. Do NOT output the final JSON action yet, just your reflection.
"""
        else:
            prompt = f"""You are an expert planning agent.

**Task:** {query}
{tools_info}

**Steps Completed So Far:**
{prefix}

**Your Initial Prediction:**
{initial_prediction}

**Your Task:**
Perform a self-reflection on your initial prediction. Critically evaluate it against the following 6 error types:

1. **E1_GOAL_MISALIGNMENT**: Does the step align with the user's query?
2.  **E2_PREMATURE_CONCLUSION**: Is the task actually finished? (Check if tool calls are missing when needed)
3.  **E3_CONSTRAINT_VIOLATION**: Does it violate any format or negative constraints? (Check for required fields like `role`, `thought`, `tool_calls`)
4.  **E4_LOGIC_ERROR**: Is there a logical flaw in the reasoning? (Check for prerequisites, redundancy, circular reasoning)
5.  **E5_TOOL_USE_ERROR**: Are tool arguments correct and valid according to the tool specification?
6.  **E6_HALLUCINATION_ERROR**: Is the step based on real information? (Check for non-existent tools or data)

**Reflection Instructions:**
1. Analyze if the predicted step is the most optimal next action.
2. Check for any potential errors based on the 6 criteria above.
3. Consider if there are better alternatives.
4. Determine if the initial prediction should be kept, modified, or completely changed.

**CRITICAL WARNINGS:**
- **DO NOT SKIP STEPS:** If the initial prediction is a necessary prerequisite (e.g., "Select file") for a future step (e.g., "Open file"), DO NOT reject it just because it doesn't finish the task. You must perform steps in order.
- **DO NOT HALLUCINATE PREREQUISITES:** If the initial prediction directly addresses the user's request, DO NOT invent missing "setup" steps unless they are strictly required by the system.
- **DO NOT OVER-CORRECT:** If the initial prediction is logical and valid, endorse it. Do not find faults where there are none.

Provide a detailed critique and reasoning. Do NOT output the final JSON action yet, just your reflection.
"""
        return prompt

    def generate_prediction_prompt(
        self, sample: Dict, prefix: str, initial_prediction: str, reflection: str
    ) -> str:
        """Generate prompt for final prediction step."""
        query = sample["query"]
        tools_info = self._format_tools(sample)

        # Get base dataset type for format requirements
        base_type = get_base_dataset_type(self.dataset_name)

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
            # FrameThinker format (dialog format with think: and action: prefixes)
            format_instruction = """
IMPORTANT: You must output your prediction in the following JSON format:
{
    "role": "assistant",
    "content": "think: your reasoning process here\\n\\naction: your actual action/tool call here"
}
Note: The content field MUST contain both `think:` and `action:` prefixes.
"""
        elif base_type == "opencua":
            # OpenCUA format
            format_instruction = """
**REQUIRED OUTPUT FORMAT:**
Your response MUST follow this exact JSON structure:
```json
{
    "role": "assistant",
    "content": "Observation: [describe what you see on screen]\\n\\nThought: [your reasoning about the next action]",
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
- Then add "\\n\\nThought: " followed by your reasoning about what action to take next and why
- Be specific about UI elements, their locations, and the current state

**Action Field Guidelines:**
1. Write ONE complete sentence describing the specific action to perform
2. Be specific about UI elements (buttons, text boxes, icons, menus, etc.)
3. Include visual details (location, appearance, labels) to identify the target element
4. State the purpose or outcome of the action
5. Do NOT use function call syntax like "click(element)" - use natural language descriptions
6. Match the detail level and style of the previous steps
"""
        else:
            # Default/unknown format
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

**Your Initial Prediction:**
{initial_prediction}

**Your Reflection:**
{reflection}

**Your Task:**
Based on your initial prediction and reflection, provide the FINAL, REFINED NEXT SINGLE STEP.
You may stick to your initial prediction if it was correct, or modify it based on your reflection.
**CRITICAL WARNINGS - READ CAREFULLY:**
1. **DO NOT SKIP STEPS:** You are evaluating the **IMMEDIATE NEXT STEP**. Do NOT replace the current necessary step with a future step. For example, if you need to generate a report *before* sending it, you MUST generate it first. Do not jump to 'sending' just because it's the final goal. **Keep the initial prediction** if it is a valid prerequisite.
2. **DO NOT OVER-CORRECT:** Do not invent missing requirements or 'nitpick' the initial prediction. If it satisfies the core user request, ENDORSE IT. Do not add complexity (e.g., accounting for minor variables like 'tomato protein') unless explicitly required.
3. **VALID JSON FORMAT:** Ensure your final JSON is valid. Escape all newlines (`\n`) inside string values. Do not output raw newlines inside strings.

{format_instruction}
{json_only_instruction}
"""
        return prompt

    def predict_sample(self, sample: Dict) -> Optional[PredictionResult]:
        """Generate prediction with self-refinement."""
        try:
            task_id = sample["index"]
            query = sample["query"]
            trajectory = sample["trajectory"]
            tools = sample.get("tools", [])
            meta_data = sample.get("meta_data", {})
            system_prompt = sample.get("system_prompt", None)

            print(f"\n{'=' * 80}")
            print(f"Predicting (Self-Refinement): {task_id}")
            print(f"{'=' * 80}")

            # Get files with types (for datasets that need them)
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

            # Step 1: Initial Prediction
            # Try to load from next1 predictions first
            if self.next1_predictions is None:
                self._load_next1_predictions()

            initial_prediction = None
            # Check if we have a pre-computed prediction
            # Try both string and original type of task_id
            task_id_str = str(task_id)
            if self.next1_predictions and task_id_str in self.next1_predictions:
                initial_prediction = self.next1_predictions[task_id_str]
                print(
                    f"✓ Loaded Initial Prediction (Next1): {initial_prediction[:100]}..."
                )

            # Fallback to generation if not found
            if not initial_prediction:
                print("→ Generating initial prediction (Fallback)...")
                initial_prompt = self.generate_initial_prediction_prompt(sample, prefix)
                initial_prediction = self.test_client.generate(
                    initial_prompt,
                    max_tokens=self.test_max_tokens,
                    files=files,
                    temperature=0.0,
                )
                print(
                    f"✓ Initial Prediction (Generated): {initial_prediction[:100]}..."
                )

            truncate_number = 3000
            # Truncate initial prediction to 3000 characters
            if initial_prediction and len(initial_prediction) > truncate_number:
                print(
                    f"✂️ Truncating initial prediction from {len(initial_prediction)} to {truncate_number} characters"
                )
                initial_prediction = initial_prediction[:truncate_number]

            # Step 2: Reflection
            reflection = meta_data.get("reflection")
            if reflection:
                print(f"✓ Loaded Saved Reflection: {reflection[:100]}...")
            else:
                print("→ Generating reflection...")
                reflection_prompt = self.generate_reflection_prompt(
                    sample, prefix, initial_prediction
                )
                reflection = self.reflection_client.generate(
                    reflection_prompt,
                    max_tokens=self.test_max_tokens,
                    files=files,
                    temperature=0.0,
                )
                print(f"✓ Reflection: {reflection[:100]}...")

            # Step 3: Final Prediction
            print("→ Generating refined prediction...")
            try:
                pred_prompt = self.generate_prediction_prompt(
                    sample, prefix, initial_prediction, reflection
                )
                predicted_next = self.test_client.generate(
                    pred_prompt,
                    max_tokens=self.test_max_tokens,
                    files=files,
                    temperature=0.0,
                )
                print(f"✓ Final Predicted: {predicted_next[:100]}...")
            except Exception as e:
                print(f"✗ Error generating refined prediction: {e}")
                print("⚠️  Saving result with empty prediction to persist reflection.")
                predicted_next = ""

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

            # Store initial prediction and reflection in meta_data
            result.meta_data["initial_prediction"] = initial_prediction
            result.meta_data["reflection"] = reflection

            return result

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return None
