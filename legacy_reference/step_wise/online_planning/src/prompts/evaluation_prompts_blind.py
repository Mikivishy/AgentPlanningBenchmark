import json
from typing import List, Dict

from .evaluation_prompts import EvaluationPrompts


class EvaluationPromptsBlind(EvaluationPrompts):
    """Dataset-specific evaluation prompts with error classification (Blind version)."""

    @staticmethod
    def build_evaluation_prompt(
        dataset_name: str,
        query: str,
        trajectory_prefix: str,
        predicted_step: str,
        reference_next_step: str,
        reference_remaining_steps: str,
        tools: List,  # Can be List[Dict] or List[str]
        prefix_file_count: int = 0,  # For opencua: number of images test model saw
        test_model: str = None,  # For framethinker: determine format based on model type
    ) -> str:
        """Build complete evaluation prompt (Blind version - no reference).

        Args:
            test_model: Name of the test model being evaluated (used to determine format for framethinker)
        """

        error_defs = EvaluationPrompts.get_error_definitions(dataset_name, test_model)
        scoring = EvaluationPrompts.get_scoring_rubric()

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
                        # Skywork format: extract from nested function object
                        func = tool["function"]
                        name = func.get("name", "unknown")
                        # Combine description and parameters as JSON string
                        desc_dict = {
                            "description": func.get("description", ""),
                            "parameters": func.get("parameters", {}),
                        }
                        desc = json.dumps(desc_dict, ensure_ascii=False)
                        tools_info += f"- {name}: {desc}\n"
                    # Check for framethinker format: {"name": ..., "description": ..., "example": ...}
                    elif "name" in tool and "example" in tool:
                        # Framethinker format: combine description and example
                        name = tool.get("name", "unknown")
                        desc = tool.get("description", "")
                        example = tool.get("example", "")
                        # Merge description and example
                        if example:
                            combined_desc = (
                                f"{desc} Example: {example}"
                                if desc
                                else f"Example: {example}"
                            )
                        else:
                            combined_desc = desc if desc else "No description"
                        tools_info += f"- {name}: {combined_desc}\n"
                    else:
                        # Standard format: {"tool_name": ..., "description": ...}
                        name = tool.get("tool_name", "unknown")
                        desc = tool.get("description", "No description")
                        tools_info += f"- {name}: {desc}\n"
                else:
                    # For string format (e.g., opencua)
                    tools_info += f"- {tool}\n"
        else:
            # If tools list is empty, tool information is already in system_prompt
            tools_info = (
                "\n**Note:** Available tools are defined in the system prompt.\n"
            )

        # For opencua, add information about image context
        image_context_info = ""
        if prefix_file_count > 0:
            image_context_info = f"""

**IMPORTANT - Image Context for OpenCUA:**
- The test model was shown the FIRST {prefix_file_count} images/screenshots (corresponding to the trajectory prefix)
- You are seeing ALL images (both prefix and remaining) to understand the full context
- Images {prefix_file_count + 1} onwards were NOT visible to the test model when making its prediction
- When evaluating, consider: Did the model make a reasonable prediction based on ONLY the first {prefix_file_count} images?
"""

        prompt = f"""You are evaluating a planning model's next step prediction.

**Dataset:** {dataset_name}

**User Query:**
{query}

**Trajectory So Far:**
{trajectory_prefix}
{tools_info}
{image_context_info}

**Predicted Next Step:**
{predicted_step}

{error_defs}

{scoring}

**Your Task:**
Evaluate the SINGLE NEXT STEP prediction with EXTREME STRICTNESS.

**CRITICAL EVALUATION REQUIREMENTS:**
1. **POTENTIAL ERROR = ERROR:** If a step *might* fail or has *potential* risks, it is an ERROR.
2. **STRICTEST INTERPRETATION:** If there is any doubt, mark it as an error.
3. **IGNORE INTENT:** Do not evaluate "planning ability" or "intent". Evaluate the strict correctness of the step.
4. **OPTIMALITY:** Suboptimal steps should be considered errors if they introduce unnecessary risk or inefficiency.
5. **SAFETY FIRST:** Any ambiguity is an error.
6. **NO LENIENCY:** Do not be lenient. Be a harsh critic.

**CRITICAL FOR E2_PREMATURE_CONCLUSION:**
- **CHECK** if the predicted step has NO tool calls (empty tool_calls array or just text content)
- **BUT** the task clearly requires executing a tool to create/generate/process something
- **EXAMPLE:** Task requires "generate HTML report", reference calls generate_html_report tool, but prediction only says "let me show you the table" or asks "do you want to export?" without actually calling any tool
- This is premature conclusion because the step talks about completing the task without actually doing it
- Text-only responses are acceptable ONLY when the task is genuinely complete or user input is needed

**CRITICAL FOR E4_LOGIC_ERROR:**
- **CHECK** if the step's description/thought claims to do data extraction/processing/cleaning
- **BUT** the actual tool call uses already-processed data that doesn't exist in previous steps
- **EXAMPLE:** Step says "extract the last 4 digits of account numbers" but tool parameters already contain extracted data like "XXXXXX0016"
- This is a logic error because the step skips necessary intermediate processing or has illogical reasoning
- The processed data must come from a previous step or be created in THIS step with appropriate tools

**CRITICAL FOR E5_TOOL_USE_ERROR:**
- **STRICT PARAMETER CHECK:** Check if parameters are optimal and correct.
- **VAGUE PARAMETERS:** Any vague or non-specific parameter is an ERROR.
- **POTENTIAL MISUSE:** If the parameter value *might* be wrong or lead to unexpected results, it is an ERROR.
- **SPECIFICATION COMPLIANCE:** Must strictly adhere to tool specifications.

**CRITICAL FOR E6_HALLUCINATION_ERROR:**
- **MUST CHECK:** Verify that every tool name used in the predicted step exists in the "Available Tools" list
- Tool names must match EXACTLY - check spelling, capitalization, and word order
- Even slight variations in tool names (different spelling, word order, or capitalization) should be flagged as hallucination
- The tool name in the predicted step must be identical to one of the tool names listed in Available Tools

Provide:

1. Error classification (True/False for each E1-E6) - BE STRICT, mark True if ANY doubt exists
2. Score (0.0, 0.2, 0.4, 0.6, 0.8, or 1.0) - Apply the strict scoring rubric
3. Detailed reasoning explaining your evaluation and why you marked each error
4. Is correct? (YES ONLY if ALL error flags are False, otherwise NO)

**Output Format (JSON):**
```json
{{
    "E1_GOAL_MISALIGNMENT": false,
    "E2_PREMATURE_CONCLUSION": false,
    "E3_CONSTRAINT_VIOLATION": false,
    "E4_LOGIC_ERROR": false,
    "E5_TOOL_USE_ERROR": false,
    "E6_HALLUCINATION_ERROR": false,
    "score": 1.0,
    "reasoning": "The predicted next step correctly identifies...",
    "is_correct": "YES"
}}
```

Provide ONLY the JSON output, no additional text.
"""
        return prompt
