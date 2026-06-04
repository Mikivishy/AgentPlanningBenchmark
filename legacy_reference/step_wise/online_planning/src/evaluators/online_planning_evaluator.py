"""
Online Planning Evaluator Factory.

This module contains the factory/router class that selects and returns
the appropriate strategy evaluator based on prediction mode.
"""

from typing import Optional, Union

from .next1_evaluator import Next1Evaluator
from .next2_evaluator import Next2Evaluator
from .next3_evaluator import Next3Evaluator
from .tool_broken_evaluator import ToolBrokenEvaluator
from .tool_redundancy_evaluator import ToolRedundancyEvaluator
from .self_refinement_evaluator import SelfRefinementEvaluator


class OnlinePlanningEvaluator:
    """
    Factory/Router class for online planning evaluators.

    This class acts as a factory that creates and returns the appropriate
    strategy evaluator (Next1 or Next2) based on the prediction_mode parameter.
    It maintains backward compatibility with existing code while enabling
    easy extension for future strategies.

    Usage:
        # Create evaluator with next1 strategy (default)
        evaluator = OnlinePlanningEvaluator.create(
            dataset_name="gaia",
            prediction_mode="next1"
        )

        # Create evaluator with next2 strategy
        evaluator = OnlinePlanningEvaluator.create(
            dataset_name="gaia",
            prediction_mode="next2"
        )

        # Run the evaluator (same interface for both strategies)
        results = evaluator.run_pipeline()
    """

    @staticmethod
    def create(
        dataset_name: str,
        test_model: str = "claude-sonnet-4-20250514",
        eval_model: str = "gemini-2.5-pro",
        max_samples: Optional[int] = None,
        test_max_tokens: int = 64000,
        eval_max_tokens: int = 64000,
        output_dir: str = "eval_results",
        resume: bool = True,
        prediction_mode: str = "next1",  # "next1", "next1_early", "next1_late", "next2", "next3", "tool_broken", "tool_redundancy", or "self_refinement"
        use_loose_prompt: bool = False,
        use_blind_prompt: bool = False,
        num_redundant_tools: int = 4,
        reflection_model: Optional[str] = None,
    ) -> Union[
        Next1Evaluator,
        Next2Evaluator,
        Next3Evaluator,
        ToolBrokenEvaluator,
        ToolRedundancyEvaluator,
        SelfRefinementEvaluator,
    ]:
        """
        Factory method to create the appropriate evaluator based on prediction_mode.

        Args:
            dataset_name: Name of the dataset to evaluate
            test_model: Model to use for predictions
            eval_model: Model to use for evaluation
            max_samples: Maximum number of samples to process (None for all)
            test_max_tokens: Max tokens for test model
            eval_max_tokens: Max tokens for eval model
            output_dir: Base directory for outputs
            resume: Whether to resume from previous progress
            prediction_mode: Strategy to use - "next1", "next1_early", "next1_late", "next2", "next3", "tool_broken", or "tool_redundancy"
            use_loose_prompt: Whether to use loose evaluation prompts
            use_blind_prompt: Whether to use blind evaluation prompts (no reference)

        Returns:
            An instance of Next1Evaluator, Next2Evaluator, Next3Evaluator, ToolBrokenEvaluator, or ToolRedundancyEvaluator

        Raises:
            ValueError: If prediction_mode is invalid
        """
        # Parse prediction_mode to extract base mode and truncation position
        base_mode = prediction_mode
        truncation_position = "middle"  # default

        if prediction_mode.startswith("next1"):
            base_mode = "next1"
            if "_early" in prediction_mode:
                truncation_position = "early"
            elif "_late" in prediction_mode:
                truncation_position = "late"
            else:
                truncation_position = "middle"

        # Validate base prediction_mode
        valid_modes = [
            "next1",
            "next2",
            "next3",
            "tool_broken",
            "tool_redundancy",
            "self_refinement",
            "self_refinement_claude",
            "refinement_without_judge",
            "refinement_without_judge_claude",
        ]
        if base_mode not in valid_modes:
            raise ValueError(
                f"Invalid prediction_mode: {prediction_mode}. "
                f"Must be one of: next1, next1_early, next1_late, next2, next3, tool_broken, tool_redundancy, self_refinement, self_refinement_claude, refinement_without_judge, refinement_without_judge_claude."
            )

        # Common parameters for all evaluators
        common_params = {
            "dataset_name": dataset_name,
            "test_model": test_model,
            "eval_model": eval_model,
            "max_samples": max_samples,
            "test_max_tokens": test_max_tokens,
            "eval_max_tokens": eval_max_tokens,
            "output_dir": output_dir,
            "resume": resume,
            "use_loose_prompt": use_loose_prompt,
            "use_blind_prompt": use_blind_prompt,
        }

        # Create and return the appropriate evaluator
        if base_mode == "next1":
            position_info = (
                f" (truncation: {truncation_position})"
                if truncation_position != "middle"
                else ""
            )
            print(
                f"🎯 Creating Next1 Evaluator (predict next single step){position_info}"
            )
            return Next1Evaluator(
                truncation_position=truncation_position, **common_params
            )
        elif base_mode == "next2":
            print("🎯 Creating Next2 Evaluator (predict next two steps)")
            return Next2Evaluator(**common_params)
        elif base_mode == "next3":
            print("🎯 Creating Next3 Evaluator (predict next three steps)")
            return Next3Evaluator(**common_params)
        elif base_mode == "tool_broken":
            print("🎯 Creating Tool-Broken Evaluator")
            return ToolBrokenEvaluator(**common_params)
        elif base_mode == "tool_redundancy":
            print("🎯 Creating Tool-Redundancy Evaluator")
            return ToolRedundancyEvaluator(
                num_redundant_tools=num_redundant_tools, **common_params
            )
        elif (
            base_mode == "self_refinement"
            or base_mode == "self_refinement_claude"
            or base_mode == "refinement_without_judge"
            or base_mode == "refinement_without_judge_claude"
        ):
            print(f"🎯 Creating Self-Refinement Evaluator (mode={base_mode})")
            return SelfRefinementEvaluator(
                reflection_model=reflection_model,
                prediction_mode=base_mode,
                **common_params,
            )

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to ensure OnlinePlanningEvaluator() calls create().

        This maintains backward compatibility with existing code that uses:
            evaluator = OnlinePlanningEvaluator(...)
        instead of:
            evaluator = OnlinePlanningEvaluator.create(...)
        """
        # If called as OnlinePlanningEvaluator(...), redirect to create()
        if cls is OnlinePlanningEvaluator:
            return cls.create(*args, **kwargs)

        # Allow subclasses to work normally
        return super().__new__(cls)
