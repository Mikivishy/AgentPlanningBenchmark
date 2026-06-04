"""Evaluation prompts for different datasets."""

from .evaluation_prompts import EvaluationPrompts
from .evaluation_prompts_strict import EvaluationPromptsStrict
from .evaluation_prompts_loose import EvaluationPromptsLoose

__all__ = ['EvaluationPrompts', 'EvaluationPromptsStrict', 'EvaluationPromptsLoose']
