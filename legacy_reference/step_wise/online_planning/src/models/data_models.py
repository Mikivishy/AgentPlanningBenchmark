"""
Data models for online planning evaluation.

This module contains all dataclass definitions used throughout the evaluation pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PredictionResult:
    """Prediction result for inference mode."""
    task_id: str
    dataset: str
    query: str
    trajectory_prefix: str
    predicted_step: str
    ground_truth_step: str
    reference_remaining_steps: str
    tools: List  # Can be List[Dict] or List[str]
    meta_data: Dict
    timestamp: str


@dataclass
class EvalResult:
    """Evaluation result for a single test case."""
    task_id: str
    dataset: str
    query: str
    predicted_step: str
    ground_truth_step: str
    score: float  # 0, 0.2, 0.4, 0.6, 0.8, 1.0
    is_correct: bool
    error_types: Dict[str, bool]  # E1-E6
    reasoning: str
    timestamp: str
    used_redundant_tools: bool = False


@dataclass
class PredictionResultNext2:
    """Prediction result for next2 inference mode."""
    task_id: str
    dataset: str
    query: str
    trajectory_prefix: str
    predicted_steps: str  # Complete prediction output (should be JSON array with 2 steps)
    ground_truth_steps: str  # Complete ground truth (should be JSON array with 2 steps)
    reference_remaining_steps: str
    tools: List  # Can be List[Dict] or List[str]
    meta_data: Dict
    timestamp: str


@dataclass
class EvalResultNext2:
    """Evaluation result for next2 test case with overall assessment."""
    task_id: str
    dataset: str
    query: str
    predicted_steps: str  # Complete prediction output (both steps together)
    ground_truth_steps: str  # Complete ground truth (both steps together)
    score: float  # Overall score for both steps: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    is_correct: bool  # Whether the overall prediction is correct
    error_types: Dict[str, bool]  # E1-E6 error types for overall assessment
    reasoning: str  # Evaluation reasoning
    timestamp: str


@dataclass
class PredictionResultNext3:
    """Prediction result for next3 inference mode."""
    task_id: str
    dataset: str
    query: str
    trajectory_prefix: str
    predicted_steps: str  # Complete prediction output (should be JSON array with 3 steps)
    ground_truth_steps: str  # Complete ground truth (should be JSON array with 3 steps)
    reference_remaining_steps: str
    tools: List  # Can be List[Dict] or List[str]
    meta_data: Dict
    timestamp: str


@dataclass
class EvalResultNext3:
    """Evaluation result for next3 test case with overall assessment."""
    task_id: str
    dataset: str
    query: str
    predicted_steps: str  # Complete prediction output (all three steps together)
    ground_truth_steps: str  # Complete ground truth (all three steps together)
    score: float  # Overall score for all three steps: 0, 0.2, 0.4, 0.6, 0.8, 1.0
    is_correct: bool  # Whether the overall prediction is correct
    error_types: Dict[str, bool]  # E1-E6 error types for overall assessment
    reasoning: str  # Evaluation reasoning
    timestamp: str


@dataclass
class Statistics:
    """Statistics for all results."""
    total: int
    accuracy: float
    avg_score: float
    score_distribution: Dict[float, int]  # {0.0: count, 0.2: count, ...}
    error_type_counts: Dict[str, int]  # {E1: count, E2: count, ...}
    error_type_rates: Dict[str, float]  # {E1: rate, E2: rate, ...}
