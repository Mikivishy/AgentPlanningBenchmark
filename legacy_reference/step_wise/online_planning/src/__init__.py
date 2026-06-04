"""
Online Planning Evaluation v2 - Modular Implementation

A refactored version of the online planning evaluation system with improved
modularity, maintainability, and ease of debugging.
"""

__version__ = "2.0.0"

from .models.data_models import (
    PredictionResult,
    EvalResult,
    PredictionResultNext2,
    EvalResultNext2,
    Statistics
)
from .clients.proxy_client import ProxyClient
from .utils.dataset_loader import DatasetLoader, get_base_dataset_type
from .prompts.evaluation_prompts import EvaluationPrompts
from .evaluators.online_planning_evaluator import OnlinePlanningEvaluator

__all__ = [
    'PredictionResult',
    'EvalResult',
    'PredictionResultNext2',
    'EvalResultNext2',
    'Statistics',
    'ProxyClient',
    'DatasetLoader',
    'get_base_dataset_type',
    'EvaluationPrompts',
    'OnlinePlanningEvaluator',
]
