"""
Base Online Planning Evaluator.

This module contains the base evaluator class with common functionality
and abstract methods for different prediction strategies.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from ..models.data_models import (
    PredictionResult,
    EvalResult,
    PredictionResultNext2,
    EvalResultNext2,
    PredictionResultNext3,
    EvalResultNext3,
    Statistics,
)
from ..clients.proxy_client import ProxyClient
from ..utils.dataset_loader import DatasetLoader, get_base_dataset_type
from ..prompts.evaluation_prompts import EvaluationPrompts


class BaseOnlinePlanningEvaluator(ABC):
    """Base class for online planning evaluators with common functionality."""

    def __init__(
        self,
        dataset_name: str,
        test_model: str = "claude-sonnet-4-20250514",
        eval_model: str = "gemini-2.5-pro",
        max_samples: Optional[int] = None,
        test_max_tokens: int = 64000,
        eval_max_tokens: int = 64000,
        output_dir: str = "eval_results",
        resume: bool = True,
        prediction_mode: str = "next1",  # "next1", "next2", "next3", etc.
        truncation_position: str = "middle",  # "early", "middle", or "late"
        use_loose_prompt: bool = False,
        use_blind_prompt: bool = False,
    ):
        self.dataset_name = dataset_name
        self.test_model = test_model
        self.eval_model = eval_model
        self.max_samples = max_samples
        self.prediction_mode = prediction_mode
        self.truncation_position = truncation_position
        self.use_loose_prompt = use_loose_prompt
        self.use_blind_prompt = use_blind_prompt

        # 如果是 gpt-4o 模型，自动设置 max_tokens 为 16384
        if test_model == "gpt-4o" and test_max_tokens > 16384:
            print(
                f"⚠️  检测到 gpt-4o 模型，将 test_max_tokens 从 {test_max_tokens} 调整为 16384"
            )
            self.test_max_tokens = 16384
        else:
            self.test_max_tokens = test_max_tokens

        self.eval_max_tokens = eval_max_tokens
        self.base_output_dir = output_dir
        self.resume = resume

        # Default data directory
        self.data_dir = "data/AgentPlanningBench/online"

        # Initialize clients
        self.test_client = ProxyClient(model=test_model)
        self.eval_client = ProxyClient(model=eval_model)

        # Create organized directory structure
        self._setup_directories()

        # For resume mode: track completed task_ids
        self.completed_task_ids: set[str] = set()
        self.resume_filepath: Optional[str] = None

    def _setup_directories(self):
        """Setup organized directory structure based on dataset and models."""
        # Normalize model names for directory paths (remove special chars)
        test_model_name = (
            self.test_model.replace("/", "_").replace(":", "_").replace(".", "_")
        )
        eval_model_name = (
            self.eval_model.replace("/", "_").replace(":", "_").replace(".", "_")
        )

        # Directory structure:
        # predictions/{test_model}/{prediction_mode}/
        # evaluations/{eval_model}/{test_model}/{prediction_mode}/
        # resume/predictions/{test_model}/{prediction_mode}/  - for prediction resume (no eval_model)
        # resume/evaluations/{eval_model}/{test_model}/{prediction_mode}/  - for evaluation resume

        # Add suffix for loose prompt to eval_model_name
        if self.use_loose_prompt:
            eval_model_name += "_LOOSE"

        # Add suffix for blind prompt to eval_model_name
        if self.use_blind_prompt:
            eval_model_name += "_BLIND"

        self.predictions_dir = os.path.join(
            self.base_output_dir, "predictions", test_model_name, self.prediction_mode
        )
        self.evaluations_dir = os.path.join(
            self.base_output_dir,
            "evaluations",
            eval_model_name,
            test_model_name,
            self.prediction_mode,
        )

        # Prediction resume: only depends on test_model, not eval_model
        self.prediction_resume_dir = os.path.join(
            self.base_output_dir,
            "resume",
            "predictions",
            test_model_name,
            self.prediction_mode,
        )
        # Evaluation resume: depends on both eval_model and test_model
        self.evaluation_resume_dir = os.path.join(
            self.base_output_dir,
            "resume",
            "evaluations",
            eval_model_name,
            test_model_name,
            self.prediction_mode,
        )

        # For backward compatibility, keep resume_dir pointing to evaluation resume dir
        self.resume_dir = self.evaluation_resume_dir

        # Create all directories
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.evaluations_dir, exist_ok=True)
        os.makedirs(self.prediction_resume_dir, exist_ok=True)
        os.makedirs(self.evaluation_resume_dir, exist_ok=True)

        # Output directory is the base output dir
        self.output_dir = self.base_output_dir

        print("📁 Directory structure:")
        print(f"   Dataset: {self.dataset_name}")
        print(f"   Test Model: {test_model_name}")
        print(f"   Eval Model: {eval_model_name}")
        print(f"   Prediction Mode: {self.prediction_mode}")
        print(f"   Predictions: {self.predictions_dir}")
        print(f"   Evaluations: {self.evaluations_dir}")
        print(f"   Prediction Resume: {self.prediction_resume_dir}")
        print(f"   Evaluation Resume: {self.evaluation_resume_dir}")

    def _get_file_paths_with_types(self, sample: Dict) -> List[Dict]:
        """
        Get file paths with their types from sample metadata.
        Files are stored in: data/AgentPlanningBench/online/files/{dataset_name}/

        Different datasets have different path extraction rules:
        - framethinker: Extract everything after 'imgs/process/X/' (e.g., video_frames/id/frame.jpg)
        - gaia: Use just the filename
        - gta: Use just the filename (even if path is like 'image/image_1.jpg')
        - opencua: Use just the filename (even if path is like 'images/s_xxx.png')
        - others: Use just the filename

        Returns:
            List of dicts with 'type' and 'path' keys
        """
        # Use base dataset type for file path determination
        base_type = get_base_dataset_type(self.dataset_name)
        files_with_types: List[Dict] = []

        # Get files from meta_data
        meta_data = sample.get("meta_data", {})
        files = meta_data.get("files", [])

        if not files:
            return files_with_types

        # Base directory for all dataset files (use actual dataset name for directory)
        base_dir = f"data/AgentPlanningBench/online/files/{self.dataset_name}"

        # Flatten nested list structure if needed (e.g., framethinker has [[{}, {}]] format)
        flattened_files = []
        for item in files:
            if isinstance(item, list):
                flattened_files.extend(item)
            else:
                flattened_files.append(item)

        for file_info in flattened_files:
            file_type = file_info.get("type", "unknown")
            file_path = file_info.get("path", "")

            if not file_path:
                continue

            # Extract the appropriate filename based on dataset type
            if base_type == "framethinker":
                # Extract everything after 'imgs/process/X/'
                # e.g., 'imgs/process/1/video_frames/id/frame.jpg' -> 'video_frames/id/frame.jpg'
                if "imgs/process/" in file_path:
                    parts = file_path.split("imgs/process/")
                    if len(parts) > 1:
                        # Skip the process number (e.g., '1/') and get the rest
                        after_process = parts[1]
                        if "/" in after_process:
                            filename = "/".join(after_process.split("/")[1:])
                        else:
                            filename = after_process
                    else:
                        filename = os.path.basename(file_path)
                else:
                    filename = os.path.basename(file_path)
            elif base_type in ["gaia", "gta", "opencua", "toolbench", "skywork"]:
                # Use just the filename (ignore any subdirectories in the path)
                filename = os.path.basename(file_path)
            else:
                # Default: use filename
                filename = os.path.basename(file_path)

            # Construct full path
            full_path = os.path.join(base_dir, filename)

            files_with_types.append({"type": file_type, "path": full_path})

        return files_with_types

    def _truncate_files_for_opencua(
        self, files: List[Dict], trajectory: str
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Truncate files for opencua dataset based on trajectory truncation point.

        For opencua, images correspond to trajectory steps. When we truncate the trajectory
        at the middle, we should also only show the first half of images to the test model,
        while the eval model should see all images (prefix + remaining).

        Args:
            files: List of file dicts with 'type' and 'path'
            trajectory: The trajectory string (list of actions)

        Returns:
            tuple: (prefix_files, all_files)
                - prefix_files: Files for the first half (for test model)
                - all_files: All files (for eval model)
        """
        base_type = get_base_dataset_type(self.dataset_name)
        if base_type != "opencua":
            # For non-opencua datasets, return all files for both
            return files, files

        # Parse trajectory to get number of steps
        try:
            if isinstance(trajectory, list):
                steps = trajectory
            elif trajectory.startswith("["):
                steps = json.loads(trajectory)
            else:
                steps = [s for s in trajectory.split("\n") if s.strip()]

            mid = len(steps) // 2

            # Truncate files at the same proportion
            # Assuming files correspond to steps
            if len(files) > 0:
                file_mid = min(mid, len(files))
                prefix_files = files[:file_mid]
                return prefix_files, files
            else:
                return files, files

        except Exception as e:
            print(f"  ⚠️  Error truncating files for opencua: {e}")
            return files, files

    def _json_only_instruction(self, expected_root: str = "object") -> str:
        """
        Build strict instructions telling the model to return raw JSON only.

        Args:
            expected_root: "object" when response must be a single JSON object,
                "array" when response must be a JSON array.
        """
        root_desc = (
            "a single JSON object" if expected_root == "object" else "a JSON array"
        )
        first_char = "{" if expected_root == "object" else "["
        last_char = "}" if expected_root == "object" else "]"

        return f"""
**STRICT RESPONSE RULES:**
- Output ONLY {root_desc} exactly as specified above.
- The first non-whitespace character MUST be '{first_char}' and the final character MUST be '{last_char}'.
- Do NOT wrap the JSON in code fences.
- Do NOT add explanations, headers, analysis, or any other text before or after the JSON.
- All reasoning must remain inside the JSON fields (e.g., thought/content/action)."""

    def _parse_evaluation(self, response: str) -> Optional[Dict]:
        """Parse evaluation response."""
        try:
            # Find JSON in response
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx == -1 or end_idx == -1:
                print(f"  ⚠️  No JSON found in evaluation response")
                return None

            json_str = response[start_idx : end_idx + 1]
            result = json.loads(json_str)

            # Validate required fields
            required_fields = ["score", "is_correct", "error_types", "reasoning"]
            for field in required_fields:
                if field not in result:
                    print(f"  ⚠️  Missing required field: {field}")
                    return None

            return result

        except Exception as e:
            print(f"  ⚠️  Error parsing evaluation: {e}")
            return None

    def calculate_statistics(
        self,
        results: Union[List[EvalResult], List[EvalResultNext2], List[EvalResultNext3]],
    ) -> Statistics:
        """Calculate statistics from evaluation results."""
        if not results:
            return Statistics(
                total=0,
                accuracy=0.0,
                avg_score=0.0,
                score_distribution={},
                error_type_counts={},
                error_type_rates={},
            )

        total = len(results)
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / total if total > 0 else 0.0

        # Score distribution
        scores = [r.score for r in results]
        avg_score = sum(scores) / total if total > 0 else 0.0

        score_distribution = {}
        for score in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            score_distribution[score] = sum(1 for s in scores if abs(s - score) < 0.01)

        # Error type statistics
        error_types = ["E1", "E2", "E3", "E4", "E5", "E6"]
        error_type_counts = {et: 0 for et in error_types}

        for result in results:
            for error_type, has_error in result.error_types.items():
                if has_error and error_type in error_type_counts:
                    error_type_counts[error_type] += 1

        error_type_rates = {
            et: count / total if total > 0 else 0.0
            for et, count in error_type_counts.items()
        }

        return Statistics(
            total=total,
            accuracy=accuracy,
            avg_score=avg_score,
            score_distribution=score_distribution,
            error_type_counts=error_type_counts,
            error_type_rates=error_type_rates,
        )

    def print_statistics(self, stats: Statistics):
        """Print statistics in a formatted way."""
        print("\n" + "=" * 80)
        print("EVALUATION STATISTICS")
        print("=" * 80)
        print(f"Total Samples: {stats.total}")
        print(f"Accuracy: {stats.accuracy:.2%}")
        print(f"Average Score: {stats.avg_score:.3f}")
        print("\nScore Distribution:")
        for score in sorted(stats.score_distribution.keys()):
            count = stats.score_distribution[score]
            percentage = (count / stats.total * 100) if stats.total > 0 else 0
            print(f"  {score:.1f}: {count:4d} ({percentage:5.1f}%)")
        print("\nError Type Statistics:")
        for error_type in sorted(stats.error_type_counts.keys()):
            count = stats.error_type_counts[error_type]
            rate = stats.error_type_rates[error_type]
            print(f"  {error_type}: {count:4d} ({rate:5.1%})")
        print("=" * 80)

    def find_latest_prediction_file(self) -> Optional[str]:
        """
        Find the latest prediction file in predictions directory for current dataset and test_model.

        Returns:
            Path to the latest prediction file, or None if no file found
        """
        if not os.path.exists(self.predictions_dir):
            return None

        # Look for prediction files matching the dataset name
        prediction_files = []
        for file in os.listdir(self.predictions_dir):
            if file.startswith(f"{self.dataset_name}_predictions") and file.endswith(
                ".json"
            ):
                filepath = os.path.join(self.predictions_dir, file)
                prediction_files.append((filepath, os.path.getmtime(filepath)))

        if not prediction_files:
            # Also check resume directory
            if os.path.exists(self.prediction_resume_dir):
                for file in os.listdir(self.prediction_resume_dir):
                    if file.startswith(
                        f"{self.dataset_name}_predictions"
                    ) and file.endswith(".json"):
                        filepath = os.path.join(self.prediction_resume_dir, file)
                        prediction_files.append((filepath, os.path.getmtime(filepath)))

        if not prediction_files:
            return None

        # Return the most recent file
        latest_file = max(prediction_files, key=lambda x: x[1])[0]
        print(f"  📂 Auto-detected prediction file: {latest_file}")
        return latest_file

    # Abstract methods to be implemented by subclasses
    @abstractmethod
    def truncate_trajectory(self, trajectory: str, system_prompt: Optional[str] = None):
        """Truncate trajectory based on prediction strategy."""
        pass

    @abstractmethod
    def generate_prediction_prompt(self, sample: Dict, prefix: str) -> str:
        """Generate prompt for test model."""
        pass

    @abstractmethod
    def predict_sample(self, sample: Dict):
        """Generate prediction for a single sample."""
        pass

    @abstractmethod
    def evaluate_prediction(self, prediction):
        """Evaluate a single prediction."""
        pass

    @abstractmethod
    def run_inference(self, dataset: Optional[List[Dict]] = None):
        """Run inference mode: generate predictions only."""
        pass

    @abstractmethod
    def run_evaluation(self, predictions):
        """Run evaluation mode: evaluate existing predictions."""
        pass

    @abstractmethod
    def run_pipeline(self, dataset: Optional[List[Dict]] = None):
        """Run pipeline mode: inference + evaluation."""
        pass

    @abstractmethod
    def save_predictions(self, predictions):
        """Save predictions to file."""
        pass

    @abstractmethod
    def load_predictions(self, filepath: str):
        """Load predictions from file."""
        pass

    @abstractmethod
    def save_results(self, results):
        """Save results to file."""
        pass
