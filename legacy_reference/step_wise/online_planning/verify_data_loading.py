
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath("/path/to/project"))

from src.evaluators.tool_broken_evaluator import ToolBrokenEvaluator
from src.evaluators.next1_evaluator import Next1Evaluator
from src.utils.dataset_loader import DatasetLoader

def test_data_dir_override():
    print("Testing ToolBrokenEvaluator data_dir override...")
    
    # Create evaluator
    evaluator = ToolBrokenEvaluator(
        dataset_name="skywork_doc",
        test_model="test-model",
        eval_model="eval-model"
    )
    
    expected_dir = "data/AgentPlanningBench/online_tool_broken"
    if evaluator.data_dir == expected_dir:
        print(f"✓ SUCCESS: ToolBrokenEvaluator.data_dir is correctly set to '{evaluator.data_dir}'")
    else:
        print(f"✗ FAILURE: ToolBrokenEvaluator.data_dir is '{evaluator.data_dir}', expected '{expected_dir}'")

def test_next1_default():
    print("\nTesting Next1Evaluator default data_dir...")
    
    # Create evaluator
    evaluator = Next1Evaluator(
        dataset_name="skywork_doc",
        test_model="test-model",
        eval_model="eval-model"
    )
    
    expected_dir = "data/AgentPlanningBench/online"
    if evaluator.data_dir == expected_dir:
        print(f"✓ SUCCESS: Next1Evaluator.data_dir is correctly set to '{evaluator.data_dir}'")
    else:
        print(f"✗ FAILURE: Next1Evaluator.data_dir is '{evaluator.data_dir}', expected '{expected_dir}'")

if __name__ == "__main__":
    test_data_dir_override()
    test_next1_default()
