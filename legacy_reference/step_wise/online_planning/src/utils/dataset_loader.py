"""
Dataset loading utilities.

This module handles loading and preprocessing of various benchmark datasets
with a unified format.
"""

import json
from typing import Dict, List
from pathlib import Path


def get_base_dataset_type(dataset_name: str) -> str:
    """
    Get the base dataset type from dataset name.
    
    For skywork variants (skywork_doc, skywork_excel, etc.), returns 'skywork'.
    For other datasets, returns the dataset name as is.
    
    Args:
        dataset_name: Full dataset name (e.g., 'skywork_doc', 'gaia')
    
    Returns:
        Base dataset type (e.g., 'skywork', 'gaia')
    """
    dataset_lower = dataset_name.lower()
    
    # Handle skywork variants
    if dataset_lower.startswith('skywork'):
        return 'skywork'
    
    # For other datasets, return as is
    return dataset_lower


class DatasetLoader:
    """Load datasets with unified format."""
    
    @staticmethod
    def load_dataset(dataset_name: str, data_dir: str = "data/AgentPlanningBench/online") -> List[Dict]:
        """Load dataset and extract all fields uniformly."""
        data_path = Path(data_dir)
        
        # Find dataset file
        dataset_file = None
        for file in data_path.iterdir():
            if file.is_file() and dataset_name.lower() in file.name.lower():
                if file.suffix in ['.json', '.jsonl']:
                    dataset_file = file
                    break
        
        if not dataset_file:
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found in {data_dir}")
        
        # Load data
        data = []
        base_type = get_base_dataset_type(dataset_name)
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    
                    # Get trajectory
                    trajectory = item.get('trajectory', '')
                    
                    # For framethinker: convert role "user" to "tool" for all steps except the first
                    if base_type == 'framethinker' and trajectory:
                        trajectory = DatasetLoader._convert_framethinker_trajectory(trajectory)
                    
                    # Ensure all fields are present
                    unified_item = {
                        'index': item.get('index', item.get('task_id', 'unknown')),
                        'query': item.get('query', item.get('question', '')),
                        'trajectory': trajectory,
                        'tools': item.get('tools', []),
                        'meta_data': item.get('meta_data', {}),
                        'system_prompt': item.get('system_prompt', None),  # Add system_prompt for framethinker
                        'redundant_tools': item.get('redundant_tools', []), # Add redundant_tools for tool_redundancy mode
                        'original_trajectory': item.get('original_trajectory', ''), # Add original_trajectory for tool_broken mode
                        'broken_tool_name': item.get('broken_tool_name', None), # Add broken_tool_name for tool_broken mode
                        'replacement_tool': item.get('replacement_tool', None), # Add replacement_tool for tool_broken mode
                        'dataset': dataset_name
                    }
                    data.append(unified_item)
        
        print(f"✓ Loaded {len(data)} samples from {dataset_file.name}")
        if base_type == 'framethinker':
            print("  ℹ️  Framethinker: Converted role 'user' → 'tool' for steps after the first")
        return data
    
    @staticmethod
    def _convert_framethinker_trajectory(trajectory: str) -> str:
        """
        Convert framethinker trajectory: change role "user" to "tool" for all steps except the first.
        
        In framethinker, the first step with role "user" is the actual user query,
        but subsequent steps with role "user" are actually tool execution results.
        
        Args:
            trajectory: Original trajectory (JSON string or list)
        
        Returns:
            Modified trajectory with role conversions applied
        """
        try:
            # Parse trajectory
            if isinstance(trajectory, str):
                if trajectory.startswith('['):
                    steps = json.loads(trajectory)
                else:
                    # Single step, return as is
                    return trajectory
            elif isinstance(trajectory, list):
                steps = trajectory
            else:
                return trajectory
            
            # Find the first user step and convert subsequent user steps to tool
            found_first_user = False
            
            for step in steps:
                if isinstance(step, dict) and 'role' in step:
                    if step['role'] == 'user':
                        if not found_first_user:
                            # This is the first user step, keep it as user
                            found_first_user = True
                        else:
                            # This is a subsequent user step, convert to tool
                            step['role'] = 'tool'
            
            # Convert back to JSON string
            return json.dumps(steps, ensure_ascii=False)
            
        except Exception as e:
            print(f"  ⚠️  Warning: Failed to convert framethinker trajectory: {e}")
            return trajectory
