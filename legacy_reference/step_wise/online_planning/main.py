"""
Main entry point for the Online Planning Evaluation system.

This script provides a command-line interface for running inference and evaluation
on various planning benchmarks.
"""

import argparse
import yaml
from pathlib import Path
from typing import Optional, List, Union
import concurrent.futures
import threading
from datetime import datetime
from src.evaluators.online_planning_evaluator import OnlinePlanningEvaluator
from src.evaluators.next1_evaluator import Next1Evaluator
from src.evaluators.next2_evaluator import Next2Evaluator
from src.evaluators.next3_evaluator import Next3Evaluator


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, looks for config.yaml in script directory.

    Returns:
        Dictionary containing configuration parameters
    """
    config_file: Path
    if config_path is None:
        # Default to config.yaml in the same directory as this script
        script_dir = Path(__file__).parent
        config_file = script_dir / "config.yaml"
    else:
        config_file = Path(config_path)

    if not config_file.exists():
        return {}

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config else {}


def run_single_config(
    config: dict,
    config_name: Optional[str] = None,
    lock: Optional[threading.Lock] = None,
    override_loose: Optional[bool] = None,
    override_blind: Optional[bool] = None,
) -> dict:
    """
    Run evaluation for a single configuration.

    Args:
        config: Configuration dictionary (can be from file or inline)
        config_name: Name for this configuration
        lock: Threading lock for synchronized output (optional)
        override_loose: Override for use_loose_prompt (optional)
        override_blind: Override for use_blind_prompt (optional)

    Returns:
        Dictionary containing results summary
    """
    if config_name is None:
        config_name = config.get("name", "unnamed")

    try:
        if not config:
            return {
                "config": config_name,
                "status": "error",
                "message": "Empty configuration",
            }

        # Extract parameters
        mode = config.get("mode", "pipeline")
        dataset = config.get("dataset")
        test_model = config.get("test_model", "claude-sonnet-4-20250514")
        eval_model = config.get("eval_model", "gemini-3-pro-preview")
        max_samples = config.get("max_samples")
        test_max_tokens = config.get("test_max_tokens", 64000)
        eval_max_tokens = config.get("eval_max_tokens", 64000)
        output_dir = config.get("output_dir", "eval_results")
        resume = config.get("resume", True)
        resume = config.get("resume", True)
        prediction_file = config.get("prediction_file")
        num_redundant_tools = config.get("num_redundant_tools", 4)

        # Determine prompt settings (overrides take precedence)
        use_loose_prompt = (
            override_loose
            if override_loose is not None
            else config.get("use_loose_prompt", False)
        )
        use_blind_prompt = (
            override_blind
            if override_blind is not None
            else config.get("use_blind_prompt", False)
        )

        # Validate required parameters
        if mode in [
            "inference",
            "pipeline",
            "inference_next1",
            "pipeline_next1",
            "inference_next1_early",
            "evaluation_next1_early",
            "pipeline_next1_early",
            "inference_next1_late",
            "evaluation_next1_late",
            "pipeline_next1_late",
            "inference_next2",
            "pipeline_next2",
            "inference_next3",
            "pipeline_next3",
            "inference_tool_broken",
            "pipeline_tool_broken",
            "evaluation_tool_broken",
            "inference_tool_redundancy",
            "pipeline_tool_redundancy",
            "evaluation_tool_redundancy",
            "inference_self_refinement",
            "pipeline_self_refinement",
            "evaluation_self_refinement",
            "inference_self_refinement_claude",
            "pipeline_self_refinement_claude",
            "evaluation_self_refinement_claude",
            "inference_refinement_without_judge",
            "pipeline_refinement_without_judge",
            "evaluation_refinement_without_judge",
            "inference_refinement_without_judge_claude",
            "pipeline_refinement_without_judge_claude",
            "evaluation_refinement_without_judge_claude",
        ]:
            if not dataset:
                return {
                    "config": config_name,
                    "status": "error",
                    "message": f"Dataset is required for {mode} mode",
                }

        # Determine prediction mode (with truncation position for next1)
        reflection_model = None
        if "tool_redundancy" in mode:
            prediction_mode = "tool_redundancy"
        elif "tool_broken" in mode:
            prediction_mode = "tool_broken"
        elif "self_refinement" in mode:
            prediction_mode = "self_refinement"
            if "claude" in mode:
                prediction_mode = "self_refinement_claude"
                reflection_model = "anthropic/claude-sonnet-4.5"
        elif "refinement_without_judge" in mode:
            prediction_mode = "refinement_without_judge"
            if "claude" in mode:
                prediction_mode = "refinement_without_judge_claude"
                reflection_model = "anthropic/claude-sonnet-4.5"
        elif "next3" in mode:
            prediction_mode = "next3"
        elif "next2" in mode:
            prediction_mode = "next2"
        elif "next1_early" in mode:
            prediction_mode = "next1_early"
        elif "next1_late" in mode:
            prediction_mode = "next1_late"
        else:
            prediction_mode = "next1"

        # Log start
        if lock:
            with lock:
                print(
                    f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {config_name}"
                )
                print(f"  Dataset: {dataset}, Model: {test_model}, Mode: {mode}")
        else:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting: {config_name}")
            print(f"  Dataset: {dataset}, Model: {test_model}, Mode: {mode}")

        # Create evaluator
        dataset_name = dataset if dataset else "unknown"
        evaluator: Union[Next1Evaluator, Next2Evaluator, Next3Evaluator] = (
            OnlinePlanningEvaluator(  # type: ignore[assignment]
                dataset_name=dataset_name,
                test_model=test_model,
                eval_model=eval_model,
                max_samples=max_samples,
                test_max_tokens=test_max_tokens,
                eval_max_tokens=eval_max_tokens,
                output_dir=output_dir,
                resume=resume,
                prediction_mode=prediction_mode,
                use_loose_prompt=use_loose_prompt,
                use_blind_prompt=use_blind_prompt,
                num_redundant_tools=num_redundant_tools,
                reflection_model=reflection_model,
            )
        )

        # Execute based on mode
        result_summary = {
            "config": config_name,
            "dataset": dataset,
            "test_model": test_model,
            "eval_model": eval_model,
            "mode": mode,
            "status": "success",
            "start_time": datetime.now().isoformat(),
        }

        if "inference" in mode and "pipeline" not in mode:
            # Run inference only
            predictions = evaluator.run_inference()  # type: ignore[arg-type]
            # Note: save_predictions is already called within run_inference() if new predictions were generated
            result_summary["num_predictions"] = len(predictions)

        elif "evaluation" in mode and "pipeline" not in mode:
            # Run evaluation only
            # Auto-detect prediction file if not specified
            if not prediction_file:
                prediction_file = evaluator.find_latest_prediction_file()  # type: ignore[attr-defined]
                if not prediction_file:
                    return {
                        "config": config_name,
                        "status": "error",
                        "message": f"No prediction file found for dataset {dataset} and model {test_model}",
                    }
                print(f"  ℹ️  Using auto-detected prediction file: {prediction_file}")

            predictions = evaluator.load_predictions(prediction_file)  # type: ignore[arg-type]
            results = evaluator.run_evaluation(predictions)  # type: ignore[arg-type]
            # Note: save_results is already called within run_evaluation() if new results were generated
            stats = evaluator.calculate_statistics(results)
            result_summary["num_results"] = len(results)
            result_summary["statistics"] = stats

        elif "pipeline" in mode:
            # Run full pipeline
            predictions = evaluator.run_inference()  # type: ignore[arg-type]
            # Note: save_predictions is already called within run_inference() if new predictions were generated
            results = evaluator.run_evaluation(predictions)  # type: ignore[arg-type]
            # Note: save_results is already called within run_evaluation() if new results were generated
            stats = evaluator.calculate_statistics(results)
            result_summary["num_predictions"] = len(predictions)
            result_summary["num_results"] = len(results)
            result_summary["statistics"] = stats

        result_summary["end_time"] = datetime.now().isoformat()

        # Log completion
        if lock:
            with lock:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed: {config_name}"
                )
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Completed: {config_name}")

        return result_summary

    except Exception as e:
        error_msg = str(e)
        if lock:
            with lock:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed: {config_name} - {error_msg}"
                )
        else:
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] ✗ Failed: {config_name} - {error_msg}"
            )

        return {
            "config": config_name,
            "status": "error",
            "message": error_msg,
            "end_time": datetime.now().isoformat(),
        }


def batch_evaluate(
    config_files: List[str],
    max_workers: int = 4,
    loose: Optional[bool] = None,
    blind: Optional[bool] = None,
) -> List[dict]:
    """
    Run evaluations for multiple configuration files in parallel.

    Args:
        config_files: List of paths to configuration files OR list of config dicts
        max_workers: Maximum number of parallel workers
        loose: Override for use_loose_prompt (optional)
        blind: Override for use_blind_prompt (optional)

    Returns:
        List of result summaries for each configuration
    """
    print(f"\n{'=' * 80}")
    print("BATCH EVALUATION MODE")
    print(f"Total configurations: {len(config_files)}")
    print(f"Max parallel workers: {max_workers}")
    print(f"{'=' * 80}\n")

    # Create a lock for synchronized output
    output_lock = threading.Lock()

    # Run configurations in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_config = {}
        for item in config_files:
            if isinstance(item, dict):
                # It's a config dictionary
                config_name = item.get("name", "unnamed")
                # Check if item has its own overrides, otherwise use global ones
                item_loose = item.get("loose", loose)
                item_blind = item.get("blind", blind)
                future = executor.submit(
                    run_single_config,
                    item,
                    config_name,
                    output_lock,
                    item_loose,
                    item_blind,
                )
                future_to_config[future] = config_name
            else:
                # It's a file path
                config = load_config(item)
                config_name = Path(item).stem
                future = executor.submit(
                    run_single_config, config, config_name, output_lock, loose, blind
                )
                future_to_config[future] = config_name

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_config):
            config_name = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                with output_lock:
                    print(f"Exception for {config_name}: {str(e)}")
                results.append(
                    {"config": config_name, "status": "error", "message": str(e)}
                )

    # Print summary
    print(f"\n{'=' * 80}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - successful
    print(f"Total: {len(results)} | Successful: {successful} | Failed: {failed}")

    if failed > 0:
        print("\nFailed configurations:")
        for r in results:
            if r["status"] == "error":
                print(f"  - {r['config']}: {r.get('message', 'Unknown error')}")

    print(f"{'=' * 80}\n")

    return results


def main():
    # Load default config from file
    default_config = load_config()

    parser = argparse.ArgumentParser(
        description="Simplified Online Planning Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (default: config.yaml in script directory)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=default_config.get("mode", "pipeline"),
        choices=[
            "inference",
            "evaluation",
            "pipeline",
            "inference_next1",
            "evaluation_next1",
            "pipeline_next1",
            "inference_next1_early",
            "evaluation_next1_early",
            "pipeline_next1_early",
            "inference_next1_late",
            "evaluation_next1_late",
            "pipeline_next1_late",
            "inference_next2",
            "evaluation_next2",
            "pipeline_next2",
            "inference_next3",
            "evaluation_next3",
            "pipeline_next3",
            "inference_tool_broken",
            "evaluation_tool_broken",
            "pipeline_tool_broken",
            "inference_tool_redundancy",
            "evaluation_tool_redundancy",
            "pipeline_tool_redundancy",
            "inference_self_refinement",
            "evaluation_self_refinement",
            "pipeline_self_refinement",
            "inference_self_refinement_claude",
            "evaluation_self_refinement_claude",
            "pipeline_self_refinement_claude",
            "inference_refinement_without_judge",
            "evaluation_refinement_without_judge",
            "pipeline_refinement_without_judge",
            "inference_refinement_without_judge_claude",
            "evaluation_refinement_without_judge_claude",
            "pipeline_refinement_without_judge_claude",
        ],
        help="Execution mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=default_config.get("dataset"),
        choices=[
            "framethinker",
            "gaia",
            "gta",
            "opencua",
            "skywork",
            "skywork_doc",
            "skywork_excel",
            "skywork_normal",
            "skywork_ppt",
            "skywork_train",
            "toolbench",
        ],
        help="Dataset to evaluate (required for inference and pipeline modes)",
    )
    parser.add_argument(
        "--prediction-file",
        type=str,
        default=default_config.get("prediction_file"),
        help="Path to prediction file (required for evaluation mode)",
    )
    parser.add_argument(
        "--test-model",
        type=str,
        default=default_config.get("test_model", "claude-sonnet-4-20250514"),
        help="Test model name",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=default_config.get("eval_model", "gemini-3-pro-preview"),
        help="Evaluation model name",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=default_config.get("max_samples"),
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--test-max-tokens",
        type=int,
        default=default_config.get("test_max_tokens", 64000),
        help="Maximum tokens for test model generation",
    )
    parser.add_argument(
        "--num-redundant-tools",
        type=int,
        default=default_config.get("num_redundant_tools", 4),
        help="Number of redundant tools to inject (for tool_redundancy mode)",
    )
    parser.add_argument(
        "--eval-max-tokens",
        type=int,
        default=default_config.get("eval_max_tokens", 64000),
        help="Maximum tokens for eval model generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,  # Changed from default_config.get('output_dir', 'eval_results') to None to detect user input
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=default_config.get("resume", True),
        help="Resume from existing predictions",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not resume from existing predictions, start fresh",
    )

    parser.add_argument(
        "--loose",
        action="store_true",
        help="Use loose evaluation prompts (less strict)",
    )

    parser.add_argument(
        "--blind",
        action="store_true",
        help="Use blind evaluation prompts (no reference trajectory)",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch mode to run multiple config files in parallel",
    )
    parser.add_argument(
        "--batch-configs",
        type=str,
        nargs="+",
        help="List of config files to run in batch mode",
    )
    parser.add_argument(
        "--batch-pattern",
        type=str,
        help='Glob pattern to match config files for batch mode (e.g., "config/gemini_*.yaml")',
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers for batch mode",
    )

    args = parser.parse_args()

    # Handle batch mode
    if args.batch:
        config_items = []

        # Collect config files from --batch-configs
        if args.batch_configs:
            config_items.extend(args.batch_configs)

        # Collect config files from --batch-pattern
        if args.batch_pattern:
            script_dir = Path(__file__).parent
            matched_files = list(script_dir.glob(args.batch_pattern))
            config_items.extend([str(f) for f in matched_files])

        if not config_items:
            parser.error(
                "--batch mode requires either --batch-configs or --batch-pattern"
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_configs = []
        for cf in config_items:
            if cf not in seen:
                seen.add(cf)
                unique_configs.append(cf)

        # Run batch evaluation
        batch_evaluate(
            unique_configs,
            max_workers=args.max_workers,
            loose=args.loose,
            blind=args.blind,
        )
        return

    # Check if single config file has batch mode enabled
    if args.config:
        config_data = load_config(args.config)
        if config_data.get("batch_mode"):
            # This is a batch configuration file
            tests = config_data.get("tests", [])
            if not tests:
                parser.error("Batch config file has no tests defined")

            max_workers = config_data.get("max_workers", args.max_workers)

            print(f"\n{'=' * 80}")
            print(f"Loading batch configuration from: {args.config}")
            print(f"Number of tests: {len(tests)}")
            print(f"{'=' * 80}\n")

            # Run batch evaluation with test configs
            if args.max_samples is not None:
                print(
                    f"  ℹ️  Overriding max_samples for all tests to: {args.max_samples}"
                )
                for test in tests:
                    test["max_samples"] = args.max_samples

            if args.output_dir is not None:
                print(f"  ℹ️  Overriding output_dir for all tests to: {args.output_dir}")
                for test in tests:
                    test["output_dir"] = args.output_dir

            if args.blind is not None:
                print(f"  ℹ️  Overriding blind for all tests to: {args.blind}")
                for test in tests:
                    test["blind"] = args.blind

            if args.loose is not None:
                print(f"  ℹ️  Overriding loose for all tests to: {args.loose}")
                for test in tests:
                    test["loose"] = args.loose

            batch_evaluate(
                tests, max_workers=max_workers, loose=args.loose, blind=args.blind
            )
            return

    # Single test mode - original logic
    # If a custom config file is specified, reload config from it
    if args.config:
        custom_config = load_config(args.config)
        # Update defaults with custom config (command line args take precedence)
        for key, value in custom_config.items():
            attr_key = key.replace("-", "_")
            if not hasattr(args, attr_key) or getattr(
                args, attr_key
            ) == default_config.get(key):
                setattr(args, attr_key, value)

    # Set default output_dir if not specified
    if args.output_dir is None:
        args.output_dir = default_config.get("output_dir", "eval_results")

    # Validate arguments based on mode
    if args.mode in [
        "inference",
        "pipeline",
        "inference_next1",
        "pipeline_next1",
        "inference_next1_early",
        "evaluation_next1_early",
        "pipeline_next1_early",
        "inference_next1_late",
        "evaluation_next1_late",
        "pipeline_next1_late",
        "inference_next2",
        "pipeline_next2",
        "inference_next3",
        "pipeline_next3",
        "inference_tool_broken",
        "pipeline_tool_broken",
        "evaluation_tool_broken",
        "inference_tool_redundancy",
        "pipeline_tool_redundancy",
        "evaluation_tool_redundancy",
        "inference_self_refinement",
        "pipeline_self_refinement",
        "evaluation_self_refinement",
        "inference_self_refinement_claude",
        "pipeline_self_refinement_claude",
        "evaluation_self_refinement_claude",
        "inference_refinement_without_judge",
        "pipeline_refinement_without_judge",
        "evaluation_refinement_without_judge",
        "inference_refinement_without_judge_claude",
        "pipeline_refinement_without_judge_claude",
        "evaluation_refinement_without_judge_claude",
    ]:
        if not args.dataset:
            parser.error(f"--dataset is required for {args.mode} mode")

    # Create evaluator (moved before validation to enable auto-detection)
    dataset_name = args.dataset if args.dataset else "unknown"

    # Determine prediction_mode based on args.mode (with truncation position for next1)
    reflection_model = None
    if "tool_redundancy" in args.mode:
        prediction_mode = "tool_redundancy"
    elif "tool_broken" in args.mode:
        prediction_mode = "tool_broken"
    elif "self_refinement" in args.mode:
        prediction_mode = "self_refinement"
        if "claude" in args.mode:
            prediction_mode = "self_refinement_claude"
            reflection_model = "anthropic/claude-sonnet-4.5"
    elif "refinement_without_judge" in args.mode:
        prediction_mode = "refinement_without_judge"
        if "claude" in args.mode:
            prediction_mode = "refinement_without_judge_claude"
            reflection_model = "anthropic/claude-sonnet-4.5"
    elif "next3" in args.mode:
        prediction_mode = "next3"
    elif "next2" in args.mode:
        prediction_mode = "next2"
    elif "next1_early" in args.mode:
        prediction_mode = "next1_early"
    elif "next1_late" in args.mode:
        prediction_mode = "next1_late"
    else:
        prediction_mode = "next1"

    evaluator: Union[Next1Evaluator, Next2Evaluator, Next3Evaluator] = (
        OnlinePlanningEvaluator(  # type: ignore[assignment]
            dataset_name=dataset_name,
            test_model=args.test_model,
            eval_model=args.eval_model,
            max_samples=args.max_samples,
            test_max_tokens=args.test_max_tokens,
            eval_max_tokens=args.eval_max_tokens,
            output_dir=args.output_dir,
            resume=args.resume,
            prediction_mode=prediction_mode,
            use_loose_prompt=args.loose,
            use_blind_prompt=args.blind,
            num_redundant_tools=args.num_redundant_tools,
            reflection_model=reflection_model,
        )
    )

    print(f"{'!' * 80}\n")
    print(f"Evaluator created for {dataset_name} with {args.mode} mode.")
    # Execute based on mode
    if "inference" in args.mode and "pipeline" not in args.mode:
        # Determine which next mode
        # Determine which next mode
        # Determine which next mode
        mode_name = (
            "Tool-Redundancy Inference"
            if "tool_redundancy" in args.mode
            else (
                "Tool-Broken Inference"
                if "tool_broken" in args.mode
                else (
                    "Self-Refinement Inference"
                    if "self_refinement" in args.mode
                    else (
                        "Refinement Without Judge Inference"
                        if "refinement_without_judge" in args.mode
                        else (
                            "Next2 Inference"
                            if "next2" in args.mode
                            else "Next1 Inference"
                            if "next1" in args.mode
                            else "Inference"
                        )
                    )
                )
            )
        )
        print(f"\n{'=' * 80}")
        print(f"MODE: {mode_name} Only")
        if args.resume:
            print("Resume: Enabled (will skip existing predictions)")
        else:
            print("Resume: Disabled (starting fresh)")
        print(f"{'=' * 80}")

        # Run inference (unified method for both next1 and next2)
        predictions = evaluator.run_inference()

        # Save predictions (unified method)
        evaluator.save_predictions(predictions)

        print(f"\n✓ {mode_name} completed. {len(predictions)} predictions generated.")

    elif "evaluation" in args.mode and "pipeline" not in args.mode:
        # Determine which next mode
        # Determine which next mode
        # Determine which next mode
        mode_name = (
            "Tool-Redundancy Evaluation"
            if "tool_redundancy" in args.mode
            else (
                "Tool-Broken Evaluation"
                if "tool_broken" in args.mode
                else (
                    "Self-Refinement Evaluation"
                    if "self_refinement" in args.mode
                    else (
                        "Refinement Without Judge Evaluation"
                        if "refinement_without_judge" in args.mode
                        else (
                            "Next2 Evaluation"
                            if "next2" in args.mode
                            else "Next1 Evaluation"
                            if "next1" in args.mode
                            else "Evaluation"
                        )
                    )
                )
            )
        )
        print(f"\n{'=' * 80}")
        print(f"MODE: {mode_name} Only")
        if args.resume:
            print("Resume: Enabled (will skip existing evaluations)")
        else:
            print("Resume: Disabled (starting fresh)")
        print(f"{'=' * 80}")

        # Auto-detect prediction file if not specified
        prediction_file_to_use = args.prediction_file
        if not prediction_file_to_use:
            prediction_file_to_use = evaluator.find_latest_prediction_file()
            if not prediction_file_to_use:
                parser.error(
                    f"No prediction file found for dataset {args.dataset} and model {args.test_model}. Please specify --prediction-file"
                )
            print(f"  ℹ️  Using auto-detected prediction file: {prediction_file_to_use}")

        # Load predictions (unified method)
        predictions = evaluator.load_predictions(prediction_file_to_use)

        # Run evaluation (unified method)
        results = evaluator.run_evaluation(predictions)

        # Calculate and print statistics (unified method)
        stats = evaluator.calculate_statistics(results)
        evaluator.print_statistics(stats)

        # Save results (unified method)
        evaluator.save_results(results)

        print(f"\n✓ {mode_name} completed. {len(results)} results generated.")

    elif "pipeline" in args.mode:
        # Determine which next mode
        # Determine which next mode
        # Determine which next mode
        mode_name = (
            "Tool-Redundancy Pipeline"
            if "tool_redundancy" in args.mode
            else (
                "Tool-Broken Pipeline"
                if "tool_broken" in args.mode
                else (
                    "Self-Refinement Pipeline"
                    if "self_refinement" in args.mode
                    else (
                        "Next2 Pipeline"
                        if "next2" in args.mode
                        else "Next1 Pipeline"
                        if "next1" in args.mode
                        else "Pipeline"
                    )
                )
            )
        )
        print(f"\n{'=' * 80}")
        print(f"MODE: {mode_name} (Inference + Evaluation)")
        if args.resume:
            print("Resume: Enabled (will skip existing results)")
        else:
            print("Resume: Disabled (starting fresh)")
        print(f"{'=' * 80}")

        # Step 1: Run inference
        print(f"\n{'=' * 80}")
        print("STEP 1: Running Inference")
        print(f"{'=' * 80}")
        predictions = evaluator.run_inference()

        # Step 2: Save predictions
        prediction_file = evaluator.save_predictions(predictions)
        print(f"✓ Predictions saved to: {prediction_file}")

        # Step 3: Run evaluation on the predictions
        print(f"\n{'=' * 80}")
        print("STEP 2: Running Evaluation")
        print(f"{'=' * 80}")
        results = evaluator.run_evaluation(predictions)

        # Step 4: Calculate and print statistics
        stats = evaluator.calculate_statistics(results)
        evaluator.print_statistics(stats)

        # Step 5: Save results
        evaluator.save_results(results)

        print(f"\n✓ {mode_name} completed. {len(results)} results generated.")


if __name__ == "__main__":
    main()
