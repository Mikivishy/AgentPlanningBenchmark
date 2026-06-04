# Configuration File Guide

## Overview

The main.py script now supports loading configuration from YAML files, making it easier to manage different evaluation scenarios without typing long command lines.

## Usage

### Method 1: Use default config.yaml
```bash
# The script automatically loads config.yaml from the script directory
cd eval/online_planning_v2
python main.py
```

### Method 2: Specify a custom config file
```bash
python main.py --config path/to/custom_config.yaml
```

### Method 3: Override config with command-line arguments
```bash
# Config file is loaded first, then command-line args override it
python main.py --config config_examples/gaia_pipeline.yaml --max-samples 10
```

## Priority Order

Command-line arguments > Custom config file > Default config.yaml > Built-in defaults

## Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `mode` | string | Execution mode: `inference`, `evaluation`, `pipeline`, `inference_next2`, `evaluation_next2`, `pipeline_next2` | `pipeline` |
| `dataset` | string | Dataset name: `gaia`, `gta`, `toolbench`, `skywork`, etc. | - |
| `prediction_file` | string | Path to prediction file (for evaluation mode) | `null` |
| `test_model` | string | Model used for generating predictions | `claude-sonnet-4-20250514` |
| `eval_model` | string | Model used for evaluation | `gemini-3-pro-preview` |
| `max_samples` | int/null | Maximum number of samples to process (`null` = all) | `null` |
| `test_max_tokens` | int | Max tokens for test model | `64000` |
| `eval_max_tokens` | int | Max tokens for eval model | `64000` |
| `output_dir` | string | Output directory for results | `eval_results` |
| `resume` | boolean | Resume from existing predictions | `true` |

## Example Configurations

### 1. GAIA Pipeline (config_examples/gaia_pipeline.yaml)
```yaml
mode: pipeline
dataset: gaia
test_model: claude-sonnet-4-20250514
eval_model: gemini-3-pro-preview
```
Run with: `python main.py --config config_examples/gaia_pipeline.yaml`

### 2. GTA Next2 Pipeline (config_examples/gta_next2_pipeline.yaml)
```yaml
mode: pipeline_next2
dataset: gta
max_samples: 100
```
Run with: `python main.py --config config_examples/gta_next2_pipeline.yaml`

### 3. Inference Only (config_examples/inference_only.yaml)
```yaml
mode: inference
dataset: toolbench
test_model: gpt-4o
max_samples: 50
```
Run with: `python main.py --config config_examples/inference_only.yaml`

### 4. Evaluation Only (config_examples/evaluation_only.yaml)
```yaml
mode: evaluation
prediction_file: eval_results/predictions/gaia_predictions.json
eval_model: gemini-3-pro-preview
```
Run with: `python main.py --config config_examples/evaluation_only.yaml`

## Creating Your Own Config

1. Copy `config.yaml` or any example to a new file
2. Modify the parameters as needed
3. Use `null` for parameters you don't need (e.g., `dataset: null` for evaluation mode)
4. Run with `python main.py --config your_config.yaml`

## Tips

- **Quick Testing**: Set `max_samples: 10` to quickly test your configuration
- **Fresh Start**: Set `resume: false` to start from scratch
- **Multiple Configs**: Keep different configs for different datasets/models
- **Override Single Param**: Use command-line args to override just one parameter without editing the config file

## Examples

```bash
# Use default config
python main.py

# Use custom config
python main.py --config my_experiment.yaml

# Use config but override max_samples
python main.py --config gaia_pipeline.yaml --max-samples 5

# Use config but disable resume
python main.py --config gaia_pipeline.yaml --no-resume

# Mix config and command-line args
python main.py --config base.yaml --dataset gta --test-model gpt-4o
```
