# Agent Planning Benchmark

This repository contains runnable evaluation code for **Agent Planning
Benchmark (APB)**.

Paper: [Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents](https://arxiv.org/abs/2606.04874)  
Dataset: [Mikivis/AgentPlanningbBenchmark](https://huggingface.co/datasets/Mikivis/AgentPlanningbBenchmark)

APB evaluates agent planning under Holistic Planning, Step-wise Planning,
Tool-Extraneous, Tool-Broken, and Unsolvable settings. The Hugging Face dataset
contains 4,209 cases:

| Split | Count |
| --- | ---: |
| `holistic_planning` | 1,109 |
| `holistic_tool_extraneous` | 750 |
| `holistic_unsolvable` | 400 |
| `step_wise_planning` | 900 |
| `step_wise_tool_extraneous` | 750 |
| `step_wise_tool_broken` | 300 |

## Repository Layout

- `evaluation/run_benchmark.py`: unified predict, judge, and score runner.
- `evaluation/apb_eval/`: shared data loading, prompt construction, model calls, judging, and scoring logic.
- `evaluation/tasks/`: per-task scripts for Holistic/offline and Step-wise/online experiments.
- `scripts/prepare_data.py`: downloads the Hugging Face dataset and extracts asset archives.
- `legacy_reference/`: cleaned reference copies of earlier experiment scripts. These are kept for tracing the original experiment logic; the supported runnable entrypoints are under `evaluation/`.

The repository is code-only. Benchmark records and assets are hosted on Hugging
Face.

## Installation

```bash
git clone https://github.com/Mikivishy/AgentPlanningBenchmark.git
cd AgentPlanningBenchmark
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

```bash
python scripts/prepare_data.py \
  --repo-id Mikivis/AgentPlanningbBenchmark \
  --output-dir data/AgentPlanningbBenchmark
```

The Hugging Face repository stores multimodal assets as tar archives:

- `assets_archives/Holistic_assets.tar`
- `assets_archives/Step-wise_assets.tar`

`scripts/prepare_data.py` extracts them automatically into `assets/Holistic/`
and `assets/Step-wise/`.

## API Configuration

The evaluator uses OpenAI-compatible chat-completions APIs for both the tested
model and the judge model. `gemini-3-pro-preview` is the default judge.

```bash
export APB_TEST_MODEL=YOUR_TEST_MODEL
export APB_TEST_BASE_URL=https://YOUR_TEST_ENDPOINT/v1
export APB_TEST_API_KEY=YOUR_TEST_API_KEY

export APB_JUDGE_MODEL=gemini-3-pro-preview
export APB_JUDGE_BASE_URL=https://YOUR_GEMINI_COMPATIBLE_ENDPOINT/v1
export APB_JUDGE_API_KEY=YOUR_JUDGE_API_KEY
```

Use `--no-response-format` if an endpoint does not support JSON response
format.

## Run All Evaluation

```bash
python evaluation/run_benchmark.py \
  --data-dir data/AgentPlanningbBenchmark \
  --splits all \
  --stage all \
  --predictions outputs/all_predictions.jsonl \
  --judgements outputs/all_judgements.jsonl \
  --scores outputs/all_scores.json
```

Stages:

- `predict`: run the tested model and write JSONL predictions.
- `judge`: judge predictions with `gemini-3-pro-preview` or the configured judge.
- `judge_score`: judge predictions and immediately aggregate scores.
- `score`: aggregate accuracy and grade by overall, split, planning regime, task family, and task category.
- `all`: run predict, judge, and score in sequence.

## Holistic / Offline Scripts

Holistic/offline all:

```bash
python evaluation/tasks/offline_all.py \
  --data-dir data/AgentPlanningbBenchmark \
  --predictions outputs/holistic_predictions.jsonl \
  --judgements outputs/holistic_judgements.jsonl \
  --scores outputs/holistic_scores.json
```

Per-task scripts:

| Task | Predict script | Judge script | Full script |
| --- | --- | --- | --- |
| Holistic Planning | `evaluation/tasks/test_holistic_planning.py` | `evaluation/tasks/judge_holistic_planning.py` | `evaluation/tasks/run_holistic_planning.py` |
| Holistic Tool-Extraneous | `evaluation/tasks/test_holistic_tool_extraneous.py` | `evaluation/tasks/judge_holistic_tool_extraneous.py` | `evaluation/tasks/run_holistic_tool_extraneous.py` |
| Holistic Unsolvable | `evaluation/tasks/test_holistic_unsolvable.py` | `evaluation/tasks/judge_holistic_unsolvable.py` | `evaluation/tasks/run_holistic_unsolvable.py` |

Example:

```bash
python evaluation/tasks/test_holistic_planning.py \
  --data-dir data/AgentPlanningbBenchmark \
  --predictions outputs/holistic_planning_predictions.jsonl

python evaluation/tasks/judge_holistic_planning.py \
  --predictions outputs/holistic_planning_predictions.jsonl \
  --judgements outputs/holistic_planning_judgements.jsonl \
  --scores outputs/holistic_planning_scores.json
```

## Step-wise / Online Scripts

Step-wise/online all:

```bash
python evaluation/tasks/online_all.py \
  --data-dir data/AgentPlanningbBenchmark \
  --predictions outputs/step_wise_predictions.jsonl \
  --judgements outputs/step_wise_judgements.jsonl \
  --scores outputs/step_wise_scores.json
```

Per-task scripts:

| Task | Predict script | Judge script | Full script |
| --- | --- | --- | --- |
| Step-wise Planning | `evaluation/tasks/test_step_wise_planning.py` | `evaluation/tasks/judge_step_wise_planning.py` | `evaluation/tasks/run_step_wise_planning.py` |
| Step-wise Tool-Extraneous | `evaluation/tasks/test_step_wise_tool_extraneous.py` | `evaluation/tasks/judge_step_wise_tool_extraneous.py` | `evaluation/tasks/run_step_wise_tool_extraneous.py` |
| Step-wise Tool-Broken | `evaluation/tasks/test_step_wise_tool_broken.py` | `evaluation/tasks/judge_step_wise_tool_broken.py` | `evaluation/tasks/run_step_wise_tool_broken.py` |

Example:

```bash
python evaluation/tasks/test_step_wise_tool_broken.py \
  --data-dir data/AgentPlanningbBenchmark \
  --predictions outputs/step_wise_tool_broken_predictions.jsonl

python evaluation/tasks/judge_step_wise_tool_broken.py \
  --predictions outputs/step_wise_tool_broken_predictions.jsonl \
  --judgements outputs/step_wise_tool_broken_judgements.jsonl \
  --scores outputs/step_wise_tool_broken_scores.json
```

## Split Aliases

`--splits all` evaluates every split. You may also pass:

- `--splits holistic` or `--splits offline`
- `--splits step_wise` or `--splits online`
- a comma-separated list, such as `holistic_planning,step_wise_tool_broken`

## Output Files

Prediction JSONL records include the original benchmark fields plus:

- `tested_model`
- `model_response`

Judgement JSONL records additionally include:

- `judge_model`
- `judgement`

The score JSON reports grouped accuracy and average grade.

## Notes

- The default runner resumes from existing JSONL outputs and skips completed records for the same model. Use `--no-resume` to rerun.
- API keys and private endpoints should be supplied through environment variables or CLI arguments and should not be committed.
- `legacy_reference/` contains sanitized copies of old scripts for reference. They may still use historical dataset names and local-path assumptions, so use `evaluation/` for the supported Hugging Face based workflow.
