# Agent Planning Benchmark

This repository contains the evaluation code for **Agent Planning Benchmark
(APB)**, a diagnostic benchmark for planning capabilities in LLM agents.

Paper: [Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents](https://arxiv.org/abs/2606.04874)  
Dataset: [Mikivis/AgentPlanningbBenchmark](https://huggingface.co/datasets/Mikivis/AgentPlanningbBenchmark)

APB evaluates whether an agent can decompose goals, select tools, reason over
constraints, predict the next step in a trajectory, handle noisy or broken tool
sets, and refuse infeasible tasks. The released dataset contains 4,209
multimodal cases covering Holistic Planning, Step-wise Planning,
Tool-Extraneous, Tool-Broken, and Unsolvable settings.

## Repository Layout

- `evaluation/run_benchmark.py`: main evaluator.
- `scripts/prepare_data.py`: downloads the Hugging Face dataset and restores assets.
- `requirements.txt`: Python dependencies.

The repository is code-only. Benchmark records and assets are hosted on
Hugging Face.

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

The Hugging Face repository stores assets as:

- `assets_archives/Holistic_assets.tar`
- `assets_archives/Step-wise_assets.tar`

`scripts/prepare_data.py` extracts them automatically, creating
`assets/Holistic/` and `assets/Step-wise/` under the data directory. The
evaluator can also auto-extract these archives when prediction starts.

## Run Evaluation

The evaluator has three stages:

1. `predict`: run the tested model on APB records.
2. `judge`: judge model outputs with `gemini-3-pro-preview`.
3. `score`: aggregate accuracy and grade metrics.

The model and judge clients use OpenAI-compatible chat-completions APIs.

```bash
export APB_TEST_MODEL=YOUR_TEST_MODEL
export APB_TEST_BASE_URL=https://YOUR_TEST_ENDPOINT/v1
export APB_TEST_API_KEY=YOUR_TEST_API_KEY

export APB_JUDGE_MODEL=gemini-3-pro-preview
export APB_JUDGE_BASE_URL=https://YOUR_GEMINI_COMPATIBLE_ENDPOINT/v1
export APB_JUDGE_API_KEY=YOUR_JUDGE_API_KEY

python evaluation/run_benchmark.py \
  --data-dir data/AgentPlanningbBenchmark \
  --splits all \
  --stage all \
  --predictions outputs/predictions.jsonl \
  --judgements outputs/judgements.jsonl \
  --scores outputs/scores.json
```

For a quick smoke test:

```bash
python evaluation/run_benchmark.py \
  --data-dir data/AgentPlanningbBenchmark \
  --splits holistic_planning \
  --stage all \
  --max-samples 5
```

## Split Names

- `holistic_planning`
- `holistic_tool_extraneous`
- `holistic_unsolvable`
- `step_wise_planning`
- `step_wise_tool_extraneous`
- `step_wise_tool_broken`

Use `--splits all` to evaluate every split, or pass a comma-separated list.

## Stage-By-Stage Usage

Generate model outputs:

```bash
python evaluation/run_benchmark.py \
  --data-dir data/AgentPlanningbBenchmark \
  --splits step_wise_planning \
  --stage predict \
  --predictions outputs/step_wise_predictions.jsonl
```

Judge existing outputs:

```bash
python evaluation/run_benchmark.py \
  --stage judge \
  --predictions outputs/step_wise_predictions.jsonl \
  --judgements outputs/step_wise_judgements.jsonl
```

Aggregate scores:

```bash
python evaluation/run_benchmark.py \
  --stage score \
  --judgements outputs/step_wise_judgements.jsonl \
  --scores outputs/step_wise_scores.json
```

Scores are grouped by overall, split, planning regime, and task family.

## Notes

- `gemini-3-pro-preview` is the default judge model.
- Use `--no-response-format` if an endpoint does not support JSON response format.
- The evaluator resumes from existing JSONL outputs and skips already processed records.
- API keys and private endpoints should be supplied through environment variables; do not commit them.
