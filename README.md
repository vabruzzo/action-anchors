# Action Anchors

Measuring the causal importance of individual Chain-of-Thought sentences for tool-calling decisions in LLM agents.

Given a reasoning model (Qwen3-8B) with access to tools, we use counterfactual resampling to identify which CoT sentences are "action anchors" — the moments where the model commits to (or abandons) a tool call.

## Method

Based on the [Thought Anchors](https://arxiv.org/abs/2505.15457) methodology, adapted for tool-calling:

1. **Calibrate**: Run each question N times to find "borderline" questions where the model uses the tool 20-80% of the time
2. **Resample**: For each borderline transcript, truncate the CoT at every sentence boundary and generate 64 continuations from each prefix
3. **Measure**: Compare each continuation's tool call to the original — compute tool choice agreement at each prefix position
4. **Score**: Importance of sentence k = agreement(k) - agreement(k-1). Sentences that cause large jumps are "action anchors"

## Tasks

- **GSM8K + Calculator**: Math word problems with an optional calculator tool
- **Factual Recall + Web Search**: Knowledge questions with an optional web search tool

## Setup

Requires Python 3.11+ and a GPU with sufficient VRAM (tested on H100).

```bash
# Install dependencies
uv sync

# Generate the GSM8K subset (downloads from HuggingFace)
uv run python -m action_anchors.tasks.gsm8k_calculator
```

## Running

```bash
# Step 1: Calibrate — find borderline questions and collect transcripts
uv run python run_calibration.py --task both --n-samples 16 --tool-use-lo 0.2 --tool-use-hi 0.8

# Step 2: Resample — generate continuations from each prefix
uv run python run_resampling.py --task both

# Step 3: Analyze — compute importance scores and generate plots
uv run python run_analysis.py
```

Use `--n-rollouts 4` on the resampling step for quick dev runs (default is 64).

## Output

Results are saved to `action_anchors/outputs/`:

- `calibration_{task}.json` — tool use rates for all questions
- `transcripts_{task}.json` — baseline transcripts for borderline questions
- `transcript_pairs_{task}.json` — paired tool-using and non-tool-using samples
- `resampling_{task}.json` — full prefix-by-prefix agreement data
- `plots/` — agreement curves, importance bar charts, entropy curves, aggregate comparisons

## Project Structure

```
action_anchors/
  agent/
    prompt_builder.py    # Builds Qwen3 chat-template prompts for vLLM
    tool_parser.py       # Parses <think>/<tool_call> output format
  resampling/
    sentence_splitter.py # Splits CoT into sentences
    resampler.py         # Core resampling engine (batched vLLM inference)
    importance.py        # Computes per-sentence importance scores
  analysis/
    metrics.py           # Agreement metrics (tool choice, arguments, entropy)
    visualize.py         # Matplotlib plotting functions
  tasks/
    base.py              # Abstract task interface
    gsm8k_calculator.py  # GSM8K + safe arithmetic eval
    factual_recall_search.py  # Factual questions + canned search responses
  data/
    factual_recall.json  # 100 factual recall questions (4 difficulty categories)
  config.yaml            # All hyperparameters

run_calibration.py       # Step 1: Calibrate and collect
run_resampling.py        # Step 2: Resample from prefixes
run_analysis.py          # Step 3: Analyze and plot
```
