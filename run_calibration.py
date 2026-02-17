"""Step 1: Calibrate and collect baseline transcripts.

For each question, run the model N times and measure how often it uses the tool.
Keep only questions where tool use rate is in a target range (e.g., 20-80%).
For each kept question, save one tool-using sample as the baseline transcript
for resampling — no separate collection step needed.
"""

import argparse
import json
from pathlib import Path

import yaml
from vllm import LLM, SamplingParams

from action_anchors.agent.prompt_builder import PromptBuilder
from action_anchors.agent.tool_parser import parse_generation
from action_anchors.resampling.sentence_splitter import split_cot_into_sentences
from action_anchors.tasks.base import BaseTask
from action_anchors.tasks.factual_recall_search import FactualRecallSearchTask
from action_anchors.tasks.gsm8k_calculator import GSM8KCalculatorTask


def calibrate_and_collect(
    llm: LLM,
    task: BaseTask,
    task_name: str,
    config: dict,
    n_samples: int = 16,
    tool_use_range: tuple[float, float] = (0.2, 0.8),
) -> list[dict]:
    """Run each question n_samples times, measure tool use rate, filter,
    and save a baseline transcript for each kept question."""
    builder = PromptBuilder(config["model"]["name"])
    examples = task.get_examples()

    # Build prompts: each question repeated n_samples times
    prompts = []
    prompt_to_example = []
    for ex in examples:
        prompt = builder.build_initial_prompt(
            task.get_system_prompt(), task.get_tools(), ex.question
        )
        prompts.extend([prompt] * n_samples)
        prompt_to_example.extend([ex] * n_samples)

    params = SamplingParams(
        max_tokens=config["collection"]["max_new_tokens"],
        temperature=config["collection"]["temperature"],
        top_p=config["collection"]["top_p"],
        stop=["<|im_end|>"],
    )

    print(f"  Running {len(examples)} questions x {n_samples} samples = {len(prompts)} generations...")
    all_outputs = llm.generate(prompts, params)

    # Aggregate by example: track stats + keep one tool-using and one non-tool sample
    results_by_id: dict[str, dict] = {}
    for ex, output in zip(prompt_to_example, all_outputs):
        if ex.id not in results_by_id:
            results_by_id[ex.id] = {
                "example": ex,
                "n_tool_use": 0,
                "n_no_tool": 0,
                "n_parse_error": 0,
                "n_total": 0,
                "tool_transcript": None,     # Sample where tool was used
                "no_tool_transcript": None,  # Sample where tool was NOT used
            }
        entry = results_by_id[ex.id]
        entry["n_total"] += 1

        raw_output = output.outputs[0].text
        parsed = parse_generation(raw_output)

        if parsed.parse_error:
            entry["n_parse_error"] += 1
            continue

        split = split_cot_into_sentences(parsed.thinking)
        n_sentences = len(split.sentences)
        cot_ok = (
            config["filtering"]["min_cot_sentences"]
            <= n_sentences
            <= config["filtering"]["max_cot_sentences"]
        )

        if not parsed.tool_calls:
            entry["n_no_tool"] += 1
            # Save first clean no-tool sample
            if entry["no_tool_transcript"] is None and cot_ok:
                entry["no_tool_transcript"] = {
                    "example_id": ex.id,
                    "question": ex.question,
                    "raw_output": raw_output,
                    "thinking": parsed.thinking,
                    "n_sentences": n_sentences,
                    "tool_calls": [],
                    "first_tool_call": None,
                    "final_answer": parsed.final_answer,
                    "ground_truth": ex.ground_truth,
                    "metadata": ex.metadata,
                }
            continue

        entry["n_tool_use"] += 1

        # Save first clean tool-using sample
        if entry["tool_transcript"] is not None or not cot_ok:
            continue

        first_tc = parsed.tool_calls[0]
        if hasattr(task, "set_current_example"):
            task.set_current_example(ex.id)
        tool_result = task.execute_tool(first_tc.name, first_tc.arguments)

        entry["tool_transcript"] = {
            "example_id": ex.id,
            "question": ex.question,
            "raw_output": raw_output,
            "thinking": parsed.thinking,
            "n_sentences": n_sentences,
            "tool_calls": [
                {"name": tc.name, "arguments": tc.arguments, "raw": tc.raw_text}
                for tc in parsed.tool_calls
            ],
            "first_tool_call": {"name": first_tc.name, "arguments": first_tc.arguments},
            "tool_result": tool_result,
            "ground_truth": ex.ground_truth,
            "metadata": ex.metadata,
        }

    # Filter by tool use rate and collect transcripts
    lo, hi = tool_use_range
    calibration = []
    transcripts = []       # Tool-using baselines (for resampling)
    transcript_pairs = []  # Both tool and no-tool samples (for analysis)
    for eid, entry in results_by_id.items():
        rate = entry["n_tool_use"] / entry["n_total"]
        ex = entry["example"]
        row = {
            "id": ex.id,
            "question": ex.question,
            "ground_truth": ex.ground_truth,
            "tool_use_rate": rate,
            "n_tool_use": entry["n_tool_use"],
            "n_no_tool": entry["n_no_tool"],
            "n_parse_error": entry["n_parse_error"],
            "n_total": entry["n_total"],
            "metadata": ex.metadata,
        }
        calibration.append(row)

        if lo <= rate <= hi and entry["tool_transcript"] is not None:
            entry["tool_transcript"]["tool_use_rate"] = rate
            transcripts.append(entry["tool_transcript"])

            pair = {
                "example_id": ex.id,
                "question": ex.question,
                "tool_use_rate": rate,
                "tool_transcript": entry["tool_transcript"],
                "no_tool_transcript": entry["no_tool_transcript"],
            }
            transcript_pairs.append(pair)

    calibration.sort(key=lambda r: r["tool_use_rate"])

    # Save calibration stats
    output_dir = Path("action_anchors/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"calibration_{task_name}.json", "w") as f:
        json.dump(calibration, f, indent=2)

    # Save tool-using transcripts (ready for resampling)
    with open(output_dir / f"transcripts_{task_name}.json", "w") as f:
        json.dump(transcripts, f, indent=2)

    # Save paired transcripts (tool vs no-tool for each question)
    with open(output_dir / f"transcript_pairs_{task_name}.json", "w") as f:
        json.dump(transcript_pairs, f, indent=2)

    # Report
    rates = [r["tool_use_rate"] for r in calibration]
    n_with_both = sum(1 for p in transcript_pairs if p["no_tool_transcript"] is not None)
    print(f"\n  Tool use rates across {len(calibration)} questions:")
    print(f"    Always uses tool (>{hi}): {sum(1 for r in rates if r > hi)}")
    print(f"    Borderline ({lo}-{hi}):     {len(transcripts)}")
    print(f"    Never uses tool (<{lo}):  {sum(1 for r in rates if r < lo)}")
    print(f"  Kept {len(transcripts)} transcripts ready for resampling")
    print(f"  {n_with_both}/{len(transcript_pairs)} have both tool + no-tool samples")

    return transcripts


def main():
    parser = argparse.ArgumentParser(description="Calibrate tool-use rates and collect baselines")
    parser.add_argument(
        "--task",
        choices=["gsm8k", "factual_recall", "both"],
        default="both",
    )
    parser.add_argument("--config", default="action_anchors/config.yaml")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=16,
        help="Number of samples per question to estimate tool use rate",
    )
    parser.add_argument(
        "--tool-use-lo",
        type=float,
        default=0.2,
        help="Minimum tool use rate to keep",
    )
    parser.add_argument(
        "--tool-use-hi",
        type=float,
        default=0.8,
        help="Maximum tool use rate to keep",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print("Loading model...")
    llm = LLM(
        model=config["model"]["name"],
        max_model_len=config["model"]["max_model_len"],
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
    )

    tool_use_range = (args.tool_use_lo, args.tool_use_hi)

    if args.task in ("gsm8k", "both"):
        print("=== Calibrating GSM8K ===")
        task = GSM8KCalculatorTask(config)
        calibrate_and_collect(llm, task, "gsm8k", config, args.n_samples, tool_use_range)

    if args.task in ("factual_recall", "both"):
        print("=== Calibrating Factual Recall ===")
        task = FactualRecallSearchTask(config)
        calibrate_and_collect(llm, task, "factual_recall", config, args.n_samples, tool_use_range)


if __name__ == "__main__":
    main()
