"""Step 2: Run the resampling experiment.

For each collected transcript:
1. Split CoT into sentences
2. For each sentence prefix (including empty baseline), generate N rollouts
3. Parse tool calls from each rollout
4. Compute agreement metrics
5. Compute importance scores
6. Save results incrementally
"""

import argparse
import json
import time
from pathlib import Path

import yaml
from vllm import LLM

from action_anchors.analysis.metrics import AgreementMetrics, compute_agreement
from action_anchors.resampling.importance import (
    SentenceImportance,
    compute_importance_profile,
    find_first_commitment_point,
)
from action_anchors.resampling.resampler import Resampler
from action_anchors.resampling.sentence_splitter import split_cot_into_sentences
from action_anchors.tasks.factual_recall_search import SEARCH_TOOL
from action_anchors.tasks.gsm8k_calculator import CALCULATOR_TOOL


def get_tools_for_task(task_name: str) -> list[dict]:
    if task_name == "gsm8k":
        return [CALCULATOR_TOOL]
    elif task_name == "factual_recall":
        return [SEARCH_TOOL]
    else:
        raise ValueError(f"Unknown task: {task_name}")


def serialize_importance(profile: list[SentenceImportance]) -> list[dict]:
    return [
        {
            "sentence_idx": si.sentence_idx,
            "sentence_text": si.sentence_text,
            "agreement": si.agreement,
            "importance": si.importance,
            "argument_agreement": si.argument_agreement,
            "entropy": si.entropy,
        }
        for si in profile
    ]


def run_experiment(
    resampler: Resampler,
    task_name: str,
    config: dict,
    n_rollouts_override: int | None = None,
):
    """Run resampling for all transcripts of a given task."""
    transcripts_path = Path("action_anchors/outputs") / f"transcripts_{task_name}.json"
    with open(transcripts_path) as f:
        transcripts = json.load(f)

    output_path = Path("action_anchors/outputs") / f"resampling_{task_name}.json"

    # Load any existing results for resume capability
    existing_ids: set[str] = set()
    all_results: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
            existing_ids = {r["example_id"] for r in all_results}
        print(f"Resuming: {len(existing_ids)} examples already processed")

    system_prompt = config["tasks"][task_name]["system_prompt"].strip()
    tools = get_tools_for_task(task_name)

    for i, transcript in enumerate(transcripts):
        example_id = transcript["example_id"]
        if example_id in existing_ids:
            print(f"  [{i + 1}/{len(transcripts)}] {example_id} — skipped (already done)")
            continue

        t0 = time.time()
        print(f"  [{i + 1}/{len(transcripts)}] {example_id} — resampling...")

        rollout_results = resampler.resample_transcript(
            system_prompt,
            tools,
            transcript["question"],
            transcript["thinking"],
            n_rollouts=n_rollouts_override,
        )

        # Compute agreement metrics for each prefix
        agreement_seq: list[AgreementMetrics] = []
        for rr in rollout_results:
            metrics = compute_agreement(transcript["first_tool_call"], rr["parsed"])
            agreement_seq.append(metrics)

        # Compute importance profile
        split = split_cot_into_sentences(transcript["thinking"])
        profile = compute_importance_profile(agreement_seq, split.sentences)
        fcp = find_first_commitment_point(profile)

        result = {
            "example_id": example_id,
            "question": transcript["question"],
            "first_tool_call": transcript["first_tool_call"],
            "n_sentences": len(split.sentences),
            "first_commitment_point": fcp,
            "baseline_agreement": agreement_seq[0].tool_choice_agreement,
            "final_agreement": agreement_seq[-1].tool_choice_agreement if agreement_seq else 0.0,
            "importance_profile": serialize_importance(profile),
            "rollout_summary": [
                {
                    "sentence_idx": rr["sentence_idx"],
                    "n_rollouts": len(rr["continuations"]),
                    "n_tool_calls": sum(1 for p in rr["parsed"] if p.tool_calls),
                    "n_same_tool": sum(
                        1
                        for p in rr["parsed"]
                        if p.tool_calls
                        and p.tool_calls[0].name == transcript["first_tool_call"]["name"]
                    ),
                }
                for rr in rollout_results
            ],
        }
        all_results.append(result)
        existing_ids.add(example_id)

        elapsed = time.time() - t0
        print(
            f"    done in {elapsed:.1f}s | "
            f"baseline_agree={result['baseline_agreement']:.2f} "
            f"final_agree={result['final_agreement']:.2f} "
            f"fcp={fcp}"
        )

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nDone. {len(all_results)} examples processed. Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run resampling experiment")
    parser.add_argument(
        "--task",
        choices=["gsm8k", "factual_recall", "both"],
        default="both",
    )
    parser.add_argument("--config", default="action_anchors/config.yaml")
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=None,
        help="Override n_rollouts (use small value like 4 for dev)",
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
    resampler = Resampler(config, llm)

    if args.task in ("gsm8k", "both"):
        print("=== Resampling GSM8K ===")
        run_experiment(resampler, "gsm8k", config, args.n_rollouts)

    if args.task in ("factual_recall", "both"):
        print("=== Resampling Factual Recall ===")
        run_experiment(resampler, "factual_recall", config, args.n_rollouts)


if __name__ == "__main__":
    main()
