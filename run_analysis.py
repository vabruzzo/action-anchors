"""Step 3: Analyze and visualize resampling results."""

import argparse
import json
from pathlib import Path

import numpy as np

from action_anchors.analysis.visualize import (
    plot_aggregate_by_position,
    plot_agreement_curve,
    plot_entropy_curve,
    plot_fcp_histogram,
    plot_importance_bars,
    plot_task_comparison,
)


def load_results(task_name: str) -> list[dict]:
    path = Path("action_anchors/outputs") / f"resampling_{task_name}.json"
    with open(path) as f:
        return json.load(f)


def print_summary(results: list[dict], task_name: str) -> None:
    """Print summary statistics for a task."""
    n = len(results)
    fcps = [r["first_commitment_point"] for r in results]
    valid_fcps = [f for f in fcps if f is not None]
    baseline_agrees = [r["baseline_agreement"] for r in results]
    final_agrees = [r["final_agreement"] for r in results]
    n_sentences = [r["n_sentences"] for r in results]

    print(f"\n{'=' * 60}")
    print(f"  {task_name.upper()} — Summary ({n} examples)")
    print(f"{'=' * 60}")
    print(f"  CoT length:           {np.mean(n_sentences):.1f} +/- {np.std(n_sentences):.1f} sentences")
    print(f"  Baseline agreement:   {np.mean(baseline_agrees):.3f} +/- {np.std(baseline_agrees):.3f}")
    print(f"  Final agreement:      {np.mean(final_agrees):.3f} +/- {np.std(final_agrees):.3f}")
    if valid_fcps:
        print(f"  First commitment pt:  {np.mean(valid_fcps):.1f} +/- {np.std(valid_fcps):.1f} (sentence index)")
        fcp_pcts = [f / s for f, s in zip(fcps, n_sentences) if f is not None]
        print(f"  FCP as % of CoT:      {np.mean(fcp_pcts):.1%}")
    print(f"  Never committed:      {n - len(valid_fcps)} / {n} examples")

    # Find most interesting examples (highest max importance)
    results_with_profiles = [r for r in results if r["importance_profile"]]
    print(f"\n  Top 5 examples by max single-sentence importance:")
    for r in sorted(results_with_profiles, key=lambda r: max(p["importance"] for p in r["importance_profile"]), reverse=True)[:5]:
        best = max(r["importance_profile"], key=lambda p: p["importance"])
        print(
            f"    {r['example_id']}: sentence {best['sentence_idx']} "
            f"(importance={best['importance']:.3f}): \"{best['sentence_text'][:80]}...\""
        )


def generate_plots(results: list[dict], task_name: str, plots_dir: Path) -> None:
    """Generate all plots for a task."""
    task_dir = plots_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Per-example plots (top 10 by max importance)
    results_with_profiles = [r for r in results if r["importance_profile"]]
    top_examples = sorted(
        results_with_profiles,
        key=lambda r: max(p["importance"] for p in r["importance_profile"]),
        reverse=True,
    )[:10]

    for r in top_examples:
        eid = r["example_id"]
        profile = r["importance_profile"]
        plot_agreement_curve(profile, eid, task_dir / f"{eid}_agreement.png")
        plot_importance_bars(profile, eid, task_dir / f"{eid}_importance.png")
        plot_entropy_curve(profile, eid, task_dir / f"{eid}_entropy.png")

    # Aggregate plots
    all_profiles = [r["importance_profile"] for r in results]
    plot_aggregate_by_position(
        all_profiles,
        task_dir / "aggregate_importance.png",
        title=f"{task_name.upper()}: Aggregate Importance by Position",
    )

    fcps = [r["first_commitment_point"] for r in results]
    plot_fcp_histogram(fcps, task_dir / "fcp_histogram.png", title=f"{task_name.upper()}: First Commitment Points")

    print(f"  Plots saved to {task_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Analyze resampling results")
    parser.parse_args()

    plots_dir = Path("action_anchors/outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    gsm8k_results = None
    factual_results = None

    # GSM8K
    gsm8k_path = Path("action_anchors/outputs/resampling_gsm8k.json")
    if gsm8k_path.exists():
        gsm8k_results = load_results("gsm8k")
        print_summary(gsm8k_results, "gsm8k")
        generate_plots(gsm8k_results, "gsm8k", plots_dir)

    # Factual recall
    factual_path = Path("action_anchors/outputs/resampling_factual_recall.json")
    if factual_path.exists():
        factual_results = load_results("factual_recall")
        print_summary(factual_results, "factual_recall")
        generate_plots(factual_results, "factual_recall", plots_dir)

    # Cross-task comparison
    if gsm8k_results and factual_results:
        print("\n=== Cross-Task Comparison ===")
        gsm_pairs = [(r["first_commitment_point"], r["n_sentences"]) for r in gsm8k_results]
        fact_pairs = [(r["first_commitment_point"], r["n_sentences"]) for r in factual_results]

        gsm_fcp_pct = [f / s for f, s in gsm_pairs if f is not None]
        fact_fcp_pct = [f / s for f, s in fact_pairs if f is not None]

        if gsm_fcp_pct and fact_fcp_pct:
            print(f"  GSM8K avg commitment at {np.mean(gsm_fcp_pct):.0%} of CoT")
            print(f"  Factual avg commitment at {np.mean(fact_fcp_pct):.0%} of CoT")

        gsm_profiles = [r["importance_profile"] for r in gsm8k_results]
        fact_profiles = [r["importance_profile"] for r in factual_results]
        plot_task_comparison(gsm_profiles, fact_profiles, plots_dir / "task_comparison.png")
        print(f"  Comparison plot saved to {plots_dir / 'task_comparison.png'}")


if __name__ == "__main__":
    main()
