"""Visualization functions for action anchors results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_agreement_curve(
    profile: list[dict],
    example_id: str,
    save_path: str | Path,
) -> None:
    """Plot tool_choice_agreement vs sentence index for one example."""
    indices = [p["sentence_idx"] for p in profile]
    agreements = [p["agreement"] for p in profile]

    plt.figure(figsize=(10, 4))
    plt.plot(indices, agreements, "b-o", markersize=4)
    plt.xlabel("Sentence Index")
    plt.ylabel("Tool Choice Agreement")
    plt.title(f"Agreement Curve: {example_id}")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_importance_bars(
    profile: list[dict],
    example_id: str,
    save_path: str | Path,
) -> None:
    """Plot per-sentence importance as a colored bar chart."""
    indices = [p["sentence_idx"] for p in profile]
    importances = [p["importance"] for p in profile]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in importances]

    plt.figure(figsize=(10, 4))
    plt.bar(indices, importances, color=colors, alpha=0.8, edgecolor="white")
    plt.xlabel("Sentence Index")
    plt.ylabel("Importance (delta agreement)")
    plt.title(f"Sentence Importance: {example_id}")
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_entropy_curve(
    profile: list[dict],
    example_id: str,
    save_path: str | Path,
) -> None:
    """Plot decision entropy vs sentence index."""
    indices = [p["sentence_idx"] for p in profile]
    entropies = [p["entropy"] for p in profile]

    plt.figure(figsize=(10, 4))
    plt.plot(indices, entropies, "r-o", markersize=4)
    plt.xlabel("Sentence Index")
    plt.ylabel("Decision Entropy (bits)")
    plt.title(f"Decision Entropy: {example_id}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_aggregate_by_position(
    all_profiles: list[list[dict]],
    save_path: str | Path,
    title: str = "Aggregate Sentence Importance by Position",
    n_bins: int = 10,
) -> None:
    """Aggregate importance across examples, normalizing by CoT length.

    Bins sentences into position buckets (0-10%, 10-20%, etc.).
    """
    bins: list[list[float]] = [[] for _ in range(n_bins)]

    for profile in all_profiles:
        n = len(profile)
        if n == 0:
            continue
        for p in profile:
            bucket = min(int(p["sentence_idx"] / n * n_bins), n_bins - 1)
            bins[bucket].append(p["importance"])

    means = [np.mean(b) if b else 0 for b in bins]
    stds = [np.std(b) / np.sqrt(len(b)) if len(b) > 1 else 0 for b in bins]
    x_labels = [f"{i * (100 // n_bins)}-{(i + 1) * (100 // n_bins)}%" for i in range(n_bins)]

    plt.figure(figsize=(10, 5))
    plt.bar(x_labels, means, yerr=stds, capsize=3, alpha=0.7, color="#3498db", edgecolor="white")
    plt.xlabel("Position in CoT (normalized)")
    plt.ylabel("Mean Importance")
    plt.title(title)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_fcp_histogram(
    fcps: list[int | None],
    save_path: str | Path,
    title: str = "Distribution of First Commitment Points",
) -> None:
    """Histogram of first commitment points."""
    valid_fcps = [f for f in fcps if f is not None]

    plt.figure(figsize=(8, 5))
    if valid_fcps:
        plt.hist(valid_fcps, bins=min(20, max(len(set(valid_fcps)), 5)), alpha=0.7, edgecolor="black", color="#9b59b6")
    plt.xlabel("First Commitment Point (sentence index)")
    plt.ylabel("Count")
    plt.title(title)
    n_none = sum(1 for f in fcps if f is None)
    if n_none > 0:
        plt.text(
            0.95, 0.95,
            f"{n_none} examples never reached 0.9 agreement",
            transform=plt.gca().transAxes,
            ha="right", va="top",
            fontsize=9, color="gray",
        )
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_task_comparison(
    gsm8k_profiles: list[list[dict]],
    factual_profiles: list[list[dict]],
    save_path: str | Path,
    n_bins: int = 10,
) -> None:
    """Compare aggregate importance between GSM8K and factual recall."""
    def bin_profiles(profiles: list[list[dict]]) -> tuple[list[float], list[float]]:
        bins: list[list[float]] = [[] for _ in range(n_bins)]
        for profile in profiles:
            n = len(profile)
            if n == 0:
                continue
            for p in profile:
                bucket = min(int(p["sentence_idx"] / n * n_bins), n_bins - 1)
                bins[bucket].append(p["importance"])
        means = [np.mean(b) if b else 0 for b in bins]
        stds = [np.std(b) / np.sqrt(len(b)) if len(b) > 1 else 0 for b in bins]
        return means, stds

    gsm_means, gsm_stds = bin_profiles(gsm8k_profiles)
    fact_means, fact_stds = bin_profiles(factual_profiles)

    x = np.arange(n_bins)
    width = 0.35
    x_labels = [f"{i * (100 // n_bins)}-{(i + 1) * (100 // n_bins)}%" for i in range(n_bins)]

    plt.figure(figsize=(12, 5))
    plt.bar(x - width / 2, gsm_means, width, yerr=gsm_stds, capsize=2, label="GSM8K (calculator)", alpha=0.7, color="#e74c3c")
    plt.bar(x + width / 2, fact_means, width, yerr=fact_stds, capsize=2, label="Factual Recall (search)", alpha=0.7, color="#3498db")
    plt.xlabel("Position in CoT (normalized)")
    plt.ylabel("Mean Importance")
    plt.title("GSM8K vs Factual Recall: When is the Tool Decision Made?")
    plt.xticks(x, x_labels)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
