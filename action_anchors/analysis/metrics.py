import math
from collections import Counter
from dataclasses import dataclass

from action_anchors.agent.tool_parser import GenerationResult


@dataclass
class AgreementMetrics:
    # Primary: does the model make the same tool choice?
    tool_choice_agreement: float

    # Secondary: even if same tool, are the arguments similar?
    argument_exact_match: float
    argument_similarity: float

    # Breakdown
    n_same_tool_same_args: int
    n_same_tool_diff_args: int
    n_different_tool: int
    n_no_tool: int
    n_parse_error: int
    n_total: int

    # Entropy of tool choice distribution
    decision_entropy: float


def compute_agreement(
    original_tool_call: dict,
    rollout_results: list[GenerationResult],
) -> AgreementMetrics:
    """Compare rollout tool calls against the original.

    Args:
        original_tool_call: {"name": str, "arguments": dict}
        rollout_results: List of parsed GenerationResults from resampled continuations
    """
    original_name = original_tool_call["name"]
    original_args = original_tool_call["arguments"]

    same_tool_same_args = 0
    same_tool_diff_args = 0
    diff_tool = 0
    no_tool = 0
    parse_error = 0

    tool_choices: list[str] = []
    arg_sims: list[float] = []

    for r in rollout_results:
        if r.parse_error:
            parse_error += 1
            tool_choices.append("__parse_error__")
            continue

        if not r.tool_calls:
            no_tool += 1
            tool_choices.append("__no_tool__")
            continue

        first_tc = r.tool_calls[0]
        tool_choices.append(first_tc.name)

        if first_tc.name != original_name:
            diff_tool += 1
        elif first_tc.arguments == original_args:
            same_tool_same_args += 1
            arg_sims.append(1.0)
        else:
            same_tool_diff_args += 1
            arg_sims.append(_compute_arg_similarity(original_args, first_tc.arguments))

    n_total = len(rollout_results)
    n_same_tool = same_tool_same_args + same_tool_diff_args

    # Entropy of tool choice distribution
    entropy = 0.0
    if n_total > 0:
        counts = Counter(tool_choices)
        for c in counts.values():
            p = c / n_total
            if p > 0:
                entropy -= p * math.log2(p)

    return AgreementMetrics(
        tool_choice_agreement=n_same_tool / n_total if n_total > 0 else 0.0,
        argument_exact_match=same_tool_same_args / n_total if n_total > 0 else 0.0,
        argument_similarity=sum(arg_sims) / len(arg_sims) if arg_sims else 0.0,
        n_same_tool_same_args=same_tool_same_args,
        n_same_tool_diff_args=same_tool_diff_args,
        n_different_tool=diff_tool,
        n_no_tool=no_tool,
        n_parse_error=parse_error,
        n_total=n_total,
        decision_entropy=entropy,
    )


def _compute_arg_similarity(orig_args: dict, rollout_args: dict) -> float:
    """Compute similarity between argument dicts.

    Uses Jaccard token overlap on the string representation as a simple baseline.
    """
    if orig_args == rollout_args:
        return 1.0

    orig_str = " ".join(str(v) for v in orig_args.values())
    roll_str = " ".join(str(v) for v in rollout_args.values())

    orig_tokens = set(orig_str.lower().split())
    roll_tokens = set(roll_str.lower().split())

    if not orig_tokens or not roll_tokens:
        return 0.0

    return len(orig_tokens & roll_tokens) / len(orig_tokens | roll_tokens)
