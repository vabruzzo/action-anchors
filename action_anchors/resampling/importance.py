from dataclasses import dataclass

from action_anchors.analysis.metrics import AgreementMetrics


@dataclass
class SentenceImportance:
    sentence_idx: int
    sentence_text: str
    agreement: float  # tool_choice_agreement at this prefix
    importance: float  # agreement[k] - agreement[k-1]
    argument_agreement: float  # argument_exact_match at this prefix
    entropy: float  # decision_entropy at this prefix


def compute_importance_profile(
    agreement_sequence: list[AgreementMetrics],
    sentences: list[str],
) -> list[SentenceImportance]:
    """Compute importance for each sentence.

    Args:
        agreement_sequence: List of metrics where:
            - agreement_sequence[0] corresponds to k=-1 (empty prefix baseline)
            - agreement_sequence[i+1] corresponds to sentence i
        sentences: List of sentence texts
    """
    results = []
    for k in range(len(sentences)):
        prev = agreement_sequence[k].tool_choice_agreement
        curr = agreement_sequence[k + 1].tool_choice_agreement
        importance = curr - prev

        results.append(
            SentenceImportance(
                sentence_idx=k,
                sentence_text=sentences[k],
                agreement=curr,
                importance=importance,
                argument_agreement=agreement_sequence[k + 1].argument_exact_match,
                entropy=agreement_sequence[k + 1].decision_entropy,
            )
        )

    return results


def find_first_commitment_point(
    profile: list[SentenceImportance],
    threshold: float = 0.9,
) -> int | None:
    """Find the first sentence index where agreement exceeds threshold.

    This is the point where the model has effectively "decided" on the tool call.
    """
    for si in profile:
        if si.agreement >= threshold:
            return si.sentence_idx
    return None
