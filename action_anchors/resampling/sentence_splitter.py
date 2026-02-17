import re
from dataclasses import dataclass


@dataclass
class SentenceSplit:
    sentences: list[str]  # Individual sentences
    prefixes: list[str]  # prefixes[k] = sentences[0] + ... + sentences[k]


def split_cot_into_sentences(thinking_text: str) -> SentenceSplit:
    """Split CoT thinking text into sentences.

    Primary strategy: split on newlines. Reasoning models (especially Qwen3)
    tend to write one thought per line, making newline splitting the most
    reliable approach.

    Within long single lines, also split on sentence-ending punctuation
    (. ? !) followed by a space and an uppercase letter, while avoiding
    splits on decimals (3.14) and common abbreviations (e.g., i.e.).
    """
    lines = thinking_text.split("\n")
    raw_sentences = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Try within-line splitting for long lines
        # Pattern: sentence-ending punctuation followed by space + uppercase letter
        # Negative lookbehind: don't split after a single digit (decimals like 3.14)
        # or after common abbreviation prefixes
        parts = re.split(
            r"(?<!\d)"  # not after digit (avoids 3.14)
            r"(?<!\be\.g)"  # not after e.g
            r"(?<!\bi\.e)"  # not after i.e
            r"(?<!\bvs)"  # not after vs
            r"(?<!\betc)"  # not after etc
            r"([.!?])"  # capture the punctuation
            r"(?=\s+[A-Z])",  # followed by whitespace + uppercase
            stripped,
        )

        # Reassemble: reattach punctuation to preceding text
        i = 0
        while i < len(parts):
            if i + 1 < len(parts) and parts[i + 1] in ".!?":
                raw_sentences.append(parts[i] + parts[i + 1])
                i += 2
            else:
                if parts[i].strip():
                    raw_sentences.append(parts[i])
                i += 1

    # Clean up
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return SentenceSplit(sentences=[], prefixes=[])

    # Build cumulative prefixes
    # Use newline as separator to preserve the format the model expects
    prefixes = []
    for k in range(len(sentences)):
        prefixes.append("\n".join(sentences[: k + 1]) + "\n")

    return SentenceSplit(sentences=sentences, prefixes=prefixes)
