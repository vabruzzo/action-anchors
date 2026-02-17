import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw_text: str


@dataclass
class GenerationResult:
    thinking: str
    tool_calls: list[ToolCall]
    final_answer: str | None
    raw_output: str
    parse_error: bool = False


def parse_generation(raw_output: str) -> GenerationResult:
    """Parse raw model output into structured result.

    Expects output in Qwen3 format:
        <think>...reasoning...</think>
        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>
    or:
        <think>...reasoning...</think>
        Direct text answer
    """
    thinking = ""
    content_after_think = raw_output

    # Extract thinking block
    think_match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        content_after_think = raw_output[think_match.end() :].strip()
    elif raw_output.strip().startswith("<think>"):
        # Think block opened but never closed (hit max tokens)
        thinking = raw_output.strip()[len("<think>") :].strip()
        return GenerationResult(
            thinking=thinking,
            tool_calls=[],
            final_answer=None,
            raw_output=raw_output,
            parse_error=True,
        )

    # Extract tool calls
    tool_calls = []
    parse_error = False
    tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
    for match in tool_call_pattern.finditer(content_after_think):
        raw_tc = match.group(1).strip()
        try:
            parsed = json.loads(raw_tc)
            tool_calls.append(
                ToolCall(
                    name=parsed["name"],
                    arguments=parsed.get("arguments", {}),
                    raw_text=raw_tc,
                )
            )
        except (json.JSONDecodeError, KeyError):
            parse_error = True

    # Final answer is content after </think> that isn't a tool call
    final_answer = None
    if not tool_calls:
        # Strip any failed tool_call tags
        cleaned = tool_call_pattern.sub("", content_after_think).strip()
        if cleaned:
            final_answer = cleaned

    return GenerationResult(
        thinking=thinking,
        tool_calls=tool_calls,
        final_answer=final_answer,
        raw_output=raw_output,
        parse_error=parse_error,
    )


def parse_continuation(cot_prefix: str, continuation: str) -> GenerationResult:
    """Parse a resampled continuation.

    The cot_prefix was injected into the prompt (text inside <think> before
    sentence k). continuation is what the model generated after that prefix.
    We reconstruct the full output and parse it.
    """
    full_output = "<think>\n" + cot_prefix + continuation
    return parse_generation(full_output)
