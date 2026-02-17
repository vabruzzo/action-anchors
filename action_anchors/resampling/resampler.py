from vllm import LLM, SamplingParams

from action_anchors.agent.prompt_builder import PromptBuilder
from action_anchors.agent.tool_parser import GenerationResult, parse_continuation
from action_anchors.resampling.sentence_splitter import split_cot_into_sentences


class Resampler:
    """Core resampling engine for measuring sentence importance.

    Uses vLLM offline inference directly (no server needed). All prefixes for
    a transcript are batched into a single llm.generate() call — vLLM schedules
    them concurrently and prefix caching ensures shared prompt tokens are
    computed once.
    """

    def __init__(self, config: dict, llm: LLM):
        self.config = config
        self.llm = llm
        self.builder = PromptBuilder(config["model"]["name"])

    def resample_transcript(
        self,
        system_prompt: str,
        tools: list[dict],
        question: str,
        thinking: str,
        n_rollouts: int | None = None,
    ) -> list[dict]:
        """Resample all sentence prefixes for a single transcript.

        All prefixes are submitted as a single batch to llm.generate() so vLLM
        can schedule them concurrently. Prefix caching means the shared base
        prompt (system + tools + user message) is computed once, and successive
        prefixes incrementally extend the cached KV.

        Returns a list of dicts, one per prefix (including k=-1 empty baseline).
        """
        n_rollouts = n_rollouts or self.config["resampling"]["n_rollouts"]
        split = split_cot_into_sentences(thinking)

        # Build all prompts upfront: empty baseline + one per sentence prefix
        prefixes_with_meta = [("", -1, "<empty>")]
        for k, prefix in enumerate(split.prefixes):
            prefixes_with_meta.append((prefix, k, split.sentences[k]))

        prompts = [
            self.builder.build_resample_prompt(system_prompt, tools, question, prefix)
            for prefix, _, _ in prefixes_with_meta
        ]

        # Single batched generation for all prefixes
        params = SamplingParams(
            n=n_rollouts,
            max_tokens=self.config["resampling"]["max_new_tokens"],
            temperature=self.config["resampling"]["temperature"],
            top_p=self.config["resampling"]["top_p"],
            stop=["<|im_end|>"],
        )
        all_outputs = self.llm.generate(prompts, params)

        # Parse results
        results = []
        for output, (prefix, k, sentence_text) in zip(all_outputs, prefixes_with_meta):
            conts = [out.text for out in output.outputs]
            parsed = [parse_continuation(prefix, c) for c in conts]
            results.append({
                "sentence_idx": k,
                "sentence_text": sentence_text,
                "prefix": prefix,
                "continuations": conts,
                "parsed": parsed,
            })

        return results
