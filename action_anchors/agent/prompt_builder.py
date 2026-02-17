from transformers import AutoTokenizer


class PromptBuilder:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_initial_prompt(
        self,
        system_prompt: str,
        tools: list[dict],
        user_message: str,
    ) -> str:
        """Build full prompt for initial generation.

        Returns raw string ending with the generation prompt
        (e.g. '<|im_start|>assistant\\n') ready for the model to generate.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return prompt

    def build_resample_prompt(
        self,
        system_prompt: str,
        tools: list[dict],
        user_message: str,
        cot_prefix: str,
    ) -> str:
        """Build prompt with partial CoT for resampling.

        Args:
            cot_prefix: Text of sentences 0..k from inside the <think> block.
                        Does NOT include the <think> tag itself.
                        Pass "" for the empty-prefix baseline (k=-1).

        Returns raw string ending with the partial CoT, ready for continuation.
        The model will continue generating more thinking and/or a tool call.
        """
        base = self.build_initial_prompt(system_prompt, tools, user_message)
        # base ends with '<|im_start|>assistant\n'
        # Model normally generates '<think>\n...' after this.
        # We inject the think tag + partial CoT.
        return base + "<think>\n" + cot_prefix

    def build_multi_turn_prompt(
        self,
        system_prompt: str,
        tools: list[dict],
        user_message: str,
        assistant_response: str,
        tool_result: str,
    ) -> str:
        """Build prompt for multi-turn: user -> assistant (tool call) -> tool result -> assistant.

        Used during baseline collection when the model calls a tool and we need
        to feed back the result for continued generation.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response},
            {"role": "user", "content": f"<tool_response>\n{tool_result}\n</tool_response>"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return prompt
