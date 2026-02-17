# Action Anchors: Implementation Plan

## Overview

Apply thought-anchors-style resampling to reasoning model chains of thought in agent settings to measure the causal importance of each CoT sentence for tool-calling decisions. Two isolated task domains: GSM8K (calculator tool) and factual recall (web search tool).

**Model**: Qwen3-8B  
**Hardware**: H100 (80GB)  
**Inference**: vLLM with prefix caching  
**Key metric**: P(same tool call | resample after sentence k) across N rollouts per sentence

---

## Architecture

```
action_anchors/
├── server/
│   └── launch_vllm.sh            # vLLM server launch script
├── tasks/
│   ├── base.py                    # Abstract task + tool definitions
│   ├── gsm8k_calculator.py        # GSM8K task with calculator tool
│   └── factual_recall_search.py   # Factual recall task with search tool
├── agent/
│   ├── prompt_builder.py          # Builds raw prompts with chat template + partial CoT
│   └── tool_parser.py             # Parses tool calls from raw model output
├── resampling/
│   ├── sentence_splitter.py       # Segments CoT into sentences
│   ├── resampler.py               # Core resampling engine
│   └── importance.py              # Computes sentence importance scores
├── analysis/
│   ├── metrics.py                 # Tool-call agreement, argument similarity, etc.
│   └── visualize.py               # Plotting importance scores
├── data/
│   ├── gsm8k_subset.json          # Curated GSM8K problems (see below)
│   └── factual_recall.json        # Factual recall questions (see below)
├── outputs/                       # Raw results, rollouts, importance scores
├── run_collection.py              # Step 1: Collect baseline tool-calling transcripts
├── run_resampling.py              # Step 2: Run resampling experiment
├── run_analysis.py                # Step 3: Analyze and visualize
└── config.yaml                    # All hyperparameters in one place
```

---

## Step 0: vLLM Server Setup

### `server/launch_vllm.sh`

```bash
#!/bin/bash
# Launch vLLM server optimized for resampling workload on H100

vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --disable-log-requests \
    --port 8000
```

**Key decisions:**
- `--enable-prefix-caching` is CRITICAL. When resampling from sentence k, the entire prefix up to that point is identical across all rollouts. Prefix caching means the KV cache for that prefix is computed once and reused for all N rollouts. This is where most of the speedup comes from.
- `--max-model-len 8192` is enough for GSM8K CoTs (usually <2K tokens) and factual recall (usually <3K tokens). Keep it conservative to maximize batch throughput.
- `--gpu-memory-utilization 0.92` — H100 has 80GB, Qwen3-8B in bf16 is ~16GB, leaving ~57GB for KV cache. This is plenty for high-throughput batched resampling.

**Do NOT use the chat/completions endpoint for resampling.** Use `/v1/completions` with raw text so we can inject partial CoT prefixes. The chat endpoint treats turns as atomic units.

---

## Step 1: Task Definitions

### Task A: GSM8K + Calculator

**Data**: Take 100 problems from GSM8K test set. Pre-filter for ones that involve at least one arithmetic operation (most do). Store in `data/gsm8k_subset.json`.

```json
[
    {
        "id": "gsm8k_001",
        "question": "Janet's ducks lay 16 eggs per day...",
        "answer": 64,
        "answer_str": "64"
    }
]
```

**Tool definition (one tool):**

```json
{
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Use this for any arithmetic calculation.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '(16 - 3) * 4 + 2'"
                }
            },
            "required": ["expression"]
        }
    }
}
```

**System prompt:**
```
You are a helpful math assistant. You have access to a calculator tool. Use it whenever you need to perform arithmetic. Think step by step, then use the calculator for each computation.
```

**What makes this interesting for action anchors:**
- Some GSM8K problems require multiple tool calls — which sentence triggers each?
- Some computations are trivial (e.g., 2+3) and the model might skip the tool. When does it decide to use vs. skip the calculator?
- The model might plan out the full approach before making any tool call — how early is the tool choice "locked in"?

### Task B: Factual Recall + Web Search

**Data**: Build 100 factual questions. Mix of:
- Questions the model likely knows (easy facts) — will it still search?
- Questions the model likely doesn't know (obscure facts) — when does it decide to search?
- Questions that are borderline — most interesting for studying the decision

Store in `data/factual_recall.json`:

```json
[
    {
        "id": "fact_001",
        "question": "What is the population of Liechtenstein?",
        "difficulty": "medium",
        "expected_tool_use": true,
        "ground_truth": "approximately 39,000"
    }
]
```

**Categories to include (roughly 25 each):**
1. **Easy, no search needed**: "What is the capital of France?" (model should know)
2. **Hard, search needed**: "What was the exact vote count in the 2023 Liechtenstein general election?" (model can't know)
3. **Borderline**: "What is the current population of Estonia?" (model might know roughly but should verify)
4. **Trick/trap**: "Who won the 2026 Super Bowl?" (future event at model's training time, must search)

**Tool definition:**

```json
{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current or factual information. Returns a short snippet with the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }
    }
}
```

**System prompt:**
```
You are a helpful assistant with access to web search. Use it when you're not confident in your answer or when the question requires current information. Think step by step about whether you need to search before answering.
```

**Simulated tool responses:**
We are NOT actually calling a search API. We pre-compute ground truth answers and return a canned response:
```
Search results for "{query}": According to [source], the answer is {ground_truth}.
```
This keeps the experiment controlled — the tool response is deterministic, so any variation in behavior comes from the CoT, not from noisy search results.

Similarly for calculator — we evaluate the expression with Python `eval()` (with safety sandboxing) and return the result.

---

## Step 2: Prompt Building and Raw Completion Interface

### `agent/prompt_builder.py`

This is the most critical piece. We need to:

1. Take the Qwen3-8B chat template and apply it manually with the system prompt, tools, user message, and any prior turns.
2. For resampling, inject a PARTIAL assistant turn that includes the `<think>` tag and the first k sentences of the CoT.
3. Return the raw string that we send to `/v1/completions`.

**Implementation approach:**

```python
from transformers import AutoTokenizer

class PromptBuilder:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def build_initial_prompt(self, system_prompt: str, tools: list[dict], user_message: str) -> str:
        """Build the full prompt for initial generation (no partial CoT)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        # Use the tokenizer's chat template to get the raw string
        # Include tool definitions in the format Qwen3 expects
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True
        )
        return prompt

    def build_resample_prompt(self, system_prompt: str, tools: list[dict],
                               user_message: str, cot_prefix: str) -> str:
        """Build prompt with partial CoT for resampling.

        cot_prefix is everything inside <think>...</think> up to and including sentence k.
        The model will continue generating from here.
        """
        base_prompt = self.build_initial_prompt(system_prompt, tools, user_message)
        # Append the partial thinking block
        # The model's response starts with <think>, so we append that + the prefix
        return base_prompt + "<think>\n" + cot_prefix
```

**IMPORTANT**: Verify the exact format Qwen3 uses for its thinking tags. Load the tokenizer and inspect `apply_chat_template` output to confirm. The thinking block might use `<think>` or might be handled differently. Check the Qwen3 documentation/HuggingFace model card.

**IMPORTANT**: Also verify how Qwen3 formats tool calls in its output. It likely uses a specific format like:

```
</think>

<tool_call>
{"name": "calculator", "arguments": {"expression": "16 * 3"}}
</tool_call>
```

Inspect several raw generations to confirm the exact format before writing the parser.

### `agent/tool_parser.py`

Parse the raw model output to extract:

```python
@dataclass
class ToolCall:
    name: str              # "calculator" or "web_search"
    arguments: dict        # {"expression": "16*3"} or {"query": "population of Estonia"}
    raw_text: str          # The raw text of the tool call block

@dataclass
class GenerationResult:
    thinking: str          # Full text inside <think>...</think>
    tool_calls: list[ToolCall]  # Parsed tool calls (could be 0 or more)
    final_answer: str | None    # If model answered without tool call
    raw_output: str        # Complete raw output
```

Parse using regex or string splitting — do NOT use an LLM for this. The format should be deterministic.

Handle edge cases:
- Model answers directly without any tool call
- Model makes multiple tool calls
- Model's CoT gets cut off by max tokens
- Malformed tool call output (discard these)

---

## Step 3: Sentence Splitter

### `resampling/sentence_splitter.py`

Split the `<think>` block into sentences. This is what we iterate over during resampling.

```python
def split_cot_into_sentences(thinking_text: str) -> list[str]:
    """Split CoT into sentences. Return list of sentences with their text."""
    # ...
```

**Approach**: Use a simple rule-based splitter. Split on sentence-ending punctuation (`. `, `! `, `? `) followed by a space or newline. Do NOT use spaCy or NLTK — keep dependencies minimal and fast.

Also handle:
- Newlines as sentence boundaries (reasoning models often use one sentence per line)
- Numbered steps like "1. First, ..." "2. Then, ..."
- Don't split on decimals like "3.14"

Return list of sentences AND cumulative prefixes:

```python
@dataclass
class SentenceSplit:
    sentences: list[str]           # Individual sentences
    prefixes: list[str]            # prefixes[k] = sentences[0] + ... + sentences[k]
    # So prefixes[k] is what you pass as cot_prefix to build_resample_prompt
```

---

## Step 4: Core Resampling Engine

### `resampling/resampler.py`

This is the performance-critical piece.

```python
class Resampler:
    def __init__(self, vllm_base_url: str = "http://localhost:8000"):
        self.base_url = vllm_base_url
        self.completions_url = f"{vllm_base_url}/v1/completions"

    async def resample_from_prefix(
        self,
        prompt_with_prefix: str,
        n_rollouts: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        """Generate n_rollouts continuations from a given prefix.

        Uses vLLM's `n` parameter to generate multiple completions in a single
        request. This is MUCH more efficient than sending n separate requests
        because vLLM can batch them and prefix caching ensures the shared prefix
        KV cache is computed only once.
        """
        payload = {
            "model": "Qwen/Qwen3-8B",
            "prompt": prompt_with_prefix,
            "n": n_rollouts,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["</s>", "<|im_end|>"],  # Verify correct stop tokens for Qwen3
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(self.completions_url, json=payload) as resp:
                result = await resp.json()
                return [choice["text"] for choice in result["choices"]]

    async def compute_sentence_importance(
        self,
        system_prompt: str,
        tools: list[dict],
        user_message: str,
        original_thinking: str,
        original_tool_calls: list[ToolCall],
        n_rollouts: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> list[SentenceImportance]:
        """Compute importance of each sentence for the tool-calling decision.

        For each sentence k in the CoT:
        1. Build prefix = sentences[0:k+1]
        2. Resample n_rollouts continuations from that prefix
        3. Parse tool calls from each continuation
        4. Measure agreement with original tool call
        """
        splitter = split_cot_into_sentences(original_thinking)
        builder = PromptBuilder()
        results = []

        for k, prefix in enumerate(splitter.prefixes):
            prompt = builder.build_resample_prompt(
                system_prompt, tools, user_message, prefix
            )
            raw_continuations = await self.resample_from_prefix(
                prompt, n_rollouts, max_new_tokens=2048, temperature=temperature, top_p=top_p
            )
            # Parse each continuation
            parsed = [parse_tool_calls(cont) for cont in raw_continuations]
            # Compute metrics against original
            importance = compute_agreement_metrics(original_tool_calls, parsed)
            results.append(SentenceImportance(
                sentence_idx=k,
                sentence_text=splitter.sentences[k],
                prefix_text=prefix,
                n_rollouts=n_rollouts,
                metrics=importance,
            ))

        return results
```

**Efficiency notes:**

1. **Use `n` parameter, not separate requests.** Sending one request with `n=64` is far more efficient than 64 separate requests. vLLM handles the batching internally and prefix caching means the shared prefix is computed once.

2. **Process sentences in order.** Sentence k's prefix is a strict subset of sentence k+1's prefix. With prefix caching, the KV cache for sentence k is reused when computing sentence k+1. So iterating in order is naturally efficient.

3. **Async all the way.** Use `asyncio` and `aiohttp` for all vLLM calls. When processing multiple examples, you can pipeline: while rollouts for example A, sentence k are generating, you can be analyzing results from example A, sentence k-1.

4. **Batch across examples too.** If you want to be really efficient, you could send prefixes from different examples in the same batch. But this is an optimization — start with one example at a time and parallelize across examples only if throughput is a bottleneck.

---

## Step 5: Metrics

### `analysis/metrics.py`

For each sentence k, given the original tool call(s) and the tool calls from N resampled rollouts, compute:

```python
@dataclass
class AgreementMetrics:
    # Primary metric: does the model make the same tool choice?
    tool_choice_agreement: float  # Fraction of rollouts with same tool name (or same "no tool")

    # Secondary: even if same tool, are the arguments similar?
    argument_exact_match: float   # Fraction with identical arguments
    argument_similarity: float    # Average similarity of arguments (e.g., for search queries, use embedding similarity; for calculator, use expression equivalence)

    # Breakdown
    n_same_tool_same_args: int
    n_same_tool_diff_args: int
    n_different_tool: int
    n_no_tool: int
    n_total: int
```

**Importance score for sentence k:**

The importance of sentence k is the CHANGE in agreement when going from prefix k-1 to prefix k:

```python
importance[k] = tool_choice_agreement[k] - tool_choice_agreement[k-1]
```

Where `tool_choice_agreement[-1]` (before any CoT) is the baseline — resample from the very beginning of the thinking block.

A large positive value at sentence k means: "After seeing this sentence, the model becomes much more committed to the original tool call." This is the action anchor.

A large negative value would mean: "This sentence actually pushed the model AWAY from the original tool call, but later sentences overcame it." Also interesting.

**Additional metrics to compute:**
- `first_commitment_point`: The smallest k where tool_choice_agreement > 0.9. This is when the model has effectively "decided."
- `decision_entropy[k]`: Entropy of the tool choice distribution at each prefix point. High entropy = model is still undecided. Low entropy = committed.

---

## Step 6: Experiment Pipeline

### `run_collection.py` — Step 1: Collect Baseline Transcripts

For each problem:
1. Run the model with tools available, temperature=0.7, generate 1 completion
2. Parse the full CoT and tool calls
3. Execute the tool call (calculator eval or canned search response)
4. If multi-turn (model uses tool result to continue), complete the full trajectory
5. Save the full transcript

Filter for examples where:
- The model actually uses a tool (discard examples where it answers directly, unless studying "no tool" decisions)
- The CoT is between 3 and 30 sentences (too short = nothing to study, too long = expensive)
- The tool call parsed correctly

**Target**: ~50 clean examples per task domain (100 total).

### `run_resampling.py` — Step 2: Resampling

For each collected transcript:
1. Split CoT into sentences
2. For each sentence k (including k=0, the baseline with empty prefix):
   - Build the prefix prompt
   - Generate N=64 rollouts
   - Parse tool calls from each rollout
   - Compute agreement metrics
3. Save all rollouts and metrics

**Compute budget estimate:**
- 100 examples × ~15 sentences average × 64 rollouts × ~500 tokens per rollout
- = ~48M tokens of generation
- Qwen3-8B on H100 at ~3000 tok/s (conservative, with batching) = ~4.4 hours
- Prefix caching makes this substantially faster in practice since most of the prompt is cached
- **Realistic estimate: 2-4 hours for the full experiment**

### `run_analysis.py` — Step 3: Analysis

Compute and plot:
1. **Per-example importance profiles**: Line plot of tool_choice_agreement vs. sentence index for each example
2. **Aggregate importance by sentence position**: Average across examples (normalized by CoT length)
3. **First commitment point distribution**: Histogram of when the model "decides"
4. **Compare GSM8K vs factual recall**: Do tool decisions get made earlier/later in different domains?
5. **Interesting case studies**: Flag examples where importance is concentrated in unexpected places

---

## Step 7: Multi-Turn Handling

Some examples will involve the model calling a tool, getting a result, then continuing. For the initial version:

**Simplify: Only study the FIRST tool call decision.**

The resampling happens within the CoT that precedes the first tool call. We measure whether that first tool call changes. This avoids the complexity of simulating full multi-turn trajectories with resampled CoTs.

Extension for later: After resampling, if the model makes the same tool call, feed it the same tool result and see if the rest of the trajectory diverges. But this is scope creep for a mini project.

---

## Step 8: Config

### `config.yaml`

```yaml
model:
  name: "Qwen/Qwen3-8B"
  vllm_url: "http://localhost:8000"
  max_model_len: 8192

resampling:
  n_rollouts: 64
  temperature: 0.7
  top_p: 0.9
  max_new_tokens: 2048

collection:
  gsm8k_n_problems: 100
  factual_recall_n_problems: 100
  min_cot_sentences: 3
  max_cot_sentences: 30

tasks:
  gsm8k:
    system_prompt: "You are a helpful math assistant. You have access to a calculator tool. Use it whenever you need to perform arithmetic. Think step by step, then use the calculator for each computation."
  factual_recall:
    system_prompt: "You are a helpful assistant with access to web search. Use it when you're not confident in your answer or when the question requires current information. Think step by step about whether you need to search before answering."
```

---

## Dependencies

```
# requirements.txt
vllm>=0.8.0
transformers
aiohttp
asyncio
numpy
pandas
matplotlib
seaborn
pyyaml
datasets  # for loading GSM8K from HuggingFace
```

---

## Build Order (for the coding agent)

**Phase 1: Infrastructure (build and test first)**
1. `server/launch_vllm.sh` — get vLLM serving Qwen3-8B
2. `agent/prompt_builder.py` — build prompts, VERIFY the chat template and tool format by generating a few examples and inspecting raw output
3. `agent/tool_parser.py` — parse tool calls from raw output, test on the examples from step 2
4. `resampling/sentence_splitter.py` — test on a few real CoTs

**Phase 2: Data**
5. `tasks/gsm8k_calculator.py` — load GSM8K, define calculator tool, implement tool execution (safe eval)
6. `tasks/factual_recall_search.py` — create factual recall dataset, define search tool, implement canned responses
7. `data/` — generate and save the datasets

**Phase 3: Collection**
8. `run_collection.py` — collect baseline transcripts, inspect them manually, filter for quality

**Phase 4: Core Experiment**
9. `resampling/resampler.py` — implement the core resampling loop
10. `analysis/metrics.py` — implement agreement metrics
11. `run_resampling.py` — run the full experiment

**Phase 5: Analysis**
12. `analysis/visualize.py` — plotting
13. `run_analysis.py` — generate all plots and summary statistics

---

## Sanity Checks (IMPORTANT — do these throughout)

- [ ] Verify Qwen3-8B actually uses the calculator/search tools reliably. If it ignores tools, tweak the system prompt or try few-shot examples.
- [ ] Verify the chat template is correct by comparing `apply_chat_template` output to Qwen3 documentation.
- [ ] Verify prefix caching is working: check vLLM logs for cache hit rates. If resampling is not faster than generating from scratch, something is wrong.
- [ ] Verify tool parsing works on 100% of clean generations. If not, fix the parser or filter out malformed outputs.
- [ ] Spot-check resampling results manually. For sentence k=0 (empty prefix), agreement should be LOW (model hasn't started reasoning yet). For k=last sentence, agreement should be HIGH (model has done all its reasoning). If this pattern doesn't hold, debug.
- [ ] Check that `n=64` in a single request actually works with vLLM without OOM or timeout. Start with `n=4` and scale up.

---

## What "Done" Looks Like

**Minimum viable result:**
- Importance profiles for 50+ examples across both domains
- Clear visualization showing where in the CoT the tool decision gets made
- Basic comparison of GSM8K vs factual recall decision dynamics
- 2-3 interesting case studies highlighted

**Stretch goals:**
- Evidence of "early commitment" (model decides tool before articulating full reasoning)
- Examples where stated reasoning doesn't match causal importance (unfaithful CoT about tool choice)
- Quantitative comparison: "In GSM8K, the tool decision is locked in by sentence X on average; in factual recall, by sentence Y"