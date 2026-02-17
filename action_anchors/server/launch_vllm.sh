#!/bin/bash
# Launch vLLM server optimized for resampling workload on H100
#
# Key flags:
# - --enable-prefix-caching: CRITICAL for resampling efficiency. The shared
#   prefix across all N rollouts is computed once and reused.
# - --max-model-len 8192: Sufficient for GSM8K (<2K) and factual recall (<3K).
# - --gpu-memory-utilization 0.92: Qwen3-8B bf16 ~16GB, leaving ~57GB for KV cache.
#
# Use /v1/completions endpoint (NOT /v1/chat/completions) for raw prompt injection.

vllm serve Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --disable-log-requests \
    --port 8000
