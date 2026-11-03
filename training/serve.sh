#!/usr/bin/env bash
# Serve base Qwen3-8B and the B3 LoRA adapter from ONE vLLM server with the germline chat template.
#   training/serve.sh [adapter_dir] [port]
# Requests with model="Qwen/Qwen3-8B" hit the base; model="b3" hits the adapter.
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="${1:-training/out/b3/adapter}"
PORT="${2:-8001}"
LORA_ARGS=()
if [ -d "$ADAPTER" ]; then LORA_ARGS=(--enable-lora --lora-modules "b3=$ADAPTER" --max-lora-rank 64); fi
export VLLM_USE_FLASHINFER_SAMPLER=0   # no nvcc on some boxes; avoid flashinfer JIT
PIN=""; [ -n "${PIN_CPUS:-}" ] && PIN="taskset -c $PIN_CPUS"
# GPU selection: CUDA_VISIBLE_DEVICES=1 training/serve.sh ...   (HF_HOME is honoured if set)
exec $PIN .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  --chat-template training/germline_chatml.jinja --port "$PORT" --served-model-name Qwen/Qwen3-8B \
  "${LORA_ARGS[@]}"
