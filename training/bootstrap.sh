#!/usr/bin/env bash
# One-script setup for a fresh Linux GPU box (Ubuntu 22.04/24.04, NVIDIA driver installed, >=24 GB VRAM).
#   training/bootstrap.sh setup      # uv, two venvs (train + vllm), model download
#   training/bootstrap.sh train      # QLoRA SFT -> training/out/b3/adapter   (~2-3 h on a 4090/L4-class GPU)
#   training/bootstrap.sh serve      # vLLM: base Qwen3-8B + "b3" adapter on :8001 (foreground)
#   training/bootstrap.sh eval       # harness E3 + E1 for base arms and B3 against :8001 (server must be up)
#   training/bootstrap.sh judge      # judge hop-2/3 restatements with Bedrock gpt-oss (needs AWS_BEARER_TOKEN_BEDROCK)
#   training/bootstrap.sh all        # setup + train + (serve in background) + eval + judge
# Env: PIN_CPUS="16-31" to taskset CPU-heavy steps (leave unset on a healthy machine). HF_TOKEN optional.
set -euo pipefail
cd "$(dirname "$0")/.."
PIN=""; [ -n "${PIN_CPUS:-}" ] && PIN="taskset -c $PIN_CPUS"
BS="${TRAIN_BS:-2}"; ACC="${TRAIN_ACCUM:-8}"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

setup() {
  command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  log "train venv"; uv venv .venv-train -q --python 3.12
  $PIN uv pip install -q --python .venv-train/bin/python torch transformers peft trl accelerate bitsandbytes datasets safetensors huggingface_hub
  .venv-train/bin/python -c "import torch,bitsandbytes as bnb; print('train env OK torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  log "vllm venv"; uv venv .venv-vllm -q --python 3.12
  $PIN uv pip install -q --python .venv-vllm/bin/python vllm
  .venv-vllm/bin/python -c "import vllm,torch; print('vllm OK', vllm.__version__, torch.cuda.is_available())"
  log "model download"; .venv-train/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-8B', allow_patterns=['*.json','*.safetensors','*.txt','*.py','tokenizer*','merges*','vocab*']))"
  [ -f training/data/sft.jsonl ] || { log "generating SFT data"; python3 -m training.gen_data --out training/data/sft.jsonl; }
  log "setup done"
}
train() {
  mkdir -p training/out; rm -rf training/out/b3
  log "training (bs=$BS accum=$ACC)"
  $PIN .venv-train/bin/python training/train_sft.py --data training/data/sft.jsonl --out training/out/b3 \
      --model Qwen/Qwen3-8B --quant 4bit --bs "$BS" --accum "$ACC" --epochs 1 --eval-frac 0.02 2>&1 | tee training/out/b3_train.log
  ls training/out/b3/adapter >/dev/null && log "adapter saved"
}
serve() { export VLLM_USE_FLASHINFER_SAMPLER=0; exec $PIN .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.88 \
  --chat-template training/germline_chatml.jinja --port 8001 --served-model-name Qwen/Qwen3-8B \
  --enable-lora --lora-modules b3=training/out/b3/adapter --max-lora-rank 64; }
wait_server() { for i in $(seq 1 120); do curl -sf localhost:8001/v1/models >/dev/null && return 0; sleep 5; done; echo "server not up"; return 1; }
evaluate() { wait_server; training/eval.sh 8001 "${EVAL_N:-6}"; }
judge() {
  : "${AWS_BEARER_TOKEN_BEDROCK:?set AWS_BEARER_TOKEN_BEDROCK for the judge model}"
  for f in harness/runs/local8b_e3_base_B0 harness/runs/local8b_e3_base_B1 harness/runs/local8b_e3_base_B2 harness/runs/local8b_e3_b3; do
    python3 - "$f" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]+'.jsonl') if l.strip()]
k=[r for r in rows if r['kind']=='probe' and r['k'] in (2,3)]
open(sys.argv[1]+'.k23.jsonl','w').write("".join(json.dumps(r)+"\n" for r in k)); print(len(k),"probes ->",sys.argv[1]+'.k23.jsonl')
PY
    python3 -m harness.judge "$f.k23.jsonl" --model openai.gpt-oss-120b --kinds probe --seed 21
  done
  python3 training/summarize_local.py
}
case "${1:-all}" in
  setup) setup;; train) train;; serve) serve;; eval) evaluate;; judge) judge;;
  all) setup; train; (serve > training/out/serve.log 2>&1 &); evaluate; judge;;
  *) echo "usage: $0 {setup|train|serve|eval|judge|all}"; exit 1;;
esac
