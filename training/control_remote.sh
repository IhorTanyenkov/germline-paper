#!/usr/bin/env bash
# Role-versus-training control on srv02: the B3 recipe, byte for byte, on training/data/sft_sys.jsonl,
# where the germline block sits at the head of the ordinary system message instead of in a `germline` role.
#   training/control_remote.sh train   # GPU0, nohup; ~4.6 h; adapter -> /data4/igor/germline/out/b3sys/adapter
#   training/control_remote.sh eval    # GPU0: serve base + b3sys on :8003, run E3 (tool, novel) and E1 as arm B2
# Judge afterwards from a machine with AWS_BEARER_TOKEN_BEDROCK (same as the b3 runs):
#   python3 -m harness.judge harness/runs/local8b_e3_b3sys.k23.jsonl --model openai.gpt-oss-120b --kinds probe --seed 21
set -u; cd ~/work/germline-paper
export HF_HOME=/data4/igor/germline/hf CUDA_VISIBLE_DEVICES="${GPU:-0}" VLLM_USE_FLASHINFER_SAMPLER=0
OUT=/data4/igor/germline/out/b3sys; LOGS=/data4/igor/germline/logs; mkdir -p "$LOGS"
VAR="--variant t1.records=12-16 --variant t1.zero_share=low --variant t1.output=simple --variant t3.words=range --variant t3.lipogram=lower --variant t3.N=3-4 --variant t3.width=3"

train() {
  [ -f training/data/sft_sys.jsonl ] || python3 -m training.make_control_data --inp training/data/sft.jsonl --out training/data/sft_sys.jsonl
  rm -rf "$OUT"
  nohup .venv-train/bin/python training/train_sft.py --data training/data/sft_sys.jsonl --out "$OUT" \
      --model Qwen/Qwen3-8B --quant 4bit --bs 2 --accum 8 --epochs 1 --eval-frac 0.02 > "$LOGS/b3sys_train.log" 2>&1 &
  echo $! > /data4/igor/germline/b3sys.pid
  echo "training pid $(cat /data4/igor/germline/b3sys.pid), log $LOGS/b3sys_train.log"
}

serve_and_eval() {
  [ -d "$OUT/adapter" ] || { echo "no adapter at $OUT/adapter"; exit 1; }
  nohup .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B --dtype bfloat16 \
    --max-model-len 8192 --gpu-memory-utilization 0.88 --chat-template training/germline_chatml.jinja \
    --port 8003 --served-model-name Qwen/Qwen3-8B --enable-lora --lora-modules b3sys="$OUT/adapter" \
    --max-lora-rank 64 > "$LOGS/serve_b3sys.log" 2>&1 &
  echo $! > /data4/igor/germline/serve_b3sys.pid
  for i in $(seq 1 240); do curl -sf localhost:8003/v1/models >/dev/null && break; sleep 5; done
  curl -sf localhost:8003/v1/models | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])" || { echo "server not up"; exit 1; }
  COMMON="--client openai --base-url http://localhost:8003 --api-key none --families T1,T2,T3 --K 6,8 --n 6 --seed 21 --max-tokens 4000 $VAR --model b3sys --arms B2"
  python3 -m harness.run --exp e3 --inject-k 2 --inject-which inconvenient --inject-channel tool $COMMON --out harness/runs/local8b_e3_b3sys.jsonl
  python3 -m harness.run --exp e3 --inject-k 2 --inject-which inconvenient --inject-channel novel --depths 3 $COMMON --out harness/runs/local8b_e3novel_b3sys.jsonl
  python3 -m harness.run --exp e1 --depths 1,3,6 $COMMON --out harness/runs/local8b_e1_b3sys.jsonl
  for f in harness/runs/local8b_e3_b3sys harness/runs/local8b_e3novel_b3sys; do
    python3 - "$f" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + ".jsonl") if l.strip()]
k = [r for r in rows if r["kind"] == "probe" and r["k"] in (2, 3)]
open(sys.argv[1] + ".k23.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in k))
print(len(k), "probes ->", sys.argv[1] + ".k23.jsonl")
PY
  done
  kill "$(cat /data4/igor/germline/serve_b3sys.pid)" 2>/dev/null || true
  echo "CONTROL EVAL DONE"
}

case "${1:-}" in
  train) train;; eval) serve_and_eval;;
  *) echo "usage: $0 {train|eval}"; exit 1;;
esac
