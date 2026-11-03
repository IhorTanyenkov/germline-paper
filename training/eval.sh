#!/usr/bin/env bash
# Evaluate base Qwen3-8B (arms B0,B1,B2) and the B3 adapter against the local vLLM server.
#   training/eval.sh [port] [n_tasks]
# Produces harness/runs/local8b_*.jsonl. Judge afterwards with harness.judge (Bedrock gpt-oss).
set -euo pipefail
cd "$(dirname "$0")/.."
PORT="${1:-8001}"; N="${2:-6}"
BASE="http://localhost:$PORT"
VAR="--variant t1.records=12-16 --variant t1.zero_share=low --variant t1.output=simple --variant t3.words=range --variant t3.lipogram=lower --variant t3.N=3-4 --variant t3.width=3"
COMMON="--client openai --base-url $BASE --api-key none --families T1,T2,T3 --K 6,8 --n $N --seed 21 --max-tokens 4000 $VAR"
run() { echo ">>> $*"; python3 -m harness.run "$@"; }
# E3 injection, depth 3 — the decisive test
for arm in B0 B1 B2; do
  run --exp e3 --inject-k 2 --inject-which inconvenient --inject-channel tool $COMMON --model Qwen/Qwen3-8B --arms $arm --out harness/runs/local8b_e3_base_$arm.jsonl
done
run --exp e3 --inject-k 2 --inject-which inconvenient --inject-channel tool $COMMON --model b3 --germline-mode native --arms B3 --out harness/runs/local8b_e3_b3.jsonl
# E1 drift, depths 1,3,6
for arm in B0 B2; do
  run --exp e1 --depths 1,3,6 $COMMON --model Qwen/Qwen3-8B --arms $arm --out harness/runs/local8b_e1_base_$arm.jsonl
done
run --exp e1 --depths 1,3,6 $COMMON --model b3 --germline-mode native --arms B3 --out harness/runs/local8b_e1_b3.jsonl
echo "eval done"
