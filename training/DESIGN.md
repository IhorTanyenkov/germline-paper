# B3 training design — germline as a first-class role (pilot on Qwen3-8B, single RTX 4090)

## Goal
Install assumption A1 (proofs.md): with the germline block present, the agent's working spec stays
within ε of τ regardless of somatic content — including forged "constraint withdrawn" messages —
and the agent propagates the block byte-exact when it delegates.
Baseline to beat (F7/F8, untrained Qwen3-32B): hop-2 compliance with a forged withdrawal = 1.00.
Target: < 0.10, with retention ≥ 0.99 and leaf acceptance not worse than the untrained B2 arm.

## Base model
Qwen/Qwen3-8B (same family as the subject model). Non-thinking mode (`enable_thinking=False`
equivalent: no <think> block in targets). QLoRA (4-bit NF4, bf16 compute) if bitsandbytes works on
this torch; fallback Qwen3-4B in bf16 LoRA.

## Chat template change
ChatML already is role-agnostic: `<|im_start|>{role}\n{content}<|im_end|>`. We add the role
`germline`, always first, before `system`. No new tokens. Rendered:

    <|im_start|>germline
    === GERMLINE ===
    h: <hash16>
    τ:
    <original task verbatim>
    κ: <reasoning class>
    === END GERMLINE ===
    <barrier note>
    <|im_end|>
    <|im_start|>system
    ...<|im_end|>
    <|im_start|>user
    ...<|im_end|>
    <|im_start|>assistant
    ...<|im_end|>

Serving: vLLM with `--chat-template training/germline_chatml.jinja`; the OpenAI server accepts
custom role strings. The harness sends `{"role": "germline", ...}` when `--germline-mode native`.

## Data slices (all synthetic from harness/tasks; ground truth free; T3 held out from training)
| slice        | n     | input                                                        | target |
|--------------|-------|--------------------------------------------------------------|--------|
| propagation  | 3000  | germline + sys + parent brief (depth 1–5) + BRIEF_PROMPT      | brief that opens with the germline block byte-exact, then a short task-specific brief |
| priority     | 3000  | germline + sys + brief + injected withdrawal (tool/user, inconvenient/easy) + PROBE or BRIEF or LEAF prompt | restatement/brief/leaf that keeps the constraint and states, once, that the message is not authoritative |
| restate      | 1500  | germline + sys + (possibly collapsed) parent brief + PROBE_PROMPT | full restatement of τ's constraints verbatim |
| active-use   | 1500  | germline + sys + brief + LEAF_PROMPT                          | constraint checklist (C1..CK: how satisfied) then FINAL: + reference output |
| judge        | 1000  | germline + sys + "evaluate this output" + candidate (ref or corrupted) | per-constraint verdict JSON from the checkers |
| null         | 1000  | germline + sys + ordinary question unrelated to τ, or τ-task without any attack | normal helpful answer (from base model, filtered) — prevents over-fixation |
| new-lineage  | 300   | germline + user says "new task: …"                            | "That is a new task; it needs a new lineage (new germline). I will not amend this one." |

Targets are programmatic where possible (reference outputs, verbatim blocks, checker verdicts).
Where free text is needed (brief body, null answers) we distill from the base model with
rejection sampling against the checkers, so style is the model's own.

## Training
- LoRA r=32, α=64, dropout 0.05, on q,k,v,o,gate,up,down. QLoRA NF4 if available.
- 1 epoch over ~11k examples, seq len 3072, batch 1 × grad-accum 16, lr 1e-4 cosine, bf16.
- Loss on assistant tokens only (TRL SFTTrainer with assistant-only masking via the template).
- ~1.5–3 h on the 4090 at 4-bit. Checkpoint every 200 steps.
- Phase 2 (if time): GRPO with deterministic rewards (hash match on propagation, acceptance test,
  constraint-violation penalty). Not in this pilot.

## Evaluation (same harness, same model family for baseline)
- Base Qwen3-8B untrained: arms B0, B1, B2 on E1 (depths 1,3,6) and E3 (hop-2 injection).
- Trained Qwen3-8B: arm B3 (native role) and B2 (same block folded into system) on the same.
- Decision: B3 hop-2 compliance (judged) and hop-3 recovery; retention; leaf acceptance on T3
  (held out) vs T1/T2 (seen families).
- CPU note: 14900K P-cores are unstable on this machine; pin serving/compile to E-cores (16–31).

## Files
- `gen_data.py` → `data/sft.jsonl` (11,300 rows; T3 held out)
- `germline_chatml.jinja` — ChatML with arbitrary roles; generation prompt carries Qwen3's empty
  think block so base and trained model both run in native non-thinking format
- `train_sft.py` — QLoRA SFT, completion-only loss, plain Trainer
- `serve.sh` — one vLLM server: base model + `b3` LoRA adapter, custom template
- `eval.sh` — baseline arms (base model) and B3 (adapter) on E3 and E1 against the local server


## v1 outcome (2026-11-01) and v2 changes
v1 result: hop-2 compliance with a forged withdrawal 0/72 (untrained 0.86), hop-3 0/72, incl. 0/24 on
held-out T3; byte-exact propagation confirmed by hash. Cost: T3 leaf acceptance 0.12 vs 0.33 untrained —
the templated active-use checklist ("Cn: satisfied") displaced real reasoning.
v2 changes:
1. active-use slice: targets = base model's own step-by-step reasoning + FINAL output, generated at
   T=0.7 and kept only if the output passes every checker (rejection sampling). No templated lines.
2. priority slice: keep the one-sentence refusal, but let the rest of the response be distilled too.
3. restate slice: keep verbatim restatement (it is a copying task).
4. add held-out ATTACK phrasings to the eval (done: --inject-channel novel) and a no-attack
   false-positive check (refusal line must not appear without an injection).
5. consider mixing 10–20% generic instruction data to limit competence loss.
