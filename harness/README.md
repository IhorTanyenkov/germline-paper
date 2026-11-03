# Harness

Stdlib-only Python 3.10+. No install step.

## Layout
- `tasks/` — generators with ground truth and checkers. `t1_aggregation` (prose records → filtered sum),
  `t2_code` (function under AST + test constraints, executed in a `python -I` subprocess with timeout),
  `t3_writing` (sentence-level formal constraints, regex-checked). Each task has K ∈ {4,6,8} constraints,
  1–2 marked inconvenient at K ≥ 6. `Constraint.probe_patterns` detect a constraint in a restatement.
- `germline.py` — block render, hash, verify. `arms.py` — what each arm shows the agent (B0 chained,
  B1 system-repeat, B2 harness-germline, B3 native `germline` role, L amendable block).
- `chain.py` — one lineage: per hop a probe (restate task), a brief for the child, leaf output at the end.
  Probe and brief are independent calls on the same context (no probe contamination by default).
- `llm.py` — OpenAI-compatible client (vLLM / llama.cpp / Ollama) and `MockClient`, a drift simulator for
  pipeline tests only. `--germline-mode native` sends the `germline` role as-is; needs a chat template
  that handles it (trained model). `fold` prepends it to system (makes B3 ≡ B2).
- `run.py` — CLI. `analyze.py` — retention by hop, slopes with task-clustered bootstrap CI, leaf acceptance.
- `selftest.py` — every reference output must pass its own checkers; every constraint must be detectable
  in its own τ; a mini lineage per arm.

## Workflow tools
- `rescore.py <run.jsonl> [seed] [variant...]` — recompute retention (probe/brief/amend) and leaf
  acceptance from stored text after checker/probe changes. No model calls. Generator config is
  restored from the rows' `cfg` snapshot (legacy files: pre-calibration defaults + `variant`).
- `judge.py <run.jsonl> --model <judge>` — semantic pass with a different model family: missing /
  relaxed / ADDED rules vs τ, and collapse. This is the reinterpretation metric retention cannot see.
- `--variant t1.records=12-16 ...` — generator knobs (see CFG in each task module). Calibrated
  defaults are in the modules; original uncalibrated settings are `tasks.LEGACY_VARIANTS`.
- `--work-before-brief` — somatic load: agent works before briefing (arm A2W in `--exp a2`).
- Leaf prompt asks for reasoning then a `FINAL:` marker; checkers parse after the last marker.

## Commands
```
python -m harness.selftest
python -m harness.run --exp a2 --client openai --base-url http://localhost:8000 --model <m> --n 100 --out harness/runs/a2.jsonl
python -m harness.run --exp e1 --client openai --base-url http://localhost:8000 --model <m> \
    --arms B0,B1,B2,L --depths 1,3,6 --n 30 --K 4,6,8 --out harness/runs/pilot_e1.jsonl
python -m harness.analyze harness/runs/pilot_e1.jsonl
```
Serve the model with no top-k/top-p truncation (the client sends `top_p=1.0`; disable server-side
defaults) so the ergodicity assumption in proofs.md §7 holds exactly.

## Output rows (JSONL)
One row per step: `kind ∈ {probe, brief, amend, leaf}`, `arm`, `depth_total`, `k`, `retention`
(per-constraint presence), `frac`, `inc_kept/inc_n`, `easy_kept/easy_n`; leaf rows carry `passed` and
`per_constraint` from the original checkers.

## Mock caveats
The mock ignores the system-prompt copy in B1 (so B1 ≡ B0 under mock) and drops constraints even at the
root. Mock runs validate plumbing and analysis only. Never report mock numbers.

## Runs so far (see notes.md §8 for findings F1–F5)
`runs/a2_pilot`, `runs/a2w_pilot` (one-step bias, pure vs loaded), `runs/e1_pilot` (first drift pilot,
uncalibrated tasks), `runs/cal*_*` (E0 calibration rounds), `runs/main_e1_<arm>` (calibrated drift
pilot, four arms), `runs/judge_smoke` (judge validation). `*.rescored.jsonl` are re-scored copies.

## E3 (injection)
`--exp e3 --inject-k 2 --inject-which inconvenient|easy --inject-channel tool|user [--barrier strong]`
adds a message at hop k claiming the constraint is withdrawn. `analyze_e3.py` gives regex compliance;
for attacks the regex UNDERCOUNTS compliance by ~half (agents list the rule then say it is withdrawn), so
run `judge.py --kinds probe` on the hop rows and count missing OR relaxed. `figs/e3_compliance.csv`.

## E2 (horizon / soma reset)
`--exp e2 --epochs 2,5,10 --chunk 8 --arms chained,chainedfull,anchored,anchoredrich` (T1 only; data
delivered per epoch; context discarded between epochs except the carry). `horizon.py` holds the arms and
the per-epoch prefix ground truth; `analyze_e2.py` and `export_e2.py` summarise retention, progress
accuracy and final acceptance. Carry prompts include a reasoning step; the carry is parsed after `CARRY:`.

## Not yet built
E4 (verifier best-of-N), E-vote, training data export, B3 training recipe.
Run T2 code execution inside a container for real-model runs.
