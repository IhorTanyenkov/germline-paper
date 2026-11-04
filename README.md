# Germline: an immutable task channel for multi-agent language model systems

Code, run logs, training data and LaTeX source for the preprint *Germline Context: An Immutable Inherited
Task Channel for Multi-Agent Language Model Systems* (Ihor Tanyenkov, 2026).

The germline is a block the harness hands to every agent in a lineage: the user's original task
statement verbatim, a reasoning class, and a hash of both. No agent can edit it. The paper measures
what happens to a task over six hops of delegation and ten epochs of context reset with and without
it, what a forged in-context "constraint withdrawn" message does, and what fine-tuning an 8B model
to read the block changes.

## Layout

| path | what |
|---|---|
| `paper/` | `main.tex`, `refs.bib`, `control.tex` (control-run numbers), `make_figs.py`, `figs/` |
| `harness/` | stdlib-only Python harness: task generators with checkers, arms, lineage runner, judge, analysis (`harness/README.md`) |
| `harness/runs/` | run logs for every reported cell, judged files included (`harness/runs/srv02/` = 8B local runs) |
| `figs/` | CSVs the figures are drawn from |
| `training/` | data generator, SFT script, chat template with the `germline` role, control-data script, serve/eval scripts (`training/DESIGN.md`) |
| `training/data/` | `sft.jsonl` (role training set) and `sft_sys.jsonl` (control: block folded into the system message) |
| `experiments.md` | experiment protocol, with the predictions written before the runs |
| `proofs.md` | the formal results with assumptions |
| `notes.md` | working notes |

## Reproducing the numbers

Every number in the paper is computed from the run logs:

```bash
python3 -m harness.analyze harness/runs/srv02/local8b_e1_b3.jsonl          # drift tables
python3 paper/make_figs.py                                                 # all figures from figs/*.csv
```

Attack compliance is judged by gpt-oss-120b (`harness/judge.py`, seed 21); the judged files are in the
logs, so the paper's counts reproduce without the judge endpoint. Re-running the subject models needs
an OpenAI-compatible endpoint (`harness/run.py --client openai`); seeds fix the task instances, outputs
reproduce in distribution.

Training (`training/train_sft.py`, QLoRA rank 32, one epoch) and the role-versus-training control
(`training/make_control_data.py`, `training/control_remote.sh`) are described in `training/DESIGN.md`
and in the paper's §6 and §9. The adapters are released separately (see the paper's first page once
the URL exists).

## Building the paper

```bash
cd paper && tectonic main.tex
```
