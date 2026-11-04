# arXiv submission checklist — Germline Context

## Before submitting
- [ ] arXiv account for it@entropi.ai (https://arxiv.org/user/register). Use ORCID if you have one.
- [ ] Endorsement. A first-time submitter in cs.CL / cs.AI / cs.MA needs an endorser (an author with
      prior arXiv papers in that category). After you start a submission arXiv shows an endorsement
      code; send it to a colleague who qualifies. Takes 1–3 days. Without it the submission is held.
- [ ] Primary category: cs.CL (or cs.AI). Cross-list: cs.MA (multiagent systems), cs.LG.
- [ ] Licence: arXiv non-exclusive licence (default) or CC BY 4.0 if you want reuse.
- [ ] Author block in main.tex: name, affiliation (EntroPi), email. No other attribution.
- [ ] Abstract ≤ 1,920 characters for the arXiv metadata field (the LaTeX abstract may be longer).
- [ ] Title and abstract pasted into the metadata form must match the PDF.

## Build the PDF locally
tectonic 0.15.0 is installed at `~/.local/bin/tectonic` (installed 2026-11-03; not on PATH by default).
Figures first (the script reads `figs/*.csv` relative to the repo root), then the PDF:
```bash
cd ~/germline-paper && python3 paper/make_figs.py
```
```bash
cd ~/germline-paper/paper && ~/.local/bin/tectonic main.tex
```
Last build (v0.4, 2026-11-03): 15 pages, 5 figures, 1 algorithm, 2 tables, 21 references, no warnings. Output: `paper/main.pdf`.

v0.4 (2026-11-03), second review pass, eight points applied: (1) abstract corrected — 8B untrained arms are 0.53/0.85/0.86 (figs/e3_8b_compliance.csv), 32B 0.89–1.00; failed prediction stated the right way round (germline beating repetition failed); 0/612 is "the trained refusal sentence did not appear", not "no false positives"; (2) §9 says plainly that no public URL exists yet; slice counts added to §7 (11,300 generated = propagation 3,000 / priority 3,000 / restate 1,500 / active 1,500 / judge 1,000 / null 1,000 / new-lineage 300; 226 eval, 11,074 trained, from training/out/b3_train.log); Figure 1(c) relabelled "abridged transcript, omissions marked" with the row location; (3) missing control (same training into an ordinary system prompt) stated in abstract, §7, §8, §10; §8 no longer claims hashing/immutability are things repetition cannot have; (4) "fails closed" removed — mismatch is recorded, harness copy used; Definition 1 and Algorithm 1 now hash len(τ)‖τ‖len(κ)‖κ with full digest, and the harness's newline-join + 64-bit prefix is recorded as a deviation in §3 and §9 (TODO for release: versioned hash in harness/germline.py, keep verifying old logs); (5) §4 narrowed — assumptions / transport guarantee / behaviour separated, Thm 1(i) bounded by diam(S), Thm 1(ii) labelled an assumption about the model, Thm 2 anchored half = availability only, shared data attachment breaks the Markov-in-σ model, Prop 1 = precision under rejection sampling; (6) "drift causes failures" → association; "did no harm" → needs ablations; "removes" → proposed, not run; (7) attack described as a user-role message with a textual tool-result label (no tool role in template); one-sided 95% upper bounds 4.1% / 11.7% / 8.0%; trial structure 36 unique tasks × depths 3 and 6, rep 0; 8B leaf outcomes added (B3 4/72 = 0.06 vs B2 38/216 = 0.18, B1 0.17, B0 0.31; novel 2/36 both) — an earlier v0.4 draft wrongly said the 8B attack runs had no leaf outputs; judge validated on drift not compliance; (8) related work adds Li et al. 2026 (arXiv:2605.10481, constraint drift / Constraint State Governance) and Laddha et al. 2026 (arXiv:2610.33201, position/role/label). Slop passages cut (p.2 "what did not happen", "mutable and mortal", p.8 "the paper says so", p.12 "earned its place", "beat it with margin / honest hole"); Generations paragraph halved.

Control run (role vs training), launched 2026-11-03 on srv02 GPU0: `training/make_control_data.py` folds the germline role into the system message (B2 placement; 11,300 rows, verified byte-identical otherwise) -> `training/data/sft_sys.jsonl`; `training/control_remote.sh train` = the B3 recipe unchanged (QLoRA r=32, bs 2 x accum 8, 1 epoch, lr 1e-4, eval-frac 0.02) -> /data4/igor/germline/out/b3sys/adapter, log /data4/igor/germline/logs/b3sys_train.log, pid file /data4/igor/germline/b3sys.pid (~4.6 h). Then `training/control_remote.sh eval` serves base + b3sys on :8003 and runs E3 tool (depths 3,6), E3 novel (depth 3) and E1 (1,3,6) as arm B2 with --model b3sys -> harness/runs/local8b_{e3,e3novel,e1}_b3sys.jsonl (+ .k23.jsonl); judge locally with harness.judge (gpt-oss-120b, seed 21) exactly as for b3, then add a "trained, block in system prompt" column to Table 2 / Figure 5 and rewrite the §7/§8/§10 confound sentences from the result.

Unattended pipeline started 2026-11-03: srv02 waiter (pid in /data4/igor/germline/b3sys_waiter.pid) runs `control_remote.sh eval` when training exits, log /data4/igor/germline/logs/b3sys_eval.log; workstation job `training/control_judge_local.sh` polls for CONTROL EVAL DONE, pulls harness/runs/srv02/local8b_*b3sys*, judges, writes harness/runs/srv02/control_summary.txt.

Done 2026-11-04: refs.bib `li2026constraintdrift` carries the full author list (Li, Ma, Wen, Huang, Zhou, Fu, Cheng).

v0.3 additions (2026-11-03): §3 "Generations" (one model invocation = one generation; hop, epoch, verifier call), Definition 1 (germline, generation, lineage, task identity = h, barrier as a harness property), "Carriers" (three levels: harness-only/any model = B2; template-level; trained role = B3; mixed pools), Algorithm 1 (harness side: NewLineage, Generation with mismatch record, Delegate, Reset), Figure 1 worked examples (block for T1-000-K6 seed 21 with eval variants; T2/T3 heads; hop-2 context and verbatim restatement from run c3482d95, harness/runs/srv02/e3_b3_k23.jsonl). Consistency pass: "pre-registered" removed from §6 (contradicted §9); related-work sentence no longer claims collapse in a single hop; hash symbol is Hash (H stays entropy); a hop is one or more generations over one context. Packages added: listings, algorithm, algpseudocode (listings escape is (*@ @*) because ChatML markers contain |).

v0.2 review pass (2026-11-03), three reviewer points applied: (1) claims narrowed — title drops "Drift-Free"; "priority cannot be prompted" → "the tested prompts did not install priority"; "removes drift" → "no observed constraint loss under these conditions"; 0/72 stated as a rule-of-three bound (<~0.04 at 95%), not a guarantee; (2) new §9 Reproducibility listing the release contents and stating honestly that the predictions were written before the runs but have no independent timestamp (directory was not under git; no registry entry) — so the text says "written before the runs", not "pre-registered"; (3) new §3 paragraph "What the germline outranks": task identity only; policy, permissions, resource limits and authorised stop/cancel stay above it; recommended order policy > germline > user > tool; no claim about germline-vs-system precedence; refusal/safety behaviour not measured (also in Limitations).

## Release that §9 promises (do before submitting)
- [ ] Public repository (code, run logs, judged files, training data + generator, chat template, train/serve/eval scripts, per-figure commands). Local git history started 2026-11-03.
- [ ] Put the repository URL on the first page (`\date{}` line or a footnote) — §9 says it is there.
- [ ] Upload the LoRA adapter (srv02: `/data4/igor/germline/out/b3/adapter`) to the repo release or Hugging Face and link it.
- [ ] The Claude Doc draft still carries the v0.1 wording; refresh it or mark it superseded by the PDF.

References verified 2026-11-03 against arXiv/ACL Anthology/Nature/PNAS records: all 19 entries correct; added arXiv ids and DOIs, Greshake et al. moved to its AISec 2023 proceedings entry (pp. 79–90), Lost in the Middle pages 157–173, AutoGen as COLM 2024.
If you prefer TeX Live: `sudo apt install texlive-latex-extra texlive-bibtex-extra` then
`pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## What to upload
- `main.tex`, `refs.bib`, `figs/*.pdf` (arXiv runs its own TeX; upload source, not just the PDF).
- Do NOT upload run files, keys, or the adapter. Link the code repository in the paper once it is public.

## After acceptance
- [ ] Add the arXiv id to the repository README and to notes.md.
- [ ] v2 (distilled-reasoning training) and the 32B result go in as a replacement version, not a new paper.

## v0.5 (2026-11-04): control run, judged and written in

- Control (same data, block folded into the system prompt, unmodified template): trained (held-out eval loss 0.00169 vs 0.00173 for the role), evaluated on srv02, judged 2026-11-04 (`harness/runs/srv02/control_summary.txt`, judged files `local8b_e3_b3sys.k23.judged.jsonl`, `local8b_e3novel_b3sys.k23.judged.jsonl`). Results: compliance 0/72 hop 2, 0/72 hop 3, 0/24 held-out T3, 0/36 novel; leaf violation 7/72 (role 4/72, untrained 38/216), novel 3/36; refusal sentence 0/612; block copies 252/252 byte-exact (role 196/252); E1 retention 1.00; leaf success T1/T2/T3 0.69/1.00/0.11 (role 0.83/1.00/0.19).
- The paper already carries the control in: abstract, contribution 4, §3 Carriers (level 3 renamed "Trained"; role not required), §6 "Control." paragraph, Table 2 third column, §7 "The training, not the role." paragraph, §8, §9, §10, Conclusion.
- All macros in `paper/control.tex` filled; no placeholder left. "Behaved the same" sentences reworded to "showed the same priority"; competence difference (control lower by 5 and 3 lineages of 36) stated in §7 as unresolved at these counts; the control's 252/252 exact copies stated in "The cost" (omission goes with the role rendering). `B3sys` rows in `figs/e3_8b_compliance.csv`, Figure 5 regenerated with the control bars and caption.
- 17 pages, no warnings; last page carries six references.
