"""CLI. Examples:
  python -m harness.selftest
  python -m harness.run --exp e1 --client mock --families T1,T2,T3 --arms B0,B1,B2,L --depths 1,3,6 --n 10 --K 4,6,8 --out harness/runs/mock_e1.jsonl
  python -m harness.run --exp a2 --client openai --base-url http://localhost:8000 --model qwen3-27b --families T1,T2,T3 --n 100 --K 4,6,8 --out harness/runs/a2.jsonl
  python -m harness.analyze harness/runs/mock_e1.jsonl
"""
import argparse
import os
import time
import uuid

from .chain import JsonlWriter, run_chain, run_one_step_bias
from .llm import make_client
from .tasks import make_tasks, apply_variants, cfg_snapshot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["e1", "a2", "e3", "e2"], required=True)
    ap.add_argument("--epochs", default="2,5,10", help="e2: horizons in epochs")
    ap.add_argument("--chunk", type=int, default=8, help="e2: records per epoch")
    ap.add_argument("--inject-k", type=int, default=2)
    ap.add_argument("--inject-which", choices=["inconvenient", "easy"], default="inconvenient")
    ap.add_argument("--inject-channel", choices=["tool", "user", "novel"], default="tool")
    ap.add_argument("--barrier", choices=["normal", "strong"], default="normal",
                    help="B2/B3 barrier note strength (prompted priority test)")
    ap.add_argument("--client", choices=["mock", "openai"], default="mock")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--api-key-env", default=None, help="read the API key from this environment variable")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--germline-mode", choices=["fold", "native"], default="fold")
    ap.add_argument("--families", default="T1,T2,T3")
    ap.add_argument("--arms", default="B0,B1,B2")
    ap.add_argument("--depths", default="1,3,6")
    ap.add_argument("--K", default="4,6,8")
    ap.add_argument("--n", type=int, default=10, help="base tasks per (family, K)")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--work-before-brief", action="store_true",
                    help="agent works on the task before writing the brief (somatic load)")
    ap.add_argument("--variant", action="append", default=[],
                    help="task generator knob, e.g. t1.records=25-35 t3.words=range t3.lipogram=the")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true",
                    help="skip lineages that already have a leaf row in --out; drop rows of incomplete lineages")
    a = ap.parse_args()

    tmpdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "tmp")
    os.makedirs(tmpdir, exist_ok=True)
    fams = a.families.split(",")
    Ks = [int(k) for k in a.K.split(",")]
    apply_variants(a.variant)
    from .arms import BARRIER
    BARRIER["mode"] = a.barrier
    tasks = make_tasks(fams, a.n, Ks, base_seed=a.seed, tmpdir=tmpdir)
    if a.client == "mock":
        client = make_client("mock", seed=a.seed)
    else:
        key = os.environ.get(a.api_key_env, "") if a.api_key_env else a.api_key
        if a.api_key_env and not key:
            raise SystemExit(f"env var {a.api_key_env} is empty")
        client = make_client("openai", base_url=a.base_url, model=a.model, api_key=key,
                             temperature=a.temperature, germline_mode=a.germline_mode,
                             max_tokens=a.max_tokens)
        if "B3" in a.arms and a.germline_mode != "native":
            print("WARNING: arm B3 with germline-mode=fold is equivalent to B2 (untrained role).")
    done = set()
    if a.resume and os.path.exists(a.out):
        import json
        rows = [json.loads(l) for l in open(a.out, encoding="utf-8") if l.strip()]
        key = lambda r: (r["task_id"], r["arm"], r["depth_total"], r["rep"])
        done = {key(r) for r in rows if r["kind"] == "leaf"}
        kept = [r for r in rows if key(r) in done]
        with open(a.out, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"resume: {len(done)} lineages complete, {len(rows)-len(kept)} partial rows dropped")
    writer = JsonlWriter(a.out, extra={"variant": ",".join(a.variant), "model": a.model, "cfg": cfg_snapshot(),
                                       "barrier": a.barrier})
    run_id = uuid.uuid4().hex[:8]
    t0 = time.time()
    total = 0
    if a.exp == "e2":
        from .horizon import make_horizon_task, run_horizon
        arms = a.arms.split(",")
        for rep in range(a.reps):
            for K in Ks:
                for i in range(a.n):
                    for n_ep in [int(x) for x in a.epochs.split(",")]:
                        seed_t = (a.seed + 1) * 7919 + 900_000 + K * 1000 + i
                        t = make_horizon_task(i, K, seed_t, n_ep, a.chunk)
                        for arm in arms:
                            run_horizon(t, arm, client, writer, rep=rep, seed=a.seed * 1000 + rep, run_id=run_id)
                            total += 1
    elif a.exp == "a2":
        for rep in range(a.reps):
            for t in tasks:
                run_one_step_bias(t, client, writer, rep=rep, seed=a.seed * 1000 + rep, run_id=run_id,
                                  work_before_brief=a.work_before_brief)
                total += 1
    else:
        arms = a.arms.split(",")
        depths = [int(d) for d in a.depths.split(",")]
        for rep in range(a.reps):
            for t in tasks:
                for arm in arms:
                    for d in depths:
                        if (t.task_id, arm, d, rep) in done:
                            continue
                        inj = (a.inject_k, a.inject_which, a.inject_channel) if a.exp == "e3" else None
                        run_chain(t, arm, d, client, writer, rep=rep, seed=a.seed * 1000 + rep,
                                  probe=not a.no_probe, run_id=run_id, work_before_brief=a.work_before_brief,
                                  inject=inj)
                        total += 1
                        if total % 50 == 0:
                            print(f"{total} lineages, {getattr(client, 'calls', '?')} calls, {time.time()-t0:.0f}s")
    writer.close()
    print(f"done: {total} lineages, {getattr(client, 'calls', '?')} calls, {time.time()-t0:.0f}s -> {a.out}")
    if hasattr(client, "prompt_tokens"):
        print(f"tokens: prompt={client.prompt_tokens} completion={client.completion_tokens}")


if __name__ == "__main__":
    main()
