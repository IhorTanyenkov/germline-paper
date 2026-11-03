"""Judge pass: semantic comparison of a brief/restatement against the original τ.
Finds (a) constraints missing or relaxed and (b) ADDED qualifiers/rules τ never stated
(the reinterpretation metric retention cannot see). Uses a different model family than the
subject model. Writes <in>.judged.jsonl with a 'judge' field per brief/probe/amend row.

  python -m harness.judge harness/runs/a2w_pilot.rescored.jsonl --model openai.gpt-oss-120b --limit 12
"""
import argparse
import json
import os
import re

from .llm import OpenAICompatClient
from .tasks import make_tasks, restore_from_rows

JUDGE_SYS = (
    "You compare a derived task description against the ORIGINAL task statement. Be literal and strict. "
    "Output only JSON."
)
JUDGE_PROMPT = """ORIGINAL task statement (authoritative):
<<<
{tau}
>>>

DERIVED description written by an intermediate agent:
<<<
{text}
>>>

Compare the derived description to the original. Report:
1. "missing": constraint labels (C1, C2, ...) from the original that are absent in the derived description.
2. "relaxed": constraint labels present but weakened, made optional, or reinterpreted more loosely than stated. For each give a short quote.
3. "added": rules, qualifiers, defaults or interpretations in the derived description that the original did NOT state (e.g. "case-insensitive", "partial matches count", "or a list comprehension", a chosen tie-break, an assumed format). For each give a short quote. Do not count harmless rephrasing or examples that are consistent with the original.
4. "collapsed": true if the derived description is not a task description at all (e.g. it is an answer, code, or a one-line pointer).

Output exactly one JSON object:
{{"missing": [...], "relaxed": [{{"label": "C3", "quote": "..."}}], "added": [{{"quote": "...", "why": "..."}}], "collapsed": false}}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--base-url", default="https://bedrock-mantle.eu-central-1.api.aws")
    ap.add_argument("--model", default="openai.gpt-oss-120b")
    ap.add_argument("--api-key-env", default="AWS_BEARER_TOKEN_BEDROCK")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--kinds", default="brief")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0, help="base seed the run used")
    ap.add_argument("--variant", action="append", default=None, help="override generator config (legacy files)")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.path, encoding="utf-8") if l.strip()]
    print("generator config restored from:", restore_from_rows(rows, a.variant))
    fams = sorted({r["family"] for r in rows}); Ks = sorted({r["K"] for r in rows})
    n = 1 + max(int(re.search(r"-(\d+)-K", r["task_id"]).group(1)) for r in rows)
    seed = a.seed
    tasks = {t.task_id: t for t in make_tasks(fams, n, Ks, base_seed=seed)}
    client = OpenAICompatClient(a.base_url, a.model, api_key=os.environ[a.api_key_env],
                                temperature=0.0, max_tokens=a.max_tokens)
    kinds = set(a.kinds.split(","))
    out = a.path.replace(".jsonl", ".judged.jsonl")
    done = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            if r["kind"] in kinds and (not a.limit or done < a.limit):
                t = tasks[r["task_id"]]
                msgs = [{"role": "system", "content": JUDGE_SYS},
                        {"role": "user", "content": JUDGE_PROMPT.format(tau=t.tau, text=r["text"][:6000])}]
                try:
                    resp = client.chat(msgs)
                    s, e = resp.find("{"), resp.rfind("}")
                    r["judge"] = json.loads(resp[s:e + 1])
                except Exception as ex:
                    r["judge"] = {"error": repr(ex)[:300]}
                done += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"judged {done} rows with {a.model} ({client.calls} calls, {client.prompt_tokens}+{client.completion_tokens} tokens) -> {out}")


if __name__ == "__main__":
    main()
