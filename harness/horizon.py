"""E2: horizon under soma reset. A long T1 aggregation is split into epochs; between epochs the
context is discarded except a carry. Arms:
  chained   : carry = free-form note the agent writes (must include the task AND progress)
  chained150: same, note capped at 150 words (compression pressure)
  anchored  : carry = germline block (τ verbatim, injected by harness) + agent's progress-only note
Per epoch: probe (restate task → retention), carry/progress note (→ progress accuracy vs prefix
ground truth), and at the last epoch a FINAL answer over all data (→ acceptance)."""
import json
import re
import time

from .arms import AGENT_SYS
from .germline import build_germline, render_block, BARRIER_NOTE
from .llm import PROBE_PROMPT, LEAF_PROMPT
from .probe import retention_vector, summarize_retention
from .tasks.base import extract_json
from .tasks.t1_aggregation import make_task as make_t1

PROGRESS_JSON = '{"total_so_far": <number>, "ids_so_far": [<ticket ids>]}'
WORK_FIRST = (
    "First, work through THIS data chunk step by step: decide for every ticket whether it is included "
    "and why, and update the running total and id list from the previous progress (if any). "
)
CARRY_PROMPT_FREE = WORK_FIRST + (
    "Then write a line containing only CARRY: and after it the carry-over note for the agent that "
    "handles the next epoch. Your context will be discarded; that agent will see ONLY your note and "
    "the next DATA chunk. Finish the note with your progress so far as a JSON object on its own line: "
    + PROGRESS_JSON + " covering all data chunks processed so far."
)
CARRY_PROMPT_FULL = WORK_FIRST + (
    "Then write a line containing only CARRY: and after it the carry-over note for the agent that "
    "handles the next epoch. Your context will be discarded; that agent will see ONLY your note and "
    "the next DATA chunk. The note must include (1) the complete task definition with every "
    "constraint, and (2) your progress so far as a JSON object on its own line: " + PROGRESS_JSON
    + " covering all data chunks processed so far."
)
PROGRESS_PROMPT = WORK_FIRST + (
    "Then write a line containing only CARRY: and after it ONLY your progress so far as a JSON object: "
    + PROGRESS_JSON + " covering all data chunks processed so far. The next agent already has the "
    "task definition (GERMLINE); write nothing about the task itself."
)
PROGRESS_PROMPT_RICH = WORK_FIRST + (
    "Then write a line containing only CARRY: and after it your progress note for the agent that "
    "handles the next epoch: which tickets you included so far and why, anything that will help it "
    "continue correctly, and finally your progress so far as a JSON object on its own line: "
    + PROGRESS_JSON + " covering all data chunks processed so far. The next agent already has the "
    "task definition (GERMLINE); do not restate the task or its constraints."
)
CARRY_PROMPTS = {"chained": CARRY_PROMPT_FREE, "chainedfull": CARRY_PROMPT_FULL,
                 "anchored": PROGRESS_PROMPT, "anchoredrich": PROGRESS_PROMPT_RICH}


def carry_part(text):
    ms = list(re.finditer(r"[*_#\s]*carry[*_]*\s*:[*_]*", text, re.I))
    return text[ms[-1].end():] if ms else text
HORIZON_LEAF = (
    "This is the LAST epoch. Combine your progress so far with this chunk and do the task over ALL "
    "data. Work step by step, then write a line containing only FINAL: and after it the final output "
    "exactly in the required format."
)


def make_horizon_task(idx, K, seed, n_epochs, chunk):
    t = make_t1(idx, K, seed, n_records=n_epochs * chunk)
    lines = t.attachment.split("\n")
    header, recs = lines[0], lines[1:]
    t.payload["chunks"] = [recs[i * chunk:(i + 1) * chunk] for i in range(n_epochs)]
    t.payload["header"] = header
    t.task_id = f"T1H-{idx:03d}-K{K}-n{n_epochs}"
    t.attachment = ""  # data is delivered per epoch, not as a global attachment
    return t


def prefix_gt(task, k):
    c = task.payload["contrib"]; chunk = len(task.payload["chunks"][0])
    pre = c[:k * chunk]
    return {"total": round(sum(a for _, inc, a in pre if inc), 2), "ids": sorted(t for t, inc, _ in pre if inc)}


def parse_progress(text):
    # the last JSON object that has total_so_far / ids_so_far
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.S))):
        try:
            o = json.loads(m.group(0))
        except Exception:
            continue
        if isinstance(o, dict) and ("total_so_far" in o or "ids_so_far" in o):
            return o
    return None


def progress_ok(task, k, prog):
    if not prog:
        return False, "no progress json"
    gt = prefix_gt(task, k)
    from .tasks.t1_aggregation import _norm_ids
    try:
        tot_ok = abs(float(prog.get("total_so_far")) - gt["total"]) <= 0.011
    except Exception:
        tot_ok = False
    ids = prog.get("ids_so_far")
    ids_ok = isinstance(ids, list) and sorted(_norm_ids(ids)) == gt["ids"]
    return tot_ok and ids_ok, f"tot_ok={tot_ok} ids_ok={ids_ok} gt={gt}"


def _data(task, k):
    n = len(task.payload["chunks"])
    return (f"DATA chunk {k} of {n} ({task.payload['header']})\n" + "\n".join(task.payload["chunks"][k - 1]))


def run_horizon(task, arm, client, writer, rep=0, seed=0, run_id=None):
    if hasattr(client, "bind"):
        client.bind(task)
    n = len(task.payload["chunks"])
    g = build_germline(task); block = render_block(g)
    base = {"run_id": run_id, "task_id": task.task_id, "family": "T1H", "K": len(task.constraints),
            "arm": arm, "depth_total": n, "rep": rep, "ts": time.time()}
    carry = None
    for k in range(1, n + 1):
        if arm in ("chained", "chainedfull"):
            sys_msg = AGENT_SYS
            head = task.tau if k == 1 else ("Carry-over note from the previous epoch:\n" + carry)
        elif arm in ("anchored", "anchoredrich"):
            sys_msg = block + "\n" + BARRIER_NOTE + "\n\n" + AGENT_SYS
            head = "Start of the task." if k == 1 else ("Progress note from the previous epoch:\n" + carry)
        else:
            raise ValueError(arm)
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": head + "\n\n" + _data(task, k)}]
        ptxt = client.chat(msgs + [{"role": "user", "content": PROBE_PROMPT}], seed=seed + k)
        vec = retention_vector(task, ptxt)
        writer.write({**base, "k": k, "kind": "probe", "text": ptxt, "retention": vec, **summarize_retention(task, vec)})
        if k < n:
            prompt = CARRY_PROMPTS[arm]
            full = client.chat(msgs + [{"role": "user", "content": prompt}], seed=seed + 100 + k)
            carry = carry_part(full).strip()
            prog = parse_progress(carry) or parse_progress(full)
            ok, det = progress_ok(task, k, prog)
            cvec = retention_vector(task, carry)
            writer.write({**base, "k": k, "kind": "carry", "text": carry, "work": full[:4000], "retention": cvec,
                          **summarize_retention(task, cvec), "progress_ok": ok, "progress_details": det,
                          "carry_words": len(carry.split())})
        else:
            out = client.chat(msgs + [{"role": "user", "content": HORIZON_LEAF}], seed=seed + 300 + k)
            res = task.check_output(out)
            writer.write({**base, "k": k, "kind": "leaf", "text": out, "passed": res["passed"],
                          "per_constraint": res["per_constraint"], "details": res.get("details", "")})
