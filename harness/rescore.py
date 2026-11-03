"""Recompute retention for probe/brief/amend rows from stored text (after probe-pattern changes).
Tasks are rebuilt deterministically from (family, K, index, base seed). Writes <in>.rescored.jsonl."""
import json
import re
import sys

from .tasks import make_tasks, restore_from_rows
from .probe import retention_vector, summarize_retention


def main(path, base_seed, override=None):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    mode = restore_from_rows(rows, override)
    print("generator config restored from:", mode)
    fams = sorted({r["family"] for r in rows})
    Ks = sorted({r["K"] for r in rows})
    n = 1 + max(int(re.search(r"-(\d+)-K", r["task_id"]).group(1)) for r in rows)
    tasks = {t.task_id: t for t in make_tasks(fams, n, Ks, base_seed=base_seed)}
    out = path.replace(".jsonl", ".rescored.jsonl")
    changed = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            if r["kind"] in ("probe", "brief", "amend"):
                t = tasks[r["task_id"]]
                vec = retention_vector(t, r["text"])
                if vec != r["retention"]:
                    changed += 1
                r["retention"] = vec
                r.update(summarize_retention(t, vec))
            elif r["kind"] == "leaf":
                t = tasks[r["task_id"]]
                res = t.check_output(r["text"])
                if res["passed"] != r["passed"]:
                    changed += 1
                r["passed"] = res["passed"]; r["per_constraint"] = res["per_constraint"]
                r["details"] = res.get("details", "")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"rescored {len(rows)} rows, {changed} retention vectors changed -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, sys.argv[3:] or None)
