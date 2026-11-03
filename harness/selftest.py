"""Generate tasks, verify every reference output passes its own checkers, verify probes detect
constraints in tau, verify germline hash round-trip, and run a tiny mock lineage per arm."""
import os
import tempfile

from .tasks import make_tasks
from .probe import retention_vector
from .germline import build_germline, render_block, verify_block
from .llm import MockClient
from .chain import run_chain, JsonlWriter


def main():
    tmp = tempfile.mkdtemp(prefix="germline_selftest_")
    tasks = make_tasks(["T1", "T2", "T3"], 5, [4, 6, 8], base_seed=1, tmpdir=tmp)
    bad = 0
    for t in tasks:
        res = t.check_output(t.reference_output())
        if not res["passed"]:
            bad += 1
            print("REFERENCE FAILS", t.task_id, res)
        vec = retention_vector(t, t.tau)
        missing = [c for c, v in vec.items() if not v]
        if missing:
            bad += 1
            print("PROBE MISSES in tau", t.task_id, missing)
        g = build_germline(t)
        assert verify_block(render_block(g), g)
        wrong = t.check_output("WRONG")
        if wrong["passed"]:
            bad += 1
            print("WRONG accepted", t.task_id)
    print(f"{len(tasks)} tasks checked, {bad} problems")

    out = os.path.join(tmp, "mini.jsonl")
    w = JsonlWriter(out)
    client = MockClient(seed=3)
    for arm in ["B0", "B1", "B2", "B3", "L"]:
        r = run_chain(tasks[0], arm, 3, client, w, run_id="selftest")
        print(f"arm {arm}: leaf passed={r['passed']}")
    w.close()
    print("mini run ->", out)
    return bad


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
