"""Export figure data as CSV from one or more (rescored) run files.
  python -m harness.export_fig harness/runs/main_e1_B0.rescored.jsonl harness/runs/main_e1_B1.rescored.jsonl ... --out figs/main_e1
Writes <out>_retention.csv (arm,k,mean,lo,hi,n), <out>_leaf.csv (arm,depth,family,acc,n),
<out>_collapse.csv (arm,k,collapsed,n), <out>_link.csv (arm,intact,acc,n)."""
import argparse
import csv
import json
import os
import random
from collections import defaultdict


def boot_mean(vals, B=500, seed=0):
    rng = random.Random(seed)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(B))
    return m, bs[int(0.025 * B)], bs[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in a.paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    arms = sorted({r["arm"] for r in rows})

    with open(a.out + "_retention.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "k", "mean", "lo", "hi", "n"])
        for arm in arms:
            for k in sorted({r["k"] for r in rows if r["kind"] == "probe"}):
                v = [r["frac"] for r in rows if r["kind"] == "probe" and r["arm"] == arm and r["k"] == k]
                if v:
                    m, lo, hi = boot_mean(v); w.writerow([arm, k, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(v)])

    with open(a.out + "_leaf.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "depth", "family", "acc", "lo", "hi", "n"])
        for arm in arms:
            for d in sorted({r["depth_total"] for r in rows}):
                for fam in ["ALL"] + sorted({r["family"] for r in rows}):
                    v = [1.0 if r["passed"] else 0.0 for r in rows if r["kind"] == "leaf" and r["arm"] == arm
                         and r["depth_total"] == d and (fam == "ALL" or r["family"] == fam)]
                    if v:
                        m, lo, hi = boot_mean(v); w.writerow([arm, d, fam, f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}", len(v)])

    with open(a.out + "_collapse.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "k", "collapsed", "n"])
        for arm in arms:
            for k in sorted({r["k"] for r in rows if r["kind"] == "brief"}):
                b = [r for r in rows if r["kind"] == "brief" and r["arm"] == arm and r["k"] == k]
                if b:
                    w.writerow([arm, k, sum(1 for r in b if r["frac"] <= 0.5), len(b)])

    with open(a.out + "_link.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "deepest_probe", "acc", "n"])
        leaves = {(r["task_id"], r["arm"], r["depth_total"], r["rep"]): r for r in rows if r["kind"] == "leaf"}
        for arm in arms:
            buckets = defaultdict(list)
            for r in rows:
                if r["kind"] == "probe" and r["arm"] == arm and r["k"] == r["depth_total"] and r["depth_total"] > 1:
                    l = leaves.get((r["task_id"], arm, r["depth_total"], r["rep"]))
                    if l:
                        buckets["intact" if r["frac"] >= 1 else "damaged"].append(1.0 if l["passed"] else 0.0)
            for key, v in buckets.items():
                w.writerow([arm, key, f"{sum(v)/len(v):.4f}", len(v)])
    print("wrote", a.out + "_{retention,leaf,collapse,link}.csv")


if __name__ == "__main__":
    main()
