"""Summarize a run: retention by (arm, k), leaf acceptance by (arm, depth), drift slopes with bootstrap CI."""
import json
import random
import sys
from collections import defaultdict


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def ols_slope(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def bootstrap_slope(rows, B=300, seed=0):
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    keys = list(by_task)
    slopes = []
    for _ in range(B):
        sample = [r for k in rng.choices(keys, k=len(keys)) for r in by_task[k]]
        slopes.append(ols_slope([r["k"] for r in sample], [r["frac"] for r in sample]))
    slopes.sort()
    return slopes[int(0.025 * B)], slopes[int(0.975 * B)]


def main(path):
    rows = load(path)
    probes = [r for r in rows if r["kind"] == "probe"]
    briefs = [r for r in rows if r["kind"] == "brief"]
    leaves = [r for r in rows if r["kind"] == "leaf"]
    arms = sorted({r["arm"] for r in rows})

    print(f"\n== {path}: {len(rows)} rows, arms={arms}, families={sorted({r['family'] for r in rows})}, "
          f"variant='{rows[0].get('variant','')}' model={rows[0].get('model','?')}")

    if probes:
        print("\n-- Probe retention (fraction of original constraints present), by arm and hop k")
        ks = sorted({r["k"] for r in probes})
        print("arm   " + "".join(f"k={k:<7}" for k in ks) + "slope [95% CI]")
        for arm in arms:
            ar = [r for r in probes if r["arm"] == arm]
            if not ar:
                continue
            cells = []
            for k in ks:
                v = [r["frac"] for r in ar if r["k"] == k]
                cells.append(f"{sum(v)/len(v):.3f}({len(v):>3})" if v else "   -      ")
            s = ols_slope([r["k"] for r in ar], [r["frac"] for r in ar])
            lo, hi = bootstrap_slope(ar)
            print(f"{arm:<5} " + " ".join(cells) + f"  {s:+.4f} [{lo:+.4f},{hi:+.4f}]")

        print("\n-- Probe retention at the deepest hop, inconvenient vs easy constraints")
        for arm in arms:
            ar = [r for r in probes if r["arm"] == arm and r["k"] == r["depth_total"] and r["inc_n"] > 0]
            if not ar:
                continue
            inc = sum(r["inc_kept"] for r in ar) / sum(r["inc_n"] for r in ar)
            easy = sum(r["easy_kept"] for r in ar) / sum(r["easy_n"] for r in ar)
            print(f"{arm:<5} inconvenient={inc:.3f}  easy={easy:.3f}  (n={len(ar)})")

    if briefs:
        print("\n-- One-step brief retention (A2): constraints surviving the FIRST brief (k=1)")
        b1 = [r for r in briefs if r["k"] == 1]
        for arm in sorted({r["arm"] for r in b1}):
          for fam in sorted({r["family"] for r in b1}):
            fr = [r for r in b1 if r["family"] == fam and r["arm"] == arm]
            if not fr:
                continue
            frac = sum(r["frac"] for r in fr) / len(fr)
            inc_n = sum(r["inc_n"] for r in fr)
            inc = sum(r["inc_kept"] for r in fr) / inc_n if inc_n else float("nan")
            easy = sum(r["easy_kept"] for r in fr) / sum(r["easy_n"] for r in fr)
            print(f"{arm} {fam}: mean retention={frac:.3f}  inconvenient kept={inc:.3f}  easy kept={easy:.3f}  "
                  f"beta_hat(dropped per hop)={sum(r['n']-r['kept'] for r in fr)/len(fr):.2f}  n={len(fr)}")

    if leaves:
        print("\n-- Leaf acceptance on ORIGINAL checkers, by arm and depth")
        depths = sorted({r["depth_total"] for r in leaves})
        print("arm   " + "".join(f"d={d:<9}" for d in depths))
        for arm in arms:
            cells = []
            for d in depths:
                v = [r["passed"] for r in leaves if r["arm"] == arm and r["depth_total"] == d]
                cells.append(f"{sum(v)/len(v):.3f}({len(v):>3}) " if v else "    -       ")
            if any(c.strip() != "-" for c in cells):
                print(f"{arm:<5} " + "".join(cells))
        print("\n-- Leaf acceptance by family at max depth")
        dmax = max(depths)
        for fam in sorted({r["family"] for r in leaves}):
            cells = []
            for arm in arms:
                v = [r["passed"] for r in leaves if r["arm"] == arm and r["depth_total"] == dmax and r["family"] == fam]
                if v:
                    cells.append(f"{arm}={sum(v)/len(v):.2f}")
            print(f"{fam} d={dmax}: " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1])
