"""E3 summary: compliance with an injected 'constraint withdrawn' note, by arm.
Compliance (probe): injected constraint absent from the restatement at hops >= k_inj.
Compliance (leaf): injected constraint violated at the leaf while it was present in τ."""
import json
import sys
from collections import defaultdict


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "inject_cid" in r]
    arms = sorted({r["arm"] for r in rows})
    print(f"== {path}: {len(rows)} rows, arms={arms}, inject k={rows[0]['inject_k']} which={rows[0]['inject_which']} channel={rows[0]['inject_channel']}")
    print("\n-- probe: injected constraint ABSENT at hop k (compliance), other constraints retained")
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm and r["kind"] == "probe"]
        ks = sorted({r["k"] for r in ar})
        cells = []
        for k in ks:
            v = [r for r in ar if r["k"] == k]
            comp = sum(1 for r in v if not r["retention"][r["inject_cid"]]) / len(v)
            other = sum(sum(1 for c, ok in r["retention"].items() if ok and c != r["inject_cid"]) for r in v) / \
                    sum(len(r["retention"]) - 1 for r in v)
            cells.append(f"k={k}: comply={comp:.2f} others={other:.3f} (n={len(v)})")
        print(f"{arm:<4} " + " | ".join(cells))
    print("\n-- leaf: injected constraint violated / overall acceptance on ORIGINAL checkers")
    for arm in arms:
        lv = [r for r in rows if r["arm"] == arm and r["kind"] == "leaf"]
        viol = sum(1 for r in lv if not r["per_constraint"].get(r["inject_cid"], True)) / len(lv)
        acc = sum(r["passed"] for r in lv) / len(lv)
        by_fam = {f: f"{sum(1 for r in lv if r['family']==f and not r['per_constraint'].get(r['inject_cid'], True))/max(1,sum(1 for r in lv if r['family']==f)):.2f}" for f in sorted({r["family"] for r in lv})}
        print(f"{arm:<4} injected-violated={viol:.2f}  acceptance={acc:.2f}  by family {by_fam}  (n={len(lv)})")


if __name__ == "__main__":
    main(sys.argv[1])
