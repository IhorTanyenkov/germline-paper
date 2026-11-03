"""E2 summary: retention by epoch, progress accuracy by epoch, final acceptance by horizon, per arm."""
import json
import sys


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    arms = sorted({r["arm"] for r in rows}); ns = sorted({r["depth_total"] for r in rows})
    print(f"== {path}: {len(rows)} rows, arms={arms}, horizons(epochs)={ns}")
    print("\n-- probe retention by epoch k (pooled over horizons)")
    ks = sorted({r["k"] for r in rows if r["kind"] == "probe"})
    for arm in arms:
        cells = []
        for k in ks:
            v = [r["frac"] for r in rows if r["kind"] == "probe" and r["arm"] == arm and r["k"] == k]
            cells.append(f"{sum(v)/len(v):.3f}({len(v):>2})" if v else "    -    ")
        print(f"{arm:<11} " + " ".join(cells))
    print("\n-- carry: progress JSON correct vs prefix ground truth, by epoch k; mean carry words")
    for arm in arms:
        cells = []
        for k in ks[:-1]:
            v = [r for r in rows if r["kind"] == "carry" and r["arm"] == arm and r["k"] == k]
            if v:
                cells.append(f"k={k}: {sum(r['progress_ok'] for r in v)/len(v):.2f} ({sum(r['carry_words'] for r in v)//len(v)}w)")
        print(f"{arm:<11} " + " | ".join(cells))
    print("\n-- carry retention (chained arms: does the note still carry the task?) by epoch")
    for arm in arms:
        cells = []
        for k in ks[:-1]:
            v = [r["frac"] for r in rows if r["kind"] == "carry" and r["arm"] == arm and r["k"] == k]
            if v:
                cells.append(f"k={k}: {sum(v)/len(v):.3f}")
        print(f"{arm:<11} " + " | ".join(cells))
    print("\n-- FINAL acceptance on original checkers, by horizon n (epochs)")
    for arm in arms:
        cells = []
        for n in ns:
            v = [r["passed"] for r in rows if r["kind"] == "leaf" and r["arm"] == arm and r["depth_total"] == n]
            cells.append(f"n={n}: {sum(v)/len(v):.2f} ({len(v)})" if v else f"n={n}: -")
        print(f"{arm:<11} " + " | ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1])
