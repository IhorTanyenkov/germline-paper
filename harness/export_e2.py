"""Export E2 (horizon) figure data: retention by epoch, progress accuracy by epoch, final acceptance by horizon."""
import csv
import json
import sys


def main(paths, out):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    arms = sorted({r["arm"] for r in rows})
    with open(out + "_retention.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "k", "mean", "n"])
        for arm in arms:
            for k in sorted({r["k"] for r in rows if r["kind"] == "probe"}):
                v = [r["frac"] for r in rows if r["kind"] == "probe" and r["arm"] == arm and r["k"] == k]
                if v: w.writerow([arm, k, f"{sum(v)/len(v):.4f}", len(v)])
    with open(out + "_progress.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "k", "progress_ok", "mean_words", "n"])
        for arm in arms:
            for k in sorted({r["k"] for r in rows if r["kind"] == "carry"}):
                v = [r for r in rows if r["kind"] == "carry" and r["arm"] == arm and r["k"] == k]
                if v: w.writerow([arm, k, f"{sum(r['progress_ok'] for r in v)/len(v):.4f}", sum(r["carry_words"] for r in v)//len(v), len(v)])
    with open(out + "_final.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm", "epochs", "acc", "n", "fail_spec", "fail_progress"])
        for arm in arms:
            for n in sorted({r["depth_total"] for r in rows}):
                L = [r for r in rows if r["kind"] == "leaf" and r["arm"] == arm and r["depth_total"] == n]
                if not L: continue
                pr = {r["task_id"]: r for r in rows if r["arm"] == arm and r["kind"] == "probe" and r["depth_total"] == n and r["k"] == n}
                ca = {r["task_id"]: r for r in rows if r["arm"] == arm and r["kind"] == "carry" and r["depth_total"] == n and r["k"] == n - 1}
                fs = sum(1 for l in L if not l["passed"] and pr.get(l["task_id"], {}).get("frac", 1) < 1)
                fp = sum(1 for l in L if not l["passed"] and ca.get(l["task_id"], {}).get("progress_ok", True) is False)
                w.writerow([arm, n, f"{sum(r['passed'] for r in L)/len(L):.4f}", len(L), fs, fp])
    print("wrote", out + "_{retention,progress,final}.csv")


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
