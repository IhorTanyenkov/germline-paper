"""Paper figures from figs/*.csv. Static PDF via matplotlib. Palette: reference categorical slots 1-4."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = {"B0": "#2a78d6", "B1": "#eb6834", "B2": "#1baf7a", "L": "#eda100", "B3": "#6250d6", "B3sys": "#9c3f8f",
     "chained": "#2a78d6", "chainedfull": "#eb6834", "anchored": "#1baf7a", "anchoredrich": "#eda100"}
NAME = {"B0": "chained", "B1": "system-prompt repeat", "B2": "germline block", "L": "editable block",
        "B3": "germline role (trained)", "chained": "free-form carry", "chainedfull": "prescriptive carry",
        "anchored": "germline + JSON progress", "anchoredrich": "germline + rich progress"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "grid.color": "#e6e5e1", "legend.frameon": False})

def rows(p): return list(csv.DictReader(open(p)))

# Fig 1: retention by hop with CI, 32B main pilot
r = rows("figs/main_e1_retention.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
for arm in ["B0", "B1", "B2", "L"]:
    d = [x for x in r if x["arm"] == arm]
    k = [int(x["k"]) for x in d]; m = [float(x["mean"]) for x in d]
    ax.fill_between(k, [float(x["lo"]) for x in d], [float(x["hi"]) for x in d], color=C[arm], alpha=0.12, lw=0)
    ax.plot(k, m, color=C[arm], lw=2, marker="o", ms=4, label=NAME[arm])
    dy = {"B2": 9, "B1": 0, "L": -9, "B0": 0}[arm]
    ax.annotate(NAME[arm], (k[-1], m[-1]), xytext=(5, dy), textcoords="offset points", va="center", fontsize=6.5, color="#0b0b0b")
ax.set_xlabel("hop"); ax.set_ylabel("constraints retained (fraction)"); ax.set_ylim(0.45, 1.04); ax.set_xticks(range(1, 7))
ax.grid(axis="y", lw=0.5); ax.set_xlim(0.8, 8.6)
fig.tight_layout(); fig.savefig("paper/figs/fig1_retention.pdf"); fig.savefig("paper/figs/fig1_retention.png"); plt.close(fig)

# Fig 2: leaf acceptance by depth
r = rows("figs/main_e1_leaf.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
arms = ["B0", "B1", "B2", "L"]; depths = [1, 3, 6]; w = 0.19
for i, arm in enumerate(arms):
    d = {int(x["depth"]): x for x in r if x["arm"] == arm and x["family"] == "ALL"}
    xs = [j + (i - 1.5) * w for j in range(3)]
    ys = [float(d[dep]["acc"]) for dep in depths]
    err = [[float(d[dep]["acc"]) - float(d[dep]["lo"]) for dep in depths], [float(d[dep]["hi"]) - float(d[dep]["acc"]) for dep in depths]]
    ax.bar(xs, ys, w - 0.02, color=C[arm], label=NAME[arm], yerr=err, error_kw={"lw": 0.8, "ecolor": "#52514e", "capsize": 1.5})
ax.set_xticks(range(3)); ax.set_xticklabels([f"depth {d}" for d in depths]); ax.set_ylabel("leaf success (original checkers)")
ax.set_ylim(0, 1.0); ax.grid(axis="y", lw=0.5); ax.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout(); fig.savefig("paper/figs/fig2_leaf.pdf"); fig.savefig("paper/figs/fig2_leaf.png"); plt.close(fig)

# Fig 3: injection compliance, 32B (a) and 8B (b)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), dpi=200, sharey=True, gridspec_kw={"width_ratios": [4, 6]})
r32 = [x for x in rows("figs/e3_compliance.csv") if x.get("hop")]
groups32 = [("B0", "normal", "chained"), ("B1", "normal", "sys-prompt\nrepeat"), ("B2", "normal", "germline\nblock"), ("B2", "strong", "germline block,\nstrong barrier")]
ax = axes[0]
for i, (arm, bar, lab) in enumerate(groups32):
    for j, hop in enumerate(("2", "3")):
        v = [x for x in r32 if x["arm"] == arm and x["barrier"] == bar and x["hop"] == hop][0]
        ax.bar(i + (j - 0.5) * 0.36, float(v["judged_compliance"]), 0.33, color=C[arm] if j == 0 else "#ffffff", edgecolor=C[arm], lw=1.2,
               hatch=None if j == 0 else "////", label=("attacked hop" if i == 0 else None) if j == 0 else ("child hop" if i == 0 else None))
        ax.text(i + (j - 0.5) * 0.36, float(v["judged_compliance"]) + 0.02, f'{float(v["judged_compliance"]):.2f}', ha="center", fontsize=5.5, color="#0b0b0b")
ax.set_xticks(range(4)); ax.set_xticklabels([g[2] for g in groups32], fontsize=6); ax.set_ylabel("judged compliance\nwith forged withdrawal")
ax.set_title("(a) Qwen3-32B, untrained (36/arm)", fontsize=7.5, loc="left"); ax.set_ylim(0, 1.25); ax.grid(axis="y", lw=0.5)
ax.legend(fontsize=6, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.0))
r8 = rows("figs/e3_8b_compliance.csv")
groups8 = [("B0", "trained-style", "chained"), ("B1", "trained-style", "sys-prompt\nrepeat"), ("B2", "trained-style", "germline\nblock"), ("B3", "trained-style", "germline\nrole (trained)"), ("B2", "novel", "block,\nnovel\nattacks"), ("B3", "novel", "role,\nnovel\nattacks")]
# the role-versus-training control (trained on the same data with the block in the system prompt) is drawn when its rows exist
if any(x["arm"] == "B3sys" for x in r8):
    groups8 = groups8[:4] + [("B3sys", "trained-style", "block in\nsys prompt\n(trained)")] + groups8[4:] + [("B3sys", "novel", "control,\nnovel\nattacks")]
ax = axes[1]
for i, (arm, att, lab) in enumerate(groups8):
    for j, hop in enumerate(("2", "3")):
        v = [x for x in r8 if x["arm"] == arm and x["attack"] == att and x["hop"] == hop][0]
        y = float(v["judged_compliance"])
        ax.bar(i + (j - 0.5) * 0.36, y, 0.33, color=C[arm] if j == 0 else "#ffffff", edgecolor=C[arm], lw=1.2, hatch=None if j == 0 else "////")
        if y > 0:
            ax.text(i + (j - 0.5) * 0.36, y + 0.02, f"{y:.2f}", ha="center", fontsize=5.5, color="#0b0b0b")
    if all(float([x for x in r8 if x["arm"] == arm and x["attack"] == att and x["hop"] == h][0]["judged_compliance"]) == 0 for h in ("2", "3")):
        ax.text(i, 0.03, "0 / 0", ha="center", fontsize=6, color="#0b0b0b", fontweight="bold")
ax.set_xticks(range(len(groups8))); ax.set_xticklabels([g[2] for g in groups8], fontsize=6)
ax.set_title("(b) Qwen3-8B: untrained arms vs trained role (72/arm, novel 36/arm)", fontsize=7.5, loc="left"); ax.grid(axis="y", lw=0.5)
fig.tight_layout(); fig.savefig("paper/figs/fig3_injection.pdf"); fig.savefig("paper/figs/fig3_injection.png"); plt.close(fig)

# Fig 4: horizon — retention by epoch and final acceptance by horizon
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4), dpi=200)
rr = rows("figs/e2v2_retention.csv"); ax = axes[0]
for arm in ["chained", "chainedfull", "anchored", "anchoredrich"]:
    d = [x for x in rr if x["arm"] == arm]
    ax.plot([int(x["k"]) for x in d], [float(x["mean"]) for x in d], color=C[arm], lw=2, marker="o", ms=3, label=NAME[arm],
            ls="--" if arm == "anchored" else "-")
ax.set_xlabel("epoch (context reset before each)"); ax.set_ylabel("constraints retained (fraction)"); ax.set_ylim(0, 1.05); ax.grid(axis="y", lw=0.5)
ax.set_title("(a) task identity across resets", fontsize=8, loc="left"); ax.legend(fontsize=6.5, loc="center right", bbox_to_anchor=(1.0, 0.5))
rf = rows("figs/e2v2_final.csv"); ax = axes[1]; w = 0.19
for i, arm in enumerate(["chained", "chainedfull", "anchored", "anchoredrich"]):
    d = {int(x["epochs"]): float(x["acc"]) for x in rf if x["arm"] == arm}
    ax.bar([j + (i - 1.5) * w for j in range(3)], [d[n] for n in (2, 5, 10)], w - 0.02, color=C[arm], label=NAME[arm])
ax.set_xticks(range(3)); ax.set_xticklabels(["2 epochs", "5 epochs", "10 epochs"]); ax.set_ylabel("final answer correct"); ax.set_ylim(0, 0.8); ax.grid(axis="y", lw=0.5)
ax.set_title("(b) final acceptance by horizon (n=12)", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig("paper/figs/fig4_horizon.pdf"); fig.savefig("paper/figs/fig4_horizon.png"); plt.close(fig)
print("figures written:", sorted(os.listdir("paper/figs")))
