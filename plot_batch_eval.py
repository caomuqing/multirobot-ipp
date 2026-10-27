#!/usr/bin/env python3
"""
Plot bar charts from batch_eval_summary.csv.

Produces one figure per metric, with:
  - X axis: target_size
  - Grouped bars: model × sensing × no_collision   (up to 8 combinations)
      Colour encodes (model, sensing); fill vs hatch encodes no_collision.
      Solid  = no_collision=True  (nc)
      Hatched = no_collision=False (coll)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(SCRIPT_DIR, "batch_eval_summary.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "batch_eval_plots.png")

METRICS = {
    "avgunc":      "Avg Uncertainty (avgunc)",
    "stdunc":      "Std Uncertainty (stdunc)",
    "minnvisit":   "Min Visit Count (minnvisit)",
    "avgnvisit":   "Avg Visit Count (avgnvisit)",
    "stdnvisit":   "Std Visit Count (stdnvisit)",
    "avggapvisit": "Avg Gap Visit (avggapvisit)",
    "stdgapvisit": "Std Gap Visit (stdgapvisit)",
}

MODEL_SHORT = {
    "centralized_nocollision_05":   "Cent",
    "decentralized_nocollision_05": "Decent",
    "centralized_nocollision_hgt003":   "Cent",
    "decentralized_nocollision_hgt003": "Decent",    
}
SENSING_SHORT = {0: "CS", 1: "DS"}   # CS = centralised sensing, DS = decentral

# Base colour per (model, sensing) — shared across nc=True/False
BASE_COLORS = {
    ("centralized_nocollision_05",   0): "#4C72B0",
    ("centralized_nocollision_05",   1): "#55A868",
    ("decentralized_nocollision_05", 0): "#C44E52",
    ("decentralized_nocollision_05", 1): "#DD8452",
    ("centralized_nocollision_hgt003", 0): "#4C72B0",
    ("centralized_nocollision_hgt003", 1): "#55A868",
    ("decentralized_nocollision_hgt003", 0): "#C44E52",
    ("decentralized_nocollision_hgt003", 1): "#DD8452",
}
# Hatch: solid for nc=True, dense hatch for nc=False
NC_HATCH = {1: "",      # no_collision=True  → solid
            0: "////"}  # no_collision=False → hatched

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["decentralized"] = df["decentralized"].astype(int)

# Handle optional no_collision column (backward-compatible)
if "no_collision" not in df.columns:
    df["no_collision"] = 1   # treat old CSVs as nc=True
df["no_collision"] = df["no_collision"].astype(int)

models  = sorted(df["model"].unique())
senses  = sorted(df["decentralized"].unique())
nc_vals = sorted(df["no_collision"].unique())

if "num_agents" in df.columns:
    df["num_agents"] = df["num_agents"].astype(int)

# X-axis: num_agents when multiple values; else target_size
if "num_agents" in df.columns and df["num_agents"].nunique() > 1:
    x_categories = sorted(int(x) for x in df["num_agents"].unique())
    x_col = "num_agents"
    x_axis_title = "Number of agents"
    x_tick_fmt = "na={}"
else:
    x_categories = sorted(int(x) for x in df["target_size"].unique())
    x_col = "target_size"
    x_axis_title = "Number of targets"
    x_tick_fmt = "tgt={}"

# Build group list in a consistent order; only include combos present in data
groups = []
for m in models:
    for s in senses:
        for nc in nc_vals:
            if len(df[(df["model"] == m) &
                      (df["decentralized"] == s) &
                      (df["no_collision"] == nc)]) > 0:
                groups.append((m, s, nc))

n_groups      = len(groups)
bar_width     = max(0.08, min(0.18, 1.0 / (n_groups + 2)))
cluster_gap   = 0.12
cluster_width = n_groups * bar_width + cluster_gap
x_centers     = np.arange(len(x_categories)) * cluster_width

# ── Plot ──────────────────────────────────────────────────────
ncols = 3
nrows = (len(METRICS) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows))
axes = axes.flatten()

for ax_idx, (metric_key, metric_label) in enumerate(METRICS.items()):
    ax = axes[ax_idx]

    for g_idx, (model, sense, nc) in enumerate(groups):
        color  = BASE_COLORS.get((model, sense), "#888888")
        hatch  = NC_HATCH[nc]
        offset = (g_idx - (n_groups - 1) / 2) * bar_width

        values = []
        for xv in x_categories:
            row = df[(df["model"] == model) &
                     (df[x_col] == xv) &
                     (df["decentralized"] == sense) &
                     (df["no_collision"] == nc)]
            values.append(float(row[metric_key].values[0]) if len(row) > 0 else np.nan)

        ms  = MODEL_SHORT.get(model, model)
        ss  = SENSING_SHORT.get(sense, str(sense))
        ncs = "nc" if nc else "coll"
        lbl = f"{ms}/{ss}/{ncs}"

        bars = ax.bar(
            x_centers + offset,
            values,
            width=bar_width * 0.92,
            color=color,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.6,
            label=lbl,
        )

        for bar, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=6, rotation=90, color="0.3",
                )

    ax.set_title(metric_label, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_axis_title, fontsize=10)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([x_tick_fmt.format(t) for t in x_categories], fontsize=10)
    ax.yaxis.set_tick_params(labelsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax * 1.25)

# Hide unused axes
for i in range(len(METRICS), len(axes)):
    axes[i].set_visible(False)

# ── Legend ────────────────────────────────────────────────────
handles = []
for model, sense, nc in groups:
    color = BASE_COLORS.get((model, sense), "#888888")
    hatch = NC_HATCH[nc]
    ms    = MODEL_SHORT.get(model, model)
    ss    = SENSING_SHORT.get(sense, str(sense))
    ncs   = "nc" if nc else "coll"
    handles.append(mpatches.Patch(
        facecolor=color, hatch=hatch,
        edgecolor="grey",
        label=f"{ms}/{ss}/{ncs}",
    ))

# Add a visual separator explaining the hatch convention
handles += [
    mpatches.Patch(facecolor="white", edgecolor="grey", label="— solid = no-collision ON"),
    mpatches.Patch(facecolor="white", edgecolor="grey", hatch="////", label="— hatch = no-collision OFF"),
]

ncol_legend = min(4, n_groups + 2)
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=ncol_legend,
    fontsize=9,
    framealpha=0.9,
    bbox_to_anchor=(0.5, 0.01),
)

fig.suptitle(
    "COMPASS Batch Evaluation Summary  (mean over evaluation episodes)\n"
    "Colour = model × sensing  |  Solid = no-collision ON  |  Hatch = no-collision OFF",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout(rect=[0, 0.10, 1, 1])
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
plt.show()
