"""Figures for the README. Palette follows a validated categorical set;
one axis per chart, legend whenever there are two or more series."""
import _common
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#c3c2b7"})


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9.5)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


raw = pd.read_csv(f"{_common.RESULTS}/05_monte_carlo_raw.csv")
base = raw["fixed 5% (baseline)"]

# --- Figure 1: distribution of the paired difference (single series, no legend)
fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=170)
fig.patch.set_facecolor(SURFACE)
diff = (raw["adaptive k=4"] - base).dropna()
ax.hist(diff, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
ax.axvline(0, color=MUTED, linewidth=1.4, linestyle="--")
ax.axvline(diff.mean(), color=ORANGE, linewidth=2.2)
ax.annotate(f"mean +{diff.mean():.1f} bps\n{(diff>0).mean()*100:.1f}% of paths worse",
            xy=(diff.mean(), ax.get_ylim()[1]*0.72), xytext=(diff.mean()+8, ax.get_ylim()[1]*0.80),
            color=ORANGE, fontsize=9.5,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
style(ax, "Adaptive execution vs constant participation, 300 paired paths",
      "Execution cost difference (bps).  Positive = adaptive is worse.", "Number of paths")
fig.tight_layout(); fig.savefig(f"{_common.RESULTS}/fig1_monte_carlo.png",
                                facecolor=SURFACE); plt.close(fig)

# --- Figure 2: ablation (single series bar, direct-labelled)
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=170)
fig.patch.set_facecolor(SURFACE)
names = ["fixed 5% (baseline)", "adaptive ramp-only", "adaptive k=4", "adaptive ease-only"]
vals = [raw[n].mean() for n in names]
cols = [MUTED, AQUA, ORANGE, ORANGE]
bars = ax.bar(range(len(names)), vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.1f}", ha="center",
            color=INK, fontsize=9.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("adaptive ", "adaptive\n").replace(" (baseline)", "\n(baseline)")
                    for n in names], color=MUTED, fontsize=9)
style(ax, "Which half of the adaptive rule is responsible?", "",
      "Mean execution cost (bps).  Lower is better.")
fig.tight_layout(); fig.savefig(f"{_common.RESULTS}/fig2_ablation.png",
                                facecolor=SURFACE); plt.close(fig)

# --- Figure 3: participation sweep
sweep = pd.read_csv(f"{_common.RESULTS}/05_participation_sweep.csv")
x = [float(s.strip('%')) for s in sweep["participation"]]
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=170)
fig.patch.set_facecolor(SURFACE)
ax.plot(x, sweep["mean_bps"], marker="o", markersize=6, color=BLUE, linewidth=2.2)
ax.annotate("falls monotonically — impact constant\nis too gentle to create a real tradeoff",
            xy=(x[-2], sweep["mean_bps"].iloc[-2]),
            xytext=(x[2]+1, sweep["mean_bps"].iloc[1]),
            color=MUTED, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
style(ax, "Execution cost vs participation rate (120 paths)",
      "Participation rate (% of forecast volume)", "Mean execution cost (bps)")
fig.tight_layout(); fig.savefig(f"{_common.RESULTS}/fig3_participation_sweep.png",
                                facecolor=SURFACE); plt.close(fig)

print("wrote fig1_monte_carlo.png, fig2_ablation.png, fig3_participation_sweep.png")
