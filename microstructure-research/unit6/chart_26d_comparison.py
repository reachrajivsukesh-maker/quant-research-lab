import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# HAC maxlags=10 numbers, both passes (beta, t) -> derive SE and 95% CI
data = {
    ('6 days\n(6.2M trades)', 'in-sample'):      (0.000032,  2.06),
    ('6 days\n(6.2M trades)', 'out-of-sample'):  (0.000024,  0.96),
    ('26 days\n(24M trades)', 'in-sample'):      (0.0000017, 0.23),
    ('26 days\n(24M trades)', 'out-of-sample'):  (0.0000197, 1.78),
}

BLUE = '#2a78d6'
ORANGE = '#eb6834'
GRID = '#e1e0d9'
INK = '#0b0b0b'
MUTED = '#898781'
SURFACE = '#fcfcfb'

fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

groups = ['6 days\n(6.2M trades)', '26 days\n(24M trades)']
x = np.arange(len(groups))
offset = 0.16

for label, color, dx in [('in-sample', BLUE, -offset), ('out-of-sample', ORANGE, offset)]:
    betas, cis_lo, cis_hi = [], [], []
    for g in groups:
        b, t = data[(g, label)]
        se = b / t if t != 0 else np.nan
        betas.append(b * 1e6)  # scale to "per-unit-OFI, in millionths" for readable axis
        cis_lo.append((b - 1.96*se) * 1e6)
        cis_hi.append((b + 1.96*se) * 1e6)
    yerr_lo = np.array(betas) - np.array(cis_lo)
    yerr_hi = np.array(cis_hi) - np.array(betas)
    ax.errorbar(x + dx, betas, yerr=[yerr_lo, yerr_hi], fmt='o', color=color, label=label,
                capsize=5, markersize=8, linewidth=1.8, elinewidth=1.8)

ax.axhline(0, color=INK, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("OFI's own coefficient (×10⁻⁶, HAC 95% CI)\ncontrolling for momentum", color=MUTED, fontsize=10)
ax.set_title("More data didn't confirm the 6-day result — it fell toward zero\nand the confidence interval widened to swallow it",
              color=INK, fontsize=11, loc='left')
ax.legend(frameon=False, loc='upper right', fontsize=10)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('crypto_6d_vs_26d_chart.png', dpi=150, facecolor=SURFACE)
print("saved")
