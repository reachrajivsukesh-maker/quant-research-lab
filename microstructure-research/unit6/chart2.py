import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

d = pd.read_csv('multi_horizon_results.csv')
d = d[d['signal'] == 'norm_signal']

BLUE = '#2a78d6'    # in-sample
ORANGE = '#eb6834'  # out-of-sample
GRID = '#e1e0d9'
INK = '#0b0b0b'
MUTED = '#898781'
SURFACE = '#fcfcfb'

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

horizons = sorted(d['horizon_min'].unique())
offset = 0.15

for label, color, dx in [('in-sample', BLUE, -offset), ('out-of-sample', ORANGE, offset)]:
    sub = d[d['sample'] == label].sort_values('horizon_min')
    x = np.array(sub['horizon_min']) + dx
    y = sub['r'].values
    yerr_lo = y - sub['ci_lo'].values
    yerr_hi = sub['ci_hi'].values - y
    ax.errorbar(x, y, yerr=[yerr_lo, yerr_hi], fmt='o', color=color, label=label,
                capsize=4, markersize=7, linewidth=1.5, elinewidth=1.5)

ax.axhline(0, color=INK, linewidth=1)
ax.set_xticks(horizons)
ax.set_xticklabels([f'{h} min' for h in horizons])
ax.set_xlabel('return horizon', color=MUTED, fontsize=10)
ax.set_ylabel('correlation, signal vs future return (95% CI)', color=MUTED, fontsize=10)
ax.set_title('SPY, 52 trading days (2026-07-15 to 2026-09-25) — norm_signal vs forward return\nnearly every CI touches or crosses zero; signs flip between in-sample and out-of-sample at 1-2min',
             color=INK, fontsize=10.5, loc='left')
ax.legend(frameon=False, loc='upper left', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('multi_horizon_chart.png', dpi=150, facecolor=SURFACE)
print("saved")
