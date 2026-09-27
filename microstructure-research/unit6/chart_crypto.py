import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

d = pd.read_csv('crypto_multi_horizon_results.csv')

BLUE = '#2a78d6'
ORANGE = '#eb6834'
RED = '#e34948'
GRID = '#e1e0d9'
INK = '#0b0b0b'
MUTED = '#898781'
SURFACE = '#fcfcfb'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

# --- panel 1: correlation by horizon, in vs out of sample ---
ax = axes[0]
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
ax.set_xlabel('forward return horizon', color=MUTED, fontsize=10)
ax.set_ylabel('correlation, true OFI vs forward return', color=MUTED, fontsize=10)
ax.set_title('BTCUSDT, 6 days, 6.19M real trades\n(a) univariate correlation by horizon',
              color=INK, fontsize=10.5, loc='left')
ax.legend(frameon=False, loc='upper right', fontsize=9)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# --- panel 2: the momentum-confound check, 1-min horizon, out-of-sample ---
ax2 = axes[1]
ax2.set_facecolor(SURFACE)
bar_labels = ['OFI alone\n(univariate)', 'OFI, controlling\nfor same-window\nmomentum']
tvals = [3.09, 0.93]
colors = [ORANGE, RED]
bars = ax2.bar(bar_labels, tvals, color=colors, width=0.55)
ax2.axhline(1.96, color=MUTED, linewidth=1, linestyle='--')
ax2.text(1.4, 2.02, '95% significance\nthreshold (t=1.96)', color=MUTED, fontsize=8, va='bottom')
for bar, v in zip(bars, tvals):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.08, f't = {v:.2f}',
              ha='center', color=INK, fontsize=10)
ax2.set_ylabel('t-statistic', color=MUTED, fontsize=10)
ax2.set_title('Out-of-sample, 1-min horizon\n(b) OFI\'s effect vanishes once momentum is controlled for',
               color=INK, fontsize=10.5, loc='left')
ax2.set_ylim(0, 3.6)
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
ax2.spines['left'].set_color(GRID)
ax2.spines['bottom'].set_color(GRID)
ax2.tick_params(colors=MUTED)
ax2.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig('crypto_ofi_chart.png', dpi=150, facecolor=SURFACE)
print("saved")
