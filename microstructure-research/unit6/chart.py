import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('decile_results_oos.csv')

BLUE = '#2a78d6'   # positive (buy-side signal -> next bar up)
RED  = '#e34948'   # negative
GRID = '#e1e0d9'
INK  = '#0b0b0b'
MUTED = '#898781'
SURFACE = '#fcfcfb'

fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)

colors = [BLUE if v >= 0 else RED for v in d['mean_bps']]
bars = ax.bar(d['decile'], d['mean_bps'], yerr=d['se_bps'], color=colors,
              width=0.62, capsize=3, error_kw={'ecolor': MUTED, 'elinewidth': 1.2})

ax.axhline(0, color=INK, linewidth=1)
ax.set_xlabel('signal decile (0 = most buy-side, 9 = most sell-side)', color=MUTED, fontsize=10)
ax.set_ylabel('mean next-bar return (bps)', color=MUTED, fontsize=10)
ax.set_title('Out-of-sample: next-1min-return by normalized signed-volume decile\nSPY, 2026-09-22 to 2026-09-25 (n=1,494 bars) — no clean monotonic pattern',
             color=INK, fontsize=11, loc='left')
ax.set_xticks(d['decile'])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('ofi_decile_chart.png', dpi=150, facecolor=SURFACE)
print("saved")
