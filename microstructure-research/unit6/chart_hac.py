import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'crypto_hac_chart.png')

BLUE = '#2a78d6'
ORANGE = '#eb6834'
RED = '#e34948'
GRID = '#e1e0d9'
INK = '#0b0b0b'
MUTED = '#898781'
SURFACE = '#fcfcfb'

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor=SURFACE)

# --- panel 1: naive OLS vs HAC t-stat for OFI's own coefficient, by sample ---
ax = axes[0]
ax.set_facecolor(SURFACE)
samples = ['in-sample', 'out-of-sample', 'full sample']
naive_t = [1.90, 0.93, None]   # full-sample naive wasn't computed before; leave gap
hac_t   = [2.05, 0.97, 2.28]   # HAC maxlags=10 values
x = np.arange(len(samples))
w = 0.32
ax.bar(x - w/2, [v if v is not None else 0 for v in naive_t], width=w, color=ORANGE, label='naive OLS SE')
ax.bar(x + w/2, hac_t, width=w, color=BLUE, label='HAC (Newey-West) SE')
ax.axhline(1.96, color=MUTED, linestyle='--', linewidth=1)
ax.text(2.05, 2.02, '95% threshold', color=MUTED, fontsize=8, va='bottom')
ax.set_xticks(x)
ax.set_xticklabels(samples)
ax.set_ylabel("OFI's own t-statistic\n(controlling for momentum)", color=MUTED, fontsize=10)
ax.set_title("(a) correcting the standard error changes the verdict\nin-sample and pooled cross the significance line under HAC",
              color=INK, fontsize=10, loc='left')
ax.legend(frameon=False, loc='upper left', fontsize=9)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color(GRID)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=MUTED)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# --- panel 2: coefficient stability -- OFI's beta vs momentum's beta, across samples ---
ax2 = axes[1]
ax2.set_facecolor(SURFACE)
ofi_beta = [0.000032, 0.000024, 0.000031]
mom_beta = [-0.0262, 0.0750, 0.0029]
# normalize both to "percent of their own max abs value" isn't honest -- instead show OFI beta directly (tiny, consistent),
# and momentum beta on its own natural scale, as two side-by-side small-multiple bars sharing zero line but different axis feel.
axt = ax2.twinx()
b1 = ax2.bar(x - w/2, ofi_beta, width=w, color=BLUE, label='OFI beta (left axis)')
b2 = axt.bar(x + w/2, mom_beta, width=w, color=RED, label='momentum beta (right axis)')
ax2.axhline(0, color=INK, linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(samples)
ax2.set_ylabel('OFI coefficient', color=BLUE, fontsize=10)
axt.set_ylabel('momentum coefficient', color=RED, fontsize=10)
ax2.set_title("(b) OFI's coefficient is stable across samples;\nmomentum's swings sign entirely", color=INK, fontsize=10, loc='left')
for spine in ['top']:
    ax2.spines[spine].set_visible(False)
    axt.spines[spine].set_visible(False)
ax2.spines['left'].set_color(BLUE)
axt.spines['right'].set_color(RED)
ax2.spines['bottom'].set_color(GRID)
ax2.tick_params(axis='y', colors=BLUE)
axt.tick_params(axis='y', colors=RED)
ax2.tick_params(axis='x', colors=MUTED)
lines = [b1, b2]
ax2.legend(lines, ['OFI beta', 'momentum beta'], frameon=False, loc='upper center',
           bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, facecolor=SURFACE, bbox_inches='tight')
print("saved")
