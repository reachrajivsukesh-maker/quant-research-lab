"""Does the adaptive execution rule work in ANY market regime?

Sweep a momentum knob from strong mean reversion to strong momentum and compare
adaptive against a constant participation rate.

PREDICTION MADE BEFORE RUNNING (and FALSIFIED): the rule is a coherent momentum
bet -- speed up when the market runs away, ease off when it moves your way -- so
it should pay once momentum exists.

ACTUAL RESULT: it loses in every regime, and the penalty GROWS monotonically with
momentum. The prediction was backwards. Four follow-up hypotheses for the
mechanism were tested and all four failed; see docs/FINDINGS.md Part III.
"""
import numpy as np, pandas as pd
from unit5_core import run, shortfall, CAP

from regime_paths import path_mom

N = 200
print(f"{'market type':>22}{'rho':>7}{'adaptive vs fixed':>20}{'95% CI':>22}{'win rate':>11}")
print("-"*82)
for rho, label in [(-0.20,"strong mean reversion"), (-0.10,"mild mean reversion"),
                   (0.0,  "random walk (ours)"),    (0.10, "mild momentum"),
                   (0.20, "strong momentum"),       (0.35, "very strong momentum")]:
    diffs = []
    for i in range(N):
        d = path_mom(9000+i, rho)
        f = shortfall(run(d, mode="fixed",    base=0.05))
        a = shortfall(run(d, mode="adaptive", base=0.05, k=4))
        if not (np.isnan(f) or np.isnan(a)): diffs.append(a-f)
    x = np.array(diffs); se = x.std(ddof=1)/np.sqrt(len(x))
    lo, hi = x.mean()-1.96*se, x.mean()+1.96*se
    verdict = "adaptive WINS" if hi < 0 else ("adaptive loses" if lo > 0 else "no difference")
    print(f"{label:>22}{rho:>7.2f}{x.mean():>+19.2f} {f'[{lo:+.2f}, {hi:+.2f}]':>22}"
          f"{(x<0).mean()*100:>10.1f}%   {verdict}")
print("""
(negative = adaptive is CHEAPER = adaptive is better)
""")
