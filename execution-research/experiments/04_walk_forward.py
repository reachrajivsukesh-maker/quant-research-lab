"""Walk-forward validation on a single path -- and why it is not enough.

Three folds is three observations. Each test window holds only 2-3 orders, so one
window that happens to contain a strong trend during an order can dominate the
whole result. Run this, then run 05 to see what adequate power looks like.
"""
import _common
import pandas as pd
from src import make_path, Config, walk_forward

CONFIGS = [Config(mode="fixed", participation=p) for p in (0.03, 0.05, 0.08)] + \
          [Config(mode="adaptive", participation=0.05, k=k) for k in (2, 4, 8)]
BASELINE = Config(mode="fixed", participation=0.05)

d = make_path(1000)
out = walk_forward(d, CONFIGS, baseline=BASELINE)
print(out.to_string(index=False))
out.to_csv(f"{_common.RESULTS}/04_walk_forward.csv", index=False)

if not out.empty:
    e = out["edge"]
    print(f"\nedges per fold : {list(e)}")
    print(f"mean           : {e.mean():.2f} bps")
    print(f"median         : {e.median():.2f} bps")
    print(f"largest fold contributes {abs(e).max()/abs(e).sum()*100:.1f}% of the total")
    print("\nA mean over three numbers where one dominates is not a result. Report")
    print("per-fold values and dispersion, or get more observations -- see 05.")
