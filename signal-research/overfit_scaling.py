"""WHY does searching hurt? Because the more combinations you try, the higher
the best one scores by luck alone. Measure it."""
import numpy as np, pandas as pd
from signal_params import path, strategy_return, GRID

N_PATHS = 150
rng = np.random.default_rng(0)

print("="*76)
print("MECHANISM: in-sample winner's score vs HOW MANY combinations you tried")
print("="*76)
print(f"{'combos tried':>14}{'best in-sample':>17}{'same, out-of-sample':>22}{'noise inflation':>18}")
for k in [1, 3, 5, 10, 20, 34]:
    tr_b, te_b = [], []
    for i in range(N_PATHS):
        c = path(7000+i); train, test = c[:300], c[300:]
        subset = GRID if k == len(GRID) else [GRID[j] for j in rng.choice(len(GRID), k, replace=False)]
        scores = {g: strategy_return(train, *g) for g in subset}
        best = max(scores, key=scores.get)
        tr_b.append(scores[best]); te_b.append(strategy_return(test, *best))
    tr_b, te_b = np.array(tr_b), np.array(te_b)
    print(f"{k:>14}{tr_b.mean():>+16.2f}%{te_b.mean():>+21.2f}%{tr_b.mean()-te_b.mean():>+17.2f}%")

print("""
   Trying ONE fixed rule: in-sample and out-of-sample roughly agree.
   Trying 34 rules and keeping the winner: in-sample looks great, out-of-sample
   is unchanged. The gap IS the overfitting, and it grows with the search.

   Nothing about the market changed between those rows. Only how hard we looked.
""")

print("="*76)
print("DOES IN-SAMPLE RANKING CARRY ANY INFORMATION AT ALL?")
print("="*76)
rhos = []
for i in range(N_PATHS):
    c = path(7000+i); train, test = c[:300], c[300:]
    tr = np.array([strategy_return(train, *g) for g in GRID])
    te = np.array([strategy_return(test,  *g) for g in GRID])
    rt = pd.Series(tr).rank().values; rs = pd.Series(te).rank().values
    rhos.append(np.corrcoef(rt, rs)[0, 1])
rhos = np.array(rhos); se = rhos.std(ddof=1)/np.sqrt(len(rhos))
print(f"   rank correlation (in-sample vs out-of-sample), averaged over {N_PATHS} paths:")
print(f"      {rhos.mean():+.4f}   95% CI [{rhos.mean()-1.96*se:+.4f}, {rhos.mean()+1.96*se:+.4f}]")
print(f"   -> {'ranking carries information' if rhos.mean()-1.96*se > 0 else 'ZERO. The in-sample ranking is noise.'}")
print("""
   This is the cleanest statement of the problem. Knowing which parameters won
   in the first half tells you nothing whatsoever about which will win in the
   second half. The search is not weakly informative -- it is exactly useless.
""")
