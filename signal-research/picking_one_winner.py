"""Two DIFFERENT questions people confuse:
  Q_A: 'Does the procedure of searching work?'  -> pick per path, 200 answers
  Q_B: 'Which single parameter pair is best?'   -> pool all paths, ONE answer
We ran Q_A. Rajiv is asking about Q_B. They need different methods."""
import numpy as np, pandas as pd
from signal_params import path, GRID, COST
from control_test import path_with_momentum

def sig_full(close, short, long):
    s = pd.Series(close)
    g = (s.rolling(short).mean() > s.rolling(long).mean()).astype(float)
    g[s.rolling(long).mean().isna()] = 0.0
    return g

def ret(close, sig, lo, hi):
    s = pd.Series(close)
    pos = sig.shift(1).fillna(0.0).iloc[lo:hi]
    r   = s.pct_change().fillna(0.0).iloc[lo:hi]
    net = pos*r - pos.diff().abs().fillna(0.0)*COST
    return float((1+net).prod()-1)*100

def experiment(maker, label, N=100):
    TR = np.zeros((N, len(GRID))); TE = np.zeros((N, len(GRID)))
    for i in range(N):
        c = maker(i)
        for j, g in enumerate(GRID):
            s = sig_full(c, *g)
            TR[i, j] = ret(c, s, 100, 300)
            TE[i, j] = ret(c, s, 300, 600)
    print(f"\n{'='*76}\n{label}   ({N} paths)\n{'='*76}")

    per_path = TR.argmax(axis=1)
    print(f"METHOD A -- pick a winner on EACH path separately "
          f"({len(set(per_path))} different winners across {N} paths)")
    print(f"   in-sample {TR[np.arange(N), per_path].mean():+7.2f}%   "
          f"out-of-sample {TE[np.arange(N), per_path].mean():+7.2f}%")

    combo_avg = TR.mean(axis=0)
    j = int(combo_avg.argmax()); one = f"{GRID[j][0]}/{GRID[j][1]}"
    b_out = TE[:, j]; se = b_out.std(ddof=1)/np.sqrt(N)
    print(f"\nMETHOD B -- average each combo across ALL paths, pick ONE overall winner")
    print(f"   winner = {one}   (average in-sample score across all paths: {combo_avg[j]:+.2f}%)")
    print(f"   applied to all unseen halves: {b_out.mean():+7.2f}%  "
          f"95% CI [{b_out.mean()-1.96*se:+.2f}, {b_out.mean()+1.96*se:+.2f}]")
    allc = TE.mean()
    print(f"   average of ALL combos out-of-sample: {allc:+7.2f}%")
    print(f"   -> pooled winner beats the average combo by {b_out.mean()-allc:+.2f} points OOS")

    h1 = TR[:N//2].mean(axis=0); h2 = TR[N//2:].mean(axis=0)
    rho = np.corrcoef(pd.Series(h1).rank(), pd.Series(h2).rank())[0,1]
    print(f"   ranking stability (first half of paths vs second half): rank corr {rho:+.3f}")

experiment(lambda i: path(7000+i), "DATA WITH NO EDGE (our actual experiment)")
experiment(lambda i: path_with_momentum(8000+i, rho=0.20), "DATA WITH A REAL EDGE PLANTED")
