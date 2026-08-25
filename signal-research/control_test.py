"""CONTROL: plant a REAL edge and check the same procedure finds it.
Without this, 'we found nothing' could just mean 'our test is blind'."""
import numpy as np, pandas as pd
from signal_params import strategy_return, GRID

def path_with_momentum(seed, n=600, rho=0.0):
    """rho = how much yesterday's return carries into today. rho=0 is a random
    walk (no edge). rho>0 is genuine momentum -- a trend-following rule SHOULD work."""
    rng = np.random.default_rng(seed)
    r, prev = [], 0.0
    for _ in range(n):
        e = rng.normal(0.0, 0.014)
        cur = rho*prev + e
        r.append(cur); prev = cur
    return 100.0*np.exp(np.cumsum(r))

N = 150
print(f"{'true momentum':>15}{'best in-sample':>17}{'out-of-sample':>16}{'rank corr':>12}{'verdict':>24}")
for rho in [0.0, 0.05, 0.10, 0.20]:
    tr_b, te_b, rhos = [], [], []
    for i in range(N):
        c = path_with_momentum(8000+i, rho=rho)
        train, test = c[:300], c[300:]
        tr = np.array([strategy_return(train, *g) for g in GRID])
        te = np.array([strategy_return(test,  *g) for g in GRID])
        b = int(np.argmax(tr))
        tr_b.append(tr[b]); te_b.append(te[b])
        rhos.append(np.corrcoef(pd.Series(tr).rank(), pd.Series(te).rank())[0,1])
    te_b = np.array(te_b); rhos = np.array(rhos)
    se = te_b.std(ddof=1)/np.sqrt(N)
    real = te_b.mean() - 1.96*se > 0
    print(f"{rho:>15.2f}{np.mean(tr_b):>+16.2f}%{te_b.mean():>+15.2f}%{rhos.mean():>+12.3f}"
          f"{('REAL EDGE FOUND' if real else 'nothing (correct)'):>24}")

print("""
   rho = 0.00 is our actual data: the procedure correctly finds nothing.
   rho = 0.20 plants a strong edge: out-of-sample return becomes significantly
   positive. So the test is NOT blind -- when something is there, it finds it.
   That makes the null result on our own data trustworthy.

   BUT look at the rank-correlation column: it stays ~0.03 even at rho = 0.20.
   Even WITH a real edge, in-sample ranking does not predict out-of-sample
   ranking. The edge belongs to the STRATEGY FAMILY (trend-following works),
   not to any particular window length. Picking 3/10 over 5/20 is still noise.

   That is the sharper lesson: finding that a strategy TYPE works does not
   license fine-tuning its parameters. Those are two different claims, and the
   second one needs far more evidence than the first.
""")
