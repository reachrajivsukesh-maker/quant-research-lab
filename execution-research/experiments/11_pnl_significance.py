"""Paired P&L differences with confidence intervals. Same path, both rules."""
import numpy as np
from _pnl import full_run
from regime_paths import path_mom

N = 200
print("Paired P&L differences (percentage points). Positive = the first rule made more.\n")
print(f"{'rho':>7}{'adaptive - fixed':>28}{'inverted - fixed':>28}{'inverted - adaptive':>30}")
print("-"*95)
for rho in [-0.20, 0.0, 0.20, 0.35]:
    F, A, I = [], [], []
    for i in range(N):
        d = path_mom(9000+i, rho)
        F.append(full_run(d,"fixed")[1]); A.append(full_run(d,"adaptive")[1]); I.append(full_run(d,"inverted")[1])
    F, A, I = map(np.array, (F, A, I))
    def g(x):
        se = x.std(ddof=1)/np.sqrt(len(x))
        star = "*" if abs(x.mean())-1.96*se > 0 else " "
        return f"{x.mean():+7.2f}{star} [{x.mean()-1.96*se:+.2f},{x.mean()+1.96*se:+.2f}]"
    print(f"{rho:>7.2f}{g(A-F):>28}{g(I-F):>28}{g(I-A):>30}")
print("\n   (* = interval excludes zero)")
print("""
Compare against the SHORTFALL ranking, which said in every regime:
       inverted BEST, fixed middle, adaptive WORST.
""")
