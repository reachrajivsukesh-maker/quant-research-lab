"""Adaptive wins on P&L in BOTH directions. That rules out a directional
exposure story. Next candidate: it simply tracks the intended position better."""
import numpy as np, pandas as pd
from _inverted import CAP, COST, IMPACT_C
from regime_paths import path_mom

def diag(sub, mode, base=0.05, k=4):
    cash, shares, ref = CAP, 0.0, None
    gaps, imp_paid, n_shares = [], 0.0, 0.0
    for _, row in sub.iterrows():
        close, sig = row["close"], row["signal"]
        volf, pdec, sigma = row["vol_forecast"], row["prev_close"], row["vol_20d"]
        if pd.isna(volf) or pd.isna(pdec) or pd.isna(sigma) or pd.isna(sig): continue
        value = cash + shares*close
        target = (value/close) if sig == 1 else 0.0
        gap = target - shares
        gaps.append(abs(gap)*close/value)          # position error as fraction of portfolio
        if abs(gap) > 1e-6:
            if ref is None: ref = pdec
            d = np.sign(gap); adv = d*(pdec-ref)/ref
            if mode == "adaptive":   u = (1+k*max(0, adv))*(0.5 if adv < -0.003 else 1)
            elif mode == "inverted": u = (1+k*max(0,-adv))*(0.5 if adv >  0.003 else 1)
            else: u = 1.0
            p = np.clip(base*np.clip(u,0.4,3.0),0.01,0.25) if mode != "fixed" else base
            Q = min(abs(gap), p*volf, 0.30*row["volume"])
            if Q > 0:
                imp = IMPACT_C*sigma*np.sqrt(Q/row["volume"])
                imp_paid += Q*close*imp; n_shares += Q
                fill = close*(1+d*imp); tv = Q*fill
                cash += (-tv-tv*COST) if d > 0 else (tv-tv*COST); shares += d*Q
        else: ref = None
    return np.mean(gaps), imp_paid, n_shares

N = 150
print(f"{'rho':>7}{'':>3}{'position tracking error':>36}{'total impact paid (Rs)':>40}")
print(f"{'':>10}{'fixed':>12}{'adaptive':>12}{'inverted':>12}{'fixed':>14}{'adaptive':>13}{'inverted':>13}")
print("-"*88)
for rho in [-0.20, 0.0, 0.20]:
    out = {m: {"g": [], "i": [], "n": []} for m in ["fixed","adaptive","inverted"]}
    for i in range(N):
        d = path_mom(9000+i, rho)
        for m in out:
            g, ip, ns = diag(d, m)
            out[m]["g"].append(g); out[m]["i"].append(ip); out[m]["n"].append(ns)
    r = f"{rho:>7.2f}{'':>3}"
    for m in ["fixed","adaptive","inverted"]: r += f"{np.mean(out[m]['g']):>12.4f}"
    for m in ["fixed","adaptive","inverted"]: r += f"{np.mean(out[m]['i']):>13,.0f}"
    print(r)
print("""
   'position tracking error' = average distance from the position the strategy
       asked for, as a fraction of the portfolio. LOWER = follows the plan better.
   'total impact paid' = rupees lost to our own order pushing the price.
""")
