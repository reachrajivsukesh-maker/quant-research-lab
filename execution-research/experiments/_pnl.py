"""Rajiv's challenge: isn't P&L vs buy-and-hold the metric that matters?
Test whether using it changes any of our conclusions."""
import numpy as np, pandas as pd
from _inverted import sf, CAP, COST, IMPACT_C
from regime_paths import path_mom

def full_run(sub, mode, base=0.05, k=4):
    """returns (shortfall_bps, final_pnl_pct)"""
    cash, shares, ref = CAP, 0.0, None
    log = []
    last = None
    for _, row in sub.iterrows():
        close, sig = row["close"], row["signal"]
        volf, pdec, sigma = row["vol_forecast"], row["prev_close"], row["vol_20d"]
        if pd.isna(volf) or pd.isna(pdec) or pd.isna(sigma) or pd.isna(sig): continue
        value = cash + shares*close
        target = (value/close) if sig == 1 else 0.0
        gap = target - shares
        if abs(gap) > 1e-6:
            if ref is None: ref = pdec
            d = np.sign(gap); adv = d*(pdec-ref)/ref
            if mode == "adaptive":  u = (1+k*max(0, adv))*(0.5 if adv < -0.003 else 1)
            elif mode == "inverted":u = (1+k*max(0,-adv))*(0.5 if adv >  0.003 else 1)
            else: u = 1.0
            p = np.clip(base*np.clip(u,0.4,3.0),0.01,0.25) if mode != "fixed" else base
            Q = min(abs(gap), p*volf, 0.30*row["volume"])
            if Q > 0:
                imp = IMPACT_C*sigma*np.sqrt(Q/row["volume"]); fill = close*(1+d*imp)
                tv = Q*fill
                cash += (-tv-tv*COST) if d > 0 else (tv-tv*COST); shares += d*Q
                log.append((ref, d, Q, fill))
        else: ref = None
        last = cash + shares*close
    lg = pd.DataFrame(log, columns=["ref","d","qty","fill"])
    return sf(lg), (last/CAP-1)*100

N = 150
print("="*96)
print("SAME EXPERIMENT, TWO METRICS. Does the answer change?")
print("="*96)
print(f"{'rho':>7}{'':>4}{'SHORTFALL (bps, lower better)':>34}{'P&L vs BUY-AND-HOLD (pp, higher better)':>44}")
print(f"{'':>11}{'fixed':>10}{'adaptive':>11}{'inverted':>11}{'fixed':>13}{'adaptive':>13}{'inverted':>13}{'buy&hold':>13}")
print("-"*96)
for rho in [-0.20, 0.0, 0.20, 0.35]:
    acc = {m: {"sf": [], "pnl": []} for m in ["fixed","adaptive","inverted"]}
    bh = []
    for i in range(N):
        d = path_mom(9000+i, rho)
        c = d["close"].dropna()
        bh.append((c.iloc[-1]/c.iloc[0]-1)*100)
        for m in acc:
            s, p = full_run(d, m)
            if not np.isnan(s): acc[m]["sf"].append(s)
            acc[m]["pnl"].append(p)
    bhm = np.mean(bh)
    row = f"{rho:>7.2f}{'':>4}"
    for m in ["fixed","adaptive","inverted"]:
        row += f"{np.mean(acc[m]['sf']):>11.2f}"
    for m in ["fixed","adaptive","inverted"]:
        row += f"{np.mean(acc[m]['pnl'])-bhm:>13.2f}"
    row += f"{bhm:>13.2f}"
    print(row)
print("""
   Left block: which rule got better PRICES.
   Right block: which rule made more MONEY than simply buying and holding.
""")
