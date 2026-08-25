"""Rajiv's proposal: INVERT the adaptive rule.

   ORIGINAL : market moves AGAINST me -> speed up   (ramp)
              market moves FOR me     -> slow down  (ease)

   INVERTED : market moves FOR me     -> speed up   (grab the discount)
              market moves AGAINST me -> slow down  (stop chasing)

`adverse` is positive when the market is punishing us, for both buys and sells.
So the inverted rule just swaps which sign triggers which response.
"""
import numpy as np, pandas as pd
CAP, COST, IMPACT_C = 15_000_000.0, 0.0015, 0.6

def run2(sub, mode="fixed", base=0.05, k=4, ramp=True, ease=True):
    """mode: fixed | adaptive | inverted"""
    cash, shares, ref = CAP, 0.0, None
    log=[]
    for _, row in sub.iterrows():
        close, sig = row["close"], row["signal"]
        volf, pdec, sigma = row["vol_forecast"], row["prev_close"], row["vol_20d"]
        if pd.isna(volf) or pd.isna(pdec) or pd.isna(sigma) or pd.isna(sig): continue
        value = cash + shares*close
        target = (value/close) if sig == 1 else 0.0
        gap = target - shares
        if abs(gap) > 1e-6:
            if ref is None: ref = pdec
            d = np.sign(gap)
            adverse = d*(pdec-ref)/ref          # >0 == market against us
            if mode == "adaptive":
                u = 1 + (k*max(0, adverse) if ramp else 0)      # speed up when AGAINST
                if ease: u *= 0.5 if adverse < -0.003 else 1    # slow down when FOR
            elif mode == "inverted":
                u = 1 + (k*max(0, -adverse) if ramp else 0)     # speed up when FOR
                if ease: u *= 0.5 if adverse > 0.003 else 1     # slow down when AGAINST
            else:
                u = 1.0
            p = np.clip(base*np.clip(u, 0.4, 3.0), 0.01, 0.25) if mode != "fixed" else base
            Q = min(abs(gap), p*volf, 0.30*row["volume"])
            if Q <= 0: continue
            imp = IMPACT_C*sigma*np.sqrt(Q/row["volume"])
            fill = close*(1 + d*imp)
            tv = Q*fill
            cash += (-tv - tv*COST) if d > 0 else (tv - tv*COST)
            shares += d*Q
            log.append((ref, d, Q, fill))
        else:
            ref = None
    return pd.DataFrame(log, columns=["ref","d","qty","fill"])

def sf(lg):
    if len(lg) == 0: return np.nan
    num = den = 0.0
    for (ref,d),g in lg.groupby(["ref","d"]):
        q = g["qty"].sum()
        if q <= 0: continue
        af = (g["qty"]*g["fill"]).sum()/q
        num += d*(af-ref)*q; den += q*ref
    return num/den*1e4 if den > 0 else np.nan

if __name__ == "__main__":
    from regime_paths import path_mom
    N = 200
    print("Execution cost vs FIXED 5%.  Negative = BETTER than fixed.  (* = CI excludes 0)\n")
    print(f"{'market':>24}{'rho':>7}{'ADAPTIVE (old)':>18}{'INVERTED (new)':>18}")
    print("-"*67)
    res = {}
    for rho, label in [(-0.20,"strong mean reversion"), (-0.10,"mild mean reversion"),
                       (0.0,"random walk"), (0.10,"mild momentum"),
                       (0.20,"strong momentum"), (0.35,"very strong momentum")]:
        A, I = [], []
        for i in range(N):
            d = path_mom(9000+i, rho)
            f = sf(run2(d, "fixed"))
            if np.isnan(f): continue
            a = sf(run2(d, "adaptive", k=4)); v = sf(run2(d, "inverted", k=4))
            if not np.isnan(a): A.append(a-f)
            if not np.isnan(v): I.append(v-f)
        def fmt(x):
            x = np.array(x); se = x.std(ddof=1)/np.sqrt(len(x))
            return f"{x.mean():+8.2f}{'*' if abs(x.mean())-1.96*se > 0 else ' '}"
        res[rho] = (np.mean(A), np.mean(I))
        print(f"{label:>24}{rho:>7.2f}{fmt(A):>18}{fmt(I):>18}")
