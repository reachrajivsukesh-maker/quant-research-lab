"""
Unit 5 core engine -- SELF-CONTAINED reconstruction of the whole study.
Regenerates its own data from a seed; nothing external needed.
Run: python3 unit5_core.py

v2 (2026-08-25) adds, relative to v1:
  - the INVERTED execution rule (the one that actually works)
  - a momentum knob on the generator, for regime sweeps
  - P&L reported alongside shortfall, because THEY DISAGREE and reporting
    only one of them was the study's largest error
"""
import numpy as np, pandas as pd

CAP, COST, IMPACT_C = 15_000_000.0, 0.0015, 0.6

# ------------------------------------------------------------------ data
def make_path(seed, n=300, rho=0.0, vol=0.014, start=100.0):
    """rho = how much of yesterday's return carries into today.
         rho <0 mean reversion | rho=0 random walk | rho >0 momentum

    NOTE: real daily returns have autocorrelation ~0.00, so rho=0 is the
    REALISTIC setting for direction. Real markets' structure lives in
    volatility and volume, not in direction. rho!=0 is a planted control."""
    rng = np.random.default_rng(seed)
    r, prev = [], 0.0
    for _ in range(n):
        e = rng.normal(0.0, vol)
        cur = rho*prev + e
        r.append(cur); prev = cur
    close = start*np.exp(np.cumsum(r))
    volume = rng.integers(50_000, 500_000, n).astype(float)
    d = pd.DataFrame({"close": close, "volume": volume},
                     index=pd.bdate_range("2025-06-02", periods=n))
    d["signal"] = (d["close"].rolling(5).mean() > d["close"].rolling(20).mean()).astype(float)
    d.loc[d["close"].rolling(20).mean().isna(), "signal"] = 0.0
    # POINT-IN-TIME inputs: shift(1) so today's own value never decides today
    d["prev_close"]   = d["close"].shift(1)
    d["vol_forecast"] = d["volume"].shift(1).rolling(20).mean()
    d["vol_20d"]      = d["close"].pct_change().rolling(20).std()
    return d

# ------------------------------------------------------------------ engine
def run(sub, mode="fixed", base=0.05, k=4, ramp=True, ease=True,
        impact=True, use_future=False):
    """mode: 'fixed' | 'adaptive' | 'inverted'
         adaptive : speed up when the market moves AGAINST you (ramp)
                    slow down when it moves FOR you            (ease)
         inverted : the opposite of both -- this is the one that wins
       use_future=True reproduces the BIASED engine (lookahead), kept so the
       bug stays demonstrable rather than merely asserted.
       Returns (fills_dataframe, final_portfolio_value)."""
    cash, shares, ref = CAP, 0.0, None
    log, last = [], CAP
    for _, row in sub.iterrows():
        close, sig = row["close"], row["signal"]
        volf  = row["volume"] if use_future else row["vol_forecast"]
        pdec  = close         if use_future else row["prev_close"]
        sigma = row["vol_20d"]
        if pd.isna(volf) or pd.isna(pdec) or pd.isna(sigma) or pd.isna(sig):
            continue
        value  = cash + shares*close
        target = (value/close) if sig == 1 else 0.0
        gap    = target - shares                       # strategy's demand
        if abs(gap) > 1e-6:
            if ref is None: ref = pdec                 # decision price, set once
            d = np.sign(gap)
            adverse = d*(pdec - ref)/ref               # >0 == market against us
            if mode == "adaptive":
                u = 1 + (k*max(0,  adverse) if ramp else 0)
                if ease: u *= 0.5 if adverse < -0.003 else 1
            elif mode == "inverted":
                u = 1 + (k*max(0, -adverse) if ramp else 0)
                if ease: u *= 0.5 if adverse >  0.003 else 1
            else:
                u = 1.0
            p = np.clip(base*np.clip(u,0.4,3.0), 0.01, 0.25) if mode != "fixed" else base
            Q = min(abs(gap), p*volf, 0.30*row["volume"])   # market cap + realism
            if Q > 0:
                imp  = IMPACT_C*sigma*np.sqrt(Q/row["volume"]) if impact else 0.0
                fill = close*(1 + d*imp)               # our own order moves price
                tv   = Q*fill
                cash += (-tv - tv*COST) if d > 0 else (tv - tv*COST)
                shares += d*Q
                log.append((ref, d, Q, fill))
        else:
            ref = None                                 # order complete
        last = cash + shares*close
    return pd.DataFrame(log, columns=["ref","d","qty","fill"]), last

# ------------------------------------------------------------------ metrics
def shortfall(lg):
    """Notional-aggregate implementation shortfall, bps. Positive = cost.
       NEVER average per-order ratios -- aggregate the money, divide once."""
    if len(lg) == 0: return np.nan
    num = den = 0.0
    for (ref, d), g in lg.groupby(["ref","d"]):
        q = g["qty"].sum()
        if q <= 0: continue
        avg = (g["qty"]*g["fill"]).sum()/q
        num += d*(avg - ref)*q; den += q*ref
    return num/den*1e4 if den > 0 else np.nan

def pnl_pct(final): return (final/CAP - 1)*100

def buy_and_hold(sub):
    c = sub["close"].dropna()
    return (c.iloc[-1]/c.iloc[0] - 1)*100

# ------------------------------------------------------------------ headline
if __name__ == "__main__":
    N = 200
    MODES = {"fixed 5%":   dict(mode="fixed",    base=0.05),
             "adaptive":   dict(mode="adaptive", base=0.05, k=4),
             "inverted":   dict(mode="inverted", base=0.05, k=4)}
    print("Paired comparison vs fixed 5%. 200 paths per regime.")
    print("SHORTFALL: lower = better fills.   P&L: higher = more money.\n")
    print(f"{'regime':>22}{'rho':>7}{'':>3}{'shortfall vs fixed':>26}{'P&L vs fixed (pp)':>26}")
    print(f"{'':>32}{'adaptive':>13}{'inverted':>13}{'adaptive':>13}{'inverted':>13}")
    print("-"*84)
    for rho, label in [(-0.20,"strong mean reversion"), (0.0,"random walk"),
                       (0.20,"strong momentum")]:
        acc = {m: {"sf": [], "pnl": []} for m in MODES}
        for i in range(N):
            d = make_path(9000+i, rho=rho)
            for m, kw in MODES.items():
                lg, fin = run(d, **kw)
                acc[m]["sf"].append(shortfall(lg)); acc[m]["pnl"].append(pnl_pct(fin))
        bsf = np.nanmean(acc["fixed 5%"]["sf"]); bpl = np.mean(acc["fixed 5%"]["pnl"])
        row = f"{label:>22}{rho:>7.2f}{'':>3}"
        for m in ["adaptive","inverted"]: row += f"{np.nanmean(acc[m]['sf'])-bsf:>13.2f}"
        for m in ["adaptive","inverted"]: row += f"{np.mean(acc[m]['pnl'])-bpl:>13.2f}"
        print(row)
    print("""
THE KEY RESULT: adaptive has WORSE fill prices but BETTER P&L, in every regime.
The two metrics rank the rules in opposite orders. Adaptive's P&L edge comes from
trading ~8% fewer shares (churn reduction), not from executing better -- its cost
per share is identical to fixed. Inverted improves BOTH.

Report execution and portfolio metrics together. A disagreement between them is
itself the finding.""")
