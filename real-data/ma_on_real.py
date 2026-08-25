"""The decisive check: does the MA strategy beat buy-and-hold on REAL prices?
If real markets were a random walk in direction, it should not."""
import numpy as np, pandas as pd
from liquidity_test import AAPL, LAKE

COST = 0.0015
def strat(close, short, long):
    s = pd.Series(close)
    sig = (s.rolling(short).mean() > s.rolling(long).mean()).astype(float)
    sig[s.rolling(long).mean().isna()] = 0.0
    pos = sig.shift(1).fillna(0.0)
    r = s.pct_change().fillna(0.0)
    net = pos*r - pos.diff().abs().fillna(0.0)*COST
    return float((1+net).prod()-1)*100, float(pos.mean())

for name, rows in [("AAPL", AAPL), ("LAKE", LAKE)]:
    c = [r[0] for r in rows]
    bh = (c[-1]/c[0]-1)*100
    print(f"\n{'='*66}\n{name}: {len(c)} real trading days\n{'='*66}")
    print(f"   buy and hold: {bh:+7.2f}%\n")
    print(f"   {'MA windows':>14}{'strategy return':>18}{'vs buy&hold':>15}{'% time invested':>18}")
    beat = 0; tried = 0
    for sh, lg in [(5,20),(3,10),(10,30),(2,20),(8,50),(15,100)]:
        v, expo = strat(c, sh, lg); tried += 1
        if v > bh: beat += 1
        print(f"   {f'{sh}/{lg}':>14}{v:>17.2f}%{v-bh:>+14.2f}%{expo*100:>17.1f}%")
    print(f"\n   beat buy-and-hold: {beat} of {tried} window choices")
