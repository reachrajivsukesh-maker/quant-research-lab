"""Shared price-path generator with a tunable momentum knob.
rho = how much of yesterday's return carries into today.
   rho < 0 -> mean reversion,  rho = 0 -> random walk,  rho > 0 -> momentum

NOTE ON TERMINOLOGY (this caused real confusion during the study):
'realistic' in this repo means realistic VOLATILITY and VOLUME behaviour.
Real daily prices have return autocorrelation ~0.00 -- they ARE approximately
random walks in DIRECTION. So rho=0 is the realistic setting for direction;
rho != 0 is a deliberately planted effect used as an experimental control."""
import numpy as np, pandas as pd

def path_mom(seed, rho=0.0, n=300):
    rng = np.random.default_rng(seed)
    r, prev = [], 0.0
    for _ in range(n):
        e = rng.normal(0.0, 0.014)
        cur = rho*prev + e
        r.append(cur); prev = cur
    close = 100.0*np.exp(np.cumsum(r))
    volume = rng.integers(50_000, 500_000, n).astype(float)
    d = pd.DataFrame({"close": close, "volume": volume},
                     index=pd.bdate_range("2025-06-02", periods=n))
    d["signal"] = (d["close"].rolling(5).mean() > d["close"].rolling(20).mean()).astype(float)
    d.loc[d["close"].rolling(20).mean().isna(), "signal"] = 0.0
    d["prev_close"]   = d["close"].shift(1)
    d["vol_forecast"] = d["volume"].shift(1).rolling(20).mean()
    d["vol_20d"]      = d["close"].pct_change().rolling(20).std()
    return d
