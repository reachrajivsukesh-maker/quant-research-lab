"""Synthetic OHLCV generation.

Real market data is not available in this environment, so every experiment runs
on synthetic paths from a known generating process. That is a limitation, but it
also buys something real: because the process is known, we can generate hundreds
of independent realisations and make statistically powered statements -- which a
single historical price series can never support.
"""
import numpy as np
import pandas as pd


def make_path(seed, n=300, drift=0.03, vol=1.2, start=100.0,
              vol_lo=50_000, vol_hi=500_000, short=5, long=20):
    """One synthetic daily OHLCV path with a moving-average crossover signal.

    Returns a DataFrame indexed by business day with point-in-time-safe columns:
      close, volume, short_ma, long_ma, signal   -- observable at end of day t
      prev_close, vol_forecast, vol_20d          -- observable BEFORE day t trades

    The .shift(1) on the last three is the whole point. Without it, a backtest
    sizes today's order using today's realised volume -- a number that does not
    exist at decision time. See docs/FINDINGS.md, Bug #1.
    """
    rng = np.random.default_rng(seed)
    close = np.maximum(np.cumsum(rng.normal(drift, vol, n)) + start, 5.0)
    volume = rng.integers(vol_lo, vol_hi, n).astype(float)

    d = pd.DataFrame({"close": close, "volume": volume},
                     index=pd.bdate_range("2025-06-02", periods=n))
    d["short_ma"] = d["close"].rolling(short).mean()
    d["long_ma"] = d["close"].rolling(long).mean()
    d["signal"] = (d["short_ma"] > d["long_ma"]).astype(float)
    d.loc[d["long_ma"].isna(), "signal"] = 0.0

    d["prev_close"] = d["close"].shift(1)
    d["vol_forecast"] = d["volume"].shift(1).rolling(20).mean()
    d["vol_20d"] = d["close"].pct_change().rolling(20).std()
    return d


def volume_forecast_quality(d):
    """How informative is the trailing-volume forecast? On this synthetic data,
    barely at all -- volume is drawn iid, so a trailing mean predicts nothing
    (corr ~= 0.02). Real market volume autocorrelates ~0.6-0.8. Reported openly
    because it means the point-in-time engine here is handicapped unrealistically.
    """
    f = d["vol_forecast"].dropna()
    a = d["volume"].loc[f.index]
    return {"correlation": float(np.corrcoef(f, a)[0, 1]),
            "mean_abs_pct_error": float((abs(a - f) / f).mean())}
