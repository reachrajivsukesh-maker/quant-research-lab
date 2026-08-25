"""Trading signals. Deliberately simple -- the research here is about EXECUTION,
not about finding alpha. A signal with no edge is in fact the right test bed:
it isolates execution cost from strategy P&L."""
import numpy as np
import pandas as pd


def ma_crossover(d, short=5, long=20):
    """Long when the fast MA is above the slow MA, flat otherwise."""
    s = (d["close"].rolling(short).mean() > d["close"].rolling(long).mean()).astype(float)
    s[d["close"].rolling(long).mean().isna()] = 0.0
    return s


def zscore_reversion(d, window=20, entry=-1.0):
    """Long when price is more than `entry` std devs below its rolling mean."""
    m = d["close"].rolling(window).mean()
    sd = d["close"].rolling(window).std()
    z = (d["close"] - m) / sd
    s = (z < entry).astype(float)
    s[z.isna()] = 0.0
    return s


def autocorrelation(x, lags=5):
    """Sample ACF, computed from the definition rather than a library call.
    Used to check whether a momentum-based execution rule has any basis."""
    x = np.asarray(pd.Series(x).dropna())
    x = x - x.mean()
    denom = (x ** 2).sum()
    return {k: float((x[k:] * x[:-k]).sum() / denom) for k in range(1, lags + 1)}
