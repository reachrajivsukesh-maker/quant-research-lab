import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n_days = 300

# Build a DatetimeIndex -- real calendar dates, not just row numbers 0,1,2,...
dates = pd.bdate_range(start="2025-06-01", periods=n_days)  # 'b' = business days only, skips weekends

# Simulate a daily close price (same random-walk-with-drift logic as before)
drift, vol = 0.03, 1.2
daily_moves = rng.normal(drift, vol, n_days)
close = 100 + np.cumsum(daily_moves)

# Build plausible open/high/low around each day's close, plus volume
open_ = close - daily_moves + rng.normal(0, 0.3, n_days)
high = np.maximum(open_, close) + np.abs(rng.normal(0.4, 0.3, n_days))
low  = np.minimum(open_, close) - np.abs(rng.normal(0.4, 0.3, n_days))
volume = rng.integers(50_000, 500_000, n_days)

df = pd.DataFrame({
    "open": open_.round(2), "high": high.round(2), "low": low.round(2),
    "close": close.round(2), "volume": volume,
}, index=dates)
df.index.name = "date"

df.to_csv("sample_ohlcv.csv")
print(df.head(8))
print("...")
print(df.tail(3))
print(f"\nshape: {df.shape}")
print(f"index type: {type(df.index)}")
