import pandas as pd

df = pd.read_csv("sample_ohlcv.csv", index_col="date", parse_dates=True)

# Short and long moving averages -- same idea as movingAvg() in the JS trading simulator,
# but pandas does the sliding-window arithmetic for you.
df["short_ma"] = df["close"].rolling(window=5).mean()
df["long_ma"]  = df["close"].rolling(window=20).mean()

# Rolling volatility -- same sliding-window idea, different aggregation (std instead of mean)
df["rolling_vol_20"] = df["close"].rolling(window=20).std()

print("First 25 rows (to show the NaN warm-up period clearly):")
print(df[["close", "short_ma", "long_ma", "rolling_vol_20"]].head(25).to_string())

print("\nRows 20-26 (right where the 20-day long_ma first gets a real value):")
print(df[["close", "short_ma", "long_ma", "rolling_vol_20"]].iloc[17:24].to_string())

df.to_csv("ohlcv_with_ma.csv")
