import pandas as pd
df = pd.read_csv("ohlcv_with_ma.csv", index_col="date", parse_dates=True)

# VWAP over a rolling 20-day window: weight each day's close by that day's volume
df["price_x_volume"] = df["close"] * df["volume"]
df["vwap_20"] = df["price_x_volume"].rolling(20).sum() / df["volume"].rolling(20).sum()

print(df[["close", "volume", "long_ma", "vwap_20"]].iloc[19:29].to_string())
