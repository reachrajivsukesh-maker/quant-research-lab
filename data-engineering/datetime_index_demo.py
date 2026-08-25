import pandas as pd
df = pd.read_csv("sample_ohlcv.csv", index_col="date", parse_dates=True)

print("1) Slicing by date directly -- works because the index is dates, not row numbers:")
print(df.loc["2025-07"].head(3))   # every row in July 2025, just by asking for "2025-07"

print("\n2) Resampling daily -> weekly (using Friday's close as each week's summary):")
weekly = df["close"].resample("W-FRI").last()
print(weekly.head(6))
