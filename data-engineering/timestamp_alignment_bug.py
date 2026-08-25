import pandas as pd

df = pd.read_csv("ohlcv_with_ma.csv", index_col="date", parse_dates=True).reset_index()

earnings = pd.DataFrame({
    "period_end":   pd.to_datetime(["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]),
    "release_date": pd.to_datetime(["2025-07-25", "2025-10-24", "2026-01-23", "2026-04-24"]),
    "eps_surprise_pct": [2.1, -1.5, 0.8, 3.2],
})
print("Earnings events:")
print(earnings.to_string(index=False))

wrong = pd.merge_asof(df, earnings[["period_end", "eps_surprise_pct"]].rename(columns={"period_end": "date"}),
                       on="date", direction="backward").rename(columns={"eps_surprise_pct": "WRONG_eps_visible"})
right = pd.merge_asof(df, earnings[["release_date", "eps_surprise_pct"]].rename(columns={"release_date": "date"}),
                       on="date", direction="backward").rename(columns={"eps_surprise_pct": "RIGHT_eps_visible"})

merged = wrong[["date", "WRONG_eps_visible"]].merge(right[["date", "RIGHT_eps_visible"]], on="date")

window = merged[(merged["date"] >= "2025-06-20") & (merged["date"] <= "2025-07-30")]
print("\nWhat each version 'knows' as it walks through late June -> late July 2025:")
print(window.to_string(index=False))
