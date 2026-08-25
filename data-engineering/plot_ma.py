import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("ohlcv_with_ma.csv", index_col="date", parse_dates=True)

plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

ax.plot(df.index, df["close"], color="#2a78d6", linewidth=1.1, label="close price")
ax.plot(df.index, df["short_ma"], color="#eb6834", linewidth=1.4, label="5-day MA")
ax.plot(df.index, df["long_ma"], color="#1baf7a", linewidth=1.4, label="20-day MA")

ax.set_title("Close price with 5-day and 20-day rolling moving averages", color="#0b0b0b", fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, color="#e1e0d9", linewidth=0.7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(colors="#898781", labelsize=8)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("ma_chart.png", facecolor=fig.get_facecolor())
print("saved")
