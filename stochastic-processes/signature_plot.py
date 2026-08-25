import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

rng = np.random.default_rng(7)

# --- Simulate one trading day at 1-second resolution ---
N_FINE = 23400          # ~6.5 hour trading day, 1-second bars
DAILY_VOL = 0.015       # true daily return volatility ~1.5%
sigma_fine = DAILY_VOL / np.sqrt(N_FINE)

true_log_returns = rng.normal(0, sigma_fine, N_FINE)
true_log_price = np.cumsum(true_log_returns)   # pure random walk, no noise

# --- Microstructure noise: bid-ask bounce-style iid noise on TOP of the true price ---
NOISE_STD = 0.00035     # ~3.5 bps per observation -- small, but constant regardless of Δt
noise = rng.normal(0, NOISE_STD, N_FINE)
observed_log_price = true_log_price + noise

# --- Sampling intervals to test (in seconds) ---
intervals_sec = [1, 5, 15, 30, 60, 300, 900, 1800, 3600, N_FINE]
labels = ["1s", "5s", "15s", "30s", "1min", "5min", "15min", "30min", "1hr", "full day"]

def realized_vol(path, step):
    sampled = path[::step]
    if sampled[-1] != path[-1]:
        sampled = np.append(sampled, path[-1])
    rets = np.diff(sampled)
    rv = np.sum(rets**2)          # realized variance for the day
    return np.sqrt(rv) * 100      # as a % daily vol estimate

vol_true = [realized_vol(true_log_price, s) for s in intervals_sec]
vol_noisy = [realized_vol(observed_log_price, s) for s in intervals_sec]

# --- Plot, styled per project palette ---
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(figsize=(8, 5.2), dpi=160)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")

x = np.arange(len(intervals_sec))
ax.plot(x, vol_true, marker="o", color="#2a78d6", linewidth=2.2, label="Idealized random walk (no noise)")
ax.plot(x, vol_noisy, marker="o", color="#eb6834", linewidth=2.2, label="Real market (with microstructure noise)")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, color="#52514e")
ax.invert_xaxis()  # finest sampling on the right, coarsest on the left -> zoom-in reads left-to-right... actually keep natural
ax.invert_xaxis()  # undo: keep left=finest (1s) to right=full day, natural reading order

ax.set_ylabel("Estimated daily volatility (%)", color="#0b0b0b", fontsize=11)
ax.set_xlabel("Sampling interval used to measure volatility", color="#0b0b0b", fontsize=11)
ax.set_title("Volatility Signature Plot: same day, measured at different zoom levels", color="#0b0b0b", fontsize=13, pad=14)

ax.grid(True, color="#e1e0d9", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#c3c2b7")
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(colors="#898781")

ax.legend(frameon=False, loc="upper right", fontsize=10, labelcolor="#0b0b0b")

ax.annotate("Flat: zooming in doesn't change\nthe true volatility estimate\n(self-similarity)",
            xy=(0.5, vol_true[0]), xytext=(2.3, vol_true[0]*1.55),
            fontsize=8.5, color="#2a78d6",
            arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1.2))

ax.annotate("Inflates sharply at very short\nintervals: bid-ask bounce & noise\ndominate at that zoom level",
            xy=(0.3, vol_noisy[0]), xytext=(1.6, vol_noisy[0]*0.55),
            fontsize=8.5, color="#eb6834",
            arrowprops=dict(arrowstyle="->", color="#eb6834", lw=1.2))

plt.tight_layout()
plt.savefig("/home/claude/quant-roadmap/unit2/signature_plot.png", facecolor=fig.get_facecolor())
print("true (no noise):", [round(v,3) for v in vol_true])
print("noisy (real-market-like):", [round(v,3) for v in vol_noisy])
