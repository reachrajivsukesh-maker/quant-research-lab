import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N = 2000

mean_level = 50.0
phi = 0.90
spring = np.zeros(N)
spring[0] = mean_level
for t in range(1, N):
    spring[t] = mean_level + phi * (spring[t-1] - mean_level) + rng.normal(0, 2)

walk = np.zeros(N)
walk[0] = 50.0
for t in range(1, N):
    walk[t] = walk[t-1] + rng.normal(0, 2)

windows = [("first 200", 0, 200), ("steps 400-600", 400, 600),
           ("steps 900-1300", 900, 1300), ("last 200", 1800, 2000)]

plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=160, sharex=True)
fig.patch.set_facecolor("#fcfcfb")

for ax, series, color, title in [
    (axes[0], spring, "#2a78d6", "Mean-reverting (\"spring\") series — stationary"),
    (axes[1], walk,   "#eb6834", "Random walk — non-stationary"),
]:
    ax.set_facecolor("#fcfcfb")
    ax.plot(series, color=color, linewidth=0.9)
    for name, a, b in windows:
        ax.axvspan(a, b, color="#898781", alpha=0.12)
        ax.axvline(a, color="#c3c2b7", linewidth=0.8, linestyle="--")
    ax.axvline(2000, color="#c3c2b7", linewidth=0.8, linestyle="--")
    ax.set_title(title, color="#0b0b0b", fontsize=12, loc="left")
    ax.grid(True, color="#e1e0d9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#898781")

# window labels on top subplot
for name, a, b in windows:
    axes[0].text((a+b)/2, axes[0].get_ylim()[1]*0.97, name, ha="center", va="top",
                 fontsize=8, color="#52514e")

axes[1].set_xlabel("time step", color="#0b0b0b")
axes[0].set_ylabel("value", color="#0b0b0b")
axes[1].set_ylabel("value", color="#0b0b0b")
plt.tight_layout()
plt.savefig("/home/claude/quant-roadmap/unit3/stationarity_demo.png", facecolor=fig.get_facecolor())
print("saved")
