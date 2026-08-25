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

# "returns" = period-to-period change, same idea as (P_t - P_{t-1})
spring_ret = np.diff(spring)
walk_ret = np.diff(walk)

windows = [("first 200", 0, 200), ("steps 400-600", 400, 600),
           ("steps 900-1300", 900, 1300), ("last 200", 1800, 1999)]

print(f"{'window':<18}{'spring-ret mean':>17}{'spring-ret std':>16}   |{'walk-ret mean':>15}{'walk-ret std':>14}")
for name, a, b in windows:
    sm, ss = spring_ret[a:b].mean(), spring_ret[a:b].std()
    wm, ws = walk_ret[a:b].mean(), walk_ret[a:b].std()
    print(f"{name:<18}{sm:>17.3f}{ss:>16.3f}   |{wm:>15.3f}{ws:>14.3f}")

plt.rcParams["font.family"] = "DejaVu Sans"
fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), dpi=160, sharex=True, sharey=True)
fig.patch.set_facecolor("#fcfcfb")

for ax, series, color, title in [
    (axes[0], spring_ret, "#2a78d6", "Returns of the mean-reverting series"),
    (axes[1], walk_ret,   "#eb6834", "Returns of the random walk"),
]:
    ax.set_facecolor("#fcfcfb")
    ax.plot(series, color=color, linewidth=0.6)
    ax.axhline(0, color="#898781", linewidth=1)
    for name, a, b in windows:
        ax.axvspan(a, b, color="#898781", alpha=0.10)
    ax.set_title(title, color="#0b0b0b", fontsize=12, loc="left")
    ax.grid(True, color="#e1e0d9", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#898781")

axes[1].set_xlabel("time step", color="#0b0b0b")
axes[0].set_ylabel("return (Δvalue)", color="#0b0b0b")
axes[1].set_ylabel("return (Δvalue)", color="#0b0b0b")
plt.tight_layout()
plt.savefig("/home/claude/quant-roadmap/unit3/returns_demo.png", facecolor=fig.get_facecolor())
print("saved")
