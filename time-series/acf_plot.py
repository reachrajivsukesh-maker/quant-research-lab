import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
N = 2000

mean_level = 50.0
phi = 0.50
spring = np.zeros(N)
spring[0] = mean_level
for t in range(1, N):
    spring[t] = mean_level + phi * (spring[t-1] - mean_level) + rng.normal(0, 2)

walk = np.zeros(N)
walk[0] = 50.0
for t in range(1, N):
    walk[t] = walk[t-1] + rng.normal(0, 2)

spring_ret = np.diff(spring)
walk_ret = np.diff(walk)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[k:] * x[:n-k]) / denom for k in range(0, max_lag+1)])

MAX_LAG = 15
acf_spring = acf(spring_ret, MAX_LAG)
acf_walk = acf(walk_ret, MAX_LAG)
n = len(spring_ret)
band = 1.96 / np.sqrt(n)  # rough "not distinguishable from zero" threshold

print("lag  spring-ACF  walk-ACF   (band = +/-%.3f)" % band)
for k in range(1, 8):
    print(f"{k:>3}  {acf_spring[k]:>10.3f}  {acf_walk[k]:>8.3f}")

fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160)
fig.patch.set_facecolor("#fcfcfb")
plt.rcParams["font.family"] = "DejaVu Sans"

# --- Left column: lag-1 scatter (return_t vs return_t-1) ---
for ax, ret, color, title in [
    (axes[0,0], spring_ret, "#2a78d6", "Mean-reverting returns: today vs. yesterday"),
    (axes[1,0], walk_ret,   "#eb6834", "Random-walk returns: today vs. yesterday"),
]:
    ax.set_facecolor("#fcfcfb")
    ax.scatter(ret[:-1], ret[1:], s=6, alpha=0.35, color=color, edgecolors="none")
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.axvline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    ax.set_xlabel("return at t-1", color="#0b0b0b", fontsize=9)
    ax.set_ylabel("return at t", color="#0b0b0b", fontsize=9)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#898781", labelsize=8)

# --- Right column: ACF bar chart ---
for ax, acf_vals, color, title in [
    (axes[0,1], acf_spring, "#2a78d6", "ACF: mean-reverting returns"),
    (axes[1,1], acf_walk,   "#eb6834", "ACF: random-walk returns"),
]:
    ax.set_facecolor("#fcfcfb")
    lags = np.arange(1, MAX_LAG+1)
    ax.bar(lags, acf_vals[1:], color=color, width=0.6)
    ax.axhspan(-band, band, color="#898781", alpha=0.15)
    ax.axhline(0, color="#c3c2b7", linewidth=0.8)
    ax.set_title(title, color="#0b0b0b", fontsize=11, loc="left")
    ax.set_xlabel("lag (days)", color="#0b0b0b", fontsize=9)
    ax.set_ylabel("autocorrelation", color="#0b0b0b", fontsize=9)
    ax.set_ylim(-0.5, 0.3)
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#898781", labelsize=8)

plt.tight_layout()
plt.savefig("/home/claude/quant-roadmap/unit3/acf_demo.png", facecolor=fig.get_facecolor())
print("saved")
