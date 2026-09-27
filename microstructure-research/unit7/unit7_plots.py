"""Unit 7 Pass 1 -- two figures from unit7_results.csv plus a fresh alpha trace."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import unit7_core as u

df = pd.read_csv("unit7_results.csv")
STRATS = ["buyhold", "ma_crossover", "qlearn_fixed", "qlearn_kalman"]
LABELS = {"buyhold": "buy&hold", "ma_crossover": "MA(5,20)",
          "qlearn_fixed": "Qlearn\nfixed-α", "qlearn_kalman": "Qlearn\nkalman-α"}
REGIMES = ["mean reversion", "random walk", "momentum"]
COLORS = {"buyhold": "#898781", "ma_crossover": "#2a78d6",
          "qlearn_fixed": "#eb6834", "qlearn_kalman": "#1baf7a"}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
for ax, regime in zip(axes, REGIMES):
    sub = df[df["regime"] == regime]
    means, los, his = [], [], []
    for s in STRATS:
        vals = sub[sub["strategy"] == s]["cum_pct"].dropna().to_numpy()
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(len(vals))
        means.append(m); los.append(1.96 * se); his.append(1.96 * se)
    x = np.arange(len(STRATS))
    ax.bar(x, means, yerr=[los, his], capsize=4,
           color=[COLORS[s] for s in STRATS], edgecolor="#0b0b0b", linewidth=0.6)
    ax.axhline(0, color="#0b0b0b", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[s] for s in STRATS], fontsize=9)
    ax.set_title(f"{regime}\n(n=150 seeds)", fontsize=10)
    ax.set_ylabel("test-segment cum. return, %  (95% CI)" if regime == REGIMES[0] else "")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
fig.suptitle("Unit 7 Pass 1 -- out-of-sample return by strategy, three synthetic regimes", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("unit7_regime_comparison.png", dpi=150)
print("wrote unit7_regime_comparison.png")

# ---- alpha trajectory: fresh run, log every step (not just last 200) ----
df0 = u.make_path(0, rho=0.0)
split = int(len(df0) * 0.7)
train = df0.iloc[:split].reset_index(drop=True)
mom_edges, vol_edges = u.fit_bin_edges(train)
tr_mom, tr_vol, tr_ret, _ = u.to_arrays(train)
rng = np.random.default_rng(5000)
agent = u.QAgent(alpha_mode="kalman")
n_epochs = 25
for ep in range(n_epochs):
    eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / (n_epochs - 1)
    u.run_episode(tr_mom, tr_vol, tr_ret, agent, mom_edges, vol_edges, eps=eps, rng=rng, learn=True)

alphas = np.array(agent.alpha_trace)
fig2, ax2 = plt.subplots(figsize=(9, 4))
ax2.plot(alphas, color="#1baf7a", linewidth=0.6, alpha=0.7, label="kalman-gain α per update")
window = 100
roll = pd.Series(alphas).rolling(window).mean()
ax2.plot(roll, color="#0b0b0b", linewidth=1.6, label=f"{window}-update rolling mean")
ax2.axhline(0.10, color="#eb6834", linewidth=1.2, linestyle="--", label="fixed α = 0.10 (comparison)")
ax2.set_xlabel("training update # (25 epochs over the 560-day train segment, seed 0, random-walk regime)")
ax2.set_ylabel("effective learning rate α")
ax2.set_title("Unit 7 Pass 1 -- Kalman-gain α over training (vs. the fixed baseline)")
ax2.legend(fontsize=9)
ax2.grid(color="#e1e0d9", linewidth=0.7)
ax2.set_axisbelow(True)
fig2.tight_layout()
fig2.savefig("unit7_alpha_trajectory.png", dpi=150)
print("wrote unit7_alpha_trajectory.png")
