"""Assemble the cost-sensitivity summary + chart from the three saved runs."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import unit7_real_btc as u

df = u.load()
split = int(len(df) * 0.7)
train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
test_arr = u.to_arrays(test)

COST_LEVELS = [0, 4, 15]
summary_rows = []

for cbps in COST_LEVELS:
    u.COST = cbps / 1e4
    bh_rets, bh_pos = u.run_buyhold(test_arr[2])
    of_rets, of_pos = u.run_ofi_sign(test_arr[0], test_arr[2])
    bh_c, bh_s, _ = u.metrics(bh_rets, bh_pos)
    of_c, of_s, _ = u.metrics(of_rets, of_pos)
    summary_rows.append({"cost_bps": cbps, "strategy": "buy&hold", "cum_pct": bh_c, "sharpe": bh_s})
    summary_rows.append({"cost_bps": cbps, "strategy": "OFI-sign (naive)", "cum_pct": of_c, "sharpe": of_s})

    qdf = pd.read_csv(f"unit7_real_btc_results_{cbps}bps.csv")
    for k, label in [("qlearn_fixed", "Qlearn fixed-α"), ("qlearn_kalman", "Qlearn kalman-α")]:
        vals = qdf[k].to_numpy()
        m = vals.mean(); se = vals.std(ddof=1) / np.sqrt(len(vals))
        sh = qdf[f"{k}_sharpe"].dropna().to_numpy()
        sh_m = sh.mean() if len(sh) else np.nan
        summary_rows.append({"cost_bps": cbps, "strategy": label, "cum_pct": m, "cum_lo": m - 1.96*se,
                              "cum_hi": m + 1.96*se, "sharpe": sh_m})

summary = pd.DataFrame(summary_rows)
summary.to_csv("unit7_real_btc_cost_sweep.csv", index=False)
print(summary.to_string(index=False))

# ---- chart: cum% vs cost, one line per strategy ----
COLORS = {"buy&hold": "#898781", "OFI-sign (naive)": "#2a78d6",
          "Qlearn fixed-α": "#eb6834", "Qlearn kalman-α": "#1baf7a"}
fig, ax = plt.subplots(figsize=(8, 5))
for strat, color in COLORS.items():
    sub = summary[summary["strategy"] == strat].sort_values("cost_bps")
    ax.plot(sub["cost_bps"], sub["cum_pct"], marker="o", color=color, label=strat, linewidth=2)
    if "cum_lo" in sub.columns and sub["cum_lo"].notna().any():
        ax.fill_between(sub["cost_bps"], sub["cum_lo"], sub["cum_hi"], color=color, alpha=0.15)
ax.axhline(0, color="#0b0b0b", linewidth=0.8)
ax.set_xlabel("assumed turnover cost, bps per unit of |Δposition|")
ax.set_ylabel("test-segment cumulative return, %")
ax.set_title("Real BTCUSDT, out-of-sample return vs. assumed trading cost\n"
             "(26-day data, same walk-forward test window as the OFI study; real 1-min std ≈ 5bps)")
ax.axvline(5, color="#c3c2b7", linestyle=":", linewidth=1)
ax.annotate("~1-min return std (5bps)", xy=(5, ax.get_ylim()[0]), xytext=(5.5, ax.get_ylim()[0]*0.9),
            fontsize=8, color="#52514e")
ax.set_ylim(-20, 15)
ax.annotate("OFI-sign wiped out\n(actual: -98% to -100%,\noff-chart)", xy=(15, -20), xytext=(9.5, -17),
            fontsize=8, color="#2a78d6", arrowprops=dict(arrowstyle="->", color="#2a78d6", lw=1))
ax.legend(fontsize=9, loc="upper right")
ax.grid(color="#e1e0d9", linewidth=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("unit7_real_btc_cost_sweep.png", dpi=150)
print("\nwrote unit7_real_btc_cost_sweep.png, unit7_real_btc_cost_sweep.csv")
