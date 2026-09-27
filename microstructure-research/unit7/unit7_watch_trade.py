"""
Unit 7 -- "watch them actually trade": full instrumented test-set rollout
(per-minute position + equity), at the FRICTIONLESS (cost=0) regime -- the
only regime in this dataset where either agent trades at all (see
unit7_diagnose.py and unit7_multiscale.py for why realistic cost forces
'never trade', confirmed now across five bar sizes too).

Framing (explicit, not hidden): this is a stress-test / demonstration of
*behavior*, not a claim of tradeable profit -- the moment any realistic
Binance fee (7.5-10bps) is included, both agents correctly go back to zero
activity, as shown separately. What this DOES show honestly: how a
fixed-alpha ("swordfish" -- fast, decisive, table-lookup speed, stable
Sharpe CI in Pass 2) agent behaves differently from a kalman-alpha
("octopus" -- adapts to instantaneous surprise, noisier outcome in Pass 2)
agent, trade for trade, on the same real market history.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unit7_core import QAgent, ACTIONS
from unit7_real_btc import load, fit_edges, to_arrays, discretize, run_ofi_sign, run_buyhold
import unit7_real_btc as u

u.COST = 0.0  # frictionless demonstration regime -- see module docstring

def run_instrumented(ofi, vol, ret, agent, ofi_edges, vol_edges):
    pos = 0
    n = len(ofi) - 1
    pos_trace = np.empty(n)
    rets = np.empty(n)
    for i in range(n):
        s = discretize(ofi[i], vol[i], pos, ofi_edges, vol_edges)
        a_idx = int(np.argmax(agent.Q[s]))  # frozen, greedy -- no exploration
        new_pos = int(ACTIONS[a_idx])
        turnover_cost = u.COST * abs(new_pos - pos)
        reward = new_pos * ret[i + 1] - turnover_cost
        rets[i] = reward; pos_trace[i] = new_pos
        pos = new_pos
    return rets, pos_trace

if __name__ == "__main__":
    df = load()
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    ofi_edges, vol_edges = fit_edges(train)
    tr_ofi, tr_vol, tr_ret = to_arrays(train)
    te_ofi, te_vol, te_ret = to_arrays(test)

    from unit7_real_btc import run_episode as train_episode
    SEED = 0
    agents = {}
    for mode, label in [("fixed", "swordfish (fixed-α)"), ("kalman", "octopus (kalman-α)")]:
        rng = np.random.default_rng(SEED + 9000)
        agent = QAgent(alpha_mode=mode)
        for ep in range(8):
            eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / 7
            train_episode(tr_ofi, tr_vol, tr_ret, agent, ofi_edges, vol_edges, eps=eps, rng=rng, learn=True)
        agents[label] = agent

    results = {}
    for label, agent in agents.items():
        rets, pos = run_instrumented(te_ofi, te_vol, te_ret, agent, ofi_edges, vol_edges)
        results[label] = (rets, pos)
    bh_rets, bh_pos = run_buyhold(te_ret)
    of_rets, of_pos = run_ofi_sign(te_ofi, te_ret)
    results["buy&hold"] = (bh_rets, bh_pos)
    results["OFI-sign (naive)"] = (of_rets, of_pos)

    minutes = test["minute"].to_numpy()[1:]  # aligned with rets/pos (next-bar indexing)
    price = test["close"].to_numpy()[1:]

    # ---- summary printed to console ----
    print(f"{'strategy':>22}{'cum% (test)':>14}{'n_trades':>12}{'activity':>11}")
    for label, (rets, pos) in results.items():
        cum = (np.prod(1 + rets) - 1) * 100
        n_trades = int(np.sum(np.abs(np.diff(np.concatenate([[0], pos]))) > 0))
        act = np.mean(np.abs(pos))
        print(f"{label:>22}{cum:>14.2f}{n_trades:>12}{act:>11.3f}")

    # ---- Chart 1: full test-period equity curves ----
    COLORS = {"buy&hold": "#898781", "OFI-sign (naive)": "#2a78d6",
              "swordfish (fixed-α)": "#eb6834", "octopus (kalman-α)": "#1baf7a"}
    fig, ax = plt.subplots(figsize=(11, 5))
    for label, (rets, pos) in results.items():
        equity = 100 * (np.cumprod(1 + rets) - 1)
        ax.plot(range(len(equity)), equity, label=label, color=COLORS[label],
                linewidth=1.8 if "α" in label else 1.2)
    ax.axhline(0, color="#0b0b0b", linewidth=0.8)
    ax.set_xlabel("test-segment minute index (≈ 11,225 minutes ≈ 7.8 days)")
    ax.set_ylabel("cumulative return, %")
    ax.set_title("Unit 7 -- real BTCUSDT test period, frictionless (cost=0) demonstration regime\n"
                 "NOT tradeable at realistic Binance fees (7.5-10bps) -- see unit7_diagnose.py for why")
    ax.legend(fontsize=9)
    ax.grid(color="#e1e0d9", linewidth=0.7); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("unit7_watch_equity_curve.png", dpi=150)
    print("\nwrote unit7_watch_equity_curve.png")

    # ---- Chart 2: zoomed position trace -- find a window with visible activity ----
    fixed_pos = results["swordfish (fixed-α)"][1]
    kalman_pos = results["octopus (kalman-α)"][1]
    fixed_active = np.abs(fixed_pos) > 0
    kalman_active = np.abs(kalman_pos) > 0
    window = 1500
    # maximize the MINIMUM of the two counts, so both agents visibly trade in the same window
    best_start, best_score = 0, -1
    for start in range(0, len(fixed_active) - window, 100):
        cf = fixed_active[start:start + window].sum()
        ck = kalman_active[start:start + window].sum()
        score = min(cf, ck)
        if score > best_score:
            best_score = score; best_start = start
    sl = slice(best_start, best_start + window)
    print(f"chosen window start={best_start}: fixed trades={fixed_active[sl].sum()}, "
          f"kalman trades={kalman_active[sl].sum()}")

    fig2, (axp, axpos) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                                       gridspec_kw={"height_ratios": [1.3, 1]})
    axp.plot(range(window), price[sl], color="#0b0b0b", linewidth=1.1)
    axp.set_ylabel("BTCUSDT close, $")
    axp.set_title(f"Unit 7 -- zoomed window (test minutes {best_start}-{best_start+window}), "
                 f"frictionless regime: watch them actually trade")
    axp.grid(color="#e1e0d9", linewidth=0.7); axp.set_axisbelow(True)

    # separate swimlanes (not overlaid) -- an offset alone was visually indistinguishable
    # since both agents mostly sit at the same +1 level at the same resolution
    axpos.step(range(window), fixed_pos[sl] * 0.8, where="post", color="#eb6834",
               linewidth=1.6, label="swordfish (fixed-α)")
    axpos.step(range(window), kalman_pos[sl] * 0.8 + 3, where="post", color="#1baf7a",
               linewidth=1.6, label="octopus (kalman-α)")
    axpos.axhline(0, color="#c3c2b7", linewidth=0.8)
    axpos.axhline(3, color="#c3c2b7", linewidth=0.8)
    axpos.set_yticks([-0.8, 0, 0.8, 2.2, 3, 3.8],
                     ["short", "flat", "long", "short", "flat", "long"])
    axpos.set_ylim(-1.3, 4.3)
    axpos.set_ylabel("position"); axpos.set_xlabel("minutes into this window")
    axpos.legend(fontsize=9, loc="upper right", ncol=2)
    axpos.grid(color="#e1e0d9", linewidth=0.7); axpos.set_axisbelow(True)
    fig2.tight_layout()
    fig2.savefig("unit7_watch_position_trace.png", dpi=150)
    print("wrote unit7_watch_position_trace.png")
