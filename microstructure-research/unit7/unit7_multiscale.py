"""
Unit 7 -- does trading become economical at a coarser bar size?
At 1-min bars, realistic Binance cost (7.5bps BNB-discounted VIP0, live-fetched
from binance.com/en/fee/schedule) is 1.5x the bar's own return std and >>the
per-bar OFI signal (0-0.3bps) -- so "never trade" is the correct answer, not a
bug (see unit7_diagnose.py). This script re-aggregates the SAME real 26-day
BTCUSDT data to coarser bars (5/15/30/60 min) -- signal and noise both change
with the timescale, cost per trade does not -- to find whether there's a
horizon where a real, tradeable edge exists. Quick single-seed scan first.
"""
import numpy as np, pandas as pd
from unit7_core import QAgent, ACTIONS
from unit7_real_btc import discretize, run_ofi_sign, run_buyhold, metrics
import unit7_real_btc as u

def load_1min():
    df = pd.read_csv("btc_1min_ofi_26d.csv")
    return df

def aggregate(df, h):
    """Non-overlapping h-minute bars from the 1-min data."""
    n = len(df) // h
    df = df.iloc[:n * h].copy()
    grp = np.arange(len(df)) // h
    agg = df.groupby(grp).agg(
        buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"),
        total_vol=("total_vol", "sum"), close=("close", "last"),
        minute=("minute", "last"),
    ).reset_index(drop=True)
    agg["ofi"] = (agg["buy_vol"] - agg["sell_vol"]) / agg["total_vol"]
    agg["ret"] = agg["close"].pct_change()
    agg["vol20"] = agg["ret"].rolling(20).std()
    return agg.dropna().reset_index(drop=True)

def run_episode(ofi, vol, ret, agent, ofi_edges, vol_edges, eps=0.0, rng=None, learn=False):
    pos = 0
    n = len(ofi) - 1
    rets = np.empty(n); pos_trace = np.empty(n)
    for i in range(n):
        s = discretize(ofi[i], vol[i], pos, ofi_edges, vol_edges)
        a_idx = agent.act(s, eps, rng)
        new_pos = int(ACTIONS[a_idx])
        turnover_cost = u.COST * abs(new_pos - pos)
        reward = new_pos * ret[i + 1] - turnover_cost
        rets[i] = reward; pos_trace[i] = new_pos
        if learn:
            s_next = discretize(ofi[i + 1], vol[i + 1], new_pos, ofi_edges, vol_edges)
            agent.update(s, a_idx, reward, s_next, learn=True)
        pos = new_pos
    return rets, pos_trace

if __name__ == "__main__":
    df1 = load_1min()
    u.COST = 0.00075  # realistic: Binance VIP0 spot, BNB discount (live-fetched)
    HORIZONS = [1, 5, 15, 30, 60]
    print(f"Realistic cost used throughout: {u.COST*1e4:.1f}bps (Binance VIP0 spot, BNB discount)\n")
    print(f"{'horizon(min)':>13}{'n bars':>9}{'bar ret std':>14}{'buyhold%':>11}{'ofisign%':>11}{'qlearn_fixed%':>15}{'activity':>10}")

    for h in HORIZONS:
        agg = aggregate(df1, h)
        split = int(len(agg) * 0.7)
        train, test = agg.iloc[:split].reset_index(drop=True), agg.iloc[split:].reset_index(drop=True)
        if len(train) < 100 or len(test) < 50:
            print(f"{h:>13}  -- too few bars, skipping")
            continue
        ofi_edges = np.quantile(train["ofi"], [1/3, 2/3])
        vol_edges = np.quantile(train["vol20"], [1/3, 2/3])
        tr_ofi, tr_vol, tr_ret = train["ofi"].to_numpy(), train["vol20"].to_numpy(), train["ret"].to_numpy()
        te_ofi, te_vol, te_ret = test["ofi"].to_numpy(), test["vol20"].to_numpy(), test["ret"].to_numpy()

        rng = np.random.default_rng(0)
        agent = QAgent(alpha_mode="fixed", alpha_fixed=0.10)
        n_epochs = 15
        for ep in range(n_epochs):
            eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / (n_epochs - 1)
            run_episode(tr_ofi, tr_vol, tr_ret, agent, ofi_edges, vol_edges, eps=eps, rng=rng, learn=True)
        test_rets, test_pos = run_episode(te_ofi, te_vol, te_ret, agent, ofi_edges, vol_edges, eps=0.0, rng=rng, learn=False)
        qc, qs, qact = metrics(test_rets, test_pos)

        bh_rets, bh_pos = run_buyhold(te_ret)
        of_rets, of_pos = run_ofi_sign(te_ofi, te_ret)
        bhc, _, _ = metrics(bh_rets, bh_pos)
        ofc, _, _ = metrics(of_rets, of_pos)

        bar_std = pd.Series(tr_ret).std()
        print(f"{h:>13}{len(agg):>9}{bar_std*1e4:>11.2f}bps{bhc:>11.2f}{ofc:>11.2f}{qc:>15.2f}{qact:>10.3f}")
