"""
Unit 7 Pass 2 -- tabular Q-learning on the REAL asset (26-day BTCUSDT 1-min
bars from Unit 6 Pass 5, `btc_1min_ofi_26d.csv`), not the synthetic regimes
Pass 1 used. Same 70/30 chronological split, same three real-feature state
Unit 6 studied (OFI) plus rolling vol, same fixed-vs-adaptive alpha question.

Dispersion here can't come from many independent price paths (there's one
real history) -- it comes from repeating TRAINING with different RNG seeds
(exploration order, tie-breaking) and evaluating each resulting frozen
policy on the identical, fixed test window. That isolates "how much does
the learned policy depend on training randomness" from "how much does it
depend on which market history we happened to draw" -- the second question
needs more real data collection, not more seeds.

Run: python3 unit7_real_btc.py
"""
import sys
import numpy as np, pandas as pd
from unit7_core import QAgent, ACTIONS, N_ACTIONS, N_VOL_BINS, N_POS, ci95

# COST is set per-run from the command line (see __main__) -- NOT hardcoded.
# Why: the original 15bps convention (Units 5 & 7 Pass 1) was calibrated for
# DAILY equity bars. At 1-minute BTC granularity the typical bar's own return
# std is ~5bps (checked directly on this data) -- so a 15bps round-trip cost
# is ~3 standard deviations of the very thing being traded. That guarantees
# "never trade" almost regardless of whether a real signal exists, which
# would make a "no edge" conclusion uninformative (cost artifact, not a
# finding about OFI). Run at multiple cost levels to separate the two.
COST = 0.0015

# ---------------------------------------------------------------- load real data
def load():
    df = pd.read_csv("btc_1min_ofi_26d.csv")
    df["ret"] = df["close"].pct_change()
    df["vol20"] = df["ret"].rolling(20).std()
    df = df.dropna().reset_index(drop=True)
    return df

def fit_edges(train):
    ofi_edges = np.quantile(train["ofi"], [1/3, 2/3])
    vol_edges = np.quantile(train["vol20"], [1/3, 2/3])
    return ofi_edges, vol_edges

def discretize(ofi, vol, pos, ofi_edges, vol_edges):
    ob = int(np.digitize(ofi, ofi_edges))
    vb = int(np.digitize(vol, vol_edges))
    pb = int(pos) + 1
    return (ob * N_VOL_BINS + vb) * N_POS + pb

def to_arrays(df):
    return df["ofi"].to_numpy(), df["vol20"].to_numpy(), df["ret"].to_numpy()

# ---------------------------------------------------------------- env loop (mirrors unit7_core)
def run_episode(ofi, vol, ret, agent, ofi_edges, vol_edges, eps=0.0, rng=None, learn=False):
    pos = 0
    n = len(ofi) - 1
    rets = np.empty(n); pos_trace = np.empty(n)
    for i in range(n):
        s = discretize(ofi[i], vol[i], pos, ofi_edges, vol_edges)
        a_idx = agent.act(s, eps, rng)
        new_pos = int(ACTIONS[a_idx])
        turnover_cost = COST * abs(new_pos - pos)
        reward = new_pos * ret[i + 1] - turnover_cost
        rets[i] = reward; pos_trace[i] = new_pos
        if learn:
            s_next = discretize(ofi[i + 1], vol[i + 1], new_pos, ofi_edges, vol_edges)
            agent.update(s, a_idx, reward, s_next, learn=True)
        pos = new_pos
    return rets, pos_trace

def run_ofi_sign(ofi, ret):
    """Naive, un-learned use of the same feature: long if OFI>0 else short.
    Directly tests: does RL beat literally the simplest possible use of the
    exact feature Unit 6 already found had no significant linear coefficient?"""
    pos = 0
    n = len(ofi) - 1
    rets = np.empty(n); pos_trace = np.empty(n)
    for i in range(n):
        new_pos = 1 if ofi[i] > 0 else -1
        turnover_cost = COST * abs(new_pos - pos)
        rets[i] = new_pos * ret[i + 1] - turnover_cost
        pos_trace[i] = new_pos
        pos = new_pos
    return rets, pos_trace

def run_buyhold(ret):
    n = len(ret) - 1
    rets = np.empty(n); pos_trace = np.ones(n)
    rets[0] = 1 * ret[1] - COST  # one entry cost
    for i in range(1, n):
        rets[i] = 1 * ret[i + 1]
    return rets, pos_trace

def metrics(rets, pos_trace):
    cum_pct = (np.prod(1 + rets) - 1) * 100
    sharpe_annualizer = np.sqrt(252 * 1440)  # 1-min bars: 1440/day, 252 trading days
    sharpe = (np.mean(rets) / np.std(rets)) * sharpe_annualizer if np.std(rets) > 0 else np.nan
    activity = np.mean(np.abs(pos_trace))
    return cum_pct, sharpe, activity

# ---------------------------------------------------------------- experiment
def one_training_seed(seed, train_arr, test_arr, ofi_edges, vol_edges, n_epochs=10):
    tr_ofi, tr_vol, tr_ret = train_arr
    te_ofi, te_vol, te_ret = test_arr
    rng = np.random.default_rng(seed + 9000)
    results = {}
    for mode in ["fixed", "kalman"]:
        agent = QAgent(alpha_mode=mode)
        for ep in range(n_epochs):
            eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / (n_epochs - 1)
            run_episode(tr_ofi, tr_vol, tr_ret, agent, ofi_edges, vol_edges, eps=eps, rng=rng, learn=True)
        test_rets, test_pos = run_episode(te_ofi, te_vol, te_ret, agent, ofi_edges, vol_edges, eps=0.0, rng=rng, learn=False)
        results[f"qlearn_{mode}"] = metrics(test_rets, test_pos)
        if mode == "kalman":
            results["_alpha_trace"] = agent.alpha_trace
    return results

if __name__ == "__main__":
    cost_bps = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
    N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    N_EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    COST = cost_bps / 1e4  # rebinds the module-level COST that run_episode etc. read at call time

    df = load()
    print(f"Loaded {len(df)} real 1-min BTCUSDT bars (26 days, Unit 6 Pass 5 data). cost={cost_bps}bps")
    split = int(len(df) * 0.7)   # same 70/30 chronological split Unit 6 Pass 5 used
    train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    print(f"train={len(train)}  test={len(test)}  (Unit 6 Pass 5 used 26,207 / 11,231 -- should match)")

    ofi_edges, vol_edges = fit_edges(train)  # fit on train only
    train_arr = to_arrays(train)
    test_arr = to_arrays(test)

    # --- non-learned baselines, computed once (deterministic, no training randomness) ---
    bh_rets, bh_pos = run_buyhold(test_arr[2])
    ofisign_rets, ofisign_pos = run_ofi_sign(test_arr[0], test_arr[2])
    baselines = {"buyhold": metrics(bh_rets, bh_pos), "ofi_sign": metrics(ofisign_rets, ofisign_pos)}

    # --- Q-learning, repeated over training seeds for training-stochasticity dispersion ---
    rows = {"qlearn_fixed": {"cum": [], "sharpe": [], "activity": []},
            "qlearn_kalman": {"cum": [], "sharpe": [], "activity": []}}
    alpha_trace_sample = None
    for seed in range(N_SEEDS):
        res = one_training_seed(seed, train_arr, test_arr, ofi_edges, vol_edges, n_epochs=N_EPOCHS)
        for k in rows:
            c, s, act = res[k]
            rows[k]["cum"].append(c); rows[k]["sharpe"].append(s); rows[k]["activity"].append(act)
        if seed == 0:
            alpha_trace_sample = res["_alpha_trace"]

    print(f"\n=== Real BTCUSDT, test segment (n={len(test_arr[2])-1} minutes), cost={cost_bps}bps, "
          f"{N_SEEDS} training seeds for Q-learning ===")
    print(f"{'strategy':>16}{'cum% mean':>12}{'95% CI':>24}{'Sharpe mean':>13}{'95% CI':>24}{'mean activity':>15}")
    for k, disp in [("buyhold", "buy&hold"), ("ofi_sign", "OFI-sign (naive)")]:
        c, s, act = baselines[k]
        print(f"{disp:>16}{c:>12.2f}{'[n/a, single path]':>24}{s:>13.2f}{'[n/a]':>24}{act:>15.3f}")
    for k, disp in [("qlearn_fixed", "Qlearn fixed-a"), ("qlearn_kalman", "Qlearn kalman-a")]:
        cm, clo, chi = ci95(rows[k]["cum"])
        sm, slo, shi = ci95(rows[k]["sharpe"])
        print(f"{disp:>16}{cm:>12.2f}{f'[{clo:.2f}, {chi:.2f}]':>24}{sm:>13.2f}{f'[{slo:.2f}, {shi:.2f}]':>24}{np.mean(rows[k]['activity']):>15.3f}")

    print(f"\nadaptive alpha (seed 0) over full training run: mean={np.mean(alpha_trace_sample):.3f} "
          f"min={np.min(alpha_trace_sample):.3f} max={np.max(alpha_trace_sample):.3f} n_updates={len(alpha_trace_sample)}")

    out = pd.DataFrame({k: rows[k]["cum"] for k in rows})
    out["seed"] = range(N_SEEDS)
    out["cost_bps"] = cost_bps
    for k in rows:
        out[f"{k}_sharpe"] = rows[k]["sharpe"]
        out[f"{k}_activity"] = rows[k]["activity"]
    fname = f"unit7_real_btc_results_{int(cost_bps)}bps.csv"
    out.to_csv(fname, index=False)
    print(f"\nWrote {fname}")
