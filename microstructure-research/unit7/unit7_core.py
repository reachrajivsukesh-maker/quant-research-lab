"""
Unit 7 -- Pass 1: tabular Q-learning trader, adaptive vs fixed learning rate.
SELF-CONTAINED (regenerates its own data), same generator/cost convention as
Unit 5's unit5_core.py, so the RL agent is judged on the same footing.
Run: python3 unit7_core.py
"""
import numpy as np, pandas as pd

# ---------------------------------------------------------------- constants
COST = 0.0015          # same 15 bps turnover cost as Unit 5 (per unit of |Δposition|)
GAMMA = 0.95            # RL discount factor
N_ACTIONS = 3           # {-1: short, 0: flat, +1: long}
ACTIONS = np.array([-1, 0, 1])
N_MOM_BINS, N_VOL_BINS, N_POS = 3, 3, 3
N_STATES = N_MOM_BINS * N_VOL_BINS * N_POS

# ---------------------------------------------------------------- data (Unit 5 convention)
def make_path(seed, n=800, rho=0.0, vol=0.014, start=100.0):
    """rho=0 kept as default: real daily returns have ~0 autocorrelation
    (Unit 5's finding). rho!=0 is a planted control for regime tests."""
    rng = np.random.default_rng(seed)
    r, prev = [], 0.0
    for _ in range(n):
        e = rng.normal(0.0, vol)
        cur = rho * prev + e
        r.append(cur); prev = cur
    close = start * np.exp(np.cumsum(r))
    d = pd.DataFrame({"close": close}, index=pd.bdate_range("2023-01-02", periods=n))
    d["ret"] = d["close"].pct_change()
    d["mom5"] = d["close"].pct_change(5)                       # momentum feature
    d["vol20"] = d["ret"].rolling(20).std()                     # vol feature
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    return d.dropna().reset_index(drop=True)

# ---------------------------------------------------------------- state discretization
def fit_bin_edges(train_df):
    """Quantile edges fit on TRAIN ONLY -- frozen and reused on test, so the
    test segment's own distribution never leaks into the state definition."""
    mom_edges = np.quantile(train_df["mom5"], [1/3, 2/3])
    vol_edges = np.quantile(train_df["vol20"], [1/3, 2/3])
    return mom_edges, vol_edges

def discretize(mom, vol, pos, mom_edges, vol_edges):
    mb = int(np.digitize(mom, mom_edges))          # 0,1,2
    vb = int(np.digitize(vol, vol_edges))          # 0,1,2
    pb = int(pos) + 1                              # -1,0,1 -> 0,1,2
    return (mb * N_VOL_BINS + vb) * N_POS + pb      # single int in [0, N_STATES)

# ---------------------------------------------------------------- Q-learning agent
class QAgent:
    """alpha_mode: 'fixed' or 'kalman'
       kalman: alpha_t = |delta_t| / (|delta_t| + m_t), m_t = running EMA of |delta|.
       m_t updated with the PRE-step value (never with delta it just produced),
       so the gain never depends on its own outcome. One extra tracked number."""
    def __init__(self, alpha_mode="fixed", alpha_fixed=0.10, beta_m=0.02,
                 alpha_min=0.01, alpha_max=0.5, eps_start=0.3, eps_end=0.05):
        self.Q = np.zeros((N_STATES, N_ACTIONS))
        self.alpha_mode = alpha_mode
        self.alpha_fixed = alpha_fixed
        self.beta_m = beta_m
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.m = 1e-4              # running |TD error| estimate, small nonzero start
        self.eps_start, self.eps_end = eps_start, eps_end
        self.alpha_trace = []      # so we can show what alpha actually did

    def act(self, s, eps, rng):
        if rng.random() < eps:
            return rng.integers(0, N_ACTIONS)
        row = self.Q[s]
        return int(rng.choice(np.flatnonzero(row == row.max())))  # tie-break randomly

    def update(self, s, a, r, s_next, learn=True):
        target = r + GAMMA * self.Q[s_next].max()
        delta = target - self.Q[s, a]
        if not learn:
            return delta
        if self.alpha_mode == "fixed":
            alpha = self.alpha_fixed
        else:
            alpha = abs(delta) / (abs(delta) + self.m)
            alpha = float(np.clip(alpha, self.alpha_min, self.alpha_max))
            self.m = (1 - self.beta_m) * self.m + self.beta_m * abs(delta)
            self.alpha_trace.append(alpha)
        self.Q[s, a] += alpha * delta
        return delta

# ---------------------------------------------------------------- environment loop
# All loops work on plain numpy arrays -- pandas .iloc row access is the
# dominant cost otherwise (checked: >50x slower for this access pattern).
def to_arrays(df):
    return df["mom5"].to_numpy(), df["vol20"].to_numpy(), df["ret"].to_numpy(), \
           (df["ma5"] > df["ma20"]).to_numpy()

def run_episode(mom, vol, ret, agent, mom_edges, vol_edges, eps=0.0, rng=None, learn=False):
    """One pass over a path's arrays for the RL agent."""
    pos = 0
    n = len(mom) - 1
    rets = np.empty(n)
    pos_trace = np.empty(n)
    for i in range(n):
        s = discretize(mom[i], vol[i], pos, mom_edges, vol_edges)
        a_idx = agent.act(s, eps, rng)
        new_pos = int(ACTIONS[a_idx])
        turnover_cost = COST * abs(new_pos - pos)
        next_ret = ret[i + 1]
        reward = new_pos * next_ret - turnover_cost
        rets[i] = reward
        pos_trace[i] = new_pos
        if learn:
            s_next = discretize(mom[i + 1], vol[i + 1], new_pos, mom_edges, vol_edges)
            agent.update(s, a_idx, reward, s_next, learn=True)
        pos = new_pos
    return rets, pos_trace

def run_baseline_signal(ma_signal, ret, kind):
    pos = 0
    n = len(ret) - 1
    rets = np.empty(n)
    pos_trace = np.empty(n)
    for i in range(n):
        new_pos = 1 if (kind == "buyhold" or ma_signal[i]) else 0
        turnover_cost = COST * abs(new_pos - pos)
        rets[i] = new_pos * ret[i + 1] - turnover_cost
        pos_trace[i] = new_pos
        pos = new_pos
    return rets, pos_trace

# ---------------------------------------------------------------- metrics
def metrics(rets, pos_trace):
    cum_pct = (np.prod(1 + rets) - 1) * 100
    sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(252) if np.std(rets) > 0 else np.nan
    activity = np.mean(np.abs(pos_trace))   # 0 = never left flat, 1 = always at |pos|=1
    return cum_pct, sharpe, activity

def ci95(x):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se

# ---------------------------------------------------------------- experiment
def one_seed(seed, rho=0.0, n_train_epochs=25):
    df = make_path(seed, rho=rho)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    mom_edges, vol_edges = fit_bin_edges(train)

    tr_mom, tr_vol, tr_ret, tr_ma = to_arrays(train)
    te_mom, te_vol, te_ret, te_ma = to_arrays(test)

    rng = np.random.default_rng(seed + 5000)
    results = {}

    for mode in ["fixed", "kalman"]:
        agent = QAgent(alpha_mode=mode)
        for ep in range(n_train_epochs):
            eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / (n_train_epochs - 1)
            run_episode(tr_mom, tr_vol, tr_ret, agent, mom_edges, vol_edges, eps=eps, rng=rng, learn=True)
        test_rets, test_pos = run_episode(te_mom, te_vol, te_ret, agent, mom_edges, vol_edges, eps=0.0, rng=rng, learn=False)
        results[f"qlearn_{mode}"] = metrics(test_rets, test_pos)
        if mode == "kalman":
            results["_alpha_trace_sample"] = agent.alpha_trace[-200:]

    for kind in ["buyhold", "ma_crossover"]:
        test_rets, test_pos = run_baseline_signal(te_ma, te_ret, kind)
        results[kind] = metrics(test_rets, test_pos)

    return results

if __name__ == "__main__":
    N_SEEDS = 150
    STRATS = ["qlearn_fixed", "qlearn_kalman", "buyhold", "ma_crossover"]
    REGIMES = [(-0.15, "mean reversion"), (0.0, "random walk"), (0.15, "momentum")]

    all_csv_rows = []
    alpha_sample_by_regime = {}

    print(f"Test-segment results over {N_SEEDS} independent synthetic paths per regime (walk-forward, 70/30 split).")
    print("Cum% and Sharpe computed on the TEST segment only; agents trained (or frozen) upstream of it.")
    print("'activity' = mean(|position|) over the test window: 0 = agent never left flat, 1 = always fully positioned.\n")

    for rho, label in REGIMES:
        rows = {k: {"cum": [], "sharpe": [], "activity": []} for k in STRATS}
        alpha_samples = None
        for seed in range(N_SEEDS):
            res = one_seed(seed, rho=rho)
            for k in STRATS:
                c, s, act = res[k]
                rows[k]["cum"].append(c); rows[k]["sharpe"].append(s); rows[k]["activity"].append(act)
                all_csv_rows.append({"regime": label, "seed": seed, "strategy": k,
                                      "cum_pct": c, "sharpe": s, "activity": act})
            if seed == 0:
                alpha_samples = res["_alpha_trace_sample"]
        alpha_sample_by_regime[label] = alpha_samples

        print(f"=== regime: {label} (rho={rho}) ===")
        print(f"{'strategy':>16}{'cum% mean':>12}{'95% CI':>24}{'Sharpe mean':>13}{'95% CI':>24}{'n_sharpe':>10}{'mean activity':>15}")
        for k, disp in [("buyhold", "buy&hold"), ("ma_crossover", "MA(5,20)"),
                        ("qlearn_fixed", "Qlearn fixed-a"), ("qlearn_kalman", "Qlearn kalman-a")]:
            cm, clo, chi = ci95(rows[k]["cum"])
            sm, slo, shi = ci95(rows[k]["sharpe"])
            n_valid = np.sum(~np.isnan(rows[k]["sharpe"]))
            print(f"{disp:>16}{cm:>12.2f}{f'[{clo:.2f}, {chi:.2f}]':>24}{sm:>13.2f}{f'[{slo:.2f}, {shi:.2f}]':>24}{n_valid:>10d}{np.mean(rows[k]['activity']):>15.3f}")
        print()

    print("Adaptive (kalman) alpha's own trajectory, seed 0, last 200 training updates, per regime:")
    for label, samples in alpha_sample_by_regime.items():
        print(f"  {label:>16}: mean={np.mean(samples):.3f}  min={np.min(samples):.3f}  max={np.max(samples):.3f}")

    pd.DataFrame(all_csv_rows).to_csv("unit7_results.csv", index=False)
    print("\nWrote unit7_results.csv")
