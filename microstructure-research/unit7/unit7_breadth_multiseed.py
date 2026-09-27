"""
Unit 7 Pass 4b -- multi-seed breadth test.

Single-seed frictionless breadth (unit7_breadth_frictionless.py) turned out
uninterpretable: 6/9 symbols showed exactly zero activity even at cost=0, with
no structural reason (checked ofi_std/ret_std -- no distinguishing pattern
between active and flat symbols). The mechanism: at cost=0, nothing forces
Q(flat)=0 to beat Q(long)/Q(short) the way realistic cost does -- which side
wins is decided by a single seed's training noise. This script removes that
by running N_SEEDS independent trainings per symbol and asking the real
question: is any symbol's frictionless activity a REPRODUCIBLE signal (same
sign, present across most seeds) or a coin-flip artifact (activity and sign
both vary randomly across seeds, averaging to ~nothing)?

Also re-checks the realistic-cost null result across a handful of seeds, to
confirm it is a structural result (should hold for EVERY seed, since it's a
Bellman-equation argument, not noise) rather than itself a single-seed fluke.

Usage: python3 unit7_breadth_multiseed.py <dir> <n_seeds_frictionless> <n_seeds_cost> <part: A|B|both>
"""
import sys, glob, os
import numpy as np, pandas as pd
from unit7_core import QAgent, ACTIONS
from unit7_real_btc import discretize, metrics
import unit7_real_btc as u


def load_symbol(path):
    df = pd.read_csv(path)
    if "ofi_notional" in df.columns:
        df["ofi"] = df["ofi_notional"]
    elif "ofi" not in df.columns:
        raise ValueError(f"{path}: no 'ofi' or 'ofi_notional' column found")
    df["ret"] = df["close"].pct_change()
    df["vol20"] = df["ret"].rolling(20).std()
    return df.dropna().reset_index(drop=True)


def fit_edges(train):
    return np.quantile(train["ofi"], [1 / 3, 2 / 3]), np.quantile(train["vol20"], [1 / 3, 2 / 3])


def run_episode(ofi, vol, ret, agent, ofi_edges, vol_edges, eps=0.0, rng=None, learn=False):
    pos = 0
    n = len(ofi) - 1
    rets = np.empty(n)
    pos_trace = np.empty(n)
    for i in range(n):
        s = discretize(ofi[i], vol[i], pos, ofi_edges, vol_edges)
        a_idx = agent.act(s, eps, rng)
        new_pos = int(ACTIONS[a_idx])
        turnover_cost = u.COST * abs(new_pos - pos)
        reward = new_pos * ret[i + 1] - turnover_cost
        rets[i] = reward
        pos_trace[i] = new_pos
        if learn:
            s_next = discretize(ofi[i + 1], vol[i + 1], new_pos, ofi_edges, vol_edges)
            agent.update(s, a_idx, reward, s_next, learn=True)
        pos = new_pos
    return rets, pos_trace


def train_test_one_seed(p, seed, n_epochs=8):
    rng = np.random.default_rng(seed)
    agent = QAgent(alpha_mode="fixed")
    for ep in range(n_epochs):
        eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / (n_epochs - 1)
        run_episode(p["tr_ofi"], p["tr_vol"], p["tr_ret"], agent, p["ofi_edges"], p["vol_edges"],
                    eps=eps, rng=rng, learn=True)
    test_rets, test_pos = run_episode(p["te_ofi"], p["te_vol"], p["te_ret"], agent,
                                       p["ofi_edges"], p["vol_edges"], eps=0.0, rng=rng, learn=False)
    return test_rets, test_pos


def load_all(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    prepared = {}
    for f in files:
        symbol = os.path.basename(f).split("_")[0].split(".")[0]
        df = load_symbol(f)
        split = int(len(df) * 0.7)
        train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
        ofi_edges, vol_edges = fit_edges(train)
        prepared[symbol] = dict(
            ofi_edges=ofi_edges, vol_edges=vol_edges,
            tr_ofi=train["ofi"].to_numpy(), tr_vol=train["vol20"].to_numpy(), tr_ret=train["ret"].to_numpy(),
            te_ofi=test["ofi"].to_numpy(), te_vol=test["vol20"].to_numpy(), te_ret=test["ret"].to_numpy(),
        )
    return prepared


def part_a(prepared, n_seeds_cost):
    print("=" * 78)
    print(f"PART A -- realistic cost (7.5bps), {n_seeds_cost} seeds per symbol")
    print("Expectation: EVERY seed, EVERY symbol -> zero activity (structural, not noise)")
    print("=" * 78)
    u.COST = 0.00075
    any_nonzero = False
    for sym, p in prepared.items():
        acts = []
        for seed in range(n_seeds_cost):
            _, pos = train_test_one_seed(p, seed)
            acts.append(np.mean(np.abs(pos)))
        max_act = max(acts)
        if max_act > 0:
            any_nonzero = True
        acts_str = ", ".join(f"{a:.4f}" for a in acts)
        print(f"  {sym:>10}: activity per seed = [{acts_str}]  max={max_act:.4f}")
    verdict = "CONFIRMED structural: zero activity at every seed, every symbol" if not any_nonzero \
        else "NOT fully structural -- some seed/symbol traded even at realistic cost"
    print(f"\n=> {verdict}\n")
    return not any_nonzero


def part_b(prepared, n_seeds_frictionless):
    print("=" * 78)
    print(f"PART B -- frictionless (cost=0), {n_seeds_frictionless} seeds per symbol")
    print("Question: is any symbol's activity a REPRODUCIBLE signal, or seed noise?")
    print("=" * 78)
    u.COST = 0.0
    consensus_rets = {}
    for sym, p in prepared.items():
        n_test = len(p["te_ret"]) - 1
        seed_rets = np.empty((n_seeds_frictionless, n_test))
        sharpes = []
        for seed in range(n_seeds_frictionless):
            rets, pos = train_test_one_seed(p, seed)
            seed_rets[seed] = rets
            _, sharpe, _ = metrics(rets, pos)
            sharpes.append(sharpe)
        consensus_rets[sym] = seed_rets.mean(axis=0)
        finite = [s for s in sharpes if np.isfinite(s)]
        n_active = len(finite)
        mean_s = np.mean(finite) if finite else float("nan")
        std_s = np.std(finite) if len(finite) > 1 else float("nan")
        frac_pos = np.mean(np.array(finite) > 0) if finite else float("nan")
        print(f"  {sym:>10}: active in {n_active}/{n_seeds_frictionless} seeds  "
              f"mean_sharpe={mean_s:>7.2f}  std={std_s:>6.2f}  frac_positive={frac_pos:.2f}")

    print("\nConsensus (seed-averaged) per-symbol result -- the noise-robust signal:")
    print(f"{'symbol':>10}{'cum%':>12}{'sharpe':>10}")
    for sym in prepared:
        r = consensus_rets[sym]
        cum = (np.prod(1 + r) - 1) * 100
        sharpe = (np.mean(r) / np.std(r)) * np.sqrt(252 * 1440) if np.std(r) > 0 else np.nan
        print(f"{sym:>10}{cum:>12.6f}{sharpe:>10.2f}")

    print()
    print("=" * 78)
    print("Breadth / Grinold's Law check on the seed-averaged (noise-robust) returns")
    print("=" * 78)
    syms = list(prepared.keys())
    min_len = min(len(consensus_rets[s]) for s in syms)
    mat = np.column_stack([consensus_rets[s][:min_len] for s in syms])
    stds = mat.std(axis=0)
    live = stds > 1e-12
    live_syms = [s for s, l in zip(syms, live) if l]
    print(f"Symbols with nonzero consensus variance after {n_seeds_frictionless}-seed averaging: "
          f"{live_syms} ({live.sum()} of {len(syms)})")
    if live.sum() >= 2:
        corr = np.corrcoef(mat[:, live], rowvar=False)
        n = int(live.sum())
        off_diag = corr[np.triu_indices(n, k=1)]
        mean_corr = np.nanmean(off_diag)
        eff_n = n / (1 + (n - 1) * mean_corr)
        print(f"Mean pairwise correlation among live symbols: {mean_corr:.4f}")
        print(f"Effective breadth among live symbols: {eff_n:.2f} (raw N={n})")
    else:
        print("Fewer than 2 symbols have nonzero consensus-averaged variance -- "
              "no correlation/breadth structure exists to measure at all.")

    port = mat.mean(axis=1)
    port_cum = (np.prod(1 + port) - 1) * 100
    port_sharpe = (np.mean(port) / np.std(port)) * np.sqrt(252 * 1440) if np.std(port) > 0 else np.nan
    print(f"\nEqual-weight 9-asset portfolio on consensus returns: cum={port_cum:.6f}%  sharpe={port_sharpe:.2f}")
    return consensus_rets


if __name__ == "__main__":
    directory = sys.argv[1]
    n_seeds_frictionless = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    n_seeds_cost = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    part = sys.argv[4] if len(sys.argv) > 4 else "both"

    prepared = load_all(directory)
    print(f"Loaded {len(prepared)} symbols: {sorted(prepared)}\n")

    if part in ("A", "both"):
        part_a(prepared, n_seeds_cost)
    if part in ("B", "both"):
        part_b(prepared, n_seeds_frictionless)
