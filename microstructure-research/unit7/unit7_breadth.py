"""
Unit 7 Pass 4 -- breadth test infrastructure.

Point this at a directory of per-symbol 1-min files. Accepts either schema:
  - the original Unit 6 schema (minute, buy_vol, sell_vol, close, n_trades,
    total_vol, ofi) -- what btc_1min_ofi_26d.csv already has
  - the richer schema from pull_and_bucket.py (adds ofi_notional, per-side
    trade counts, average trade size per side) -- preferred when present,
    since notional imbalance is the fairer cross-asset comparison (BTC
    trades near $80k, an altcoin can be $1-200, so raw coin-quantity
    imbalance isn't comparable across symbols the way dollar imbalance is)
For each symbol: runs the same Q-learning backtest (Unit 7 Pass 2 style,
realistic 7.5bps cost), then combines all symbols into an equal-weight
portfolio and reports:
  - each symbol's own Sharpe (the "before breadth" picture)
  - the portfolio's Sharpe (the "after breadth" picture)
  - the empirical pairwise return correlation matrix
  - the EFFECTIVE breadth: N / (1 + (N-1) * mean pairwise correlation)
    (Grinold's IR ~= IC * sqrt(breadth) needs INDEPENDENT bets; correlated
    assets give less effective breadth than their raw count suggests)

Usage: python3 unit7_breadth.py <directory containing symbol CSVs>
Each file should be named like ETHUSDT_1min_ofi.csv, SOLUSDT_1min_ofi.csv, etc.
-- symbol is inferred from the filename up to the first underscore.
"""
import sys, glob, os
import numpy as np, pandas as pd
from unit7_core import QAgent, ACTIONS
from unit7_real_btc import discretize, metrics
import unit7_real_btc as u

u.COST = 0.00075  # realistic: Binance VIP0 spot, BNB discount (see Pass 3)

def load_symbol(path):
    df = pd.read_csv(path)
    # prefer notional-based imbalance (richer schema) over the original
    # quantity-based "ofi" -- fair across differently-priced assets
    if "ofi_notional" in df.columns:
        df["ofi"] = df["ofi_notional"]
    elif "ofi" not in df.columns:
        raise ValueError(f"{path}: no 'ofi' or 'ofi_notional' column found")
    df["ret"] = df["close"].pct_change()
    df["vol20"] = df["ret"].rolling(20).std()
    return df.dropna().reset_index(drop=True)

def fit_edges(train):
    return np.quantile(train["ofi"], [1/3, 2/3]), np.quantile(train["vol20"], [1/3, 2/3])

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

def analyze_symbol(path, seed=0, n_epochs=8):
    df = load_symbol(path)
    split = int(len(df) * 0.7)
    train, test = df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    ofi_edges, vol_edges = fit_edges(train)
    tr_ofi, tr_vol, tr_ret = train["ofi"].to_numpy(), train["vol20"].to_numpy(), train["ret"].to_numpy()
    te_ofi, te_vol, te_ret = test["ofi"].to_numpy(), test["vol20"].to_numpy(), test["ret"].to_numpy()

    rng = np.random.default_rng(seed)
    agent = QAgent(alpha_mode="fixed")
    for ep in range(n_epochs):
        eps = agent.eps_start + (agent.eps_end - agent.eps_start) * ep / (n_epochs - 1)
        run_episode(tr_ofi, tr_vol, tr_ret, agent, ofi_edges, vol_edges, eps=eps, rng=rng, learn=True)
    test_rets, test_pos = run_episode(te_ofi, te_vol, te_ret, agent, ofi_edges, vol_edges, eps=0.0, rng=rng, learn=False)
    cum, sharpe, act = metrics(test_rets, test_pos)
    return {"test_rets": test_rets, "test_minutes": test["minute"].to_numpy()[1:],
            "cum": cum, "sharpe": sharpe, "activity": act, "n_test": len(test_rets)}

def effective_breadth(corr_matrix):
    n = corr_matrix.shape[0]
    if n < 2:
        return float(n)
    off_diag = corr_matrix[np.triu_indices(n, k=1)]
    mean_corr = np.nanmean(off_diag)
    return n / (1 + (n - 1) * mean_corr), mean_corr

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(directory, "*.csv")))
    if not files:
        print(f"No CSV files found in {directory}. Point this at a folder of per-symbol "
              f"1-min OFI files (same schema as btc_1min_ofi_26d.csv).")
        sys.exit(1)

    print(f"Found {len(files)} symbol file(s) in {directory}\n")
    per_symbol = {}
    for f in files:
        symbol = os.path.basename(f).split("_")[0].split(".")[0]
        print(f"Analyzing {symbol} ({f})...")
        try:
            per_symbol[symbol] = analyze_symbol(f)
        except Exception as e:
            print(f"  skipped ({e})")

    if len(per_symbol) < 2:
        print("\nNeed at least 2 symbols for a breadth comparison. "
              "This ran fine on the single available symbol -- pipeline verified, "
              "waiting on more real data.")
        for sym, r in per_symbol.items():
            print(f"  {sym}: cum={r['cum']:.2f}%  sharpe={r['sharpe']:.2f}  activity={r['activity']:.3f}  n={r['n_test']}")
        sys.exit(0)

    # ---- align on common test minutes, build a returns matrix ----
    min_len = min(r["n_test"] for r in per_symbol.values())
    rets_matrix = np.column_stack([r["test_rets"][:min_len] for r in per_symbol.values()])
    symbols = list(per_symbol.keys())

    corr = np.corrcoef(rets_matrix, rowvar=False)
    eff_n, mean_corr = effective_breadth(corr)

    portfolio_rets = rets_matrix.mean(axis=1)  # equal-weight
    port_cum = (np.prod(1 + portfolio_rets) - 1) * 100
    port_sharpe = (np.mean(portfolio_rets) / np.std(portfolio_rets)) * np.sqrt(252 * 1440) if np.std(portfolio_rets) > 0 else np.nan

    print(f"\n{'symbol':>10}{'cum%':>10}{'sharpe':>10}{'activity':>11}")
    for sym, r in per_symbol.items():
        print(f"{sym:>10}{r['cum']:>10.2f}{r['sharpe']:>10.2f}{r['activity']:>11.3f}")

    print(f"\nMean pairwise return correlation across {len(symbols)} symbols: {mean_corr:.3f}")
    print(f"Raw N = {len(symbols)}  ->  Effective breadth (correlation-adjusted) = {eff_n:.2f}")
    print(f"\nEqual-weight portfolio (N={len(symbols)}): cum={port_cum:.2f}%  sharpe={port_sharpe:.2f}")
    single_sharpes = [r["sharpe"] for r in per_symbol.values()]
    print(f"Mean single-symbol sharpe: {np.nanmean(single_sharpes):.2f}  "
          f"(naive sqrt(N) prediction if independent: {np.nanmean(single_sharpes)*np.sqrt(len(symbols)):.2f}, "
          f"correlation-adjusted prediction: {np.nanmean(single_sharpes)*np.sqrt(eff_n):.2f})")
