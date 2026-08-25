"""Out-of-sample validation.

Two designs are implemented, and the second exists because the first was shown
to be inadequate:

  walk_forward  -- train/test folds carved from ONE price path. Only 3 folds fit
                   in 300 days, each test window holds 2-3 orders, so a single
                   trending window can dominate the result. Kept for comparison.

  monte_carlo   -- many INDEPENDENT paths, paired (both configs see the identical
                   path, so path-level luck cancels). This is what gives usable
                   confidence intervals.
"""
import numpy as np
import pandas as pd

from .data import make_path
from .engine import backtest
from .metrics import implementation_shortfall


def score(d, cfg):
    _, fills = backtest(d, cfg)
    return implementation_shortfall(fills)


def walk_forward(d, configs, train=120, test=60, step=60, baseline=None):
    """Select the best config in-sample, apply it once to the unseen test block,
    and compare against a fixed baseline you'd have used without tuning."""
    folds, s = [], 0
    while s + train + test <= len(d):
        folds.append((s, s + train, s + train + test)); s += step

    rows = []
    for i, (a, b, c) in enumerate(folds, 1):
        tr, te = d.iloc[a:b], d.iloc[b:c]
        in_sample = {j: score(tr, cfg) for j, cfg in enumerate(configs)}
        valid = {j: v for j, v in in_sample.items() if not np.isnan(v)}
        if not valid:
            continue
        best = min(valid, key=valid.get)
        rows.append(dict(
            fold=i,
            train_start=str(d.index[a].date()), test_start=str(d.index[b].date()),
            chosen=str(configs[best]), train_bps=round(in_sample[best], 2),
            oos_chosen=round(score(te, configs[best]), 2),
            oos_baseline=round(score(te, baseline), 2) if baseline else np.nan))
    out = pd.DataFrame(rows)
    if not out.empty and baseline:
        out["edge"] = (out["oos_baseline"] - out["oos_chosen"]).round(2)
    return out


def monte_carlo(variants, n_paths=300, seed0=1000, baseline_key=None):
    """Run every variant on every path. Returns a DataFrame, one row per path."""
    paths = [make_path(seed0 + i) for i in range(n_paths)]
    data = {name: np.array([score(p, cfg) for p in paths])
            for name, cfg in variants.items()}
    return pd.DataFrame(data)


def paired_summary(df, baseline_key):
    """Paired comparison of each variant against the baseline, across paths.
    Reports the WIN RATE alongside the mean -- a mean alone hides whether a small
    average difference is a near-universal loss or a coin flip."""
    base = df[baseline_key].values
    rows = []
    for col in df.columns:
        diff = df[col].values - base
        diff = diff[~np.isnan(diff)]
        n = len(diff)
        m, sd = diff.mean(), diff.std(ddof=1) if n > 1 else np.nan
        se = sd / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(
            variant=col, mean_bps=round(float(np.nanmean(df[col])), 2),
            vs_baseline=round(float(m), 2),
            ci_low=round(float(m - 1.96 * se), 2) if n > 1 else np.nan,
            ci_high=round(float(m + 1.96 * se), 2) if n > 1 else np.nan,
            t_stat=round(float(m / se), 2) if se and se > 0 else np.nan,
            win_rate_pct=round(float((diff < 0).mean() * 100), 1), n_paths=n))
    return pd.DataFrame(rows)
