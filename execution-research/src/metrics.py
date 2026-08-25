"""Execution-cost and performance metrics.

The headline metric is implementation shortfall: how much worse the average fill
was than the price we saw when we decided to trade.

CRITICAL (see FINDINGS.md Bug #2): aggregate the RUPEES, then divide once. Do not
average per-order percentages -- a 7-share order and a 364,000-share order are not
equally informative, and equal-weighting them understated true cost by 3x.
"""
import numpy as np
import pandas as pd

from .engine import fills_to_frame


def _orders(fills):
    """Group fills into orders. One order = a run of fills sharing a decision
    price and direction."""
    if not fills:
        return pd.DataFrame(columns=["ref", "direction", "qty", "avg_fill",
                                     "shortfall_bps", "cost", "notional"])
    f = fills_to_frame(fills)
    out = []
    for (ref, d), g in f.groupby(["ref", "direction"]):
        q = g["qty"].sum()
        if q <= 0:
            continue
        avg = (g["qty"] * g["price"]).sum() / q
        out.append(dict(ref=ref, direction=d, qty=q, avg_fill=avg,
                        shortfall_bps=d * (avg - ref) / ref * 1e4,
                        cost=d * (avg - ref) * q, notional=q * ref))
    return pd.DataFrame(out)


def implementation_shortfall(fills):
    """Aggregate execution cost in basis points. Positive = cost. THE metric."""
    o = _orders(fills)
    if o.empty or o["notional"].sum() <= 0:
        return np.nan
    return float(o["cost"].sum() / o["notional"].sum() * 1e4)


def shortfall_equal_weighted(fills):
    """The BROKEN version, kept deliberately so experiments/03 can demonstrate
    the bug rather than just assert it."""
    o = _orders(fills)
    return np.nan if o.empty else float(o["shortfall_bps"].mean())


def order_table(fills):
    return _orders(fills)


def sharpe(values, periods=252):
    r = values.pct_change().dropna()
    return np.nan if r.std() == 0 else float(r.mean() / r.std() * np.sqrt(periods))


def total_return_pct(values):
    return float((values.iloc[-1] / values.iloc[0] - 1) * 100)
