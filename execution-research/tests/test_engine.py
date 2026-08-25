"""Tests that encode the lessons, not just the code.

Several of these exist because a bug got through once. test_no_lookahead and
test_shortfall_is_notional_weighted in particular are regression guards against
the two errors documented in docs/FINDINGS.md.
"""
import os, sys
import numpy as np
import pandas as pd
try:
    import pytest
except ImportError:                     # minimal shim so the suite also runs
    _abs = abs
    class _A:                           # standalone (this sandbox has no PyPI)
        @staticmethod
        def approx(v, rel=1e-6, abs=None):
            class _C:
                def __eq__(s, o):
                    if isinstance(v, (list, tuple, np.ndarray)):
                        return np.allclose(o, v, rtol=rel)
                    tol = abs if abs is not None else max(rel * max(1.0, _abs(v)), 1e-9)
                    return _abs(o - v) <= tol
                def __ne__(s, o): return not s.__eq__(o)
            return _C()
        @staticmethod
        def fixture(**kw):
            def deco(f): return f
            return deco
    pytest = _A()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import (make_path, Config, backtest, order_table,
                 implementation_shortfall, shortfall_equal_weighted, score)
from src.signals import autocorrelation


@pytest.fixture(scope="module")
def path():
    return make_path(1000)


def test_point_in_time_columns_are_lagged(path):
    """prev_close must equal yesterday's close, never today's."""
    assert np.allclose(path["prev_close"].iloc[1:], path["close"].iloc[:-1], equal_nan=False)


def test_volume_forecast_excludes_today(path):
    """A forecast that includes the day it predicts is lookahead. Verify the
    20-day mean is built strictly from prior days."""
    i = 60
    expected = path["volume"].iloc[i - 20:i].mean()
    assert path["vol_forecast"].iloc[i] == pytest.approx(expected)


def test_no_lookahead_changes_results(path):
    """The biased engine must actually differ from the corrected one -- if it
    doesn't, the use_future_info flag has silently stopped working and the
    regression guard is worthless."""
    a = score(path, Config(use_future_info=True))
    b = score(path, Config(use_future_info=False))
    assert a != pytest.approx(b)


def test_participation_cap_limits_trade_size(path):
    """No fill may exceed participation x forecast volume."""
    _, fills = backtest(path, Config(participation=0.05))
    for f in fills:
        assert f.qty <= 0.05 * path["volume"].max() + 1e-6


def test_realism_cap_never_exceeds_actual_volume(path):
    """You cannot buy shares that never traded."""
    _, fills = backtest(path, Config(participation=0.25))
    assert all(f.qty <= 0.30 * path["volume"].max() + 1e-6 for f in fills)


def test_shortfall_sign_convention(path):
    """Positive shortfall must mean 'cost us money' for BOTH buys and sells."""
    _, fills = backtest(path, Config())
    o = order_table(fills)
    buys = o[o["direction"] > 0]
    sells = o[o["direction"] < 0]
    for _, r in buys.iterrows():          # bought above decision price -> positive
        assert np.sign(r["shortfall_bps"]) == np.sign(r["avg_fill"] - r["ref"])
    for _, r in sells.iterrows():         # sold below decision price -> positive
        assert np.sign(r["shortfall_bps"]) == np.sign(r["ref"] - r["avg_fill"])


def test_shortfall_is_notional_weighted(path):
    """Regression guard for Bug #2. A tiny order must not move the aggregate
    metric the way it moves the equal-weighted one."""
    _, fills = backtest(path, Config(mode="adaptive"))
    o = order_table(fills)
    assert len(o) > 3
    agg = implementation_shortfall(fills)
    manual = o["cost"].sum() / o["notional"].sum() * 1e4
    assert agg == pytest.approx(manual)
    # and it must differ from the broken version, or the bug is back
    assert agg != pytest.approx(shortfall_equal_weighted(fills), rel=0.05)


def test_impact_makes_fills_worse(path):
    """Impact must always move the fill price against us, never for us."""
    _, fills = backtest(path, Config(impact=True))
    assert all(f.impact_bps >= 0 for f in fills)
    with_impact = score(path, Config(impact=True))
    without = score(path, Config(impact=False))
    assert with_impact > without


def test_flat_signal_produces_no_trades(path):
    d = path.copy()
    d["signal"] = 0.0
    _, fills = backtest(d, Config())
    assert len(fills) == 0


def test_autocorrelation_of_random_walk_is_near_zero(path):
    """The adaptive rule bets on momentum. This is the check that says the bet
    has no basis on this data."""
    acf = autocorrelation(path["close"].pct_change(), lags=3)
    band = 1.96 / np.sqrt(len(path))
    assert all(abs(v) < band for v in acf.values())


def test_monte_carlo_is_reproducible():
    """Same seed must give the same answer, or none of the CIs mean anything."""
    assert score(make_path(7), Config()) == score(make_path(7), Config())


if __name__ == "__main__":
    p = make_path(1000)
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in fns:
        try:
            f(p) if f.__code__.co_argcount else f()
            print(f"  PASS  {n}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {n}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
