"""Event-driven backtest engine.

Design rule that everything else depends on: the STRATEGY decides *what position
to hold*; the EXECUTION ALGORITHM decides *how fast to get there*. They meet in
exactly one line:

    Q = min(|target - held|, participation * volume_forecast, realism_cap)

Keeping those separable is what made the whole research programme possible --
every experiment varies one side while holding the other fixed.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

INITIAL_CAPITAL = 15_000_000.0
COST_PCT = 0.0015      # broker + exchange, round-trip fraction per side
IMPACT_C = 0.6         # square-root impact constant; literature range 0.3-1.0
REALISM_CAP = 0.30     # never fill more than 30% of the volume that ACTUALLY traded


@dataclass
class Fill:
    ref: float          # decision price -- the price when this order began
    direction: float    # +1 buy, -1 sell
    qty: float
    price: float        # price actually transacted at, impact included
    participation: float
    impact_bps: float


@dataclass
class Config:
    mode: str = "fixed"            # "fixed" | "adaptive"
    participation: float = 0.05
    k: float = 4.0                 # adaptive urgency sensitivity
    ramp: bool = True              # speed up on adverse moves
    ease: bool = True              # slow down on favourable moves
    impact: bool = True
    use_future_info: bool = False  # True reproduces the BIASED engine (see FINDINGS #1)
    capital: float = INITIAL_CAPITAL
    cost_pct: float = COST_PCT
    impact_c: float = IMPACT_C

    def __str__(self):
        if self.mode == "fixed":
            return f"fixed {self.participation*100:.0f}%"
        parts = [] if (self.ramp and self.ease) else \
                (["ramp-only"] if self.ramp else ["ease-only"])
        tag = (" " + ",".join(parts)) if parts else ""
        return f"adaptive {self.participation*100:.0f}% k={self.k:g}{tag}"


def _participation_rate(cfg, direction, price_dec, ref):
    """Fixed: a constant. Adaptive: scaled by how far the market has moved AGAINST
    the live order since it began.

    `adverse` is positive whenever the market is punishing us, for BOTH sides:
    buying into a rising price, or selling into a falling one. The `direction`
    multiplier is what makes one formula cover both.
    """
    if cfg.mode == "fixed":
        return cfg.participation, np.nan

    adverse = direction * (price_dec - ref) / ref
    u = 1.0 + (cfg.k * max(0.0, adverse) if cfg.ramp else 0.0)
    if cfg.ease:
        u *= 0.5 if adverse < -0.003 else 1.0
    u = float(np.clip(u, 0.4, 3.0))
    return float(np.clip(cfg.participation * u, 0.01, 0.25)), adverse


def backtest(d, cfg=None, signal_col="signal"):
    """Run one configuration over one price path. Returns (portfolio_value, fills)."""
    cfg = cfg or Config()
    cash, shares, ref = cfg.capital, 0.0, None
    fills, values, index = [], [], []

    for ts, row in d.iterrows():
        close = row["close"]
        sig = row[signal_col]

        # ---- the two information sources under audit --------------------
        if cfg.use_future_info:
            vol_for_cap = row["volume"]      # today's REALISED total -- not knowable yet
            price_dec = close                # same-bar decide-and-fill
        else:
            vol_for_cap = row["vol_forecast"]
            price_dec = row["prev_close"]
        sigma = row["vol_20d"]

        if pd.isna(vol_for_cap) or pd.isna(price_dec) or pd.isna(sigma) or pd.isna(sig):
            values.append(cash + shares * close); index.append(ts); continue

        value = cash + shares * close
        target = (value / close) if sig == 1 else 0.0
        gap = target - shares                      # WANTED -- strategy's output

        if abs(gap) > 1e-6:
            if ref is None:
                ref = price_dec                    # anchor the order
            direction = float(np.sign(gap))
            part, _ = _participation_rate(cfg, direction, price_dec, ref)

            allowed = part * vol_for_cap           # ALLOWED -- market's constraint
            qty = min(abs(gap), allowed, REALISM_CAP * row["volume"])
            if qty > 0:
                impact = (cfg.impact_c * sigma * np.sqrt(qty / row["volume"])
                          if cfg.impact else 0.0)
                fill_price = close * (1 + direction * impact)
                notional = qty * fill_price
                cash += (-notional - notional * cfg.cost_pct) if direction > 0 \
                    else (notional - notional * cfg.cost_pct)
                shares += direction * qty
                fills.append(Fill(ref, direction, qty, fill_price, part, impact * 1e4))
        else:
            ref = None                             # order complete, reset anchor

        values.append(cash + shares * close); index.append(ts)

    return pd.Series(values, index=index), fills


def fills_to_frame(fills):
    return pd.DataFrame([f.__dict__ for f in fills])
