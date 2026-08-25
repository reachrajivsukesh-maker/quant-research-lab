"""Does the volume-participation cap ever actually bind?

At retail size the cap is dead code -- it runs but never changes a decision.
At institutional size it binds on ~9 days in 10. Same engine, same market: the
only thing that changed is order size relative to daily volume.
"""
import _common
import pandas as pd
from src import make_path


def cap_binding_audit(d, capital, participation):
    """Walk the path recording, each day, how many shares the strategy WANTED
    versus how many the participation cap ALLOWED."""
    cash, shares = capital, 0.0
    wanted_days = bound_days = 0
    for _, row in d.iterrows():
        if pd.isna(row["vol_forecast"]) or pd.isna(row["prev_close"]):
            continue
        value = cash + shares * row["close"]
        target = (value / row["close"]) if row["signal"] == 1 else 0.0
        gap = target - shares
        if abs(gap) <= 1e-6:
            continue
        wanted_days += 1
        allowed = participation * row["vol_forecast"]
        if abs(gap) > allowed + 1e-9:
            bound_days += 1
        qty = min(abs(gap), allowed, 0.30 * row["volume"])
        direction = 1.0 if gap > 0 else -1.0
        shares += direction * qty
        cash -= direction * qty * row["close"]
    return wanted_days, bound_days


if __name__ == "__main__":
    d = make_path(1000)
    rows = []
    for capital, part, label in [(100_000.0, 0.08, "retail"),
                                 (15_000_000.0, 0.05, "institutional")]:
        wanted, bound = cap_binding_audit(d, capital, part)
        rows.append(dict(regime=label, capital=capital, cap_pct=part * 100,
                         days_wanted=wanted, days_cap_bound=bound,
                         pct_bound=round(bound / max(wanted, 1) * 100, 1)))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(f"{_common.RESULTS}/01_order_size_vs_liquidity.csv", index=False)
    print("\nNot an institution-vs-retail distinction: it is order size relative to")
    print("that stock's liquidity. An illiquid microcap flips a retail order into")
    print("the institutional regime.")
