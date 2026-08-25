"""The cost metric was averaging per-order percentages.

A 7-share order and a 364,000-share order each got one vote. Equal-weighting
understated true execution cost by roughly 3x. The fix is not a clever weighting
scheme -- it is to stop averaging ratios: aggregate the rupees, divide once.
"""
import _common
import pandas as pd
from src import make_path, Config, backtest, order_table, \
    implementation_shortfall, shortfall_equal_weighted

d = make_path(1000)
_, fills = backtest(d, Config(mode="adaptive"))

orders = order_table(fills).sort_values("qty")
orders_out = orders.assign(
    qty=orders["qty"].round(1), ref=orders["ref"].round(2),
    avg_fill=orders["avg_fill"].round(4),
    shortfall_bps=orders["shortfall_bps"].round(2),
    cost=orders["cost"].round(0), notional=orders["notional"].round(0))
print(orders_out.to_string(index=False))
orders_out.to_csv(f"{_common.RESULTS}/03_order_breakdown.csv", index=False)

eq = shortfall_equal_weighted(fills)
ag = implementation_shortfall(fills)
print(f"\nequal-weighted mean of per-order bps : {eq:8.2f} bps   <- BROKEN")
print(f"rupee-aggregate (cost / notional)   : {ag:8.2f} bps   <- correct")
print(f"understatement factor               : {ag/eq:8.2f}x")

smallest = orders.nsmallest(1, "qty").iloc[0]
largest = orders.nlargest(1, "qty").iloc[0]
print(f"\nsmallest order: {smallest['qty']:>12,.1f} shares, cost {smallest['cost']:>12,.0f}")
print(f"largest  order: {largest['qty']:>12,.1f} shares, cost {largest['cost']:>12,.0f}")
print("Under equal weighting these two carried identical weight.")
