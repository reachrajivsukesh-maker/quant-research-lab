"""Untangling which data was used where, and what 'realistic' actually meant."""
import numpy as np, pandas as pd
from liquidity_test import AAPL
from adaptive_in_momentum import path_mom
from unit5_core import make_path

def acf(x,k=1):
    x=np.asarray(x,float); x=x-x.mean(); return float((x[k:]*x[:-k]).sum()/(x**2).sum())

print("="*80)
print("THREE generators were used at different points. They are NOT the same.")
print("="*80)
aapl = np.array([r[0] for r in AAPL]); r_aapl = np.diff(np.log(aapl))
d0 = make_path(1000); r0 = np.diff(np.log(d0["close"].values))
dm = path_mom(9000, 0.0); rm = np.diff(np.log(dm["close"].values))

print(f"""
   1. make_path        -- the ORIGINAL. Random walk + iid volume.
                          Used for the main execution study.
   2. calibrated gen   -- added volatility clustering, volume persistence,
                          volume/volatility coupling, fitted to AAPL/KO/SPY.
                          Used ONLY for the real-data validation section.
   3. path_mom         -- random walk with a tunable momentum knob, iid volume.
                          Used for EVERYTHING in the last several messages.

   Point 3 is the thing I did not make clear: the recent regime sweeps,
   the inverted rule, the P&L comparison -- all of those ran on generator 3,
   NOT the calibrated one.
""")
print("="*80)
print("BUT here is the more important point: what did 'calibrated' actually match?")
print("="*80)
print(f"\n{'series':>34}{'|ret| ACF':>13}{'RETURN ACF':>14}")
print(f"{'':>34}{'(clustering)':>13}{'(direction)':>14}")
print("-"*61)
print(f"{'REAL AAPL':>34}{acf(np.abs(r_aapl)):>13.3f}{acf(r_aapl):>14.3f}")
print(f"{'original make_path':>34}{acf(np.abs(r0)):>13.3f}{acf(r0):>14.3f}")
print(f"{'path_mom (rho=0)':>34}{acf(np.abs(rm)):>13.3f}{acf(rm):>14.3f}")
print(f"\n   noise band for these samples: about +/- {1.96/np.sqrt(len(r_aapl)):.3f}")
print("""
   REAL AAPL's RETURN autocorrelation is ~0.006 -- indistinguishable from zero.

   THAT IS THE POINT. Real markets ARE approximately random walks in DIRECTION.
   Tomorrow's up-or-down is not predictable from today's. This is the efficient
   market property, and it is why the daily-bar world is so hard.

   What real markets have that our first generator lacked is structure in
   VOLATILITY and VOLUME -- not in direction:
       |ret| ACF   : real 0.03-0.14   vs   original ~0.00   <- we fixed this
       volume ACF  : real 0.37-0.64   vs   original ~0.00   <- we fixed this
       RETURN ACF  : real ~0.00       vs   original ~0.00   <- nothing to fix

   So 'made it realistic' meant: realistic volatility and volume behaviour.
   It did NOT mean, and could not mean, 'gave it a predictable direction'.
""")
