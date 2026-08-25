# Real Market Data

Live daily and 1-minute bars pulled from Twelve Data. Used to check assumptions
the simulated studies rest on.

## What real markets actually have

| property | real (AAPL/KO/SPY) | our original generator |
|---|---|---|
| **return** autocorrelation (direction) | **~0.00** | ~0.00 |
| \|return\| autocorrelation (volatility clustering) | 0.03–0.14 | −0.01 |
| log-volume autocorrelation | 0.37–0.64 | −0.04 |
| corr(log volume, \|return\|) | 0.48–0.57 | 0.10 |

**The first row is the important one.** Real daily prices are approximately random
walks in *direction* — tomorrow's up-or-down is not predictable from today's. What
real markets have that our generator lacked is structure in **volatility and
volume**, not in direction.

So "calibrated to real markets" in this repo means realistic volatility and volume
dynamics. It does not mean, and could not mean, predictable direction.

## Does the strategy work on real prices? (`ma_on_real.py`)

```
AAPL, 150 real trading days.  Buy and hold: +19.59%
   5/20     +8.03%   (-11.57% vs buy&hold)   49.3% invested
   3/10    +10.13%   ( -9.46%)               57.5% invested
   8/50     +0.83%   (-18.76%)               50.0% invested
   beat buy-and-hold: 0 of 6 window choices

LAKE, 150 real trading days.  Buy and hold: +26.59%
   beat buy-and-hold: 0 of 6 window choices
```

Zero for twelve across two very different stocks. The strategy loses on real data
for the same reason it loses on simulated data — it bets on direction, and
direction is not predictable at daily frequency.

## Liquidity and the volume/volatility link (`liquidity_test.py`)

```
stock       avg volume  daily vol %  corr(V, |ret|)  noise band
AAPL        51,209,726         1.79           0.558       0.163
LAKE           119,179         3.95           0.353       0.161
```

The volume/volatility link is real in both, but **weaker in the illiquid name** —
fewer participants means noisier linkage. Note LAKE's daily volatility is more
than double AAPL's: thin markets move more for less reason.

## Why daily bars are structurally inadequate

An edge is detectable when |ACF| > 1.96/√n:

```
to detect an ACF of      you need n =
              0.10              384     ~1.5 years
              0.05            1,537     ~6 years
              0.02            9,604     use intraday
```

Real tradable edges sit around 0.01–0.05. A year of daily bars is 252
observations; a year of 1-minute bars is ~98,000. That is the entire reason
serious work happens intraday.

## Data quality warning

Stocklake's SPY volume drops ~66x from 2026-07-21 (it splices feeds with
different conventions). Twelve Data is correct. Always diff a volume series
before trusting it.
