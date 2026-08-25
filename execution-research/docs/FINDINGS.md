# Findings: three bugs, two retractions, one result

This document is the point of the project. The engine is ordinary; what is worth
reading is the sequence of measurement errors that each *reversed the conclusion*,
and the two claims that had to be withdrawn after being stated.

Every number below is reproduced by the scripts in `experiments/`.

---

## Bug #1 — Lookahead bias in the execution layer

**The false alarm.** The adaptive rule computes

```python
adverse = direction * (price_decision - ref_price) / ref_price
```

This looks like it needs to know where the price is heading. It doesn't:
`ref_price` is a past anchor and `price_decision` is the present, so it measures a
move that has already happened. Reactive, not predictive. Clean.

**The actual bug**, which sat in *both* the adaptive and the "blind" fixed engine:

```python
allowed = participation * volume        # volume = today's TOTAL realised volume
```

That number does not exist at the moment of the decision. On an unusually heavy
day the engine automatically traded more, having "known" in advance. A second,
subtler instance: today's close was used both to *decide* the participation rate
and as the *fill price* — same-bar decide-and-fill.

**Fix.**

```python
prev_close   = close.shift(1)                        # decide from yesterday's close
vol_forecast = volume.shift(1).rolling(20).mean()    # size from a lagged forecast
```

plus a hard `0.30 × actual_volume` backstop so forecast error cannot produce
physically impossible fills.

**Effect** (`experiments/02_lookahead_audit.py`) — the adaptive engine degrades
more than the fixed one when the future information is removed. The general
lesson: *the more inputs a strategy consumes, the more surface area it has for
leakage, and leakage always flatters.*

**Caveat, reported openly.** On this synthetic data the volume forecast correlates
**0.014** with actual volume (42.6% mean absolute error), because volume is drawn
iid. Real market volume autocorrelates ~0.6–0.8. So this *overstates* the biased
engine's advantage and handicaps the corrected one. The direction of the
conclusion holds; the magnitude is a data artifact.

---

## Bug #2 — The cost metric averaged ratios

Implementation shortfall per order is

```
shortfall_bps = direction × (avg_fill − decision_price) / decision_price × 10⁴
```

The `direction` term makes positive mean "cost us money" for buys and sells alike.

The bug was in the aggregation: the per-order percentages were averaged equally.
From `experiments/03_metric_correction.py`:

| | shares | cost |
|---|---|---|
| smallest order | 0.1 | −₹1 |
| largest order | 144,386 | ₹196,056 |

Both carried the same weight. Result:

```
equal-weighted mean of per-order bps :    24.03 bps   <- BROKEN
rupee-aggregate (cost / notional)    :   160.95 bps   <- correct
understatement factor                :      6.70x
```

**The fix is not a cleverer weighting.** It is to stop averaging ratios: aggregate
the money, then divide once.

> **Rule:** never average a ratio across units of wildly different size. Sum the
> numerators, sum the denominators, divide.

`shortfall_equal_weighted()` is deliberately kept in `src/metrics.py` so the
experiment can demonstrate the bug rather than assert it, and
`test_shortfall_is_notional_weighted` guards against its return.

---

## Bug #3 — The metric had no counterweight

Every walk-forward fold selecting the *highest* participation rate was the tell.
In a simulator that fills at the close regardless of size, faster is always better
by construction: shortfall measures drift while filling, so filling instantly
drives it to zero.

We had removed the entire reason execution algorithms exist, then asked which
execution algorithm was best.

**Fix — the square-root impact law**, which holds well empirically across markets
and asset classes:

```
impact_fraction = C × σ × √(Q / V)
fill_price      = close × (1 + direction × impact_fraction)
```

`Q/V` is the trade as a fraction of the day's volume. The square root matters:
doubling size multiplies impact by ≈1.41, not 2 — impact is sublinear. `σ` is
daily volatility; `C` ≈ 0.3–1.0 empirically, here 0.6.

---

## Retraction #1 — "tuning was worth +102 bps"

A three-fold walk-forward produced per-fold edges of roughly **128, 16, and 52
bps**. Reporting the mean implies a typical, repeatable gain. No fold delivered
the mean, and the largest single fold contributes **65%** of the total.

The mechanism, traced in the original data: one fold's test window contained a
long buy order into a sustained trend. The faster configuration finished before
the trend arrived; the slower one was still buying through it. Same signal, same
market — one config simply finished first. Luck of timing, not edge.

> **Rule:** never headline a mean over three observations where one dominates.
> Report per-fold values, the median, and the dispersion.

---

## Retraction #2 — "the optimum is around 12% participation"

That U-shape was a single-path artifact. Across 300 paths
(`experiments/05_monte_carlo.py`) the sweep falls monotonically:

```
participation  mean_bps
           3%    222.66
           5%    180.10
           8%    148.16
          12%    129.32
          16%    120.67
          20%    116.79
          25%    114.50
```

No interior optimum at all. The root cause is that `C = 0.6` makes impact too
weak to outweigh drift; a genuine Almgren–Chriss-style interior optimum only
emerges at `C ≥ 2`. So every "8% beat 5%" result was really the model saying
*speed is nearly free here* — a statement about a hand-picked constant, not about
markets.

---

## The result that survived

The adaptive rule:

```python
u = 1 + k * max(0, adverse)              # RAMP: speed up on adverse moves
u *= 0.5 if adverse < -0.003 else 1      # EASE: slow down on favourable moves
```

Three hypotheses were tested.

**H1 — "it's just slower on average": rejected.** Mean participation was 4.70 /
4.92 / 5.34% for k = 2/4/8, right around the 5% baseline. Running *fixed* at
adaptive's own average rate scored better by 21–36 bps. So holding total speed
constant, the choice of *when* to be fast was actively harmful — worse than
choosing at random.

**H2 — "both halves are bad": rejected; it is the easing rule.** Ablation over
300 paths:

| variant | mean bps | vs baseline | 95% CI | win rate |
|---|---|---|---|---|
| fixed 5% (baseline) | 180.89 | — | — | — |
| fixed 8% | 148.62 | −32.26 | [−35.01, −29.52] | 93.0% |
| adaptive k=4 | 198.04 | **+17.16** | [+15.60, +18.71] | 5.3% |
| adaptive ramp-only | 179.87 | −1.02 | [−1.83, −0.21] | 62.0% |
| adaptive ease-only | 199.23 | **+18.35** | [+16.89, +19.80] | 2.0% |

Ramp-only is trivially *better* than fixed. Easing carries essentially all the
damage.

**H3 — why easing specifically.** Three compounding reasons:

1. **Structurally asymmetric.** The ramp nudges 5.00% → ~5.6%. Easing *halves* to
   2.5%. One is a nudge, the other a cliff — equally-wrong rules would still not
   do equal damage.
2. **It buys nothing.** Its bet is that favourable conditions persist, which
   requires momentum. Measured lag-1 autocorrelation on this series is −0.007
   against a ±0.113 noise band (`test_autocorrelation_of_random_walk_is_near_zero`).
   A certain cost — more days of drift exposure — traded for a coin flip.
3. **Worse than a coin flip in-sample.** On the days we traded, the correlation
   between an adverse move and the *further* adverse move was −0.42: the rule
   slows down right before the price turns against us and speeds up right before
   it comes back. **Caveat:** this is a selection effect, not a market property —
   orders terminate when the signal flips, and the signal flips after sustained
   moves. Real for this strategy's order lifecycle; not generalisable.

**One line:** sophistication with the sign wrong is worse than no sophistication.

---

## Why Monte Carlo replaced walk-forward

More folds carved from one path would not have helped — overlapping windows are
not independent samples, and the true sample unit is *orders*, not days. Each
60-day test window held 2–3 orders, so effective n ≈ 3.

| | 3-fold walk-forward | 300-path Monte Carlo |
|---|---|---|
| observations | 3 | 300 |
| result | edges of 128 / 16 / 52 bps | +17.16 bps |
| precision | one fold = 65% of the mean | ±1.5 bps |
| conclusion | "cannot say" | "loses on 94.7% of paths" |

The comparison is **paired** — both configs run on the identical path, so
path-level luck cancels. And the **win rate is reported alongside the mean**,
because 5.3% is not a small average loss, it is a near-universal one, and a mean
alone hides that distinction.

---

## What can and cannot be claimed

**Can:** within this simulated world, the adaptive urgency rule as specified is
worse than a constant participation rate — mean +17.16 bps, 95% CI [15.60, 18.71],
losing on 94.7% of 300 independent paths, with the damage traced by ablation to
one component.

**Cannot:** anything about real markets. All 300 paths share iid returns, no
volatility clustering, no volume autocorrelation, no order-book dynamics, and a
hand-picked impact constant. **Monte Carlo eliminates sampling error, not model
error** — 300,000 paths would tighten the interval to nothing while telling you no
more about reality. And the one condition under which the adaptive rule *could*
work — genuine short-horizon momentum — is excluded by construction, so this test
was never capable of vindicating it.

---

## The meta-lesson

A strategy that looked defensible was knocked down four times: a lookahead
question, a metric-weighting bug, a missing impact model, and an underpowered
validation design. Each fix changed or reversed the conclusion.

That sequence is the work. Most of quantitative research is killing your own ideas
faster than the market kills them for you.
