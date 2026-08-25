# Findings, Part III — regimes, an inverted rule, and a metric that was wrong

Appended after Parts I and II. This part contains the study's largest correction:
**the headline result of Part I is a statement about one metric, and it reverses
under another.**

---

## 1. Does the adaptive rule work in any market regime?

Swept a momentum knob across six regimes, 200 paths each, measured by
implementation shortfall (`experiments/08_regime_sweep.py`):

```
                  market    rho    ADAPTIVE (old)    INVERTED (new)
   strong mean reversion  -0.20           +17.15*            -7.77*
     mild mean reversion  -0.10           +19.07*           -10.07*
             random walk   0.00           +20.71*            -8.02*
           mild momentum   0.10           +24.13*            -7.74*
         strong momentum   0.20           +25.58*           -10.82*
    very strong momentum   0.35           +28.96*            -9.19
```

Prediction before running: the rule is a momentum bet, so it should pay once
momentum exists. **Falsified** — it loses everywhere and the penalty *grows*
monotonically with momentum.

Cross-regime ablation confirms the ease component is the culprit in every regime
(+18.85 → +32.70 as momentum rises) while the ramp is ~0 throughout.

**Four mechanism hypotheses were tested and all four failed:** dips reversing
during uptrends (measured the opposite), buy/sell asymmetry (both sides equal),
adaptive filling later (it fills *earlier*), ease-only filling later (delays don't
track penalties). **The effect is robust; the mechanism is unresolved.** That is a
legitimate position and better than a fifth plausible story.

---

## 2. The inverted rule

Proposed by Rajiv: invert the logic. Speed up when the market moves *for* you
(grab the discount), slow down when it moves *against* you (stop chasing).
Implemented in `experiments/09_inverted_rule.py`.

**It wins in every regime, by 8–11 bps on shortfall.** A ~30 bps swing from the
original.

Critically it is *not* just trading faster — its average participation is **3.4%**,
slower than the 5% baseline. A constant 3.4% rate costs +27 to +56 bps more than
fixed 5%. So the gap between inverted and a speed-matched constant is 32–57 bps of
genuine day-selection.

---

## 3. The correction: shortfall and P&L disagree

Rajiv asked whether P&L versus buy-and-hold is the metric that actually matters.
Testing it reversed the study's headline.

Paired P&L differences, 200 paths (`experiments/11_pnl_significance.py`):

```
    rho            adaptive - fixed            inverted - fixed
  -0.20        +0.50* [+0.25,+0.75]        +0.79* [+0.37,+1.20]
   0.00        +0.49* [+0.20,+0.77]        +0.78* [+0.27,+1.29]
   0.20        +0.41* [+0.08,+0.75]        +0.64  [-0.01,+1.28]
   0.35        +0.82* [+0.40,+1.24]        +0.30  [-0.47,+1.06]
```

**Adaptive makes significantly MORE money than fixed in all four regimes**, while
having significantly WORSE fill prices in all four. The two metrics rank the rules
in opposite orders.

### Why — and it deflates the P&L result too

Cost decomposition (`experiments/12_cost_decomposition.py`):

```
    rho               total shares traded               impact per share (Rs)
                 fixed    adaptive    inverted       fixed    adaptive    inverted
   0.00      1,750,707   1,613,238   1,518,899       0.207       0.208       0.178
```

Adaptive trades **8% fewer shares at identical cost per share.** Its P&L advantage
is churn reduction, not execution skill — it reaches the target faster and so does
less repeated rebalancing.

Inverted trades 13% fewer shares **and** genuinely improves cost per share
(0.178 vs 0.207). That one is a real execution effect.

### The corrected claim

> Adaptive execution achieves systematically worse fill prices than a constant
> rate (+20 bps, robust across regimes) while producing modestly better total P&L
> (+0.4 to +0.8 pp) purely through reduced turnover. The two effects point in
> opposite directions.

---

## 4. Why shortfall alone was the wrong verdict

Shortfall measures the price you paid. It does not measure whether you ended up
holding what the strategy asked for.

```
    rho   signal edge/day   exposure: fixed   exposure: inverted   P&L gain
  -0.20          -0.0187%             0.706                0.643      +0.80%
   0.00          -0.0075%             0.706                0.642      +0.61%
   0.20          +0.0103%             0.707                0.646      +0.43%
   0.35          +0.0333%             0.707                0.639      -0.02%
```

Inverted is always ~6 points under-invested. When the signal is worthless that is
free; as the signal gains edge it costs more, and the P&L advantage decays to zero.

> An execution improvement measured in shortfall is only worth what it claims **if
> being out of position is cheap.** A patient execution algorithm is nearly free
> for a mediocre strategy and expensive for a good one.

**This is the fourth instance in this project of judging something by a metric
adjacent to what actually matters** — after averaging ratios instead of money,
and forecast accuracy instead of execution cost. Report execution and portfolio
metrics together, always; a disagreement between them is itself the finding.

---

## 5. Terminology correction

"Calibrated to real markets" in Part II meant realistic **volatility and volume**
dynamics. Real daily returns have autocorrelation ~0.00 — real markets *are*
approximately random walks in direction, so there was never directional
predictability to add. Verified on real prices (`real-data/ma_on_real.py`): the
strategy beat buy-and-hold on **0 of 12** window/stock combinations.
