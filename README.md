# Quant Research Lab

Self-directed research toward quantitative trading, from a mechanical engineering
background.

**Rajiv Sukesh** · B.Tech Mechanical Engineering, IIT Madras

---

## [→ Execution Cost Research](execution-research/)

A backtesting engine and an execution-algorithm study. Findings are documented
across three parts in [`docs/`](execution-research/docs/), including every
measurement error that had to be corrected to reach them.

| | what went wrong | effect |
|---|---|---|
| Bug #1 | order sizing used the day's *realised* volume — unknowable at decision time | lookahead bias in both engines |
| Bug #2 | cost metric averaged per-order ratios; a 0.1-share order outweighed a 144,386-share order | true cost understated **6.7×** |
| Bug #3 | no market-impact model, so the metric silently rewarded pure speed | "optimal" answer was always "trade fastest" |
| Bug #4 | warm-up asymmetry in the train/test split | overfitting gap inflated 17% |
| Retraction #1 | headlined a mean over 3 folds where one contributed 78% | withdrawn |
| Retraction #2 | read an interior optimum off a single path | withdrawn |
| Retraction #3 | judged a component by forecast accuracy, not by its cost | withdrawn |
| Correction #4 | reported a shortfall result as *the* finding; P&L reverses it | claim narrowed |

## [→ Signal Parameter Research](signal-research/)

Searching 34 moving-average window pairs on data built to contain no edge produces
**+11.4% in-sample and −1.6% out-of-sample**. The illusion scales with how hard
you look. In-sample ranking predicts out-of-sample ranking at **+0.003** — and
stays there even when a real edge is planted.

> Establishing that a strategy *type* works does not license fine-tuning its
> parameters.

## [→ Real Market Data](real-data/)

Live data from Twelve Data, used to check the assumptions the simulations rest on.
Real daily returns have autocorrelation ~0.00 — markets are approximately random
walks in *direction*; their structure is in volatility and volume. The strategy
beat buy-and-hold on **0 of 12** real stock/window combinations.

## Supporting work

| Directory | What it contains |
|---|---|
| [`time-series/`](time-series/) | Stationarity, autocorrelation and the ADF test derived by hand |
| [`data-engineering/`](data-engineering/) | pandas mechanics; a `merge_asof` lookahead demo that leaks earnings 25 days early |
| [`stochastic-processes/`](stochastic-processes/) | Brownian motion and a volatility signature plot |
| [`interactive/`](interactive/) | Browser-based trading simulator and study tools |

## Stack

Python · pandas · NumPy · Matplotlib · pytest

## What this is

A learning record, deliberately honest about being one. Several results are
negative, four claims were corrected or retracted after being made, and the
limitations sections are written to be read.

The judgement being demonstrated is not "I built a strategy that works" — it is
that I can find the reasons one doesn't, including in my own measurements.
