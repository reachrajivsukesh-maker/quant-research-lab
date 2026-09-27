# Market Microstructure & RL Trading Research

Two connected research questions on real market data: does order-flow imbalance (OFI) predict
short-horizon returns, and can a reinforcement-learning agent exploit it once realistic
transaction costs are included? Both are answered with real trade-level data, walk-forward
discipline, and — where an initial result turned out to be fragile — an explicit correction rather
than a quiet redo.

## Order-flow imbalance (OFI) — [`unit6/`](unit6/)

**Question:** does the relative volume of aggressive buys vs. sells in the last minute predict the
next minute's return?

**Full write-up:** [`ofi_research_note.pdf`](ofi_research_note.pdf) — abstract, methods, results,
and a discussion section built around two self-corrections. Code and results below are the
supporting material; the PDF is the citable summary.

Five passes, each correcting the one before it:

| Pass | Data | Result |
|---|---|---|
| 1-2 | `signed_volume` proxy, SPY OHLCV, 13 then 52 days | Sign flipped between in-sample/out-of-sample at nearly every horizon — nothing survived. |
| 3 | Real trade-level Binance BTCUSDT, 6 days (6.19M trades), true aggressor-side OFI from `is_buyer_maker` | An apparent out-of-sample edge was first diagnosed as a momentum confound — but using plain OLS standard errors despite already-known serial correlation. |
| 4 | Same data, corrected with HAC/Newey-West standard errors | Reversed the conclusion again: OFI's coefficient looked small but stable and significant in-sample/pooled. |
| 5 | 4.3x more data — 26 days, 24.0M trades, identical pre-declared spec | **The coefficient collapsed 5-15x and lost cross-sample consistency. The Pass 4 "stable, real, underpowered" read was itself a small-sample artifact.** |

**Conclusion: no validated OFI signal, on either proxy or real trade-level data, after five tests, one
self-caught statistical bug, and one purpose-built resolving test at 4x the data.**

**Code:** `unit6/` — `analyze.py`/`analyze2.py` (proxy passes), `analyze_crypto.py` (Pass 3),
`analyze_crypto_hac.py` (Pass 4, HAC correction), `analyze_crypto_26d.py` (Pass 5, the resolving
test), plus matching `chart_*.py` scripts. **Results:** `unit6/results/` — regression tables,
decile analysis, and the charts referenced in the PDF.

## Can an RL agent trade this, net of real costs? — [`unit7/`](unit7/)

**Question:** if OFI has any exploitable structure at all, does a tabular Q-learning agent, trained
and tested walk-forward on the same real data, turn it into a strategy that survives Binance's
actual trading fees?

**Result, in one line: no — checked five independent ways, across 9 assets, confirmed robust to
random seed.**

1. **Design & baseline** (fixed-α vs. Kalman-style adaptive-α Q-learning; 27-state, 3-action) —
   benchmarked on the same backtester/cost model as the [execution cost research](../execution-research/),
   first on synthetic regimes, then corrected to the real BTCUSDT data once flagged that the
   synthetic pass wasn't the actual target.
2. **Real BTCUSDT, cost sweep (0/4/15bps):** at any realistic cost, both Q-learning variants
   unanimously return exactly 0.00%, across 20 training seeds.
3. **Mechanism, in the literal Q-table:** entry cost (7.5-10bps, Binance's live fee schedule) is
   25-35x the best per-state signal (0.29bps) — Q-learning correctly drives every non-flat action's
   value below Q(flat)=0. Swept bar size 1→60 minutes: cost matters *more*, not less, at finer
   granularity; no horizon clears cost with adequate sample support.
4. **Breadth test, 9 liquid pairs, real data:** realistic-cost null result generalizes to all 9
   assets (45 independent training runs, zero activity every time). The frictionless
   diversification/Grinold's-Law mechanism check needed a second pass — a single-seed run was
   uninterpretable, so it was re-run with 5-8 independent seeds per symbol. Most apparent
   single-seed "activity" washed out to noise once averaged; the one symbol that looked like a
   real reproducible signal (BNB, positive in 6/6 active seeds) had a computed ~9% probability of
   arising by pure chance once corrected for scanning 9 symbols — not below any real significance
   bar, and also one of the 9 that showed zero activity at realistic cost anyway.

**Code:** `unit7/` — `unit7_core.py` (Q-learning agent), `unit7_real_btc.py` (walk-forward
train/test on real BTC), `unit7_diagnose.py` (the literal Q-table mechanism check),
`unit7_multiscale.py` (1-60min horizon sweep), `unit7_watch_trade.py` (frictionless behavioral
demonstration), `unit7_breadth.py` / `unit7_breadth_multiseed.py` (the 9-asset breadth test and its
multi-seed resolution). `pull_and_bucket.py` and the original single-symbol `bucket_trades.py`
(both one level up, alongside the shared `btc_1min_ofi.csv` dataset both studies train/test on) pull
and bucket real Binance trade data for any set of symbols — run `pull_and_bucket.py` yourself to
regenerate the per-minute data; raw daily dumps and the 8-symbol breadth-test CSVs are not committed
here, only results, to keep this repo lean and reproducible from source. **Results:**
`unit7/results/` — cost-sweep tables/chart, the position-trace and equity-curve charts, and
`breadth_test_summary.csv` (the full multi-seed breadth numbers).

## Honest framing, stated once, applies throughout
Every "0.00%" or "zero activity" result above is a *correct, structural* answer given the stated
cost assumption (Binance's real published fee schedule) — not a bug, and checked mechanistically
each time rather than asserted. Frictionless (cost=0) numbers appear only as signal-existence /
mechanism checks, explicitly never as a tradability claim. This distinction is enforced consistently
across both studies.
