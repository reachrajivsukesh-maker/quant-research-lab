# Time-Series Data Engineering

pandas mechanics for financial data, and one demonstration that matters more than
the rest.

| Script | What it shows |
|---|---|
| `build_sample_data.py` | Synthetic 300-day OHLCV on a business-day `DatetimeIndex` |
| `datetime_index_demo.py` | Date-based slicing and resampling (`.resample("W-FRI")`) |
| `rolling_windows.py` | Rolling means and volatility, including the NaN warm-up and why `.rolling()` is lookahead-safe by default |
| `vwap_demo.py` | Volume-weighted average price vs a plain moving average |
| **`timestamp_alignment_bug.py`** | **The important one** |

## The lookahead demo

Quarterly earnings merged onto a price series two ways, using
`pd.merge_asof(..., direction="backward")`:

- keyed on **`period_end`** — leaks the earnings surprise **25 days before it was
  published**
- keyed on **`release_date`** — correctly shows `NaN` until the announcement

The wrong version looks completely normal and produces a backtest that appears to
work. This is the same class of error as Bug #1 in the main project, which is why
it is here: point-in-time correctness is a discipline, not a one-time check.
