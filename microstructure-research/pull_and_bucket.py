#!/usr/bin/env python3
"""
Pull daily trade dumps from Binance's public archive (data.binance.vision --
no API key needed) for one or more symbols and a date range, and bucket them
into 1-minute files with a RICHER feature set than the original OFI-only
convention: per-side volume AND notional, per-side trade counts, average
trade size per side. Run this on your own machine (Terminal.app) -- Binance
is blocked from every Claude-controlled path.

Requires: pip install pandas requests

Usage:
    python3 pull_and_bucket.py --symbols ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT \
        --start 2026-08-31 --end 2026-09-25 --outdir ./breadth_data

Output: one file per symbol, e.g. ETHUSDT_1min_ofi.csv, with columns:
    minute, buy_qty, sell_qty, total_qty, buy_notional, sell_notional,
    total_notional, n_buy_trades, n_sell_trades, n_trades, close,
    ofi_qty, ofi_notional, avg_buy_trade_size, avg_sell_trade_size

ofi_qty matches the original convention ((buy_qty-sell_qty)/total_qty);
ofi_notional is the same idea in dollar terms, the fairer cross-asset measure.

Raw daily dumps are cached in <outdir>/raw/ (zip + extracted csv) so a
re-run or a symbol you add later doesn't re-download days you already have.
Each day's raw file is ~tens to hundreds of MB for a liquid pair -- expect
this to take a while and use real disk space; delete <outdir>/raw/ afterwards
if you want it back (the bucketed files are what actually matters).
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.binance.vision/data/spot/daily/trades"
# Binance's public dumps have historically shipped as headerless CSVs with
# these columns; some newer dumps include a header row. Handle both.
COLUMNS = ["id", "price", "qty", "quoteQty", "time", "isBuyerMaker", "isBestMatch"]


def daterange(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d")
    d1 = datetime.strptime(end, "%Y-%m-%d")
    d = d0
    while d <= d1:
        yield d.strftime("%Y-%m-%d")
        d += timedelta(days=1)


def download_day(symbol: str, day: str, raw_dir: Path) -> Path | None:
    """Downloads one day's trade dump zip if not already cached; returns the
    path to the extracted CSV, or None if that day doesn't exist (e.g. a
    symbol that didn't trade yet, or a request beyond today)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{symbol}-trades-{day}.csv"
    if csv_path.exists():
        return csv_path

    url = f"{BASE_URL}/{symbol}/{symbol}-trades-{day}.zip"
    resp = requests.get(url, timeout=60)
    if resp.status_code != 200:
        print(f"  [{symbol} {day}] not found (HTTP {resp.status_code}) -- skipping", file=sys.stderr)
        return None

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        inner_name = zf.namelist()[0]
        zf.extract(inner_name, raw_dir)
        extracted = raw_dir / inner_name
        if extracted != csv_path:
            extracted.rename(csv_path)
    return csv_path


def load_trades(csv_path: Path) -> pd.DataFrame:
    # Peek at the first line to detect a header row (headerless files start
    # with a numeric trade id).
    with open(csv_path) as f:
        first_line = f.readline().strip()
    has_header = not first_line.split(",")[0].isdigit()

    if has_header:
        df = pd.read_csv(csv_path)
        df.columns = [c[0].lower() + c[1:] for c in df.columns]  # normalize case
    else:
        df = pd.read_csv(csv_path, header=None, names=COLUMNS)

    # isBuyerMaker: True means the BUY side was resting (maker) and a SELL
    # order came in and matched it -- so this trade was SELLER-initiated.
    # False means a BUY order was the aggressor.
    df["isBuyerMaker"] = df["isBuyerMaker"].astype(str).str.lower().isin(["true", "1"])

    # time can be ms or us epoch depending on dump vintage -- ms is ~13
    # digits, us is ~16; detect from magnitude rather than assuming.
    sample_t = df["time"].iloc[0]
    unit = "us" if sample_t > 10**14 else "ms"
    df["ts"] = pd.to_datetime(df["time"], unit=unit)
    return df


def bucket_1min(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("ts")
    df["is_buy_aggr"] = ~df["isBuyerMaker"]  # aggressor was the buyer
    df["buy_qty"] = df["qty"].where(df["is_buy_aggr"], 0.0)
    df["sell_qty"] = df["qty"].where(~df["is_buy_aggr"], 0.0)
    df["buy_notional"] = df["quoteQty"].where(df["is_buy_aggr"], 0.0)
    df["sell_notional"] = df["quoteQty"].where(~df["is_buy_aggr"], 0.0)
    df["n_buy"] = df["is_buy_aggr"].astype(int)
    df["n_sell"] = (~df["is_buy_aggr"]).astype(int)

    agg = df.resample("1min").agg(
        buy_qty=("buy_qty", "sum"), sell_qty=("sell_qty", "sum"),
        buy_notional=("buy_notional", "sum"), sell_notional=("sell_notional", "sum"),
        n_buy_trades=("n_buy", "sum"), n_sell_trades=("n_sell", "sum"),
        close=("price", "last"), n_trades=("price", "size"),
    )
    agg = agg[agg["n_trades"] > 0].copy()
    agg["total_qty"] = agg["buy_qty"] + agg["sell_qty"]
    agg["total_notional"] = agg["buy_notional"] + agg["sell_notional"]
    agg["ofi_qty"] = (agg["buy_qty"] - agg["sell_qty"]) / agg["total_qty"]
    agg["ofi_notional"] = (agg["buy_notional"] - agg["sell_notional"]) / agg["total_notional"]
    agg["avg_buy_trade_size"] = (agg["buy_notional"] / agg["n_buy_trades"]).where(agg["n_buy_trades"] > 0)
    agg["avg_sell_trade_size"] = (agg["sell_notional"] / agg["n_sell_trades"]).where(agg["n_sell_trades"] > 0)

    agg = agg.reset_index().rename(columns={"ts": "minute"})
    cols = ["minute", "buy_qty", "sell_qty", "total_qty", "buy_notional", "sell_notional",
            "total_notional", "n_buy_trades", "n_sell_trades", "n_trades", "close",
            "ofi_qty", "ofi_notional", "avg_buy_trade_size", "avg_sell_trade_size"]
    return agg[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="comma-separated, e.g. ETHUSDT,SOLUSDT")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--outdir", default="./breadth_data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    raw_dir = outdir / "raw"
    outdir.mkdir(parents=True, exist_ok=True)

    for symbol in [s.strip().upper() for s in args.symbols.split(",")]:
        print(f"\n=== {symbol} ===")
        day_frames = []
        for day in daterange(args.start, args.end):
            print(f"  fetching {day}...", end=" ", flush=True)
            csv_path = download_day(symbol, day, raw_dir)
            if csv_path is None:
                continue
            trades = load_trades(csv_path)
            bucketed = bucket_1min(trades)
            day_frames.append(bucketed)
            print(f"{len(trades):,} trades -> {len(bucketed)} minutes")

        if not day_frames:
            print(f"  no data collected for {symbol}, skipping output file")
            continue

        full = pd.concat(day_frames, ignore_index=True).sort_values("minute")
        out_path = outdir / f"{symbol}_1min_ofi.csv"
        full.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(full)} total minutes)")

    print(f"\nDone. Send the *_1min_ofi.csv files in {outdir} back over -- "
          f"the raw/ subfolder is safe to delete once you've confirmed the outputs look right.")


if __name__ == "__main__":
    main()
