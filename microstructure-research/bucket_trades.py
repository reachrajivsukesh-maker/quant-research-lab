import pandas as pd
import numpy as np
import glob, os

files = sorted(glob.glob(os.path.expanduser("~/mnt/Documents/qrl/microstructure-research/raw/*.csv")))
print("files found:", len(files))

cols = ["agg_id","price","qty","first_id","last_id","ts_us","is_buyer_maker","is_best_match"]
all_buckets = []

for fp in files:
    df = pd.read_csv(fp, header=None, names=cols)
    df["ts"] = pd.to_datetime(df["ts_us"], unit="us")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.strip().str.lower() == "true"
    # is_buyer_maker True -> resting order was a buy -> the AGGRESSOR (taker) was the SELLER
    # is_buyer_maker False -> the AGGRESSOR (taker) was the BUYER
    df["aggressor_buy_qty"]  = np.where(~df["is_buyer_maker"], df["qty"], 0.0)
    df["aggressor_sell_qty"] = np.where(df["is_buyer_maker"], df["qty"], 0.0)
    df["minute"] = df["ts"].dt.floor("1min")

    g = df.groupby("minute").agg(
        buy_vol=("aggressor_buy_qty", "sum"),
        sell_vol=("aggressor_sell_qty", "sum"),
        close=("price", "last"),
        n_trades=("agg_id", "count"),
    ).reset_index()
    all_buckets.append(g)
    print(fp.split("/")[-1], "->", len(g), "minute buckets,", len(df), "raw trades")

out = pd.concat(all_buckets, ignore_index=True).sort_values("minute").drop_duplicates(subset="minute")
out["total_vol"] = out["buy_vol"] + out["sell_vol"]
out["ofi"] = (out["buy_vol"] - out["sell_vol"]) / out["total_vol"]

outpath = os.path.expanduser("~/mnt/Documents/qrl/microstructure-research/btc_1min_ofi.csv")
out.to_csv(outpath, index=False)
print("TOTAL minute buckets:", len(out))
print("date range:", out["minute"].min(), "to", out["minute"].max())
print("total raw trades processed:", out["n_trades"].sum())
print("saved to", outpath, "size:", os.path.getsize(outpath), "bytes")
