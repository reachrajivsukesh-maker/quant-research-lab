"""FIX: compute the moving averages on the FULL series, THEN split.

The bug: I sliced the price series first and computed moving averages on each
half separately. That restarts the warm-up, so a 100-day slow window sat in
cash for the first 99 of the 300 test days -- a third of the window.

The fix is also more realistic: in reality you always have price history from
before your test period, so your averages are already warmed up on day one.
"""
import numpy as np, pandas as pd
from signal_params import path, GRID, COST

def signal_full(close, short, long):
    """moving averages over the WHOLE series -- no restart at the split"""
    s = pd.Series(close)
    sig = (s.rolling(short).mean() > s.rolling(long).mean()).astype(float)
    sig[s.rolling(long).mean().isna()] = 0.0
    return sig

def ret_from(close, sig, lo, hi):
    """return over rows [lo:hi] using a signal computed on the full series"""
    s = pd.Series(close)
    pos = sig.shift(1).fillna(0.0).iloc[lo:hi]
    r   = s.pct_change().fillna(0.0).iloc[lo:hi]
    net = pos*r - pos.diff().abs().fillna(0.0)*COST
    return float((1+net).prod()-1)*100

rows = []
for i in range(200):
    c = path(7000+i)
    sigs = {g: signal_full(c, *g) for g in GRID}
    tr = {g: ret_from(c, sigs[g], 100, 300) for g in GRID}   # skip 100-day warm-up once
    b  = max(tr, key=tr.get)
    rows.append(dict(path=i, winner=f"{b[0]}/{b[1]}", in_sample=tr[b],
                     out_of_sample=ret_from(c, sigs[b], 300, 600),
                     default_oos=ret_from(c, sigs[(5,20)], 300, 600)))
df = pd.DataFrame(rows)

print("="*74)
print("RE-RUN WITH THE WARM-UP FIXED")
print("="*74)
def rep(lbl, x):
    se = x.std(ddof=1)/np.sqrt(len(x))
    print(f"   {lbl:<34}{x.mean():+8.2f}%   median {x.median():+7.2f}%   "
          f"95% CI [{x.mean()-1.96*se:+.2f}, {x.mean()+1.96*se:+.2f}]")
rep("BEST params, in-sample",        df["in_sample"])
rep("SAME params, out-of-sample",    df["out_of_sample"])
rep("default 5/20, out-of-sample",   df["default_oos"])

d = df["out_of_sample"] - df["default_oos"]
se = d.std(ddof=1)/np.sqrt(len(d))
print(f"\n   searched minus default, out-of-sample: {d.mean():+.2f}%  "
      f"95% CI [{d.mean()-1.96*se:+.2f}, {d.mean()+1.96*se:+.2f}]")
print(f"   searched beat default on {(d>0).mean()*100:.1f}% of paths")
print(f"   distinct winners: {df['winner'].nunique()} of {len(GRID)}; "
      f"most frequent won {df['winner'].value_counts().iloc[0]/2:.1f}% of the time")

print(f"""
{"="*74}
COMPARISON: does the conclusion change?
{"="*74}
                                    BUGGY version      FIXED version
   best in-sample                      +14.13%          {df['in_sample'].mean():+8.2f}%
   same params out-of-sample            -1.39%          {df['out_of_sample'].mean():+8.2f}%
   gap (the overfitting)                 15.52           {df['in_sample'].mean()-df['out_of_sample'].mean():8.2f}
   searched vs default                  -0.31%          {d.mean():+8.2f}%
""")
df.to_csv("fixed_warmup_results.csv", index=False)
