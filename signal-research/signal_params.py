"""UNIT 5, FINAL TOPIC -- are the 5/20 MA windows any good, and would
searching for better ones actually help?

The setup is deliberately rigged so we KNOW the right answer: paths are pure
random walks with ZERO drift. No moving-average rule can have an edge on them.
So any in-sample profit we find is 100% noise, and we can measure exactly how
much of it survives out of sample. Answer should be: none.
"""
import numpy as np, pandas as pd

COST = 0.0015          # cost per side, charged when the position flips

def path(seed, n=600):
    """zero-drift random walk -- by construction there is NO edge to find"""
    rng = np.random.default_rng(seed)
    return 100.0*np.exp(np.cumsum(rng.normal(0.0, 0.014, n)))

def strategy_return(close, short, long):
    """Long when fast MA > slow MA, else flat. Returns total % over the period."""
    s = pd.Series(close)
    sig = (s.rolling(short).mean() > s.rolling(long).mean()).astype(float)
    sig[s.rolling(long).mean().isna()] = 0.0
    pos = sig.shift(1).fillna(0.0)                 # act tomorrow on today's signal
    ret = s.pct_change().fillna(0.0)
    gross = (pos*ret)
    flips = pos.diff().abs().fillna(0.0)           # cost only when we actually trade
    net = gross - flips*COST
    return float((1+net).prod() - 1)*100

SHORTS = [2, 3, 5, 8, 10, 15]
LONGS  = [10, 20, 30, 50, 80, 100]
GRID   = [(s, l) for s in SHORTS for l in LONGS if s < l]
print(f"Grid: {len(GRID)} moving-average combinations")
print(f"Truth: zero-drift random walks -- the correct answer is that NONE of them work.\n")

N_PATHS = 200
rows = []
for i in range(N_PATHS):
    c = path(7000+i)
    train, test = c[:300], c[300:]
    tr = {g: strategy_return(train, *g) for g in GRID}
    best = max(tr, key=tr.get)
    rows.append(dict(
        path=i,
        best_params=f"{best[0]}/{best[1]}",
        train_best=tr[best],
        test_best=strategy_return(test, *best),
        train_default=tr[(5,20)],
        test_default=strategy_return(test, 5, 20),
        train_avg=np.mean(list(tr.values())),
    ))
df = pd.DataFrame(rows)
df.to_csv("signal_params_results.csv", index=False)

print("="*74)
print("WHAT THE SEARCH FINDS IN-SAMPLE vs WHAT IT DELIVERS OUT-OF-SAMPLE")
print("="*74)
def line(label, x):
    se = x.std(ddof=1)/np.sqrt(len(x))
    print(f"{label:<44}{x.mean():>+9.2f}%  95% CI [{x.mean()-1.96*se:+6.2f}, {x.mean()+1.96*se:+6.2f}]")

line("BEST params, measured in-sample",        df["train_best"])
line("SAME params, applied out-of-sample",     df["test_best"])
print()
line("default 5/20, in-sample",                df["train_default"])
line("default 5/20, out-of-sample",            df["test_default"])
print()
line("average of ALL 30 combos, in-sample",    df["train_avg"])

print(f"""
The in-sample winner averages {df['train_best'].mean():+.2f}% -- looks like a real strategy.
Out of sample the same parameters give {df['test_best'].mean():+.2f}%.
{df['train_best'].mean() - df['test_best'].mean():.2f} percentage points of the apparent edge was pure noise.
""")

print("="*74)
print("DOES PICKING THE BEST BEAT JUST USING THE DEFAULT?")
print("="*74)
d = df["test_best"] - df["test_default"]
se = d.std(ddof=1)/np.sqrt(len(d))
print(f"   searched-best minus default, out-of-sample: {d.mean():+.2f}%")
print(f"   95% CI [{d.mean()-1.96*se:+.2f}, {d.mean()+1.96*se:+.2f}]")
print(f"   -> {'searching HELPED' if d.mean()-1.96*se > 0 else 'searching was WORTHLESS (zero is inside)'}")
print(f"   the search picked a different winner on {df['best_params'].nunique()} of {N_PATHS} paths -- "
      f"no stable answer exists")
print("\n   most common 'best' parameters found:")
print(df["best_params"].value_counts().head(5).to_string())
