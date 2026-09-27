import pandas as pd
import numpy as np

df = pd.read_csv('btc_1min_ofi.csv')
df['minute'] = pd.to_datetime(df['minute'])
df = df.sort_values('minute').reset_index(drop=True)

# sanity: is the 1-min grid actually continuous (no gaps)?
gaps = df['minute'].diff().dropna()
print("Gap check: expected all gaps == 60s. Value counts of gap sizes (seconds):")
print((gaps.dt.total_seconds()).value_counts().head())
print()

HORIZONS = [1, 2, 5, 10]
for h in HORIZONS:
    df[f'ret_{h}'] = (df['close'].shift(-h) - df['close']) / df['close']

def corr_ci(x, y):
    mask = (~x.isna()) & (~y.isna()) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
    return r, lo, hi, n

print(f"Total minute buckets: {len(df)}  ({df['minute'].min()} to {df['minute'].max()})")
print(f"Total raw trades represented: {df['n_trades'].sum():,}")
print()

n = len(df)
split = int(n * 0.7)
in_sample = df.iloc[:split]
out_sample = df.iloc[split:]
print(f"In-sample:  {len(in_sample)} min ({in_sample['minute'].min()} to {in_sample['minute'].max()})")
print(f"Out-sample: {len(out_sample)} min ({out_sample['minute'].min()} to {out_sample['minute'].max()})\n")

results = []
for h in HORIZONS:
    for label, sub in [('full', df), ('in-sample', in_sample), ('out-of-sample', out_sample)]:
        r, lo, hi, n_ = corr_ci(sub['ofi'], sub[f'ret_{h}'])
        results.append({'horizon_min': h, 'sample': label, 'r': round(r,4),
                         'ci_lo': round(lo,4), 'ci_hi': round(hi,4), 'n': n_})

res_df = pd.DataFrame(results)
res_df.to_csv('crypto_multi_horizon_results.csv', index=False)
print(res_df.to_string(index=False))
