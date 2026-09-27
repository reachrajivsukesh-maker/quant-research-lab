import pandas as pd
import numpy as np

df = pd.read_csv('spy_1min_combined.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)
df['date'] = df['datetime'].dt.date

df['signed_volume'] = np.sign(df['open'] - df['close']) * df['volume']
df['direction_only'] = np.sign(df['open'] - df['close'])
df['avg_vol_30'] = df['volume'].rolling(30, min_periods=10).mean()
df['norm_signal'] = df['signed_volume'] / df['avg_vol_30']

HORIZONS = [1, 2, 5, 10]
for h in HORIZONS:
    same_day = df['date'] == df.groupby('date')['date'].shift(-h)
    fut_close = df.groupby('date')['close'].shift(-h)
    df[f'ret_{h}'] = np.where(same_day, (fut_close - df['close']) / df['close'], np.nan)

def corr_ci(x, y):
    mask = (~x.isna()) & (~y.isna())
    x, y = x[mask], y[mask]
    n = len(x)
    r = np.corrcoef(x, y)[0, 1]
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
    return r, lo, hi, n

print(f"Combined sample: {len(df)} bars, {df['date'].nunique()} trading days "
      f"({df['date'].min()} to {df['date'].max()})\n")

n = len(df)
split = int(n * 0.7)
in_sample = df.iloc[:split]
out_sample = df.iloc[split:]
print(f"In-sample:  {in_sample['date'].nunique()} days ({in_sample['date'].min()} to {in_sample['date'].max()})")
print(f"Out-sample: {out_sample['date'].nunique()} days ({out_sample['date'].min()} to {out_sample['date'].max()})\n")

results = []
for h in HORIZONS:
    for label, sub in [('full', df), ('in-sample', in_sample), ('out-of-sample', out_sample)]:
        for sigcol, signame in [('norm_signal', 'norm_signal'), ('direction_only', 'direction_only')]:
            r, lo, hi, n_ = corr_ci(sub[sigcol], sub[f'ret_{h}'])
            results.append({'horizon_min': h, 'sample': label, 'signal': signame,
                             'r': round(r, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4), 'n': n_})

res_df = pd.DataFrame(results)
res_df.to_csv('multi_horizon_results.csv', index=False)
print("=== norm_signal correlations, all horizons ===")
print(res_df[res_df['signal'] == 'norm_signal'].to_string(index=False))
print()
print("=== direction_only correlations, all horizons ===")
print(res_df[res_df['signal'] == 'direction_only'].to_string(index=False))

df.to_csv('spy_1min_processed_full.csv', index=False)
