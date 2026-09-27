import pandas as pd
import numpy as np

df = pd.read_csv('spy_1min_raw.csv', sep=';')
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime').reset_index(drop=True)  # chronological
df['date'] = df['datetime'].dt.date

# signal: signed volume (seller-aggressive = positive, matches earlier convention)
df['signed_volume'] = np.sign(df['open'] - df['close']) * df['volume']

# baseline: pure direction, no volume weighting at all
df['direction_only'] = np.sign(df['open'] - df['close'])

# normalized signal: signed_volume scaled by a trailing 30-bar average volume
# (so magnitude is comparable across the sample, not dominated by the MOC-auction-sized bars)
df['avg_vol_30'] = df['volume'].rolling(30, min_periods=10).mean()
df['norm_signal'] = df['signed_volume'] / df['avg_vol_30']

# next-bar return, WITHIN THE SAME DAY ONLY (this is the lookahead-hygiene point from Unit 5/4:
# pairing bar 15:59 of day N with bar 09:30 of day N+1 would mix an overnight gap into an
# intraday microstructure signal -- different regime entirely, would just add noise/bias)
df['next_close'] = df.groupby('date')['close'].shift(-1)
df['same_day_next'] = df['date'] == df.groupby('date')['date'].shift(-1)
df['next_ret'] = np.where(df['same_day_next'], (df['next_close'] - df['close']) / df['close'], np.nan)

clean = df.dropna(subset=['next_ret', 'norm_signal']).copy()
print(f"Total bars pulled: {len(df)}")
print(f"Trading days covered: {df['date'].nunique()} ({df['date'].min()} to {df['date'].max()})")
print(f"Usable bars after dropping day-boundary/warm-up rows: {len(clean)}")
print()

# split chronologically: first 70% in-sample, last 30% out-of-sample (Unit 5 walk-forward discipline)
n = len(clean)
split = int(n * 0.7)
in_sample = clean.iloc[:split]
out_sample = clean.iloc[split:]
print(f"In-sample: {len(in_sample)} bars ({in_sample['date'].min()} to {in_sample['date'].max()})")
print(f"Out-of-sample: {len(out_sample)} bars ({out_sample['date'].min()} to {out_sample['date'].max()})")
print()

def corr_with_ci(x, y, label):
    r = np.corrcoef(x, y)[0, 1]
    n = len(x)
    # Fisher z-transform for a 95% CI on the correlation
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
    print(f"{label}: r = {r:.4f}, 95% CI [{lo:.4f}, {hi:.4f}], n = {n}")
    return r

print("=== Full-sample correlations (signal vs next-bar return) ===")
corr_with_ci(clean['norm_signal'], clean['next_ret'], "normalized signed-volume")
corr_with_ci(clean['direction_only'], clean['next_ret'], "direction-only baseline (no volume)")
print()

print("=== In-sample vs out-of-sample (walk-forward check) ===")
corr_with_ci(in_sample['norm_signal'], in_sample['next_ret'], "norm_signal, IN-sample")
corr_with_ci(out_sample['norm_signal'], out_sample['next_ret'], "norm_signal, OUT-of-sample")
print()

# decile analysis on the OUT-OF-SAMPLE data only (avoids in-sample bin edges leaking into the "result")
out_sample = out_sample.copy()
out_sample['decile'] = pd.qcut(out_sample['norm_signal'], 10, labels=False, duplicates='drop')
decile_stats = out_sample.groupby('decile')['next_ret'].agg(['mean', 'std', 'count'])
decile_stats['se'] = decile_stats['std'] / np.sqrt(decile_stats['count'])
decile_stats['mean_bps'] = decile_stats['mean'] * 10000
decile_stats['se_bps'] = decile_stats['se'] * 10000
print("=== Out-of-sample: mean next-bar return by signal decile (bps) ===")
print(decile_stats[['count', 'mean_bps', 'se_bps']].round(3))

decile_stats.to_csv('decile_results_oos.csv')
clean.to_csv('spy_1min_processed.csv', index=False)
