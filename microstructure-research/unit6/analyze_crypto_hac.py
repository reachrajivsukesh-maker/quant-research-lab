import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('btc_1min_ofi.csv')
df['minute'] = pd.to_datetime(df['minute'])
df = df.sort_values('minute').reset_index(drop=True)
df['ret_next'] = (df['close'].shift(-1) - df['close']) / df['close']
df['ret_lag1'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)

n = len(df)
split = int(n * 0.7)
in_sample = df.iloc[:split]
out_sample = df.iloc[split:]

def fit_hac(sub, label, maxlags):
    d = sub.dropna(subset=['ret_next', 'ofi', 'ret_lag1'])
    y = d['ret_next'].values
    X = sm.add_constant(d[['ofi', 'ret_lag1']].values)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    b_ofi, t_ofi, p_ofi = model.params[1], model.tvalues[1], model.pvalues[1]
    b_mom, t_mom, p_mom = model.params[2], model.tvalues[2], model.pvalues[2]
    print(f"{label:14s} (n={len(d):5d}, HAC lags={maxlags:2d}):  "
          f"OFI b={b_ofi:+.6f} t={t_ofi:+.2f} p={p_ofi:.3f}   |   "
          f"momentum b={b_mom:+.4f} t={t_mom:+.2f} p={p_mom:.3f}")
    return dict(label=label, maxlags=maxlags, n=len(d), b_ofi=b_ofi, t_ofi=t_ofi, p_ofi=p_ofi,
                b_mom=b_mom, t_mom=t_mom, p_mom=p_mom)

print("=== Declared-upfront joint model: ret_next = a + b*OFI + c*ret_lag1 ===")
print("=== HAC (Newey-West) standard errors -- correct for the serial correlation we found ===\n")

results = []
for maxlags in [5, 10, 20]:
    print(f"--- HAC maxlags = {maxlags} ---")
    results.append(fit_hac(df, 'full sample', maxlags))
    results.append(fit_hac(in_sample, 'in-sample', maxlags))
    results.append(fit_hac(out_sample, 'out-of-sample', maxlags))
    print()

pd.DataFrame(results).to_csv('crypto_hac_results.csv', index=False)
