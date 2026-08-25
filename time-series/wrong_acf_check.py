import numpy as np

rng = np.random.default_rng(42)
N = 2000
walk = np.zeros(N)
walk[0] = 50.0
for t in range(1, N):
    walk[t] = walk[t-1] + rng.normal(0, 2)

walk_ret = np.diff(walk)

def acf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    denom = np.sum(x**2)
    return np.array([np.sum(x[k:] * x[:n-k]) / denom for k in range(0, max_lag+1)])

print("ACF computed correctly, on the RETURNS (stationary):")
acf_correct = acf(walk_ret, 10)
for k in range(1, 11):
    print(f"  lag {k:>2}: {acf_correct[k]:>7.3f}")

print("\nACF computed WRONGLY, directly on the raw LEVELS (non-stationary):")
acf_wrong = acf(walk, 10)
for k in range(1, 11):
    print(f"  lag {k:>2}: {acf_wrong[k]:>7.3f}")
