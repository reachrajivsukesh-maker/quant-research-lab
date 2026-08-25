import numpy as np

rng = np.random.default_rng(42)
N = 2000
mean_level = 50.0
phi = 0.50

spring = np.zeros(N)
spring[0] = mean_level
for t in range(1, N):
    spring[t] = mean_level + phi * (spring[t-1] - mean_level) + rng.normal(0, 2)

walk = np.zeros(N)
walk[0] = 50.0
for t in range(1, N):
    walk[t] = walk[t-1] + rng.normal(0, 2)

def adf_regression(x):
    """Fit: delta_x_t = alpha + beta * x_(t-1) + eps_t   via OLS, by hand."""
    x_lag = x[:-1]          # x_(t-1)
    dx = np.diff(x)         # delta_x_t = x_t - x_(t-1)
    n = len(dx)

    x_mean, dx_mean = x_lag.mean(), dx.mean()
    beta_hat = np.sum((x_lag - x_mean) * (dx - dx_mean)) / np.sum((x_lag - x_mean)**2)
    alpha_hat = dx_mean - beta_hat * x_mean

    resid = dx - (alpha_hat + beta_hat * x_lag)
    sigma2 = np.sum(resid**2) / (n - 2)
    se_beta = np.sqrt(sigma2 / np.sum((x_lag - x_mean)**2))

    t_stat = beta_hat / se_beta
    return alpha_hat, beta_hat, se_beta, t_stat

# MacKinnon's standard Dickey-Fuller critical values (constant, no trend case)
CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}

for name, series, theory in [("Mean-reverting (spring) series", spring, -(1-phi)),
                               ("Random walk", walk, 0.0)]:
    a, b, se, t = adf_regression(series)
    print(f"\n{name}:")
    print(f"  fitted beta = {b:.4f}   (theoretical beta = {theory:.4f})")
    print(f"  std error   = {se:.4f}")
    print(f"  t-statistic = {t:.3f}")
    verdict = "significant at 1%" if t < CRIT["1%"] else \
              "significant at 5%" if t < CRIT["5%"] else \
              "significant at 10%" if t < CRIT["10%"] else \
              "NOT significant -- cannot reject unit-root/random-walk null"
    print(f"  vs Dickey-Fuller critical values (1%={CRIT['1%']}, 5%={CRIT['5%']}, 10%={CRIT['10%']}): {verdict}")
