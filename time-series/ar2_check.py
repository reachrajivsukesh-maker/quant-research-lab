import numpy as np

rng = np.random.default_rng(42)
N = 2000
mean_level = 50.0
phi = 0.50

spring = np.zeros(N)
spring[0] = mean_level
for t in range(1, N):
    spring[t] = mean_level + phi * (spring[t-1] - mean_level) + rng.normal(0, 2)

# Fit x_t = c + phi1*x_(t-1) + phi2*x_(t-2) + eps_t  via OLS (2 predictors)
y  = spring[2:]        # x_t
x1 = spring[1:-1]      # x_(t-1)
x2 = spring[:-2]       # x_(t-2)

X = np.column_stack([np.ones_like(x1), x1, x2])   # design matrix: [1, x_(t-1), x_(t-2)]
coeffs, residuals_ssr, rank, sv = np.linalg.lstsq(X, y, rcond=None)
c_hat, phi1_hat, phi2_hat = coeffs

resid = y - X @ coeffs
n, k = len(y), X.shape[1]
sigma2 = np.sum(resid**2) / (n - k)
cov_matrix = sigma2 * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(cov_matrix))

print("True generating process: AR(1) with true phi = 0.50 (phi2 should be ~0, no real effect)\n")
print(f"c_hat    = {c_hat:.4f}")
print(f"phi1_hat = {phi1_hat:.4f}   SE = {se[1]:.4f}   t-stat = {phi1_hat/se[1]:.3f}   (yesterday's effect)")
print(f"phi2_hat = {phi2_hat:.4f}   SE = {se[2]:.4f}   t-stat = {phi2_hat/se[2]:.3f}   (2-days-ago's EXTRA effect, beyond phi1)")
