# Time-Series Statistics from First Principles

Every method here is built from its definition rather than called from a library
(`statsmodels` was unavailable, which turned out to be a good constraint).

| Script | What it shows |
|---|---|
| `stationarity_demo.py` / `stationarity_plot.py` | AR(1) vs random walk simulated side by side; mean and variance measured across windows to show one is stable and the other is not |
| `returns_plot.py` | Why returns, not price levels, are the stationary object |
| `acf_plot.py` | Autocorrelation computed from the definition, with the noise band |
| `wrong_acf_check.py` | Running ACF on non-stationary levels yields a meaningless ~0.997 — the classic trap |
| `variance_subset_check.py` | Hand-verification that the variance of a sum of independent shocks is the sum of variances |
| `adf_manual.py` | The Augmented Dickey–Fuller test derived as a regression: Δx_t = α + βx_(t−1) + ε_t, with standard error, degrees of freedom and t-statistic computed by hand |
| `ar2_check.py` | AR(2) partial effects confirmed empirically by regression |

The payoff for the main project: the ADF and ACF machinery here is what
established that the price series has no exploitable autocorrelation — which is
the reason the momentum-based execution rule had no basis.
