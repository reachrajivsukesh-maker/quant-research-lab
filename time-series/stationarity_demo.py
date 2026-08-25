import numpy as np

rng = np.random.default_rng(42)
N = 2000

# --- Series A: mean-reverting ("spring") process ---
# x_t = mean + phi*(x_{t-1} - mean) + noise,  phi < 1 pulls it back toward `mean`
mean_level = 50.0
phi = 0.90
spring = np.zeros(N)
spring[0] = mean_level
for t in range(1, N):
    spring[t] = mean_level + phi * (spring[t-1] - mean_level) + rng.normal(0, 2)

# --- Series B: random walk (non-stationary) ---
walk = np.zeros(N)
walk[0] = 50.0
for t in range(1, N):
    walk[t] = walk[t-1] + rng.normal(0, 2)

# Several DIFFERENT windows -- deliberately not just "first vs last"
windows = {
    "first 200":    (0, 200),
    "steps 400-600": (400, 600),
    "steps 900-1300": (900, 1300),
    "last 200":     (1800, 2000),
}

print(f"{'window':<18}{'spring mean':>14}{'spring std':>13}   |{'walk mean':>12}{'walk std':>11}")
for name, (a, b) in windows.items():
    sm, ss = spring[a:b].mean(), spring[a:b].std()
    wm, ws = walk[a:b].mean(), walk[a:b].std()
    print(f"{name:<18}{sm:>14.2f}{ss:>13.2f}   |{wm:>12.2f}{ws:>11.2f}")
