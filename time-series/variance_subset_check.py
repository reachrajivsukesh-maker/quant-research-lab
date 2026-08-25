import numpy as np

rng = np.random.default_rng(42)
N = 2000
mean_level = 50.0
phi = 0.50

spring = np.zeros(N)
spring[0] = mean_level
for t in range(1, N):
    spring[t] = mean_level + phi * (spring[t-1] - mean_level) + rng.normal(0, 2)

spring_ret = np.diff(spring)   # this is the same return series used in the ACF demo

full_var = spring_ret.var()
print(f"Full series (all {len(spring_ret)} returns): variance = {full_var:.4f}\n")

print("Using just the FIRST N returns of the series:")
for n in [1, 2, 5, 10, 30, 100, 300, 1000]:
    chunk = spring_ret[:n]
    if n == 1:
        print(f"  n=1     : variance = {chunk.var():.4f}   <- undefined/meaningless, only 1 number, no spread to measure")
    else:
        print(f"  n={n:<5} : variance = {chunk.var():.4f}   (vs full-series {full_var:.4f})")

print("\nUsing a chunk from the MIDDLE of the series instead (steps 900-1000), to show it's not just about 'the start':")
mid_chunk = spring_ret[900:1000]
print(f"  middle 100 returns: variance = {mid_chunk.var():.4f}   (vs full-series {full_var:.4f})")
