"""Monte Carlo over independent paths -- and an ablation of the adaptive rule.

More folds carved from ONE path would not have helped: overlapping windows are
not independent samples. Since the generating process is known, generate many
independent realisations instead and compare PAIRED (both configs on the same
path, so path-level luck cancels).

The ablation then asks WHICH HALF of the adaptive rule is responsible:
  ramp  -- speed up when the market moves against the live order
  ease  -- slow down when it moves in our favour
"""
import _common
import pandas as pd
from src import Config, monte_carlo, paired_summary

N_PATHS = 300
VARIANTS = {
    "fixed 5% (baseline)": Config(mode="fixed", participation=0.05),
    "fixed 8%":            Config(mode="fixed", participation=0.08),
    "adaptive k=4":        Config(mode="adaptive", participation=0.05, k=4),
    "adaptive ramp-only":  Config(mode="adaptive", participation=0.05, k=4, ease=False),
    "adaptive ease-only":  Config(mode="adaptive", participation=0.05, k=4, ramp=False),
}

if __name__ == "__main__":
    print(f"Running {len(VARIANTS)} variants over {N_PATHS} independent paths...")
    raw = monte_carlo(VARIANTS, n_paths=N_PATHS)
    raw.to_csv(f"{_common.RESULTS}/05_monte_carlo_raw.csv", index=False)

    summary = paired_summary(raw, "fixed 5% (baseline)")
    print("\nPaired vs fixed 5%. Negative vs_baseline = BETTER. win_rate = % of paths beaten.\n")
    print(summary.to_string(index=False))
    summary.to_csv(f"{_common.RESULTS}/05_monte_carlo_summary.csv", index=False)

    print("\nParticipation sweep (is there an interior optimum?):")
    sweep = monte_carlo({f"{p*100:.0f}%": Config(mode="fixed", participation=p)
                         for p in (0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25)},
                        n_paths=120)
    s = pd.DataFrame({"participation": sweep.columns,
                      "mean_bps": sweep.mean().round(2).values,
                      "std": sweep.std(ddof=1).round(2).values})
    print(s.to_string(index=False))
    s.to_csv(f"{_common.RESULTS}/05_participation_sweep.csv", index=False)
    print("\nIf this falls monotonically, the impact constant is too gentle to")
    print("create a real speed-vs-impact tradeoff -- a statement about the model,")
    print("not about markets. See docs/FINDINGS.md, Retraction #2.")
