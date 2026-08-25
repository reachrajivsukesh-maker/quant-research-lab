"""Lookahead bias in the execution layer.

The adaptive rule's `adverse` term is NOT lookahead -- it compares a past anchor
to the present, so it reacts rather than predicts. The real bug was elsewhere and
sat in BOTH engines:

    allowed = participation * volume        # <-- today's TOTAL realised volume

That number does not exist at decision time. On a heavy day the engine "knew" to
trade more. Fixed by sizing from a lagged 20-day forecast and deciding from the
prior close.
"""
import _common
import pandas as pd
from src import make_path, Config, backtest, implementation_shortfall, \
    total_return_pct, volume_forecast_quality

d = make_path(1000)
rows = []
for mode in ["fixed", "adaptive"]:
    for future in [True, False]:
        v, fills = backtest(d, Config(mode=mode, use_future_info=future))
        rows.append(dict(
            engine=f"{mode} / {'BIASED' if future else 'point-in-time'}",
            final_value=round(v.iloc[-1], 0),
            return_pct=round(total_return_pct(v), 2),
            shortfall_bps=round(implementation_shortfall(fills), 2)))

out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv(f"{_common.RESULTS}/02_lookahead_audit.csv", index=False)

q = volume_forecast_quality(d)
print(f"\nCAVEAT -- volume forecast correlation with actual: {q['correlation']:.3f}")
print(f"          mean absolute error: {q['mean_abs_pct_error']*100:.1f}%")
print("Synthetic volume is drawn iid, so a trailing mean predicts nothing. Real")
print("market volume autocorrelates ~0.6-0.8, so this OVERSTATES the biased")
print("engine's advantage and handicaps the corrected one. Direction of the")
print("conclusion holds; the magnitude is a data artifact.")
