import polars as pl

input_file = "INTC_features.csv"
output_file = "INTC_tier1.csv"

df = pl.read_csv(input_file).with_columns(
    pl.col("Volatility_15m")
    .rolling_mean(window_size=60)
    .alias("Volatility_Baseline_60m")
).drop_nans().drop_nulls(subset=["Volatility_Baseline_60m"])

df = df.with_columns(
    pl.when(pl.col("Volatility_15m") > 1.2 * pl.col("Volatility_Baseline_60m"))
    .then(pl.lit("Fragile"))
    .otherwise(pl.lit("Normal"))
    .alias("Regime")
)

df.write_csv(output_file)

counts = df.group_by("Regime").len().to_dicts()
summary = {row["Regime"]: row["len"] for row in counts}
print(f"Normal: {summary.get('Normal', 0)}")
print(f"Fragile: {summary.get('Fragile', 0)}")