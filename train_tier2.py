import polars as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

features = ["Volatility_15m", "BuySellRatio_15m"]

# Lowest Close across the next 15 rows, excluding the current row.
future_min_close = pl.min_horizontal(
    [pl.col("Close").shift(-step) for step in range(1, 16)]
)

df = pl.read_csv("INTC_tier1.csv").with_columns(
    pl.when(pl.col("Close").shift(-15).is_not_null())
    .then((future_min_close < pl.col("Close") * 0.995).cast(pl.Int8))
    .otherwise(None)
    .alias("Crash_Target")
)

# Keep Fragile rows with a target and usable feature values.
data = (
    df.filter(pl.col("Regime") == "Fragile")
    .drop_nans(subset=[*features, "Crash_Target"])
    .drop_nulls(subset=[*features, "Crash_Target"])
    .filter(pl.all_horizontal([pl.col(c).is_finite() for c in features]))
)

split_at = int(data.height * 0.8)
if split_at == 0 or split_at == data.height:
    raise ValueError("Not enough usable Fragile rows for an 80/20 time-ordered split.")

train = data.slice(0, split_at)
test = data.slice(split_at)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(train.select(features).to_numpy(), train["Crash_Target"].to_numpy())

predictions = model.predict(test.select(features).to_numpy())
accuracy = accuracy_score(test["Crash_Target"].to_numpy(), predictions)
print(f"Test accuracy: {accuracy:.4f}")