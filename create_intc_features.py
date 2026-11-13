import csv
import polars as pl

input_file = "INTC_1min_data.csv"
output_file = "INTC_features.csv"

# Support ordinary CSV headers and yfinance's two-row ticker headers.
with open(input_file, newline="", encoding="utf-8-sig") as f:
    header = next(csv.reader(f))
    second_row = next(csv.reader(f), [])

if second_row and second_row[0] == "Ticker":
    if not {"Open", "Close", "Volume"}.issubset(header):
        raise ValueError("CSV must contain Open, Close, and Volume columns.")
    header[0] = "Datetime"  # yfinance may label the timestamp column "Price"
    df = pl.read_csv(input_file, skip_rows=2, has_header=False, new_columns=header)
else:
    df = pl.read_csv(input_file)
    if not {"Open", "Close", "Volume"}.issubset(df.columns):
        raise ValueError("CSV must contain Open, Close, and Volume columns.")

df = df.with_columns(
    pl.when(pl.col("Close") > pl.col("Open"))
    .then(pl.col("Volume"))
    .otherwise(0)
    .alias("BuyVolume"),
    pl.when(pl.col("Close") <= pl.col("Open"))
    .then(pl.col("Volume"))
    .otherwise(0)
    .alias("SellVolume"),
).with_columns(
    pl.col("Close").rolling_std(window_size=15).alias("Volatility_15m"),
    pl.col("BuyVolume").rolling_sum(window_size=15).alias("BuyVolume_15m"),
    pl.col("SellVolume").rolling_sum(window_size=15).alias("SellVolume_15m"),
).with_columns(
    (pl.col("BuyVolume_15m") / pl.col("SellVolume_15m"))
    .alias("BuySellRatio_15m")
).drop_nans().drop_nulls()

df.write_csv(output_file)
print(f"Saved {df.height} rows to {output_file}")