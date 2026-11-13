from __future__ import annotations

from typing import Any

import polars as pl
from data_sampler import DEFAULT_UNIVERSE, ingest_multi_asset_universe


def route_regimes(
    bars_map: dict[str, pl.DataFrame],
    profiles_map: dict[str, dict[str, Any]],
) -> dict[str, pl.DataFrame]:
    """Add rolling volatility, buy/sell ratio, threshold, and regime per asset."""
    routed: dict[str, pl.DataFrame] = {}

    for ticker, bars in bars_map.items():
        profile = profiles_map[ticker]
        baseline_return_std = profile["baseline_bar_return_std"]
        dynamic_multiplier = 1.0 + (baseline_return_std * 100)

        processed = (
            bars.with_columns(
                pl.col("Close").rolling_std(window_size=15).alias("Volatility_15m"),
                pl.when(pl.col("Close") > pl.col("Open"))
                .then(pl.col("Volume"))
                .otherwise(0)
                .alias("_BuyVolume"),
                pl.when(pl.col("Close") <= pl.col("Open"))
                .then(pl.col("Volume"))
                .otherwise(0)
                .alias("_SellVolume"),
            )
            .with_columns(
                pl.col("_BuyVolume").rolling_sum(window_size=15).alias("_BuyVolume_15m"),
                pl.col("_SellVolume").rolling_sum(window_size=15).alias("_SellVolume_15m"),
            )
            .with_columns(
                (pl.col("_BuyVolume_15m") / pl.col("_SellVolume_15m"))
                .alias("BuySellRatio_15m"),
                pl.col("Volatility_15m")
                .rolling_mean(window_size=60)
                .alias("Baseline_Vol"),
            )
            .with_columns(
                (pl.col("Baseline_Vol") * dynamic_multiplier).alias("Dynamic_Threshold")
            )
            .with_columns(
                pl.when(pl.col("Volatility_15m") > pl.col("Dynamic_Threshold"))
                .then(pl.lit("Fragile"))
                .otherwise(pl.lit("Normal"))
                .alias("Regime")
            )
            .drop(["_BuyVolume", "_SellVolume", "_BuyVolume_15m", "_SellVolume_15m"])
            .drop_nans()
            .drop_nulls()
        )
        routed[ticker] = processed

    return routed


def print_summary(regimes_map: dict[str, pl.DataFrame]) -> None:
    headers = ("TICKER", "TOTAL BARS", "NORMAL COUNT", "FRAGILE COUNT", "% OF FRAGILE BARS")
    rows = []
    for ticker, df in regimes_map.items():
        total = df.height
        fragile = df.filter(pl.col("Regime") == "Fragile").height
        normal = df.filter(pl.col("Regime") == "Normal").height
        fragile_pct = (100 * fragile / total) if total else 0.0
        rows.append((ticker, str(total), str(normal), str(fragile), f"{fragile_pct:.2f}%"))

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    bars_map, profiles_map = ingest_multi_asset_universe(DEFAULT_UNIVERSE)
    regime_map = route_regimes(bars_map, profiles_map)
    print_summary(regime_map)

