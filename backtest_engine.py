import pandas as pd
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from data_sampler import DEFAULT_UNIVERSE, ingest_multi_asset_universe
from regime_router import route_regimes
from hybrid_predictor import train_hybrid_models


def run_historical_backtest():
    """Evaluate each trained tree on the chronological holdout period."""
    bars_map, profiles_map = ingest_multi_asset_universe(DEFAULT_UNIVERSE)
    regime_map = route_regimes(bars_map, profiles_map)
    model_artifacts = train_hybrid_models(regime_map)
    backtest_results = {}

    for ticker, artifacts in model_artifacts.items():
        if ticker not in regime_map:
            continue

        frame = regime_map[ticker]
        feature_names = artifacts["feature_names"]

        # Match training: future window is calculated on the routed frame, then
        # the fragile rows are selected and cleaned before the chronological split.
        future_min = pl.min_horizontal(
            [pl.col("Close").shift(-period) for period in range(1, 16)]
        )
        labeled = frame.with_columns(
            pl.when(pl.col("Close").shift(-15).is_not_null())
            .then((future_min < pl.col("Close") * 0.995).cast(pl.Int8))
            .otherwise(None)
            .alias("Crash_Target")
        )

        missing_volume_features = [
            name for name in ("_BuyVolume", "_SellVolume")
            if name in feature_names and name not in labeled.columns
        ]
        if missing_volume_features:
            labeled = labeled.with_columns(
                pl.when(pl.col("Close") > pl.col("Open"))
                .then(pl.col("Volume"))
                .otherwise(0)
                .alias("_BuyVolume"),
                pl.when(pl.col("Close") <= pl.col("Open"))
                .then(pl.col("Volume"))
                .otherwise(0)
                .alias("_SellVolume"),
            )

        fragile = (
            labeled.filter(pl.col("Regime") == "Fragile")
            .drop_nans(subset=feature_names + ["Crash_Target"])
            .drop_nulls(subset=feature_names + ["Crash_Target"])
            .filter(pl.all_horizontal([pl.col(c).is_finite() for c in feature_names]))
        )
        pandas_frame = fragile.select(feature_names + ["Crash_Target"]).to_pandas()
        X = pandas_frame[feature_names]
        y = pandas_frame["Crash_Target"].to_numpy(dtype=np.int8)

        # Use the same split arguments as training to reproduce its unseen 20%.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        predictions = artifacts["tree_model"].predict(X_test)

        true_positives = int(np.sum((predictions == 1) & (y_test == 1)))
        false_positives = int(np.sum((predictions == 1) & (y_test == 0)))
        false_negatives = int(np.sum((predictions == 0) & (y_test == 1)))

        backtest_results[ticker] = {
            "total_test_rows": int(len(y_test)),
            "kill_switch_triggers": int(np.sum(predictions == 1)),
            "crashes_avoided": true_positives,
            "false_alarms": false_positives,
            "missed_crashes": false_negatives,
        }

    return backtest_results


def print_summary(results):
    headers = (
        "TICKER", "TEST ROWS", "KILL SWITCH TRIGGERS",
        "CRASHES AVOIDED", "FALSE ALARMS", "MISSED CRASHES",
    )
    rows = [
        (
            ticker,
            str(metrics["total_test_rows"]),
            str(metrics["kill_switch_triggers"]),
            str(metrics["crashes_avoided"]),
            str(metrics["false_alarms"]),
            str(metrics["missed_crashes"]),
        )
        for ticker, metrics in results.items()
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    results = run_historical_backtest()
    print_summary(results)
