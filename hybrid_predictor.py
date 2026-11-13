from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

from data_sampler import DEFAULT_UNIVERSE, ingest_multi_asset_universe
from regime_router import route_regimes

FEATURES = ["Volatility_15m", "_BuyVolume", "_SellVolume"]


def train_hybrid_models(regime_map: dict[str, pl.DataFrame]) -> dict[str, dict]:
    """Train the XGBoost classifier and cosine-neighbor engine for each ticker."""
    artifacts: dict[str, dict] = {}

    for ticker, frame in regime_map.items():
        # Compute the forward crash target before filtering so the next 15 periods
        # refer to the complete time series, including rows outside Fragile regimes.
        future_min = pl.min_horizontal(
            [pl.col("Close").shift(-period) for period in range(1, 16)]
        )
        labeled = frame.with_columns(
            pl.when(pl.col("Close").shift(-15).is_not_null())
            .then((future_min < pl.col("Close") * 0.995).cast(pl.Int8))
            .otherwise(None)
            .alias("Crash_Target")
        )

        # route_regimes currently removes these intermediate columns, so recreate
        # them from candle direction and volume when needed.
        missing_volume_features = [name for name in ("_BuyVolume", "_SellVolume") if name not in labeled.columns]
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
            .drop_nans(subset=FEATURES + ["Crash_Target"])
            .drop_nulls(subset=FEATURES + ["Crash_Target"])
            .filter(pl.all_horizontal([pl.col(c).is_finite() for c in FEATURES]))
        )
        if fragile.height < 2:
            continue

        pandas_frame = fragile.select(FEATURES + ["Crash_Target"]).to_pandas()
        X = pandas_frame[FEATURES]
        y = pandas_frame["Crash_Target"].to_numpy(dtype=np.int8)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        if len(X_train) < 5:
            continue

        pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
        tree = XGBClassifier(
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric="logloss",
        )
        tree.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, tree.predict(X_test))

        vector_engine = NearestNeighbors(n_neighbors=5, metric="cosine")
        vector_engine.fit(X_train.to_numpy())

        artifacts[ticker] = {
            "tree_model": tree,
            "vector_engine": vector_engine,
            "feature_names": FEATURES.copy(),
            "test_accuracy": float(accuracy),
        }

    return artifacts


def print_summary(artifacts: dict[str, dict]) -> None:
    headers = ("TICKER", "XGBOOST ACCURACY (%)", "VECTOR ENGINE STATUS")
    rows = [
        (ticker, f"{item['test_accuracy'] * 100:.2f}%", "Ready (5 neighbors)")
        for ticker, item in artifacts.items()
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    bars_map, profiles_map = ingest_multi_asset_universe(DEFAULT_UNIVERSE)
    regime_map = route_regimes(bars_map, profiles_map)
    model_artifacts = train_hybrid_models(regime_map)
    print_summary(model_artifacts)

