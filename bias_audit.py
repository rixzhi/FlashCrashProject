import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from data_sampler import DEFAULT_UNIVERSE, ingest_multi_asset_universe
from regime_router import route_regimes
from hybrid_predictor import train_hybrid_models

FEATURES = ["Volatility_15m", "_BuyVolume", "_SellVolume"]


def audit_bias():
    """Audit chronological splitting and backward-only volatility per ticker."""
    bars_map, profiles_map = ingest_multi_asset_universe(DEFAULT_UNIVERSE)
    regime_map = route_regimes(bars_map, profiles_map)
    results = {}

    for ticker, routed in regime_map.items():
        raw = bars_map[ticker]
        if "Datetime" not in routed.columns or "Datetime" not in raw.columns:
            results[ticker] = {"chrono_split": "FAIL", "feature_leakage": "FAIL"}
            continue

        # Match the labeled, cleaned Fragile rows used by hybrid_predictor.
        future_min = pl.min_horizontal(
            [pl.col("Close").shift(-step) for step in range(1, 16)]
        )
        labeled = routed.with_columns(
            pl.when(pl.col("Close").shift(-15).is_not_null())
            .then((future_min < pl.col("Close") * 0.995).cast(pl.Int8))
            .otherwise(None)
            .alias("Crash_Target")
        )
        if "_BuyVolume" not in labeled.columns or "_SellVolume" not in labeled.columns:
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
            results[ticker] = {"chrono_split": "FAIL", "feature_leakage": "FAIL"}
            continue

        audit_rows = fragile.select(["Datetime", "Close", *FEATURES, "Crash_Target"]).to_pandas()
        train_rows, test_rows = train_test_split(
            audit_rows, test_size=0.2, shuffle=False
        )
        chrono_pass = train_rows["Datetime"].max() < test_rows["Datetime"].min()

        # Fixed seed makes the sampled audit reproducible.
        sample = test_rows.sample(n=1, random_state=42).iloc[0]
        timestamp = sample["Datetime"]
        preceding = raw.filter(pl.col("Datetime") <= pl.lit(timestamp)).tail(15)
        if preceding.height != 15:
            leakage_pass = False
        else:
            manual_std = float(np.std(preceding["Close"].to_numpy(), ddof=1))
            stored_std = float(sample["Volatility_15m"])
            leakage_pass = bool(np.isclose(manual_std, stored_std, rtol=1e-9, atol=1e-12))

        results[ticker] = {
            "chrono_split": "PASS" if chrono_pass else "FAIL",
            "feature_leakage": "PASS" if leakage_pass else "FAIL",
        }

    return results


def print_audit_table(results):
    headers = ("TICKER", "CHRONO SPLIT", "FEATURE LEAKAGE")
    rows = [
        (ticker, item["chrono_split"], item["feature_leakage"])
        for ticker, item in results.items()
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


if __name__ == "__main__":
    audit_results = audit_bias()
    print_audit_table(audit_results)
