from datetime import date, datetime
import json

import numpy as np

from data_sampler import DEFAULT_UNIVERSE, ingest_multi_asset_universe
from regime_router import route_regimes
from hybrid_predictor import train_hybrid_models

# These are the feature names and order used when hybrid_predictor trains its models.
FEATURES = ["Volatility_15m", "_BuyVolume", "_SellVolume"]
VOLUME_COLUMN_ALIASES = {
    "_BuyVolume": ("_BuyVolume", "BuyVolume", "Buy_Volume", "BuyVolume_15m", "_BuyVolume_15m"),
    "_SellVolume": ("_SellVolume", "SellVolume", "Sell_Volume", "SellVolume_15m", "_SellVolume_15m"),
}
TIMESTAMP_COLUMNS = ("timestamp", "Timestamp", "Datetime", "Date", "datetime")


def _as_string(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def run_risk_scan(tickers=DEFAULT_UNIVERSE):
    """Run the pipeline and return current per-ticker risk decisions."""
    bars_map, profiles_map = ingest_multi_asset_universe(tickers)
    regime_map = route_regimes(bars_map, profiles_map)
    model_artifacts = train_hybrid_models(regime_map)

    scan_results = []
    for ticker in tickers:
        if ticker not in regime_map or regime_map[ticker].is_empty():
            continue

        latest = regime_map[ticker].tail(1).to_dicts()[0]
        regime = latest["Regime"]
        if regime == "Normal":
            kill_switch = False
            tree_pred = 0
            vector_risk = "Low"
        elif regime == "Fragile":
            if ticker not in model_artifacts:
                raise RuntimeError(
                    f"No trained model artifacts for fragile ticker {ticker}; "
                    "cannot complete risk scan safely."
                )

            artifacts = model_artifacts[ticker]
            model_features = artifacts.get("feature_names", FEATURES)
            feature_values = {"Volatility_15m": latest["Volatility_15m"]}

            for feature_name, aliases in VOLUME_COLUMN_ALIASES.items():
                source_name = next((name for name in aliases if name in latest), None)
                if source_name is not None:
                    feature_values[feature_name] = latest[source_name]
                elif feature_name == "_BuyVolume" and latest["Close"] > latest["Open"]:
                    feature_values[feature_name] = latest["Volume"]
                elif feature_name == "_SellVolume" and latest["Close"] <= latest["Open"]:
                    feature_values[feature_name] = latest["Volume"]
                else:
                    feature_values[feature_name] = 0

            features = np.asarray(
                [[feature_values[name] for name in model_features]], dtype=float
            )
            tree_pred = int(artifacts["tree_model"].predict(features)[0])
            distances, _ = artifacts["vector_engine"].kneighbors(features, n_neighbors=1)
            vector_risk = "High" if float(distances[0, 0]) < 0.05 else "Moderate"
            kill_switch = tree_pred == 1 or vector_risk == "High"
        else:
            raise ValueError(f"Unexpected regime {regime!r} for ticker {ticker}.")

        timestamp_col = next(
            (name for name in TIMESTAMP_COLUMNS if name in latest), None
        )
        if timestamp_col is None:
            timestamp_col = next(
                (name for name in latest if name not in {
                    "Open", "High", "Low", "Close", "Volume", "Regime",
                    "Volatility_15m", "BuySellRatio_15m", "Baseline_Vol",
                    "Dynamic_Threshold", *FEATURES,
                }),
                None,
            )
        if timestamp_col is None:
            raise KeyError(f"No timestamp column found for ticker {ticker}.")

        scan_results.append({
            "ticker": str(ticker),
            "timestamp": _as_string(latest[timestamp_col]),
            "current_price": float(latest["Close"]),
            "regime": regime,
            "tree_prediction": int(tree_pred),
            "vector_risk": vector_risk,
            "kill_switch_triggered": bool(kill_switch),
        })

    return scan_results


if __name__ == "__main__":
    results = run_risk_scan()
    print(json.dumps(results, indent=4))
