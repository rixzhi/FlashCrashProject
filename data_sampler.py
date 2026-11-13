import concurrent.futures
from typing import Dict, List, Tuple
import pandas as pd
import polars as pl
import yfinance as yf

DEFAULT_UNIVERSE = ["INTC", "AAPL", "NVDA", "TSLA", "BTC-USD"]


def fetch_single_ticker(ticker: str, period: str = "7d", interval: str = "1m") -> Tuple[str, pl.DataFrame]:
    """
    Fetches 1-minute historical intraday data for a single ticker via yfinance
    and converts it into a clean Polars DataFrame.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        pdf = yf_ticker.history(period=period, interval=interval)

        if pdf.empty:
            print(f"[WARN] No data returned for {ticker}")
            return ticker, pl.DataFrame()

        pdf = pdf.reset_index()

        # Normalize the timestamp column name across yfinance versions
        time_col = "Datetime" if "Datetime" in pdf.columns else "Date"
        pdf = pdf.rename(columns={time_col: "Datetime"})

        # Ensure timezone-naive UTC timestamps for clean Polars compatibility
        pdf["Datetime"] = pd.to_datetime(pdf["Datetime"], utc=True).dt.tz_localize(None)

        keep_cols = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        pdf = pdf[keep_cols].dropna()

        df = pl.from_pandas(pdf)
        df = df.with_columns([
            pl.col("Open").cast(pl.Float64),
            pl.col("High").cast(pl.Float64),
            pl.col("Low").cast(pl.Float64),
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Float64),
        ])
        return ticker, df

    except Exception as e:
        print(f"[ERROR] Failed fetching {ticker}: {e}")
        return ticker, pl.DataFrame()


def build_dynamic_volume_bars(df: pl.DataFrame, ticker: str) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    Converts rigid 1-minute clock bars into Information-Driven Dynamic Volume Bars
    and computes the asset's scaling profile (baseline volume & return volatility).
    """
    if df.is_empty():
        return df, {}

    # Filter non-zero volume rows to compute an accurate liquidity baseline
    non_zero_vol = df.filter(pl.col("Volume") > 0)["Volume"]
    if len(non_zero_vol) > 0:
        median_1m_vol = float(non_zero_vol.median())
        mean_1m_vol = float(non_zero_vol.mean())
    else:
        median_1m_vol = 1.0
        mean_1m_vol = 1.0

    # Set dynamic volume bucket threshold to 1.35x median 1-minute volume.
    # Calm minutes accumulate until they hit this bucket; high-volume panic minutes trigger immediately.
    vol_bucket_size = max(median_1m_vol * 1.35, 1.0)

    # Assign each row to a dynamic volume bar bucket
    df_bars = (
        df.with_columns(
            pl.when(pl.col("Volume") > 0)
            .then(pl.col("Volume"))
            .otherwise(median_1m_vol * 0.25)  # Fallback weight if feed reports 0 volume (e.g., off-peak crypto ticks)
            .alias("Effective_Volume")
        )
        .with_columns(
            (pl.col("Effective_Volume").cum_sum() // vol_bucket_size)
            .cast(pl.Int64)
            .alias("vol_bar_id")
        )
        .group_by("vol_bar_id", maintain_order=True)
        .agg([
            pl.col("Datetime").last().alias("Datetime"),
            pl.col("Open").first().alias("Open"),
            pl.col("High").max().alias("High"),
            pl.col("Low").min().alias("Low"),
            pl.col("Close").last().alias("Close"),
            pl.col("Volume").sum().alias("Volume"),
            pl.len().alias("Minutes_Compressed"),
        ])
        .drop("vol_bar_id")
    )

    # Calculate per-asset return volatility profile for dynamic threshold scaling in Tier 1 & Tier 2
    returns_std = float(
        df_bars.select(pl.col("Close").pct_change().std()).item() or 0.001
    )

    asset_profile = {
        "ticker": ticker,
        "raw_clock_bars": df.height,
        "dynamic_volume_bars": df_bars.height,
        "compression_ratio_pct": round((1.0 - (df_bars.height / max(df.height, 1))) * 100, 2),
        "median_1m_volume": round(median_1m_vol, 2),
        "mean_1m_volume": round(mean_1m_vol, 2),
        "volume_bucket_threshold": round(vol_bucket_size, 2),
        "baseline_bar_return_std": round(returns_std, 6),
    }

    return df_bars, asset_profile


def ingest_multi_asset_universe(
    tickers: List[str] = DEFAULT_UNIVERSE,
    period: str = "7d",
    interval: str = "1m",
) -> Tuple[Dict[str, pl.DataFrame], Dict[str, Dict[str, float]]]:
    """
    Concurrently downloads 1-minute data for multiple tickers and transforms each
    into Dynamic Volume Bars alongside its Asset Scaling Profile.
    """
    sampled_bars_dict: Dict[str, pl.DataFrame] = {}
    asset_profiles: Dict[str, Dict[str, float]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
        future_to_ticker = {
            executor.submit(fetch_single_ticker, ticker, period, interval): ticker
            for ticker in tickers
        }
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker, raw_df = future.result()
            if not raw_df.is_empty():
                bars_df, profile = build_dynamic_volume_bars(raw_df, ticker)
                sampled_bars_dict[ticker] = bars_df
                asset_profiles[ticker] = profile

    return sampled_bars_dict, asset_profiles


if __name__ == "__main__":
    print("🚀 Running Step 1: Multi-Asset Concurrent Ingestion & Dynamic Volume Bar Sampler...\n")
    bars_map, profiles_map = ingest_multi_asset_universe(DEFAULT_UNIVERSE)

    print(f"{'TICKER':<10} | {'CLOCK BARS':<11} | {'VOL BARS':<9} | {'NOISE COMPRESSED':<17} | {'VOL BUCKET':<14} | {'BASE RET STD'}")
    print("-" * 88)
    for t in DEFAULT_UNIVERSE:
        if t in profiles_map:
            p = profiles_map[t]
            print(
                f"{t:<10} | {p['raw_clock_bars']:<11} | {p['dynamic_volume_bars']:<9} | "
                f"{p['compression_ratio_pct']:>6.2f}%          | {p['volume_bucket_threshold']:<14.1f} | {p['baseline_bar_return_std']:.5f}"
            )

    if "INTC" in bars_map:
        print("\n🔍 Sample Dynamic Volume Bars for INTC (last 5 bars):")
        print(bars_map["INTC"].tail(5))