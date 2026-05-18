"""Market-microstructure-adjacent features computable from OHLCV."""

from __future__ import annotations

import polars as pl

from qfs.registry import feature_view


@feature_view(
    name="realized_vol",
    version="v1",
    inputs=["close"],
    description="Annualised rolling 20-day vol of log returns.",
)
def realized_vol_v1(df: pl.DataFrame) -> pl.DataFrame:
    period = 20
    out = df.sort("symbol", "event_time").with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("symbol")).log().alias("_r")
    )
    out = out.with_columns(
        (pl.col("_r").rolling_std(period, min_samples=period).over("symbol") * (252**0.5)).alias(
            "rv20"
        )
    )
    return out.select("symbol", "event_time", "rv20")


@feature_view(
    name="hl_spread",
    version="v1",
    inputs=["high", "low", "close"],
    description="High-low range as fraction of close — bar-level liquidity proxy.",
)
def hl_spread_v1(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns(((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_spread"))
    return out.select("symbol", "event_time", "hl_spread")
