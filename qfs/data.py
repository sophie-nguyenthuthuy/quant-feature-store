"""yfinance OHLCV loader returning a Polars frame with our canonical schema."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import yfinance as yf


def load_ohlcv(
    symbols: list[str],
    start: str | datetime,
    end: str | datetime,
    publish_lag_minutes: int = 0,
) -> pl.DataFrame:
    """Fetch daily OHLCV and stamp knowledge_time = event_time + publish_lag.

    For daily bars, a small positive lag (e.g. one minute) realistically
    models that the bar is only known after the close print is published.
    """
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    rows: list[pl.DataFrame] = []
    for sym in symbols:
        if (sym, "Close") in raw.columns:
            sub = raw[sym]
        else:
            sub = raw
        sub = sub.dropna(subset=["Close"]).reset_index()
        if sub.empty:
            continue
        sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
        df = pl.from_pandas(sub).select(
            pl.lit(sym).alias("symbol"),
            pl.col("date").cast(pl.Datetime("us")).alias("event_time"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        df = df.with_columns(
            (pl.col("event_time") + pl.duration(minutes=publish_lag_minutes)).alias(
                "knowledge_time"
            )
        )
        rows.append(df)
    return pl.concat(rows) if rows else pl.DataFrame()
