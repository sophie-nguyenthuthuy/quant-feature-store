"""Bitemporal corporate-action adjustments.

A split announced today reaches *back in time* and reshapes the historical
price series. Naive systems silently use the latest-known adjusted prices
across the entire backtest — which means an Aug 2020 backtest of AAPL
'sees' the August 4:1 split before it happened. This is a leakage bug
that production pipelines ship with constantly.

The fix is to treat each split as a bitemporal fact:
  - ex_date         = when the split takes effect in the world
  - knowledge_time  = when we recorded it (announcement, late-arriving
                      data correction, etc.)

`adjusted_ohlcv_as_of(bars, splits, as_of)` returns the OHLCV frame as
it would have looked when consulted at `as_of` — using only splits whose
knowledge_time AND ex_date are both <= `as_of`. The same raw bars yield
different adjusted prices at different `as_of` values; that asymmetry
*is* the correctness property.

Dividends use the same model with a subtractive adjustment; not
implemented here to keep the demo tight, but the function would have an
identical shape (filter to known-and-effective, subtract sum).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl


def adjusted_ohlcv_as_of(
    bars: pl.DataFrame,
    splits: pl.DataFrame,
    as_of: datetime,
) -> pl.DataFrame:
    """Apply split-adjustments to historical OHLCV using only splits known
    and ex-effective by `as_of`.

    bars: must have columns (symbol, event_time, open, high, low, close).
    splits: must have columns (symbol, ex_date, knowledge_time, ratio).
            A 4:1 forward split has ratio = 4.0.

    Returns bars with columns adj_open / adj_high / adj_low / adj_close
    added. Bars after a split's ex_date are unchanged; bars before are
    divided by the cumulative product of all eligible split ratios that
    occurred strictly after the bar's event_time.
    """
    required_bars = {"symbol", "event_time", "open", "high", "low", "close"}
    missing = required_bars - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    required_splits = {"symbol", "ex_date", "knowledge_time", "ratio"}
    missing = required_splits - set(splits.columns)
    if missing:
        raise ValueError(f"splits missing columns: {missing}")

    as_of_lit = pl.lit(as_of).cast(pl.Datetime("us"))

    eligible = splits.filter(
        (pl.col("knowledge_time") <= as_of_lit)
        & (pl.col("ex_date") <= as_of_lit)
    ).select("symbol", "ex_date", "ratio")

    if eligible.is_empty():
        return bars.with_columns(
            pl.col("open").alias("adj_open"),
            pl.col("high").alias("adj_high"),
            pl.col("low").alias("adj_low"),
            pl.col("close").alias("adj_close"),
        )

    # For each (symbol, event_time), compute the cumulative split factor =
    # product of 1/ratio over all eligible splits with ex_date > event_time.
    # Cross-join per symbol, filter, group, product.
    joined = bars.select("symbol", "event_time").join(eligible, on="symbol", how="left")
    factors = (
        joined.filter(pl.col("ex_date") > pl.col("event_time"))
        .with_columns((1.0 / pl.col("ratio")).alias("inv_ratio"))
        .group_by("symbol", "event_time")
        .agg(pl.col("inv_ratio").product().alias("factor"))
    )
    out = bars.join(factors, on=["symbol", "event_time"], how="left").with_columns(
        pl.col("factor").fill_null(1.0)
    )
    out = out.with_columns(
        (pl.col("open") * pl.col("factor")).alias("adj_open"),
        (pl.col("high") * pl.col("factor")).alias("adj_high"),
        (pl.col("low") * pl.col("factor")).alias("adj_low"),
        (pl.col("close") * pl.col("factor")).alias("adj_close"),
    ).drop("factor")

    return out
