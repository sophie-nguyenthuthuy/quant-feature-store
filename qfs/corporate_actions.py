"""Bitemporal corporate-action adjustments.

A split announced today reaches *back in time* and reshapes the historical
price series. Naive systems silently use the latest-known adjusted prices
across the entire backtest — which means an Aug 2020 backtest of AAPL
'sees' the August 4:1 split before it happened. This is a leakage bug
that production pipelines ship with constantly.

The fix is to treat each corporate action as a bitemporal fact:
  - ex_date         = when the action takes effect in the world
  - knowledge_time  = when we recorded it (announcement, late-arriving
                      data correction, etc.)

`adjusted_ohlcv_as_of(bars, as_of, splits=..., dividends=...)` returns
the OHLCV frame as it would have looked when consulted at `as_of` —
using only actions whose knowledge_time AND ex_date are both <= `as_of`.
The same raw bars yield different adjusted prices at different `as_of`
values; that asymmetry *is* the correctness property.

Splits use a multiplicative factor (1/ratio). Dividends use a
subtractive factor: each cash dividend with ex_date > bar.event_time
reduces the bar's adjusted close by the dividend amount. This is the
"back-adjusted price" convention common in quant research — simpler
and more transparent than the multiplicative dividend-adjustment some
vendors use (factor = (close_ex - div) / close_ex propagated back).
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

_BAR_COLS = ("open", "high", "low", "close")


def adjusted_ohlcv_as_of(
    bars: pl.DataFrame,
    as_of: datetime,
    splits: pl.DataFrame | None = None,
    dividends: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Apply split AND dividend adjustments using only actions known and
    ex-effective by `as_of`.

    bars: must have columns (symbol, event_time, open, high, low, close).
    splits: optional, columns (symbol, ex_date, knowledge_time, ratio).
            A 4:1 forward split has ratio = 4.0.
    dividends: optional, columns (symbol, ex_date, knowledge_time, amount).
            amount is the per-share cash dividend in price units.

    Returns bars with adj_open / adj_high / adj_low / adj_close columns.
    Splits are applied multiplicatively (price / ratio for bars before
    the ex_date); dividends subtractively (price - sum of post-bar
    dividends). When both are present, splits are applied first, then
    dividends in the (now-split-adjusted) price space.
    """
    required_bars = {"symbol", "event_time", *_BAR_COLS}
    missing = required_bars - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {missing}")
    as_of_lit = pl.lit(as_of).cast(pl.Datetime("us"))

    # ---- Step 1: split-adjust ----
    split_factor = _split_factor(bars, splits, as_of_lit)
    out = bars.join(split_factor, on=["symbol", "event_time"], how="left").with_columns(
        pl.col("split_factor").fill_null(1.0)
    )
    out = out.with_columns(
        *[(pl.col(c) * pl.col("split_factor")).alias(f"adj_{c}") for c in _BAR_COLS]
    ).drop("split_factor")

    # ---- Step 2: dividend-adjust (subtractive, in split-adjusted price) ----
    div_adj = _dividend_adjustment(bars, dividends, as_of_lit)
    out = out.join(div_adj, on=["symbol", "event_time"], how="left").with_columns(
        pl.col("div_adj").fill_null(0.0)
    )
    out = out.with_columns(
        *[(pl.col(f"adj_{c}") - pl.col("div_adj")).alias(f"adj_{c}") for c in _BAR_COLS]
    ).drop("div_adj")

    return out


def _split_factor(
    bars: pl.DataFrame,
    splits: pl.DataFrame | None,
    as_of_lit: pl.Expr,
) -> pl.DataFrame:
    """Return frame keyed (symbol, event_time) with column split_factor."""
    if splits is None or splits.is_empty():
        return bars.select("symbol", "event_time").with_columns(pl.lit(1.0).alias("split_factor"))
    required = {"symbol", "ex_date", "knowledge_time", "ratio"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"splits missing columns: {missing}")

    eligible = splits.filter(
        (pl.col("knowledge_time") <= as_of_lit) & (pl.col("ex_date") <= as_of_lit)
    ).select("symbol", "ex_date", "ratio")
    if eligible.is_empty():
        return bars.select("symbol", "event_time").with_columns(pl.lit(1.0).alias("split_factor"))

    joined = bars.select("symbol", "event_time").join(eligible, on="symbol", how="left")
    return (
        joined.filter(pl.col("ex_date") > pl.col("event_time"))
        .with_columns((1.0 / pl.col("ratio")).alias("inv_ratio"))
        .group_by("symbol", "event_time")
        .agg(pl.col("inv_ratio").product().alias("split_factor"))
    )


def _dividend_adjustment(
    bars: pl.DataFrame,
    dividends: pl.DataFrame | None,
    as_of_lit: pl.Expr,
) -> pl.DataFrame:
    """Return frame keyed (symbol, event_time) with column div_adj — the
    cumulative cash to subtract from bars at that event_time."""
    if dividends is None or dividends.is_empty():
        return bars.select("symbol", "event_time").with_columns(pl.lit(0.0).alias("div_adj"))
    required = {"symbol", "ex_date", "knowledge_time", "amount"}
    missing = required - set(dividends.columns)
    if missing:
        raise ValueError(f"dividends missing columns: {missing}")

    eligible = dividends.filter(
        (pl.col("knowledge_time") <= as_of_lit) & (pl.col("ex_date") <= as_of_lit)
    ).select("symbol", "ex_date", "amount")
    if eligible.is_empty():
        return bars.select("symbol", "event_time").with_columns(pl.lit(0.0).alias("div_adj"))

    joined = bars.select("symbol", "event_time").join(eligible, on="symbol", how="left")
    return (
        joined.filter(pl.col("ex_date") > pl.col("event_time"))
        .group_by("symbol", "event_time")
        .agg(pl.col("amount").sum().alias("div_adj"))
    )
