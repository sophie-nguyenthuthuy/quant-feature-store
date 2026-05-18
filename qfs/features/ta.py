"""Technical-analysis features.

Each view is a pure function over an OHLCV frame and returns a frame
ready to write to the store: (symbol, event_time, <feature cols>).
The store stamps knowledge_time at write time.

v1 uses simple-MA flavour; v2 uses Wilder-smoothed RSI to demonstrate
that versions can coexist.
"""

from __future__ import annotations

import polars as pl

from qfs.registry import feature_view


def _ema(s: pl.Expr, span: int) -> pl.Expr:
    return s.ewm_mean(span=span, adjust=False, min_samples=span)


@feature_view(
    name="rsi",
    version="v1",
    inputs=["close"],
    description="RSI(14) — simple rolling average of gains/losses.",
)
def rsi_v1(df: pl.DataFrame) -> pl.DataFrame:
    period = 14
    out = df.sort("symbol", "event_time").with_columns(
        pl.col("close").diff().over("symbol").alias("_d")
    )
    out = out.with_columns(
        pl.when(pl.col("_d") > 0).then(pl.col("_d")).otherwise(0.0).alias("_g"),
        pl.when(pl.col("_d") < 0).then(-pl.col("_d")).otherwise(0.0).alias("_l"),
    )
    out = out.with_columns(
        pl.col("_g").rolling_mean(period, min_samples=period).over("symbol").alias("_ag"),
        pl.col("_l").rolling_mean(period, min_samples=period).over("symbol").alias("_al"),
    )
    out = out.with_columns((100.0 - 100.0 / (1.0 + pl.col("_ag") / pl.col("_al"))).alias("rsi14"))
    return out.select("symbol", "event_time", "rsi14")


@feature_view(
    name="rsi",
    version="v2",
    inputs=["close"],
    description="RSI(14) — Wilder smoothing (EMA with alpha=1/period).",
)
def rsi_v2(df: pl.DataFrame) -> pl.DataFrame:
    period = 14
    alpha = 1.0 / period
    out = df.sort("symbol", "event_time").with_columns(
        pl.col("close").diff().over("symbol").alias("_d")
    )
    out = out.with_columns(
        pl.when(pl.col("_d") > 0).then(pl.col("_d")).otherwise(0.0).alias("_g"),
        pl.when(pl.col("_d") < 0).then(-pl.col("_d")).otherwise(0.0).alias("_l"),
    )
    out = out.with_columns(
        pl.col("_g")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        .over("symbol")
        .alias("_ag"),
        pl.col("_l")
        .ewm_mean(alpha=alpha, adjust=False, min_samples=period)
        .over("symbol")
        .alias("_al"),
    )
    out = out.with_columns((100.0 - 100.0 / (1.0 + pl.col("_ag") / pl.col("_al"))).alias("rsi14"))
    return out.select("symbol", "event_time", "rsi14")


@feature_view(
    name="macd",
    version="v1",
    inputs=["close"],
    description="MACD(12, 26, 9) — line, signal, histogram.",
)
def macd_v1(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("symbol", "event_time").with_columns(
        _ema(pl.col("close"), 12).over("symbol").alias("_ema12"),
        _ema(pl.col("close"), 26).over("symbol").alias("_ema26"),
    )
    out = out.with_columns((pl.col("_ema12") - pl.col("_ema26")).alias("macd"))
    out = out.with_columns(_ema(pl.col("macd"), 9).over("symbol").alias("macd_signal"))
    out = out.with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"))
    return out.select("symbol", "event_time", "macd", "macd_signal", "macd_hist")


@feature_view(
    name="bollinger",
    version="v1",
    inputs=["close"],
    description="Bollinger bands(20, 2) — mid, upper, lower, %B.",
)
def bollinger_v1(df: pl.DataFrame) -> pl.DataFrame:
    period, k = 20, 2.0
    out = df.sort("symbol", "event_time").with_columns(
        pl.col("close").rolling_mean(period, min_samples=period).over("symbol").alias("bb_mid"),
        pl.col("close").rolling_std(period, min_samples=period).over("symbol").alias("_sd"),
    )
    out = out.with_columns(
        (pl.col("bb_mid") + k * pl.col("_sd")).alias("bb_upper"),
        (pl.col("bb_mid") - k * pl.col("_sd")).alias("bb_lower"),
    )
    out = out.with_columns(
        ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower"))).alias(
            "bb_pctb"
        )
    )
    return out.select("symbol", "event_time", "bb_mid", "bb_upper", "bb_lower", "bb_pctb")


@feature_view(
    name="atr",
    version="v1",
    inputs=["high", "low", "close"],
    description="Average True Range(14) — Wilder smoothing.",
)
def atr_v1(df: pl.DataFrame) -> pl.DataFrame:
    period = 14
    out = df.sort("symbol", "event_time").with_columns(
        pl.col("close").shift(1).over("symbol").alias("_prev_close"),
    )
    out = out.with_columns(
        pl.max_horizontal(
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("_prev_close")).abs(),
            (pl.col("low") - pl.col("_prev_close")).abs(),
        ).alias("_tr")
    )
    out = out.with_columns(
        pl.col("_tr")
        .ewm_mean(alpha=1.0 / period, adjust=False, min_samples=period)
        .over("symbol")
        .alias("atr14")
    )
    return out.select("symbol", "event_time", "atr14")


@feature_view(
    name="stochastic",
    version="v1",
    inputs=["high", "low", "close"],
    description="Stochastic oscillator(14, 3) — %K (fast) and %D (slow).",
)
def stochastic_v1(df: pl.DataFrame) -> pl.DataFrame:
    k_period, d_period = 14, 3
    out = df.sort("symbol", "event_time").with_columns(
        pl.col("low").rolling_min(k_period, min_samples=k_period).over("symbol").alias("_ll"),
        pl.col("high").rolling_max(k_period, min_samples=k_period).over("symbol").alias("_hh"),
    )
    out = out.with_columns(
        (100.0 * (pl.col("close") - pl.col("_ll")) / (pl.col("_hh") - pl.col("_ll"))).alias(
            "stoch_k"
        )
    )
    out = out.with_columns(
        pl.col("stoch_k")
        .rolling_mean(d_period, min_samples=d_period)
        .over("symbol")
        .alias("stoch_d")
    )
    return out.select("symbol", "event_time", "stoch_k", "stoch_d")


@feature_view(
    name="obv",
    version="v1",
    inputs=["close", "volume"],
    description="On-Balance Volume — running signed sum of volume on up/down bars.",
)
def obv_v1(df: pl.DataFrame) -> pl.DataFrame:
    out = df.sort("symbol", "event_time").with_columns(
        pl.col("close").diff().over("symbol").alias("_d")
    )
    out = out.with_columns(
        pl.when(pl.col("_d") > 0)
        .then(pl.col("volume"))
        .when(pl.col("_d") < 0)
        .then(-pl.col("volume"))
        .otherwise(0.0)
        .alias("_signed_vol")
    )
    out = out.with_columns(pl.col("_signed_vol").cum_sum().over("symbol").alias("obv"))
    return out.select("symbol", "event_time", "obv")
