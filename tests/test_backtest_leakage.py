"""Backtest-level proof that the store prevents leakage.

We build a perfect-foresight feature: 'tomorrow's return'. Then write it
TWICE into the store:

  - Honest version:  knowledge_time = event_time + 1 day (next morning,
                     after we observe tomorrow's bar).
  - Lying version:   knowledge_time = event_time (pretends we knew it
                     at today's close).

We run the SAME 'long if feature > 0' strategy against each. The lying
version produces a wildly-positive Sharpe; the honest version produces
near-zero because the feature is never available in time to act on. The
gap is the proof that the store's leakage prevention is doing real work.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from qfs import FeatureStore, backtest


def _make_bars(seed: int = 0, n: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    price = 100.0 * np.cumprod(1.0 + rets)
    base = datetime(2024, 1, 1)
    rows = [
        ("TST", base + timedelta(days=i), base + timedelta(days=i), price[i])
        for i in range(n)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "knowledge_time": pl.Datetime("us"),
            "close": pl.Float64,
        },
        orient="row",
    )


def _perfect_foresight_feature(bars: pl.DataFrame) -> pl.DataFrame:
    return bars.sort("event_time").with_columns(
        ((pl.col("close").shift(-1) - pl.col("close")) / pl.col("close")).alias("tomorrow_ret")
    ).drop_nulls("tomorrow_ret").select("symbol", "event_time", "tomorrow_ret")


def long_if_positive(features: pl.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in features.iter_rows(named=True):
        v = None
        for k, val in row.items():
            if k.endswith("__tomorrow_ret"):
                v = val
                break
        if v is None:
            out[row["symbol"]] = 0.0
        else:
            out[row["symbol"]] = 1.0 if v > 0 else -1.0
    return out


def test_lying_lag_beats_honest_lag(tmp_path):
    bars = _make_bars()
    feat = _perfect_foresight_feature(bars)

    # Lying: stamp knowledge_time = event_time. Pretends we knew tomorrow's
    # return at today's close.
    lying_store = FeatureStore(tmp_path / "lying")
    lying = feat.with_columns(pl.col("event_time").alias("knowledge_time"))
    lying_store.write(lying, "ff", "v1")

    # Honest: stamp knowledge_time = event_time + 1 day. We could only
    # truly know 'tomorrow's return' after tomorrow's close prints.
    honest_store = FeatureStore(tmp_path / "honest")
    honest = feat.with_columns(
        (pl.col("event_time") + pl.duration(days=1)).alias("knowledge_time")
    )
    honest_store.write(honest, "ff", "v1")

    lying_res = backtest(lying_store, bars, [("ff", "v1")], long_if_positive)
    honest_res = backtest(honest_store, bars, [("ff", "v1")], long_if_positive)

    # The lying version cheats successfully — large positive total return.
    # The honest version reflects realistic latency — near zero.
    assert lying_res.total_return > 0.5, lying_res.summary()
    assert abs(honest_res.total_return) < 0.2, honest_res.summary()
    assert lying_res.sharpe > 5.0 * max(abs(honest_res.sharpe), 0.1), (
        f"lying={lying_res.sharpe}, honest={honest_res.sharpe}"
    )
