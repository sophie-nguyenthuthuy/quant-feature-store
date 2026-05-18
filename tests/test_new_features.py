"""Smoke tests for ATR, Stochastic, OBV — verify they compute and produce
sensible warmup behavior. Numerical correctness is implicit via the
standard formulas; the heavier guarantee is that each view registers
cleanly and the compute fn returns the documented column set."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

import qfs.features  # noqa: F401
from qfs.registry import registry


def _ohlcv(n: int = 40):
    base = datetime(2024, 1, 1)
    rows = []
    price = 100.0
    vol = 1_000_000.0
    for i in range(n):
        price *= 1.0 + ((-1) ** i) * 0.01
        rows.append(
            ("AAPL", base + timedelta(days=i), price - 0.5, price + 1.0, price - 1.0, price, vol)
        )
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
        orient="row",
    )


def test_atr_computes_and_warms_up():
    out = registry.get("atr", "v1").compute(_ohlcv())
    assert set(out.columns) == {"symbol", "event_time", "atr14"}
    # First 13 rows null (warmup); thereafter positive.
    vals = out["atr14"].to_list()
    assert all(v is None for v in vals[:13])
    assert all(v is not None and v > 0 for v in vals[14:])


def test_stochastic_computes_in_range():
    out = registry.get("stochastic", "v1").compute(_ohlcv())
    assert set(out.columns) == {"symbol", "event_time", "stoch_k", "stoch_d"}
    # %K must be in [0, 100] when defined.
    for k in out["stoch_k"].to_list():
        if k is not None:
            assert 0 <= k <= 100


def test_obv_is_monotone_in_sign_changes():
    out = registry.get("obv", "v1").compute(_ohlcv())
    assert set(out.columns) == {"symbol", "event_time", "obv"}
    obv = out["obv"].to_list()
    # First row is null (no diff for first bar) or zero; subsequent rows
    # change by ±volume.
    diffs = [
        b - a for a, b in zip(obv[1:-1], obv[2:], strict=False) if a is not None and b is not None
    ]
    # Every diff must be a signed multiple of volume = 1_000_000.
    for d in diffs:
        assert abs(d) in (0.0, 1_000_000.0)
