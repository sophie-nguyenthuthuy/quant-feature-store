"""Two versions of the same feature must coexist and remain independently
queryable. Old backtests stay reproducible against v1; new research uses v2."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from qfs import FeatureStore
import qfs.features  # noqa: F401  — register all views
from qfs.registry import registry
from qfs.data import load_ohlcv  # noqa: F401


def _toy_ohlcv():
    base = datetime(2024, 1, 1)
    rows = []
    price = 100.0
    for i in range(40):
        price *= 1.0 + ((-1) ** i) * 0.01
        rows.append(("AAPL", base + timedelta(days=i), price, price + 1, price - 1, price, 1_000_000.0))
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
            "close": pl.Float64, "volume": pl.Float64,
        },
        orient="row",
    )


def test_rsi_v1_and_v2_coexist(tmp_path):
    store = FeatureStore(tmp_path / "qfs")
    ohlcv = _toy_ohlcv()

    rsi_v1 = registry.get("rsi", "v1").compute(ohlcv)
    rsi_v2 = registry.get("rsi", "v2").compute(ohlcv)
    store.write(rsi_v1, "rsi", "v1", knowledge_time=datetime(2024, 3, 1))
    store.write(rsi_v2, "rsi", "v2", knowledge_time=datetime(2024, 3, 1))

    q = pl.DataFrame({"symbol": ["AAPL"], "as_of": [datetime(2024, 3, 1)]})
    got = store.get_point_in_time(q, [("rsi", "v1"), ("rsi", "v2")])
    v1 = got["rsi__v1__rsi14"][0]
    v2 = got["rsi__v2__rsi14"][0]
    assert v1 is not None and v2 is not None
    # Different smoothing -> different values, but both in (0, 100).
    assert 0 < v1 < 100 and 0 < v2 < 100
    assert v1 != v2
