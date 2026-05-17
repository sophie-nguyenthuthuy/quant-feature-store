"""Point-in-time correctness tests.

These are the load-bearing tests for the entire project. If any of them
fails, the store is unsafe for backtests.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from qfs import FeatureStore


@pytest.fixture
def store(tmp_path):
    return FeatureStore(tmp_path / "qfs")


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "knowledge_time": pl.Datetime("us"),
            "x": pl.Float64,
        },
        orient="row",
    )


def test_basic_read(store):
    t0 = datetime(2024, 1, 1)
    df = _frame([("AAPL", t0, t0, 1.0), ("AAPL", t0 + timedelta(days=1), t0 + timedelta(days=1), 2.0)])
    store.write(df, "f", "v1")
    q = pl.DataFrame({"symbol": ["AAPL"], "as_of": [t0 + timedelta(days=2)]})
    got = store.get_point_in_time(q, [("f", "v1")])
    assert got["f__v1__x"].to_list() == [2.0]


def test_no_lookahead_on_event_time(store):
    """A row stamped tomorrow must not be visible today."""
    t0 = datetime(2024, 1, 1)
    df = _frame([
        ("AAPL", t0,                     t0,                     1.0),
        ("AAPL", t0 + timedelta(days=5), t0 + timedelta(days=5), 999.0),  # the future
    ])
    store.write(df, "f", "v1")
    q = pl.DataFrame({"symbol": ["AAPL"], "as_of": [t0 + timedelta(days=2)]})
    got = store.get_point_in_time(q, [("f", "v1")])
    assert got["f__v1__x"].to_list() == [1.0]


def test_no_lookahead_on_knowledge_time(store):
    """A row whose event_time is in the past but that we did not yet KNOW
    must also be excluded. This is what generic feature stores get wrong.
    Models a macro revision: the figure 'happened' Jan 1 but was published
    Jan 10. A backtest as-of Jan 5 must not see it."""
    df = _frame([
        ("CPI", datetime(2024, 1, 1), datetime(2024, 1, 1),  100.0),   # first print
        ("CPI", datetime(2024, 1, 1), datetime(2024, 1, 10), 101.5),   # revision
    ])
    store.write(df, "macro", "v1")
    q = pl.DataFrame({"symbol": ["CPI"], "as_of": [datetime(2024, 1, 5)]})
    got = store.get_point_in_time(q, [("macro", "v1")])
    # As of Jan 5 we only knew the first print.
    assert got["macro__v1__x"].to_list() == [100.0]

    q2 = pl.DataFrame({"symbol": ["CPI"], "as_of": [datetime(2024, 1, 15)]})
    got2 = store.get_point_in_time(q2, [("macro", "v1")])
    # By Jan 15 the revision is known and is preferred (latest knowledge_time
    # for the same event_time).
    assert got2["macro__v1__x"].to_list() == [101.5]


def test_missing_feature_returns_null(store):
    t0 = datetime(2024, 1, 1)
    df = _frame([("AAPL", t0 + timedelta(days=10), t0 + timedelta(days=10), 1.0)])
    store.write(df, "f", "v1")
    q = pl.DataFrame({"symbol": ["AAPL"], "as_of": [t0]})
    got = store.get_point_in_time(q, [("f", "v1")])
    assert got["f__v1__x"].to_list() == [None]


def test_multi_symbol_and_multi_query(store):
    t0 = datetime(2024, 1, 1)
    df = _frame([
        ("AAPL", t0,                     t0,                     1.0),
        ("AAPL", t0 + timedelta(days=2), t0 + timedelta(days=2), 2.0),
        ("MSFT", t0,                     t0,                     10.0),
        ("MSFT", t0 + timedelta(days=2), t0 + timedelta(days=2), 20.0),
    ])
    store.write(df, "f", "v1")
    q = pl.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "as_of":  [t0 + timedelta(days=1), t0 + timedelta(days=3),
                   t0 + timedelta(days=1)],
    })
    got = store.get_point_in_time(q, [("f", "v1")])
    assert got["f__v1__x"].to_list() == [1.0, 2.0, 10.0]
