"""Operational metadata: knowledge_lag distribution + restatement count
are the most useful health signals for a feature pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from qfs import FeatureStore


def _row(symbol, e, k, x):
    return (symbol, e, k, x)


def test_audit_basic(tmp_path):
    store = FeatureStore(tmp_path)
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame(
        [
            _row("AAPL", t0, t0 + timedelta(minutes=1), 1.0),
            _row("AAPL", t0 + timedelta(days=1), t0 + timedelta(days=1, minutes=1), 2.0),
            _row("MSFT", t0, t0 + timedelta(minutes=2), 10.0),
        ],
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "knowledge_time": pl.Datetime("us"),
            "x": pl.Float64,
        },
        orient="row",
    )
    store.write(df, "f", "v1")
    audits = store.audit()
    assert len(audits) == 1
    a = audits[0]
    assert a.view == "f" and a.version == "v1"
    assert a.row_count == 3
    assert a.n_symbols == 2
    assert a.knowledge_lag_p50 >= timedelta(minutes=1)
    assert a.knowledge_lag_p50 <= timedelta(minutes=2)
    assert a.n_restated_keys == 0


def test_audit_detects_restatement(tmp_path):
    """A restatement = same (symbol, event_time), different knowledge_time.
    This is the macro-revision pattern; audit should count it."""
    store = FeatureStore(tmp_path)
    t0 = datetime(2024, 1, 1)
    df = pl.DataFrame(
        [
            ("CPI", t0, t0, 100.0),
            ("CPI", t0, t0 + timedelta(days=10), 101.5),
        ],
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "knowledge_time": pl.Datetime("us"),
            "x": pl.Float64,
        },
        orient="row",
    )
    store.write(df, "macro", "v1")
    [a] = store.audit()
    assert a.n_restated_keys == 1
