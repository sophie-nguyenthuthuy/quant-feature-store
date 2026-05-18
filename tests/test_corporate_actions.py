"""Corporate actions are bitemporal: the same raw bar yields different
adjusted prices depending on WHEN you ask. These tests pin down the
property for splits AND for dividends."""

from __future__ import annotations

from datetime import datetime

import polars as pl

from qfs.corporate_actions import adjusted_ohlcv_as_of


def _bars():
    # Synthetic AAPL-like 4:1 split between day 3 and day 4.
    rows = [
        ("AAPL", datetime(2020, 8, 28), 395.0, 405.0, 390.0, 400.0),
        ("AAPL", datetime(2020, 8, 29), 405.0, 415.0, 400.0, 410.0),
        ("AAPL", datetime(2020, 8, 30), 415.0, 425.0, 410.0, 420.0),
        ("AAPL", datetime(2020, 8, 31), 99.0, 102.0, 98.0, 100.0),  # split day
        ("AAPL", datetime(2020, 9, 1), 101.0, 106.0, 100.0, 105.0),
    ]
    return pl.DataFrame(
        rows,
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Datetime("us"),
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
        },
        orient="row",
    )


def _splits():
    return pl.DataFrame(
        [("AAPL", datetime(2020, 8, 31), datetime(2020, 7, 30), 4.0)],
        schema={
            "symbol": pl.Utf8,
            "ex_date": pl.Datetime("us"),
            "knowledge_time": pl.Datetime("us"),
            "ratio": pl.Float64,
        },
        orient="row",
    )


def _dividends():
    return pl.DataFrame(
        [("AAPL", datetime(2020, 8, 31), datetime(2020, 8, 1), 1.0)],
        schema={
            "symbol": pl.Utf8,
            "ex_date": pl.Datetime("us"),
            "knowledge_time": pl.Datetime("us"),
            "amount": pl.Float64,
        },
        orient="row",
    )


# ---- splits ----


def test_pre_announcement_view_unchanged():
    """Before the split was announced, no historical adjustment applies."""
    out = adjusted_ohlcv_as_of(_bars(), as_of=datetime(2020, 6, 1), splits=_splits())
    assert out["adj_close"].to_list() == [400.0, 410.0, 420.0, 100.0, 105.0]


def test_announced_but_not_ex_view_unchanged():
    """The split is known but has not yet taken effect — historical prices
    still reflect their unsplit form. Adjusting them now would inject the
    future into the past."""
    out = adjusted_ohlcv_as_of(_bars(), as_of=datetime(2020, 8, 15), splits=_splits())
    assert out["adj_close"].to_list() == [400.0, 410.0, 420.0, 100.0, 105.0]


def test_post_ex_view_retroactively_adjusts():
    """Once the split has happened, ALL bars before ex_date are divided
    by the ratio. Bars on or after ex_date are unchanged."""
    out = adjusted_ohlcv_as_of(_bars(), as_of=datetime(2020, 9, 30), splits=_splits())
    assert out["adj_close"].to_list() == [100.0, 102.5, 105.0, 100.0, 105.0]
    assert out["adj_open"].to_list() == [395.0 / 4, 405.0 / 4, 415.0 / 4, 99.0, 101.0]


def test_no_eligible_splits_returns_passthrough():
    out = adjusted_ohlcv_as_of(
        _bars(),
        as_of=datetime(2030, 1, 1),
        splits=_splits().clear(),
    )
    assert out["adj_close"].to_list() == out["close"].to_list()


# ---- dividends ----


def test_dividend_pre_ex_view_unchanged():
    """Before the dividend goes ex, historical prices are not yet
    back-adjusted."""
    out = adjusted_ohlcv_as_of(
        _bars(),
        as_of=datetime(2020, 8, 15),
        dividends=_dividends(),
    )
    assert out["adj_close"].to_list() == [400.0, 410.0, 420.0, 100.0, 105.0]


def test_dividend_post_ex_subtracts_from_history():
    """Once the dividend has been paid, every prior bar gets its prices
    reduced by the dividend amount. Bars on/after ex_date are unchanged."""
    out = adjusted_ohlcv_as_of(
        _bars(),
        as_of=datetime(2020, 9, 30),
        dividends=_dividends(),
    )
    assert out["adj_close"].to_list() == [399.0, 409.0, 419.0, 100.0, 105.0]


def test_splits_and_dividends_combined():
    """Splits adjust first, then dividends in the split-adjusted price
    space. Pre-split close 400 -> /4 -> 100 -> -1 div -> 99."""
    out = adjusted_ohlcv_as_of(
        _bars(),
        as_of=datetime(2020, 9, 30),
        splits=_splits(),
        dividends=_dividends(),
    )
    assert out["adj_close"].to_list() == [99.0, 101.5, 104.0, 100.0, 105.0]


def test_no_actions_returns_passthrough():
    out = adjusted_ohlcv_as_of(_bars(), as_of=datetime(2030, 1, 1))
    assert out["adj_close"].to_list() == out["close"].to_list()
