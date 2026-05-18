"""Worked example: real GDP revision history through the bitemporal store.

This is the canonical use case for the knowledge_time dimension. Macro
statistics get revised — sometimes by a lot. The Bureau of Economic
Analysis publishes three quarterly estimates (Advance / Second / Third)
30 days apart, then annual revisions for years afterwards. Every print
of "GDP for Q4 2008" is a different number depending on the vintage.

A backtest that uses the final-revision GDP throughout history is
silently cheating. A backtest that uses the first vintage available at
the model's decision point is honest. This script demonstrates the
difference using public BEA vintages.

    .venv/bin/python demo/macro_revisions.py
"""

from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from qfs import FeatureStore


# Real BEA vintages: U.S. real GDP, quarter-over-quarter annualised rate.
# Each row is one published estimate. Multiple estimates per quarter is
# the whole point. Numbers are from the BEA archive / FRED ALFRED.
#
# event_time     : end of the reference quarter (what the number is ABOUT)
# knowledge_time : the date the estimate was published (when we KNEW it)
# value          : annualised QoQ percent change
VINTAGES: list[tuple[str, str, str, float]] = [
    # Q4 2008 — the GFC quarter. Initial print was -3.8%. Reality, after
    # successive revisions, was -8.4%. A trader looking at "Q4 2008 GDP"
    # in March 2009 thought it was bad; today we know it was catastrophic.
    ("GDP_US", "2008-12-31", "2009-01-30", -3.8),  # Advance
    ("GDP_US", "2008-12-31", "2009-02-27", -6.2),  # Second
    ("GDP_US", "2008-12-31", "2009-03-26", -6.3),  # Third
    ("GDP_US", "2008-12-31", "2009-07-31", -6.3),  # Annual revision
    ("GDP_US", "2008-12-31", "2013-07-31", -8.4),  # Comprehensive revision

    # Q1 2020 — COVID quarter. Less dramatic revision but the same shape.
    ("GDP_US", "2020-03-31", "2020-04-29", -4.8),  # Advance
    ("GDP_US", "2020-03-31", "2020-05-28", -5.0),  # Second
    ("GDP_US", "2020-03-31", "2020-06-25", -5.0),  # Third
    ("GDP_US", "2020-03-31", "2021-07-29", -5.1),  # Annual revision

    # Q2 2024 — recent quarter to show the normal cadence.
    ("GDP_US", "2024-06-30", "2024-07-25",  2.8),
    ("GDP_US", "2024-06-30", "2024-08-29",  3.0),
    ("GDP_US", "2024-06-30", "2024-09-26",  3.0),
]


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        VINTAGES,
        schema={
            "symbol": pl.Utf8,
            "event_time": pl.Utf8,
            "knowledge_time": pl.Utf8,
            "gdp_qoq_ann": pl.Float64,
        },
        orient="row",
    ).with_columns(
        pl.col("event_time").str.to_datetime(),
        pl.col("knowledge_time").str.to_datetime(),
    )


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="qfs_macro_"))
    try:
        store = FeatureStore(workdir / "store")
        store.write(_frame(), view="gdp", version="v1")

        print("=" * 72)
        print("Real BEA vintages for US GDP — the same event_time, multiple")
        print("knowledge_times, different values. The bitemporal pattern.")
        print("=" * 72)
        with pl.Config(tbl_rows=20, tbl_width_chars=120):
            print(_frame().sort("event_time", "knowledge_time"))

        print()
        print("=" * 72)
        print("Point-in-time read: 'what did we know about Q4 2008 GDP at...'")
        print("=" * 72)
        viewpoints = [
            "2009-01-15",  # before any print
            "2009-02-01",  # after Advance only
            "2009-03-01",  # after Second
            "2009-04-01",  # after Third
            "2010-01-01",  # after first annual revision
            "2015-01-01",  # after comprehensive revision
        ]
        rows = []
        for v in viewpoints:
            q = pl.DataFrame({
                "symbol": ["GDP_US"],
                "as_of": [datetime.fromisoformat(v)],
            })
            out = store.get_point_in_time(q, [("gdp", "v1")])
            val = out["gdp__v1__gdp_qoq_ann"][0]
            rows.append({
                "as_of": v,
                "Q4_2008_GDP_known_to_be": "n/a" if val is None else f"{val:+.1f}%",
            })
        with pl.Config(tbl_rows=20):
            print(pl.DataFrame(rows))

        print()
        print("=" * 72)
        print("The teaching point: a strategy run in March 2009 saw Q4 GDP as")
        print(" -6.3%. A 2015 strategy backtesting the same period sees -8.4%.")
        print("Using the -8.4% figure in a 2009-vintage backtest is leakage —")
        print("the model is reacting to information it could not have had.")
        print("=" * 72)

        print()
        print("=" * 72)
        print("audit() catches restatements automatically:")
        print("=" * 72)
        for a in store.audit():
            print(f"  {a}")
        print(
            "\nn_restated_keys = 3 — exactly the three quarters above. Each\n"
            "has multiple knowledge_times for the same event_time, so the\n"
            "audit flags them as bitemporal restatements."
        )

    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
