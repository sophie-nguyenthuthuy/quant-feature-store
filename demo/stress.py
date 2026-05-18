"""Stress test: scale from 4 to 30 large-caps and measure each phase.

Purpose is to surface scaling cliffs before a real user hits them. Each
phase is timed independently so the bottleneck is obvious.

    .venv/bin/python demo/stress.py
"""

from __future__ import annotations

import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

import qfs.features  # noqa: F401
from qfs import FeatureStore, backtest
from qfs.data import load_ohlcv
from qfs.registry import registry

UNIVERSE = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "BRK-B",
    "AVGO",
    "JPM",
    "LLY",
    "V",
    "XOM",
    "WMT",
    "MA",
    "PG",
    "JNJ",
    "HD",
    "ORCL",
    "COST",
    "BAC",
    "MRK",
    "ABBV",
    "CVX",
    "KO",
    "PEP",
    "ADBE",
    "CRM",
    "MCD",
    "TMO",
]
START = "2023-01-01"
END = "2025-01-01"


@contextmanager
def timed(label: str, results: dict):
    t0 = time.perf_counter()
    yield
    results[label] = time.perf_counter() - t0
    print(f"  {label:20s} {results[label]:6.2f}s")


def buy_and_hold(features: pl.DataFrame) -> dict[str, float]:
    return {row["symbol"]: 1.0 for row in features.iter_rows(named=True)}


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data-stress"
    if root.exists():
        shutil.rmtree(root)
    store = FeatureStore(root)
    results: dict[str, float] = {}

    print(f"universe: {len(UNIVERSE)} tickers, {START} -> {END}")

    with timed("load_ohlcv", results):
        bars = load_ohlcv(UNIVERSE, START, END, publish_lag_minutes=1)
    print(f"    bars: {len(bars):,}")

    with timed("compute+write all", results):
        for v in registry.all():
            feat = v.compute(bars).join(
                bars.select("symbol", "event_time", "knowledge_time"),
                on=["symbol", "event_time"],
                how="left",
            )
            store.write(feat, v.name, v.version)

    with timed("audit all views", results):
        audits = store.audit()
    print(f"    views: {len(audits)}, total rows: {sum(a.row_count for a in audits):,}")

    with timed("point-in-time pull", results):
        queries = bars.select(
            "symbol",
            pl.col("knowledge_time").alias("as_of"),
        )
        feats = store.get_point_in_time(queries, [(v.name, v.version) for v in registry.all()])
    print(f"    feat rows: {feats.height:,}")

    with timed("backtest buy&hold", results):
        result = backtest(store, bars, [("rsi", "v2")], buy_and_hold)
    print(f"    total_return = {result.total_return:+.2%}, sharpe = {result.sharpe:.2f}")

    total = sum(results.values())
    print(f"\ntotal: {total:.2f}s")
    print(f"\nbottleneck: {max(results, key=results.get)} ({max(results.values()):.2f}s)")


if __name__ == "__main__":
    main()
