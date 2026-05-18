"""End-to-end demo: ingest, register features, point-in-time backtest.

Run from project root:
    uv run python demo/backtest.py

Pulls ~2 years of daily bars for a handful of tickers, materialises every
registered feature view, and runs a leakage-safe pull aligned to bar
closes — exactly how a research notebook would consume the store.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

import qfs.features  # noqa: F401  — side-effect registration
from qfs import FeatureStore
from qfs.data import load_ohlcv
from qfs.registry import registry

SYMBOLS = ["AAPL", "MSFT", "SPY", "NVDA"]
START = "2023-01-01"
END = "2025-01-01"


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data"
    store = FeatureStore(root)

    print(f"loading OHLCV for {SYMBOLS} {START} -> {END}")
    ohlcv = load_ohlcv(SYMBOLS, START, END, publish_lag_minutes=1)
    if ohlcv.is_empty():
        raise SystemExit("no data returned (offline?)")
    print(f"  {len(ohlcv):,} bars")

    # Compute every registered view and write it.
    # knowledge_time is taken from the OHLCV bar itself (event_time + 1min),
    # since a feature derived from a bar is only knowable after that bar prints.
    for view in registry.all():
        feat = view.compute(ohlcv)
        feat = feat.join(
            ohlcv.select("symbol", "event_time", "knowledge_time"),
            on=["symbol", "event_time"],
            how="left",
        )
        store.write(feat, view.name, view.version)
        print(f"  wrote {view.name}@{view.version}: {len(feat):,} rows")

    # Build query frame: each bar's close becomes a research point. We pull
    # features as-of the bar's knowledge_time, NOT its event_time — this is
    # the realistic moment a strategy could act.
    queries = ohlcv.select(
        "symbol",
        pl.col("knowledge_time").alias("as_of"),
    )
    feature_refs = [(v.name, v.version) for v in registry.all()]
    df = store.get_point_in_time(queries, feature_refs)

    last = df.sort("as_of").tail(5)
    print("\nlast 5 feature rows:")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(last)

    # Sanity: oldest queries should have null features (warmup).
    oldest = df.sort("as_of").head(1)
    null_count = sum(
        1 for c in oldest.columns if c not in ("symbol", "as_of") and oldest[c][0] is None
    )
    print(f"\nfeatures null at first bar (warmup, expected non-zero): {null_count}")


if __name__ == "__main__":
    main()
