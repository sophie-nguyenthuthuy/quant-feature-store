"""RSI mean-reversion strategy run through the leakage-safe backtest.

Long when RSI(14) < 30, short when > 70, flat otherwise. Naive and not
expected to make money on a 4-ticker liquid universe — the point is to
show the end-to-end loop: ingest -> store -> point-in-time pull ->
strategy -> simulated PnL.

Run from project root after demo/backtest.py has populated ./data:
    .venv/bin/python demo/strategy_rsi_meanrev.py
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from qfs import FeatureStore, backtest
import qfs.features  # noqa: F401
from qfs.data import load_ohlcv
from qfs.registry import registry


SYMBOLS = ["AAPL", "MSFT", "SPY", "NVDA"]
START = "2023-01-01"
END = "2025-01-01"


def rsi_meanrev(features: pl.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in features.iter_rows(named=True):
        rsi = row.get("rsi__v2__rsi14")
        if rsi is None:
            out[row["symbol"]] = 0.0
        elif rsi < 30:
            out[row["symbol"]] = 1.0
        elif rsi > 70:
            out[row["symbol"]] = -1.0
        else:
            out[row["symbol"]] = 0.0
    return out


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data"
    store = FeatureStore(root)

    # Ensure features exist; ingest if the store is empty.
    if not store.list_views():
        print("store empty — ingesting first")
        ohlcv = load_ohlcv(SYMBOLS, START, END, publish_lag_minutes=1)
        for v in registry.all():
            feat = v.compute(ohlcv).join(
                ohlcv.select("symbol", "event_time", "knowledge_time"),
                on=["symbol", "event_time"], how="left",
            )
            store.write(feat, v.name, v.version)
        bars = ohlcv
    else:
        bars = load_ohlcv(SYMBOLS, START, END, publish_lag_minutes=1)

    print(f"running RSI mean-reversion on {SYMBOLS}, {bars['event_time'].min()} -> {bars['event_time'].max()}")
    result = backtest(
        store=store,
        bars=bars,
        features=[("rsi", "v2")],
        strategy=rsi_meanrev,
    )
    print(result.summary())


if __name__ == "__main__":
    main()
