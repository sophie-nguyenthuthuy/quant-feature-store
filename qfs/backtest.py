"""Walk-forward backtest engine that uses the store's point-in-time read.

The strategy function is called once per bar with the features pulled
as-of that bar's knowledge_time. Whatever target positions it returns
take effect at the NEXT bar's close — there is no way for the strategy
to act on information it could not have known at the moment it ran.

This is the simplest possible simulator that still has the right
leakage-prevention shape. It deliberately does NOT model:
  - transaction costs / slippage
  - fractional fills, position sizing models, margin
  - leverage caps, exposure limits
  - intraday execution
Add those for real research; for proving the store works, they're noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

from qfs.store import FeatureStore


# strategy(features_at_t) -> {symbol: target_position}
# features_at_t is a polars frame with one row per symbol, columns are
# the joined feature columns from the store.
Strategy = Callable[[pl.DataFrame], dict[str, float]]


@dataclass
class BacktestResult:
    equity: pl.DataFrame      # columns: as_of, equity
    positions: pl.DataFrame   # columns: as_of, symbol, position
    returns: pl.DataFrame     # columns: as_of, ret
    total_return: float
    sharpe: float
    max_drawdown: float
    turnover: float           # mean absolute change in position per bar

    def summary(self) -> str:
        return (
            f"total_return = {self.total_return:+.2%}\n"
            f"sharpe       = {self.sharpe:.2f}\n"
            f"max_drawdown = {self.max_drawdown:.2%}\n"
            f"turnover     = {self.turnover:.3f}"
        )


def _sharpe(rets: np.ndarray, periods_per_year: int = 252) -> float:
    if rets.size < 2:
        return 0.0
    sd = rets.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(rets.mean() / sd * np.sqrt(periods_per_year))


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def backtest(
    store: FeatureStore,
    bars: pl.DataFrame,
    features: list[tuple[str, str]],
    strategy: Strategy,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Run a walk-forward backtest.

    bars: OHLCV with columns symbol, event_time, knowledge_time, close.
          Defines the universe and the time grid; the close of each bar
          is used to mark equity.
    features: list of (view, version) pairs to pull at each step.
    strategy: function (features_frame) -> {symbol: target_position}.
    """
    required = {"symbol", "event_time", "knowledge_time", "close"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"bars missing columns: {missing}")

    bars = bars.sort("event_time", "symbol")
    timeline = bars.select("event_time", "knowledge_time").unique(
        subset=["event_time"]
    ).sort("event_time")

    # One pull for the whole backtest: ask the store for features at every
    # (symbol, knowledge_time). Vectorised join >> per-bar loop.
    queries = (
        bars.select("symbol", pl.col("knowledge_time").alias("as_of"))
        .unique()
        .sort("as_of", "symbol")
    )
    feats = store.get_point_in_time(queries, features)
    feats = feats.with_columns(pl.col("as_of").alias("knowledge_time"))

    # Per-bar prices keyed by knowledge_time so we can mark equity at each
    # decision point. Returns are close-to-close on the bar AFTER the one
    # whose features the strategy used.
    prices = bars.select("symbol", "knowledge_time", "close").sort(
        "knowledge_time", "symbol"
    )

    timeline_rows = timeline.to_dicts()
    symbols = sorted(bars["symbol"].unique().to_list())
    positions: dict[str, float] = {s: 0.0 for s in symbols}

    equity = initial_equity
    equity_rows: list[dict] = []
    position_rows: list[dict] = []
    return_rows: list[dict] = []
    turnover_acc = 0.0
    turnover_n = 0

    prev_prices: dict[str, float] = {}
    prev_positions = dict(positions)

    for tl in timeline_rows:
        kt = tl["knowledge_time"]

        # Realise returns from the position taken at the PREVIOUS step,
        # using the price change between previous bar and this bar.
        cur_prices = {
            r["symbol"]: r["close"]
            for r in prices.filter(pl.col("knowledge_time") == kt).iter_rows(named=True)
        }
        if prev_prices:
            bar_ret = 0.0
            n_active = 0
            for sym, pos in prev_positions.items():
                p0 = prev_prices.get(sym)
                p1 = cur_prices.get(sym)
                if p0 is None or p1 is None or pos == 0.0:
                    continue
                bar_ret += pos * (p1 - p0) / p0
                n_active += 1
            if n_active > 0:
                bar_ret /= n_active   # equal-weighted across active positions
            equity *= (1.0 + bar_ret)
            return_rows.append({"as_of": kt, "ret": bar_ret})

        equity_rows.append({"as_of": kt, "equity": equity})

        # Decide new positions using features knowable at kt.
        feats_now = feats.filter(pl.col("knowledge_time") == kt).drop("knowledge_time")
        if feats_now.height == 0:
            target = dict(positions)
        else:
            target = strategy(feats_now)

        for sym in symbols:
            new = float(target.get(sym, 0.0))
            turnover_acc += abs(new - positions[sym])
            positions[sym] = new
            position_rows.append({"as_of": kt, "symbol": sym, "position": new})
        turnover_n += 1

        prev_prices = cur_prices
        prev_positions = dict(positions)

    eq_df = pl.DataFrame(equity_rows)
    pos_df = pl.DataFrame(position_rows)
    ret_df = pl.DataFrame(return_rows) if return_rows else pl.DataFrame(
        {"as_of": [], "ret": []}
    )

    rets = ret_df["ret"].to_numpy() if ret_df.height else np.array([])
    eq_arr = eq_df["equity"].to_numpy() if eq_df.height else np.array([])
    total = float(eq_arr[-1] / eq_arr[0] - 1.0) if eq_arr.size else 0.0
    turnover = (turnover_acc / max(turnover_n, 1)) / max(len(symbols), 1)

    return BacktestResult(
        equity=eq_df,
        positions=pos_df,
        returns=ret_df,
        total_return=total,
        sharpe=_sharpe(rets),
        max_drawdown=_max_drawdown(eq_arr),
        turnover=turnover,
    )
