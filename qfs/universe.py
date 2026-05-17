"""Survivorship-bias-free investable universe.

The default backtest mistake: filter to currently-listed S&P 500 names
and run the strategy 10 years back. That universe excludes everything
that went bankrupt, got acquired, or fell out of the index — exactly
the names whose poor returns matter most for honest evaluation.

The fix is to record universe membership bitemporally:

  (symbol, included_from, included_to, knowledge_time)

A symbol is investable at `as_of` t when:
  - some row exists with knowledge_time <= t (we knew the membership)
  - that row's included_from <= t
  - that row's included_to is NULL or > t

For each (symbol, included_from) the LATEST knowledge_time row wins —
this lets us record corrections without rewriting history.

Backtest harnesses should consult `members_as_of(t)` BEFORE pulling
features at t. If they don't, the universe is silently survivor-only.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl


_SCHEMA = {
    "symbol": pl.Utf8,
    "included_from": pl.Datetime("us"),
    "included_to": pl.Datetime("us"),
    "knowledge_time": pl.Datetime("us"),
}


class Universe:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "members.parquet"

    def add(
        self,
        symbol: str,
        included_from: datetime,
        included_to: datetime | None = None,
        knowledge_time: datetime | None = None,
    ) -> None:
        kt = knowledge_time or datetime.utcnow()
        row = pl.DataFrame(
            [(symbol, included_from, included_to, kt)],
            schema=_SCHEMA,
            orient="row",
        )
        if self._path.exists():
            existing = pl.read_parquet(self._path)
            row = pl.concat([existing, row], how="vertical_relaxed")
        row.write_parquet(self._path)

    def _all(self) -> pl.DataFrame:
        if not self._path.exists():
            return pl.DataFrame(schema=_SCHEMA)
        return pl.read_parquet(self._path)

    def members_as_of(self, as_of: datetime) -> list[str]:
        df = self._all()
        if df.is_empty():
            return []
        as_of_lit = pl.lit(as_of).cast(pl.Datetime("us"))
        # Only consider rows we knew at as_of.
        known = df.filter(pl.col("knowledge_time") <= as_of_lit)
        if known.is_empty():
            return []
        # For each (symbol, included_from), take the latest knowledge_time.
        latest = (
            known.sort("knowledge_time", descending=True)
            .group_by("symbol", "included_from")
            .agg(pl.all().first())
        )
        # Symbol is investable if any latest row has included_from <= as_of
        # AND (included_to is null OR included_to > as_of).
        active = latest.filter(
            (pl.col("included_from") <= as_of_lit)
            & (
                pl.col("included_to").is_null()
                | (pl.col("included_to") > as_of_lit)
            )
        )
        return sorted(active["symbol"].unique().to_list())
