"""Bitemporal feature store with point-in-time reads.

Every row carries two timestamps:
  - event_time:     when the fact occurred in the world (e.g. bar close)
  - knowledge_time: when we became aware of it (ingestion / publication)

A point-in-time read at time t returns, per entity, the row maximising
event_time then knowledge_time subject to BOTH timestamps <= t. This is
the only join that makes backtests honest: it never returns a value the
strategy could not have known at time t.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

ENTITY_COL = "symbol"
EVENT_COL = "event_time"
KNOWLEDGE_COL = "knowledge_time"


@dataclass(frozen=True)
class FeatureRef:
    view: str
    version: str

    @property
    def path_segment(self) -> str:
        return f"{self.view}__{self.version}"


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(":memory:")
        # Fast-path eligibility is data-dependent and stable until the next
        # write into that view. Cache to avoid re-probing on every read.
        self._fast_path_cache: dict[str, bool] = {}

    def _view_dir(self, ref: FeatureRef) -> Path:
        return self.root / ref.path_segment

    def write(
        self,
        df: pl.DataFrame,
        view: str,
        version: str,
        knowledge_time: datetime | None = None,
    ) -> None:
        """Append a batch of feature rows.

        df must contain the entity column and event_time. If knowledge_time
        is not a column it is stamped from the argument (defaulting to now).
        """
        if ENTITY_COL not in df.columns:
            raise ValueError(f"missing column {ENTITY_COL!r}")
        if EVENT_COL not in df.columns:
            raise ValueError(f"missing column {EVENT_COL!r}")

        if KNOWLEDGE_COL not in df.columns:
            kt = knowledge_time or datetime.utcnow()
            df = df.with_columns(pl.lit(kt).alias(KNOWLEDGE_COL))

        df = df.with_columns(
            pl.col(EVENT_COL).cast(pl.Datetime("us")),
            pl.col(KNOWLEDGE_COL).cast(pl.Datetime("us")),
        )

        ref = FeatureRef(view, version)
        out_dir = self._view_dir(ref)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        df.write_parquet(out_dir / f"batch_{stamp}.parquet")
        self._fast_path_cache.pop(ref.path_segment, None)

    def get_point_in_time(
        self,
        entity_df: pl.DataFrame,
        features: Iterable[tuple[str, str]],
        as_of_col: str = "as_of",
    ) -> pl.DataFrame:
        """Point-in-time feature retrieval.

        entity_df: must contain `symbol` and `as_of` (timestamp) columns.
        features: iterable of (view, version) pairs.

        Returns entity_df augmented with one column per feature, prefixed
        as `<view>__<version>__<col>`. Rows for which no eligible feature
        row exists become null — the strategy should treat null as 'feature
        not yet available' and abstain.
        """
        if ENTITY_COL not in entity_df.columns:
            raise ValueError(f"entity_df missing {ENTITY_COL!r}")
        if as_of_col not in entity_df.columns:
            raise ValueError(f"entity_df missing {as_of_col!r}")

        entity = entity_df.with_columns(pl.col(as_of_col).cast(pl.Datetime("us"))).with_row_index(
            "__qid"
        )
        self._con.register("entity", entity)

        result = entity
        for view, version in features:
            ref = FeatureRef(view, version)
            view_dir = self._view_dir(ref)
            parquet_glob = str(view_dir / "*.parquet")
            if not list(view_dir.glob("*.parquet")):
                raise FileNotFoundError(f"no data for {ref.path_segment}")

            sql = self._build_pit_query(parquet_glob, as_of_col, ref.path_segment)
            feat_df = pl.from_arrow(self._con.execute(sql).to_arrow_table())

            prefix = ref.path_segment + "__"
            renames = {c: prefix + c for c in feat_df.columns if c != "__qid"}
            feat_df = feat_df.rename(renames)
            result = result.join(feat_df, on="__qid", how="left")

        return result.drop("__qid")

    def _build_pit_query(self, parquet_glob: str, as_of_col: str, cache_key: str) -> str:
        """Pick the read path.

        DuckDB ASOF JOIN supports only ONE inequality, so we can't directly
        express the bitemporal "max event_time AND max knowledge_time"
        condition in one pass. The slow path uses a window function with
        two inequalities in the join predicate; it is always correct.

        The fast path uses ASOF on knowledge_time alone. It is correct
        only when:
          (1) no restatements: (symbol, event_time) is unique
          (2) STRICTLY increasing publishing: per symbol, knowledge_time
              is strictly greater for later event_times (so max kt
              uniquely picks max et — no ties for ASOF to misresolve)
          (3) no predictive timestamps: knowledge_time >= event_time
              (so kt <= as_of implies et <= as_of)

        These conditions cover the overwhelming common case (real-time
        feeds with a per-row publishing lag). Macro revisions, constant-
        timestamp backfills, and late-arriving corrections automatically
        fall back to the slow path.

        Detection is one cheap SQL probe per view, cached for subsequent
        reads (invalidated on the next write to that view).
        """
        if cache_key in self._fast_path_cache:
            fast_path_ok = self._fast_path_cache[cache_key]
        else:
            probe = self._con.execute(f"""
                WITH f AS (SELECT * FROM read_parquet('{parquet_glob}')),
                     lagged AS (
                         SELECT
                             {KNOWLEDGE_COL},
                             {EVENT_COL},
                             LAG({KNOWLEDGE_COL}) OVER (
                                 PARTITION BY {ENTITY_COL} ORDER BY {EVENT_COL}
                             ) AS prev_kt
                         FROM f
                     )
                SELECT
                    (SELECT COUNT(*) <> COUNT(DISTINCT ({ENTITY_COL}, {EVENT_COL}))
                       FROM f) AS has_restatements,
                    (SELECT BOOL_OR({KNOWLEDGE_COL} < {EVENT_COL}) FROM f) AS has_predictive,
                    (SELECT BOOL_OR({KNOWLEDGE_COL} <= prev_kt) FROM lagged) AS not_strict
            """).fetchone()
            fast_path_ok = not any(probe)
            self._fast_path_cache[cache_key] = fast_path_ok

        if fast_path_ok:
            return f"""
            WITH feats AS (
                SELECT * FROM read_parquet('{parquet_glob}')
            )
            SELECT
                e.__qid,
                f.* EXCLUDE ({ENTITY_COL}, {EVENT_COL}, {KNOWLEDGE_COL})
            FROM entity e
            ASOF LEFT JOIN feats f
                ON e.{ENTITY_COL} = f.{ENTITY_COL}
               AND e.{as_of_col} >= f.{KNOWLEDGE_COL}
            """

        return f"""
        WITH feats AS (
            SELECT * FROM read_parquet('{parquet_glob}')
        ),
        candidates AS (
            SELECT
                e.__qid,
                f.* EXCLUDE ({ENTITY_COL}, {EVENT_COL}, {KNOWLEDGE_COL}),
                ROW_NUMBER() OVER (
                    PARTITION BY e.__qid
                    ORDER BY f.{EVENT_COL} DESC, f.{KNOWLEDGE_COL} DESC
                ) AS rn
            FROM entity e
            LEFT JOIN feats f
                ON e.{ENTITY_COL} = f.{ENTITY_COL}
               AND f.{EVENT_COL} <= e.{as_of_col}
               AND f.{KNOWLEDGE_COL} <= e.{as_of_col}
        )
        SELECT * EXCLUDE (rn) FROM candidates WHERE rn = 1 OR rn IS NULL
        """

    def list_views(self) -> list[FeatureRef]:
        out = []
        for p in self.root.iterdir():
            if p.is_dir() and "__" in p.name:
                view, version = p.name.split("__", 1)
                out.append(FeatureRef(view, version))
        return out

    def audit(self, view: str | None = None, version: str | None = None) -> list[ViewAudit]:
        """Per-view operational stats: row count, time ranges, the
        distribution of knowledge_lag (= knowledge_time - event_time),
        and count of restated keys (same (symbol, event_time), multiple
        knowledge_times). The lag distribution is the single most useful
        number for catching data-pipeline regressions — when a view's
        p95 lag jumps from 1 hour to 1 day overnight, the upstream feed
        broke."""
        refs = self.list_views()
        if view is not None:
            refs = [r for r in refs if r.view == view and (version is None or r.version == version)]
        out: list[ViewAudit] = []
        for ref in refs:
            view_dir = self._view_dir(ref)
            files = sorted(view_dir.glob("*.parquet"))
            if not files:
                continue
            df = pl.read_parquet([str(f) for f in files])
            lag_us = (df[KNOWLEDGE_COL] - df[EVENT_COL]).dt.total_microseconds()
            restated = (
                df.group_by(ENTITY_COL, EVENT_COL)
                .agg(pl.col(KNOWLEDGE_COL).count().alias("n"))
                .filter(pl.col("n") > 1)
                .height
            )
            out.append(
                ViewAudit(
                    view=ref.view,
                    version=ref.version,
                    row_count=df.height,
                    n_symbols=df[ENTITY_COL].n_unique(),
                    event_time_min=df[EVENT_COL].min(),
                    event_time_max=df[EVENT_COL].max(),
                    knowledge_time_min=df[KNOWLEDGE_COL].min(),
                    knowledge_time_max=df[KNOWLEDGE_COL].max(),
                    knowledge_lag_mean=timedelta(microseconds=int(lag_us.mean())),
                    knowledge_lag_p50=timedelta(microseconds=int(lag_us.median())),
                    knowledge_lag_p95=timedelta(microseconds=int(lag_us.quantile(0.95))),
                    knowledge_lag_max=timedelta(microseconds=int(lag_us.max())),
                    n_restated_keys=restated,
                )
            )
        return out


@dataclass
class ViewAudit:
    view: str
    version: str
    row_count: int
    n_symbols: int
    event_time_min: datetime
    event_time_max: datetime
    knowledge_time_min: datetime
    knowledge_time_max: datetime
    knowledge_lag_mean: timedelta
    knowledge_lag_p50: timedelta
    knowledge_lag_p95: timedelta
    knowledge_lag_max: timedelta
    n_restated_keys: int

    def __str__(self) -> str:
        return (
            f"{self.view}@{self.version}: "
            f"rows={self.row_count:,} symbols={self.n_symbols} "
            f"event=[{self.event_time_min:%Y-%m-%d}..{self.event_time_max:%Y-%m-%d}] "
            f"lag p50={self.knowledge_lag_p50} p95={self.knowledge_lag_p95} "
            f"restated={self.n_restated_keys}"
        )
