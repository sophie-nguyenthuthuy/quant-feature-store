# qfs — quant feature store

![ci](https://github.com/sophie-nguyenthuthuy/quant-feature-store/actions/workflows/ci.yml/badge.svg)

A small, opinionated feature store for trading signals. Optimised for the
one thing most general-purpose feature stores get wrong: **point-in-time
correctness during backtests.**

## Why this exists

Generic feature stores (Feast, Tecton, in-house clones) handle the
operational side of features well but treat backtests as a second-class
citizen. For trading, "what did the model know at time t?" is the entire
game — and it is harder than it looks.

Three subtle ways the future leaks into a backtest:

1. **Event-time leakage** — a feature row whose underlying bar happens
   *after* the query timestamp.
2. **Knowledge-time leakage** — a feature row whose underlying fact
   *predates* the query timestamp but was not published / scraped /
   computed until *later*. Macro releases get revised. Sentiment scrapes
   lag. On-chain data is subject to reorgs.
3. **Corporate-action leakage** — a split announced today reaches back
   in time and reshapes the historical price series. Naive systems use
   the latest-known adjusted prices everywhere, so a backtest of AAPL
   August 2020 'sees' the 4:1 split before it happened.

`qfs` treats every fact as bitemporal (`event_time` + `knowledge_time`)
and only returns rows where both timestamps are <= the query time.
Corporate actions are the same model applied to splits. There is one
read path and it is leakage-safe by construction.

## The load-bearing tests

[`tests/test_point_in_time.py`](tests/test_point_in_time.py)
encodes the macro-revision scenario most stores silently fail.

[`tests/test_backtest_leakage.py`](tests/test_backtest_leakage.py)
runs the same strategy with a perfect-foresight feature stamped two
ways: honest (knowledge_time = next morning) and lying (knowledge_time =
today's close). The lying version produces a fake-positive Sharpe; the
honest version is flat. The gap is the proof the store earns its keep.

[`tests/test_corporate_actions.py`](tests/test_corporate_actions.py)
pins down that pre-announcement / post-announcement / post-ex-date views
of the SAME raw bars yield three different adjusted price series.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
.venv/bin/pytest -v                            # 17 tests, ~2s
.venv/bin/python demo/backtest.py              # ingest + materialise
.venv/bin/python demo/strategy_rsi_meanrev.py  # leakage-safe backtest
.venv/bin/python demo/macro_revisions.py       # GDP vintages worked example
.venv/bin/python demo/stress.py                # 30-ticker scaling profile
```

The ingest demo pulls 2 years of daily OHLCV for `AAPL MSFT SPY NVDA`,
computes every registered feature view, and runs a point-in-time pull
against every bar. The strategy demo runs a naive RSI mean-reversion
through the backtest engine end-to-end; expect a negative Sharpe — the
point is the pipeline, not the alpha. The macro-revisions demo shows
real BEA vintages of US GDP being queried at different `as_of` dates
(Q4 2008 reads as -3.8%, -6.2%, -6.3%, or -8.4% depending on when you
ask). The stress script times each phase on a 30-ticker S&P-large-cap
universe — ~5s end-to-end, no scaling cliffs in the qfs code itself
(yfinance network fetch dominates).

## Using the store

```python
import polars as pl
from datetime import datetime
from qfs import FeatureStore, backtest

store = FeatureStore("./data")

# Append-only writes. knowledge_time is stamped on each row.
store.write(my_features_df, view="rsi", version="v1")

# Point-in-time read. Returns one row per (symbol, as_of), with features
# joined as `<view>__<version>__<col>`. Rows where no eligible feature
# row exists become null — your strategy treats null as "abstain".
out = store.get_point_in_time(
    pl.DataFrame({"symbol": ["AAPL"], "as_of": [datetime(2024, 6, 1)]}),
    features=[("rsi", "v1"), ("macd", "v1"), ("bollinger", "v1")],
)

# Walk-forward backtest. Strategy is called once per bar with features
# pulled as-of that bar's knowledge_time. Returns realise on the NEXT
# bar's close — no path for the strategy to act on info it could not
# have known.
def my_strategy(features: pl.DataFrame) -> dict[str, float]:
    return {row["symbol"]: 1.0 if row["rsi__v1__rsi14"] < 30 else 0.0
            for row in features.iter_rows(named=True)}

result = backtest(store, bars=ohlcv, features=[("rsi", "v1")], strategy=my_strategy)
print(result.summary())
```

## Built-in feature views

All registered via `@feature_view(name, version)` decorator in
[`qfs/features/`](qfs/features/).

| view            | version | description                                  |
|-----------------|---------|----------------------------------------------|
| `rsi`           | v1      | 14-period RSI, simple rolling average        |
| `rsi`           | v2      | 14-period RSI, Wilder smoothing              |
| `macd`          | v1      | MACD(12, 26, 9) line / signal / histogram    |
| `bollinger`     | v1      | Bollinger(20, 2) mid / upper / lower / %B    |
| `realized_vol`  | v1      | Annualised 20-day vol of log returns         |
| `hl_spread`     | v1      | High-low range / close — bar liquidity proxy |

`rsi@v1` and `rsi@v2` coexist by design. Old backtests stay reproducible
against `v1`; new research uses `v2`. Versions never overwrite each
other.

## Survivorship-bias-free universe

The default backtest mistake: filter to *currently listed* names and
run the strategy ten years back. That universe excludes everything
that went bankrupt, was acquired, or fell out of the index — exactly
the names whose poor returns matter most.

```python
from qfs import Universe

u = Universe("./data/universe")
u.add("LEH",  datetime(2000, 1, 1), included_to=datetime(2008, 9, 15),
      knowledge_time=datetime(2000, 1, 1))
u.add("AAPL", datetime(2000, 1, 1), included_to=None,
      knowledge_time=datetime(2000, 1, 1))

u.members_as_of(datetime(2007, 6, 1))  # -> ["AAPL", "LEH"]
u.members_as_of(datetime(2010, 1, 1))  # -> ["AAPL"]
```

Membership is bitemporal — corrections (e.g. recording the Lehman
delisting on 2008-10-01) don't leak backwards. Backtest harnesses
should call `members_as_of(t)` before pulling features at t.

## Audit your pipelines

```python
for a in store.audit():
    print(a)
# rsi@v2: rows=2,008 symbols=4 event=[2023-01-03..2024-12-31]
#         lag p50=0:01:00 p95=0:01:00 restated=0
```

`audit()` reports per-view row count, symbol count, event-time range,
the **knowledge-lag distribution** (mean / p50 / p95 / max), and the
number of restated keys. The lag distribution is the single best
signal for catching a regressed upstream feed: when a view's p95 lag
jumps from 1 hour to 1 day overnight, the pipeline broke.

## Corporate actions

Splits are modelled as bitemporal facts:

```python
from qfs.corporate_actions import adjusted_ohlcv_as_of

# splits frame: (symbol, ex_date, knowledge_time, ratio)
adj_jul = adjusted_ohlcv_as_of(bars, splits, as_of=datetime(2020, 7, 1))
adj_oct = adjusted_ohlcv_as_of(bars, splits, as_of=datetime(2020, 10, 1))
# adj_jul.close: AAPL on Aug 28 shows $400 (unsplit world)
# adj_oct.close: AAPL on Aug 28 shows $100 (retroactively adjusted)
```

The same raw bars, the same query, two different answers — because the
*viewpoint date* is part of the query. That asymmetry is the whole
point: a backtest run in July 2020 should not 'see' the August split.

Dividends fit the same model with a subtractive adjustment; not
implemented here to keep the demo tight.

## Design notes

- **Storage:** append-only Parquet, one directory per
  `(view, version)`. DuckDB reads the directory as a single table.
- **Read path:** a window-function query that, per request row, picks
  the latest feature row with `event_time <= as_of` *and*
  `knowledge_time <= as_of`. Slower than DuckDB `ASOF JOIN` but
  expresses both inequalities correctly. Easy to swap later.
- **No mutation:** restating a fact means writing a new row with a
  newer `knowledge_time`. Old reads remain reproducible.
- **Backtest engine:** the simulator deliberately omits transaction
  costs, slippage, fractional fills, leverage caps, and intraday
  execution. None of those are necessary to demonstrate the leakage
  prevention. Add them for real research.

## What is deliberately not built

These are the natural next layers, none of which are needed to prove the
correctness claim:

- streaming ingest / online serving
- distributed compute (everything fits in DuckDB on one box)
- dividends as a separate adjustment (same model as splits)
- transaction-cost / market-impact models in the backtest
- ACL / multi-tenant isolation
- a UI

## Project layout

```
qfs/
  store.py             # FeatureStore — bitemporal read/write + audit()
  registry.py          # FeatureView + @feature_view decorator
  backtest.py          # walk-forward simulator, Sharpe / DD / turnover
  corporate_actions.py # bitemporal split adjustments
  universe.py          # survivorship-bias-free membership
  features/
    ta.py              # RSI, MACD, Bollinger
    market.py          # realized vol, hl-spread
  data.py              # yfinance loader
tests/
  test_point_in_time.py        # leakage tests — the proof
  test_backtest_leakage.py     # end-to-end leakage proof
  test_corporate_actions.py    # bitemporal split adjustments
  test_universe.py             # delisted-ticker visibility
  test_audit.py                # operational metadata
  test_versioning.py           # v1 / v2 coexistence
demo/
  backtest.py                  # ingest + materialise + point-in-time pull
  strategy_rsi_meanrev.py      # end-to-end strategy through the engine
  macro_revisions.py           # real BEA GDP vintages — bitemporal example
  stress.py                    # 30-ticker scaling profile
.github/workflows/ci.yml       # pytest on push / PR
```
