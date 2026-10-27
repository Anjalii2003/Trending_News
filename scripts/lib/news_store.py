"""Derived store of the news engine: PostgreSQL schema `news_engine` (default). Source systems are never written.

Everything the engine derives - articles' derived facts, geography, event clusters, metrics, stories, alerts, checkpoints, run
logs - lives in that one schema, written by the engine's role (`news_engine_app`, SELECT on the source tables it reads, rights on
this schema only) and read by the dashboard's role (`news_engine_reader`, SELECT only). There is no local database file.

Reliability model
  * every write path is an UPSERT keyed on a natural key -> reruns cannot create duplicates
  * a stage's work for a day is committed in ONE transaction together with its `checkpoint` row, so a crash leaves either
    (data + status) or neither
  * `run_log` records each run; a run left 'running' by a crash is marked 'abandoned' by the next run that acquires the lock
  * an exclusive file lock (`NEWS_LOCK`) and a PostgreSQL advisory lock keep two runs from overlapping, on any host
  * a read-only store (the dashboard) is one REPEATABLE READ transaction with writes refused: a consistent snapshot, never blocked by
    the writer and never blocking it

The class keeps the interface the analytics modules were written against (`df`, `scalar`, `upsert`, `tx`, `con.execute`,
`con.register`, ...). `Conn` translates the few DuckDB-isms of those call sites (`?` placeholders, registered DataFrames as
temporary tables) so the statistics code is unchanged.
"""
from __future__ import annotations

import datetime as dt
import decimal
import fcntl
import hashlib
import json
import os
import re
import socket
import threading
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extensions
import psycopg2.extras

from . import news_config as C
from . import news_pgstore as PGS

SCHEMA_VERSION = 2

DDL = """
CREATE TABLE IF NOT EXISTS state (key text PRIMARY KEY, value text, updated_at timestamptz);

CREATE TABLE IF NOT EXISTS run_log (
  run_id text PRIMARY KEY, mode text, scope text, trigger text, status text,
  started_at timestamptz, finished_at timestamptz, heartbeat_at timestamptz, date_from date, date_to date,
  articles_ingested integer, articles_unique_new integer, clusters_touched integer,
  totals jsonb, summary jsonb, error text, host text, params jsonb);
ALTER TABLE run_log ADD COLUMN IF NOT EXISTS params jsonb;
CREATE INDEX IF NOT EXISTS run_log_started ON run_log (started_at DESC);

CREATE TABLE IF NOT EXISTS checkpoint (
  stage text, scope text, day date, status text, n_in bigint, n_out bigint, run_id text, updated_at timestamptz, error text,
  PRIMARY KEY (stage, scope, day));

CREATE TABLE IF NOT EXISTS pending_label (
  source_table text, article_id bigint, first_seen_at timestamptz, last_checked_at timestamptz, attempts integer,
  published_at timestamptz, PRIMARY KEY (source_table, article_id));

CREATE TABLE IF NOT EXISTS taxonomy_node (
  code text PRIMARY KEY, tier text, name text, parent text, sort_order integer, is_catch_all boolean, in_scope boolean,
  taxonomy_version text, loaded_at timestamptz);

-- articles that were seen but are NOT used for trends, with the reason (never silently dropped)
CREATE TABLE IF NOT EXISTS excluded_article (
  source_table text, article_id bigint, reason text, published_at timestamptz, run_id text, seen_at timestamptz,
  PRIMARY KEY (source_table, article_id));

CREATE TABLE IF NOT EXISTS article_fact (
  source_table text, article_id bigint, os_id text, os_index text, published_at timestamptz, day date, ingested_at timestamptz,
  lag_hours double precision, date_confidence text, date_source text, date_precision text,
  has_label boolean, domain_id text, topic_id text, category_id text, subtopic_id text,
  category_conf double precision, subtopic_conf double precision, taxonomy_version text,
  content_hash text, is_duplicate boolean, dup_of text, source_domain text, url text, title text, gdelt_event_id text,
  cluster_id text, cluster_sim double precision, cluster_note text, run_id text, processed_at timestamptz,
  PRIMARY KEY (source_table, article_id));

CREATE TABLE IF NOT EXISTS article_geo (
  source_table text, article_id bigint, ev_country_iso text, ev_adm1_code text, ev_place text,
  ev_lat double precision, ev_lon double precision, ev_precision text, geo_source text, geo_inconsistent boolean,
  geo_snapped boolean, loc_status text, country_code_unmapped boolean, mention_countries_iso text[],
  PRIMARY KEY (source_table, article_id));

CREATE TABLE IF NOT EXISTS event_cluster (
  event_id text PRIMARY KEY, first_seen timestamptz, last_seen timestamptz, n_articles integer, n_publishers integer,
  n_copies integer, is_event boolean, loc_country_iso text, loc_adm1_code text, loc_confidence text, loc_share double precision,
  n_located integer, rep_source_table text, rep_article_id bigint, updated_at timestamptz);
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS domain_id text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS topic_id text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS category_id text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS rep_title text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS rep_url text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS loc_precision text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS loc_status text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS centroid_sum bytea;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS is_open boolean;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS created_run text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS updated_run text;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS ref_sim double precision;
ALTER TABLE event_cluster ADD COLUMN IF NOT EXISTS ref_votes text;

CREATE TABLE IF NOT EXISTS event_metrics (
  event_id text PRIMARY KEY, computed_at timestamptz, as_of timestamptz, n_articles integer, n_publishers integer,
  n_copies integer, syndication_share double precision, first_seen timestamptz, last_seen timestamptz,
  duration_hours double precision, active_days integer, peak_day date, articles_24h integer, articles_prev_24h integer,
  growth_24h double precision, accel_6h double precision, publisher_pie double precision, cross_source_ratio double precision,
  half_life_days double precision, half_life_ok boolean, persistence_note text, support_ok boolean, support_note text);

CREATE TABLE IF NOT EXISTS event_aoi_test (
  aoi_id text, event_id text, kind text, as_of timestamptz, age_hours double precision, n_articles_at_age integer,
  n_publishers_at_age integer, n_ref integer, stratum text, p double precision, q double precision, effect double precision,
  support_ok boolean, support_note text, alert_flag boolean, PRIMARY KEY (aoi_id, event_id, kind));

CREATE TABLE IF NOT EXISTS event_development (
  event_id text, dev_id text, aoi_id text, detected_asof timestamptz, n_new integer, n_new_publishers integer,
  shift double precision, shift_p double precision, shift_q double precision, coherence double precision, rep_title text,
  alert_flag boolean, member_keys text, PRIMARY KEY (dev_id, aoi_id));

CREATE TABLE IF NOT EXISTS aoi_def (
  aoi_id text PRIMARY KEY, kind text, key text, name text, mode text, min_precision text, adhoc boolean, iso2 text,
  parent_aoi text, geometry_json text, registered_at timestamptz);

CREATE TABLE IF NOT EXISTS aoi_day (
  aoi_id text, day date, n_articles integer, n_publishers integer, n_events_started integer, n_events_active integer,
  n_located_event integer, n_mention_only integer, n_scope_unlocated integer, n_scope_total integer, in_window boolean,
  PRIMARY KEY (aoi_id, day));

CREATE TABLE IF NOT EXISTS node_aoi_day (
  aoi_id text, level text, node_code text, day date, in_window boolean,
  "V" double precision, exposure double precision, "V_rate" double precision, "share_V" double precision,
  growth double precision, growth_7d double precision, accel double precision, accel_smooth double precision,
  momentum double precision, baseline_mu double precision, baseline_theta double precision, baseline_days integer,
  baseline_mean double precision, baseline_dist text, baseline_method text, surprise double precision,
  surprise_p double precision, surprise_q double precision, alert_flag boolean, burst_state double precision,
  burst_intensity double precision, burst_method text, burst_support_ok boolean, burst_episode_id text,
  n_publishers integer, publisher_pie double precision, support_n double precision, support_ok boolean,
  PRIMARY KEY (aoi_id, level, node_code, day));

CREATE TABLE IF NOT EXISTS node_aoi_series (
  aoi_id text, level text, node_code text, total_v double precision, n_days integer, ar1_phi double precision,
  ar1_half_life_days double precision, ar1_ljung_box_p double precision, ar1_n_obs integer, ar1_fit_ok boolean,
  PRIMARY KEY (aoi_id, level, node_code));

CREATE TABLE IF NOT EXISTS burst_episode (
  episode_id text PRIMARY KEY, aoi_id text, level text, node_code text, start_day date, peak_day date, end_day date,
  "peak_V" double precision, "total_V" double precision, n_days integer, level_k integer, burst_method text,
  onset_slope double precision, half_life_days double precision, decay_ok boolean);

CREATE TABLE IF NOT EXISTS alert (
  alert_id text PRIMARY KEY, kind text, aoi_id text, entity_type text, entity_id text, day date, as_of timestamptz,
  p double precision, q double precision, effect double precision, severity text, status text, rationale text,
  first_seen_at timestamptz, last_seen_at timestamptz, retracted_at timestamptz);
ALTER TABLE alert ADD COLUMN IF NOT EXISTS support_json text;
ALTER TABLE alert ADD COLUMN IF NOT EXISTS first_run text;
ALTER TABLE alert ADD COLUMN IF NOT EXISTS last_run text;
CREATE INDEX IF NOT EXISTS alert_status ON alert (status, as_of DESC);

CREATE TABLE IF NOT EXISTS corpus_day (
  day date PRIMARY KEY, n_dated_total bigint, n_action_country bigint, n_action_geo bigint, n_mention_country bigint,
  source text, measured_at timestamptz);

CREATE TABLE IF NOT EXISTS data_quality (
  run_id text, key text, value text, recorded_at timestamptz, PRIMARY KEY (run_id, key));

CREATE TABLE IF NOT EXISTS metric_meta (
  metric_key text PRIMARY KEY, display_name text, family text, formula text, units text, grain text, direction text,
  caveats text, support_gate text);

-- stories (events of the same story merged) and the rankings computed from them: materialised after each successful run
CREATE TABLE IF NOT EXISTS story (
  story_id text PRIMARY KEY, as_of timestamptz, title text, n_events integer, n_articles integer, n_publishers integer,
  first_seen timestamptz, last_seen timestamptz, domain_id text, topic_id text, category_id text, subtopic_id text,
  tax_status text, country_iso text, place text, n_located integer, n_in_country integer, loc_share double precision,
  ref_sim double precision, ref_votes text, run_id text);
CREATE TABLE IF NOT EXISTS story_event (story_id text, event_id text, PRIMARY KEY (story_id, event_id));
CREATE INDEX IF NOT EXISTS story_event_event ON story_event (event_id);
CREATE TABLE IF NOT EXISTS trending_ranking (
  as_of timestamptz, window_key text, country_iso text, tax_code text, rank integer, story_id text, score double precision,
  n_articles_window integer, n_publishers_window integer, run_id text, PRIMARY KEY (window_key, country_iso, tax_code, rank));

CREATE INDEX IF NOT EXISTS article_fact_day ON article_fact (day);
CREATE INDEX IF NOT EXISTS article_fact_cluster ON article_fact (cluster_id);
CREATE INDEX IF NOT EXISTS article_fact_hash ON article_fact (content_hash) WHERE NOT is_duplicate;
CREATE INDEX IF NOT EXISTS article_fact_pub ON article_fact (published_at DESC);
CREATE INDEX IF NOT EXISTS article_fact_todo ON article_fact (published_at) WHERE cluster_id IS NULL AND cluster_note IS NULL;
CREATE INDEX IF NOT EXISTS article_geo_country ON article_geo (ev_country_iso);
CREATE INDEX IF NOT EXISTS article_geo_adm1 ON article_geo (ev_adm1_code);
CREATE INDEX IF NOT EXISTS article_geo_mention ON article_geo USING gin (mention_countries_iso);
CREATE INDEX IF NOT EXISTS event_cluster_last ON event_cluster (last_seen DESC);
CREATE INDEX IF NOT EXISTS event_cluster_event ON event_cluster (is_event) WHERE is_event;
CREATE INDEX IF NOT EXISTS node_aoi_day_aoi ON node_aoi_day (aoi_id);
CREATE INDEX IF NOT EXISTS event_aoi_test_aoi ON event_aoi_test (aoi_id);
"""

# compatibility views for downstream readers of the first (mirror) release of this schema; created separately because a
# CREATE OR REPLACE VIEW must not collide with the old mirror TABLES of the same name (see deploy/migrate_to_postgres.py)
VIEWS = """
CREATE OR REPLACE VIEW article_cluster AS
  SELECT source_table, article_id, cluster_id AS event_id, published_at, day, is_duplicate, has_label, processed_at AS synced_at
  FROM article_fact;
CREATE OR REPLACE VIEW trend_node_day AS
  SELECT aoi_id, level, node_code, day, "V" AS v, exposure, growth, baseline_mu, surprise, surprise_p, surprise_q,
         alert_flag, n_publishers, support_ok
  FROM node_aoi_day;
"""

PK = {  # table -> primary key columns, used by upsert()
    "state": ["key"], "run_log": ["run_id"], "checkpoint": ["stage", "scope", "day"],
    "pending_label": ["source_table", "article_id"], "taxonomy_node": ["code"],
    "excluded_article": ["source_table", "article_id"], "article_fact": ["source_table", "article_id"],
    "article_geo": ["source_table", "article_id"], "event_cluster": ["event_id"],
    "event_metrics": ["event_id"], "event_aoi_test": ["aoi_id", "event_id", "kind"],
    "event_development": ["dev_id", "aoi_id"], "aoi_def": ["aoi_id"], "aoi_day": ["aoi_id", "day"],
    "node_aoi_day": ["aoi_id", "level", "node_code", "day"],
    "node_aoi_series": ["aoi_id", "level", "node_code"], "burst_episode": ["episode_id"],
    "alert": ["alert_id"], "corpus_day": ["day"], "data_quality": ["run_id", "key"],
    "metric_meta": ["metric_key"], "story": ["story_id"], "story_event": ["story_id", "event_id"],
    "trending_ranking": ["window_key", "country_iso", "tax_code", "rank"],
}
TABLES = list(PK)
PAGE = 5000


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class LockBusy(RuntimeError):
    pass


# --------------------------------------------------------------------------
# value adaptation
# --------------------------------------------------------------------------
for _t in (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64):
    psycopg2.extensions.register_adapter(_t, lambda v: psycopg2.extensions.AsIs(int(v)))
for _t in (np.float32, np.float64):
    psycopg2.extensions.register_adapter(_t, lambda v: psycopg2.extensions.AsIs("NULL") if v != v else psycopg2.extensions.AsIs(repr(float(v))))
psycopg2.extensions.register_adapter(np.bool_, lambda v: psycopg2.extensions.AsIs("true" if v else "false"))
psycopg2.extensions.register_adapter(pd.Timestamp, lambda v: psycopg2.extensions.adapt(v.to_pydatetime()))
psycopg2.extensions.register_adapter(type(pd.NaT), lambda v: psycopg2.extensions.AsIs("NULL"))
psycopg2.extensions.register_adapter(type(pd.NA), lambda v: psycopg2.extensions.AsIs("NULL"))
psycopg2.extensions.register_adapter(np.ndarray, lambda v: psycopg2.extensions.adapt(v.tolist()))
psycopg2.extensions.register_adapter(set, lambda v: psycopg2.extensions.adapt(sorted(v)))


def _py(v: Any) -> Any:
    """pandas / numpy value -> Python value for psycopg2 (NaN, NaT, pd.NA -> NULL)."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


_CASED = re.compile(r'(?<![\w"])(V|V_rate|share_V|peak_V|total_V)(?![\w"])')   # columns created with exact case (quoted identifiers)


def _translate(sql: str) -> str:
    """DuckDB-style SQL of the analytics modules -> psycopg2: `?` placeholders, literal `%`, DuckDB cast names, and the
    mixed-case column names (PostgreSQL folds unquoted identifiers to lower case)."""
    parts, buf, inq = [], [], False
    for ch in sql:
        if ch == "'":
            if not inq:
                parts.append((False, "".join(buf)))
                buf = []
            buf.append(ch)
            if inq:
                parts.append((True, "".join(buf)))
                buf = []
            inq = not inq
        else:
            buf.append(ch)
    parts.append((inq, "".join(buf)))
    out = []
    for lit, seg in parts:
        if lit:
            out.append(seg.replace("%", "%%"))
            continue
        seg = seg.replace("%", "%%").replace("?", "%s")
        seg = re.sub(r"::DOUBLE\b", "::double precision", seg)
        seg = re.sub(r"\bAS DOUBLE\)", "AS DOUBLE PRECISION)", seg)
        seg = re.sub(r"::INT\b", "::integer", seg)
        out.append(_CASED.sub(lambda m: '"' + m.group(1) + '"', seg))
    return "".join(out)


def _pg_type(series: pd.Series) -> str:
    if isinstance(series.dtype, pd.DatetimeTZDtype):
        return "timestamptz"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "timestamp"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "bigint"
    if pd.api.types.is_float_dtype(series):
        return "double precision"
    nn = series.dropna()
    if not len(nn):
        return "text"
    v = nn.iloc[0]
    if isinstance(v, (bool, np.bool_)):
        return "boolean"
    if isinstance(v, (int, np.integer)):
        return "bigint"
    if isinstance(v, (float, np.floating)):
        return "double precision"
    if isinstance(v, (bytes, bytearray, memoryview)):
        return "bytea"
    if isinstance(v, dt.datetime):
        return "timestamptz" if v.tzinfo else "timestamp"
    if isinstance(v, dt.date):
        return "date"
    if isinstance(v, (list, tuple, np.ndarray)):
        return "text[]"
    return "text"


class Result:
    def __init__(self, cur):
        self._cur = cur
        self.description = cur.description

    def fetchall(self) -> list[tuple]:
        return [tuple(bytes(x) if isinstance(x, memoryview) else x for x in r) for r in self._cur.fetchall()] if self.description else []

    def fetchone(self):
        r = self._cur.fetchone() if self.description else None
        return None if r is None else tuple(bytes(x) if isinstance(x, memoryview) else x for x in r)

    def df(self) -> pd.DataFrame:
        if not self.description:
            return pd.DataFrame()
        cols = [d[0] for d in self.description]
        rows = self._cur.fetchall()
        df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
        for d in self.description:                     # type fix-ups so frames look like the analytics code expects
            c, oid = d[0], d[1]
            if oid in (1184, 1114):                    # timestamptz / timestamp
                df[c] = pd.to_datetime(df[c], utc=(oid == 1184))
            elif oid == 1082:                          # date -> datetime64 (as DuckDB returned it)
                df[c] = pd.to_datetime(df[c])
            elif oid == 17:                            # bytea
                df[c] = [None if x is None else bytes(x) for x in df[c]]
            elif oid == 1700:                          # numeric (sum of bigint ...) -> float / int
                df[c] = pd.to_numeric(df[c].map(lambda x: None if x is None else float(x) if isinstance(x, decimal.Decimal) else x))
        return df


class Conn:
    """The subset of the DuckDB connection API the analytics modules use, on a psycopg2 connection."""

    _WRITE = re.compile(r"^\s*(insert|update|delete|drop|create|alter|truncate|grant|revoke|copy|vacuum|reindex|comment|call|do)\b", re.I)

    def __init__(self, conn, schema: str, read_only: bool = False):
        self._c = conn
        self.schema = schema
        self.read_only = read_only
        self._registered: set[str] = set()
        self._coltypes: Optional[dict] = None

    # ---- statements -----------------------------------------------------------------------------------------------
    def execute(self, sql: str, params: Optional[Sequence] = None) -> Result:
        if self.read_only and self._WRITE.match(sql):
            raise PermissionError("this store is read-only")       # (temp tables for register() are made internally)
        cur = self._c.cursor()
        cur.execute(_translate(sql), tuple(_py(p) for p in params) if params else ())
        return Result(cur)

    def executemany_values(self, sql: str, rows: Iterable[Sequence], page: int = PAGE) -> int:
        """`sql` contains a single `VALUES %s`; rows are sent in pages."""
        n, buf = 0, []
        with self._c.cursor() as cur:
            for r in rows:
                buf.append(tuple(_py(v) for v in r))
                if len(buf) >= page:
                    psycopg2.extras.execute_values(cur, sql, buf, page_size=page)
                    n += len(buf)
                    buf = []
            if buf:
                psycopg2.extras.execute_values(cur, sql, buf, page_size=page)
                n += len(buf)
        return n

    # ---- DataFrames as temporary tables ------------------------------------------------------------------------------
    def _type_of(self, series: pd.Series, name: str) -> str:
        """Column type for a temp table. A column with no values carries no type information, so take the type of the schema's
        column of that name (e.g. an all-NULL `loc_share` must stay double precision, not text)."""
        if series.dtype == object and series.dropna().empty:
            if self._coltypes is None:
                cur = self._c.cursor()
                cur.execute("SELECT a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid "
                            "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = %s AND a.attnum > 0 AND NOT a.attisdropped "
                            "AND c.relkind = 'r'", (self.schema,))
                self._coltypes = {}
                for col, typ in cur.fetchall():
                    self._coltypes.setdefault(col, typ)
            return self._coltypes.get(name, "text")
        return _pg_type(series)

    def register(self, name: str, df: pd.DataFrame) -> None:
        """Make `df` queryable as table `name` for this session (DuckDB `register`). Replaces an earlier registration."""
        self.unregister(name)
        cols = list(df.columns)
        cur = self._c.cursor()
        cur.execute("CREATE TEMP TABLE {} ({})".format(name, ", ".join(f'"{c}" {self._type_of(df[c], c)}' for c in cols)))
        if len(df):
            self.executemany_values(f'INSERT INTO {name} ({", ".join(chr(34) + c + chr(34) for c in cols)}) VALUES %s',
                                    df.itertuples(index=False, name=None))
        self._registered.add(name)

    def unregister(self, name: str) -> None:
        self._c.cursor().execute(f"DROP TABLE IF EXISTS pg_temp.{name}")
        self._registered.discard(name)

    def close(self) -> None:
        try:
            self._c.close()
        except Exception:
            pass


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
_ENSURED: set[str] = set()
_ENSURE_LOCK = threading.Lock()


def schema_for(path: Optional[Path | str]) -> str:
    """Production: the configured schema (default news_engine). Tests set NEWS_STORE_SCHEMA_FROM_PATH=1 so every temporary
    path gets its own throw-away schema (the old per-file isolation)."""
    if path is not None and os.environ.get("NEWS_STORE_SCHEMA_FROM_PATH") == "1":
        return "ne_" + hashlib.sha1(str(path).encode()).hexdigest()[:12]
    return PGS.schema_name()


class Store:
    def __init__(self, path: Path | str | None = None, read_only: bool = False, schema: Optional[str] = None):
        self.read_only = read_only
        self.schema = schema or schema_for(path)
        self.path = Path(path) if path else None
        s = PGS.derived_settings()
        raw = psycopg2.connect(host=s["host"], port=s["port"], user=s["user"], password=s["password"], dbname=s["dbname"],
                               connect_timeout=30, application_name="news_engine_store",
                               options=f"-c search_path={self.schema} -c TimeZone=UTC -c statement_timeout=900000")
        raw.autocommit = True
        self.con = Conn(raw, self.schema, read_only)
        self._in_tx = False
        self._on_close = None
        if read_only:
            # one consistent snapshot per store. (Not "READ ONLY": PostgreSQL then forbids the temp tables register() uses; writes
            # are refused by Conn and, in production, by the dashboard role's grants.)
            self.con.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
        else:
            self._ensure_schema()
            self.set_kv("schema_version", str(SCHEMA_VERSION))

    def _ensure_schema(self) -> None:
        key = f"{self.schema}@{PGS.derived_settings()['host']}:{PGS.derived_settings()['port']}"
        with _ENSURE_LOCK:
            if key in _ENSURED:
                return
            if os.environ.get("NEWS_STORE_SCHEMA_FROM_PATH") == "1":
                self.con.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            self.con.execute("SELECT pg_advisory_lock(hashtext('news_engine.ddl'))")
            try:
                self.con.execute(DDL)
                if not self._is_table("article_cluster") and not self._is_table("trend_node_day"):
                    self.con.execute(VIEWS)
            finally:
                self.con.execute("SELECT pg_advisory_unlock(hashtext('news_engine.ddl'))")
            _ENSURED.add(key)

    def _is_table(self, name: str) -> bool:
        return self.scalar("SELECT 1 FROM information_schema.tables WHERE table_schema = ? AND table_name = ? AND table_type = 'BASE TABLE'",
                           [self.schema, name]) is not None

    # ----- low-level ----------------------------------------------------------
    def close(self) -> None:
        try:
            if self.read_only:
                self.con.execute("ROLLBACK")
        except Exception:
            pass
        self.con.close()
        cb, self._on_close = self._on_close, None
        if cb:
            cb()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    @contextmanager
    def tx(self):
        """One transaction. Everything inside commits together or not at all."""
        self.con.execute("BEGIN")
        try:
            yield self.con
        except BaseException:
            self.con.execute("ROLLBACK")
            raise
        else:
            self.con.execute("COMMIT")

    def df(self, sql: str, params: Optional[list] = None) -> pd.DataFrame:
        return self.con.execute(sql, params).df()

    def scalar(self, sql: str, params: Optional[list] = None):
        r = self.con.execute(sql, params).fetchone()
        return None if r is None else r[0]

    def upsert(self, table: str, df: pd.DataFrame, replace_all: bool = True) -> int:
        """INSERT .. ON CONFLICT (pk) DO UPDATE, in pages. Columns are matched BY NAME.
        replace_all=True (default, the old `INSERT OR REPLACE`): columns the frame omits are set to NULL on conflict, so pass full
        rows when replacing. replace_all=False updates only the frame's columns and leaves the others as they are."""
        if df is None or len(df) == 0:
            return 0
        known = self.columns(table)
        unknown = set(df.columns) - set(known)
        if unknown:
            raise KeyError(f"{table}: unknown columns {sorted(unknown)}")
        pk = PK[table]
        dup = df.duplicated(subset=pk).sum()
        if dup:
            raise ValueError(f"{table}: {dup} duplicate primary keys in the batch")
        cols = list(df.columns)
        q = lambda c: '"' + c + '"'
        other = [c for c in (cols if not replace_all else known) if c not in pk]
        sets = ", ".join(f"{q(c)} = EXCLUDED.{q(c)}" if c in cols else f"{q(c)} = NULL" for c in other)
        sql = (f"INSERT INTO {table} ({', '.join(q(c) for c in cols)}) VALUES %s ON CONFLICT ({', '.join(q(c) for c in pk)}) DO "
               + (f"UPDATE SET {sets}" if sets else "NOTHING"))
        return self.con.executemany_values(sql, df.itertuples(index=False, name=None))

    def delete_where(self, table: str, where: str, params: Optional[list] = None) -> None:
        self.con.execute(f"DELETE FROM {table} WHERE {where}", params)

    def columns(self, table: str) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT column_name FROM information_schema.columns WHERE table_schema = ? AND table_name = ? "
                                               "ORDER BY ordinal_position", [self.schema, table]).fetchall()]

    # ----- key/value ------------------------------------------------------------
    def get_kv(self, key: str) -> Optional[str]:
        return self.scalar("SELECT value FROM state WHERE key = ?", [key])

    def set_kv(self, key: str, value: Optional[str]) -> None:
        self.con.execute("INSERT INTO state (key, value, updated_at) VALUES (?, ?, ?) ON CONFLICT (key) DO UPDATE "
                         "SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at", [key, value, now_utc()])

    def all_state(self) -> dict:
        return {r[0]: r[1] for r in self.con.execute("SELECT key, value FROM state").fetchall()}

    # ----- single writer across hosts ---------------------------------------------------------
    def try_lock(self, name: str = "news_engine.service") -> bool:
        """Session advisory lock: a second service (any host) is refused instead of overlapping."""
        return bool(self.scalar("SELECT pg_try_advisory_lock(hashtext(?))", [name]))

    def unlock(self, name: str = "news_engine.service") -> None:
        self.scalar("SELECT pg_advisory_unlock(hashtext(?))", [name])

    # ----- runs ------------------------------------------------------------------------------
    def start_run(self, mode: str, scope: str, date_from, date_to, params: dict) -> str:
        run_id = f"{now_utc():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
        # a crash leaves a 'running' row behind; the lock proves nobody is actually running
        self.con.execute("UPDATE run_log SET status='abandoned', finished_at=? WHERE status='running'", [now_utc()])
        self.con.execute("INSERT INTO run_log (run_id, mode, scope, trigger, status, started_at, heartbeat_at, date_from, date_to, params, host) "
                         "VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?)",
                         [run_id, mode, scope, (params or {}).get("trigger"), now_utc(), now_utc(), date_from, date_to,
                          json.dumps(params, default=str), socket.gethostname()])
        return run_id

    def finish_run(self, run_id: str, status: str, summary: dict, error: Optional[str] = None, *, ingested: Optional[int] = None,
                   unique_new: Optional[int] = None, touched: Optional[int] = None, totals: Optional[dict] = None) -> None:
        self.con.execute("UPDATE run_log SET status=?, finished_at=?, heartbeat_at=?, summary=?, error=?, articles_ingested=?, "
                         "articles_unique_new=?, clusters_touched=?, totals=? WHERE run_id=?",
                         [status, now_utc(), now_utc(), json.dumps(summary, default=str), (error or None) and error[:2000], ingested,
                          unique_new, touched, None if totals is None else json.dumps(totals, default=str), run_id])

    # ----- per-day stage checkpoints -----------------------------------------------
    def mark_stage(self, stage: str, scope: str, day: dt.date, status: str, run_id: str,
                   n_in: int = 0, n_out: int = 0, error: Optional[str] = None) -> None:
        self.con.execute("INSERT INTO checkpoint (stage, scope, day, status, n_in, n_out, run_id, updated_at, error) "
                         "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT (stage, scope, day) DO UPDATE SET status = EXCLUDED.status, "
                         "n_in = EXCLUDED.n_in, n_out = EXCLUDED.n_out, run_id = EXCLUDED.run_id, updated_at = EXCLUDED.updated_at, "
                         "error = EXCLUDED.error", [stage, scope, day, status, n_in, n_out, run_id, now_utc(), error])

    def stage_done(self, stage: str, scope: str, day: dt.date) -> bool:
        return self.scalar("SELECT status FROM checkpoint WHERE stage=? AND scope=? AND day=?", [stage, scope, day]) == "done"

    def record_quality(self, run_id: str, key: str, value) -> None:
        self.con.execute("INSERT INTO data_quality (run_id, key, value, recorded_at) VALUES (?,?,?,?) ON CONFLICT (run_id, key) "
                         "DO UPDATE SET value = EXCLUDED.value, recorded_at = EXCLUDED.recorded_at",
                         [run_id, key, json.dumps(value, default=str), now_utc()])


@contextmanager
def run_lock(path: Path = C.NEWS_LOCK, wait: bool = False):
    """Exclusive advisory lock so two runs never overlap. Released automatically on process
    death, so a crashed run cannot wedge the scheduler."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            raise LockBusy(f"another news run holds {path}")
        fh.seek(0)
        fh.truncate()
        fh.write(f"pid={os.getpid()} since={now_utc().isoformat()}\n")
        fh.flush()
        yield
    finally:
        try:
            fcntl.flock(fh, fcntl.LOCK_UN)
        finally:
            fh.close()
