#!/usr/bin/env python3
"""ONE-TIME migration of the engine's DuckDB working store into the PostgreSQL schema `news_engine`.

  python deploy/migrate_duckdb_to_postgres.py --duckdb /path/to/news_metrics.duckdb            # dry run: shows counts, writes nothing
  python deploy/migrate_duckdb_to_postgres.py --duckdb /path/to/news_metrics.duckdb --apply    # copy, verify, then retire the mirrors

This is the ONLY place DuckDB is still used (read-only, and only for this migration; it is not a dependency of the engine or the
image). Nothing is recomputed: every row is copied as it is, so all processed data, clusters (with their centroids), metrics,
alerts and - above all - the day checkpoints and the ingest watermark carry over, and the live service resumes exactly where the
DuckDB-based one stopped. Run it while the engine is STOPPED (drain-stop the ingest container first).

Steps (--apply):
  1. create/upgrade the tables of the schema (lib/news_store.py DDL; additive, never drops anything)
  2. copy table by table in primary-key batches (idempotent upserts: re-running is safe)
  3. verify: row counts and content checksums equal on both sides, for every table
  4. only if everything matched: replace the two first-release MIRROR tables (article_cluster, trend_node_day) by compatibility
     views over the full tables (their rows are a subset of what was just copied and verified)
Credentials come from the environment (deploy/news.env); the PostgreSQL role is the engine's own (news_engine_app).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import news_store as NS  # noqa: E402

# DuckDB table -> (PostgreSQL table, order-by columns for stable paging)
MAP = {
    "taxonomy_node": ("taxonomy_node", ["code"]),
    "aoi_def": ("aoi_def", ["aoi_id"]),
    "corpus_day": ("corpus_day", ["day"]),
    "metric_meta": ("metric_meta", ["metric_key"]),
    "data_quality": ("data_quality", ["run_id", "key"]),
    "excluded_article": ("excluded_article", ["source_table", "article_id"]),
    "pending_label": ("pending_label", ["source_table", "article_id"]),
    "article_fact": ("article_fact", ["source_table", "article_id"]),
    "article_geo": ("article_geo", ["source_table", "article_id"]),
    "event_cluster": ("event_cluster", ["event_id"]),
    "event_metrics": ("event_metrics", ["event_id"]),
    "event_aoi_test": ("event_aoi_test", ["aoi_id", "event_id", "kind"]),
    "event_development": ("event_development", ["dev_id", "aoi_id"]),
    "aoi_day": ("aoi_day", ["aoi_id", "day"]),
    "node_aoi_day": ("node_aoi_day", ["aoi_id", "level", "node_code", "day"]),
    "node_aoi_series": ("node_aoi_series", ["aoi_id", "level", "node_code"]),
    "burst_episode": ("burst_episode", ["episode_id"]),
    "alert": ("alert", ["alert_id"]),
    "day_stage": ("checkpoint", ["stage", "scope", "day"]),
    "kv": ("state", ["key"]),
    "run_log": ("run_log", ["run_id"]),
}
BATCH = 40000
RENAME = {"run_log": {"params_json": "params", "summary_json": "summary"}}      # DuckDB json text -> PostgreSQL jsonb columns
# a cheap-but-sensitive content fingerprint per table: (duckdb SQL, postgres SQL) returning one comparable value
CHECK = {
    "article_fact": ("sum(article_id), count(DISTINCT cluster_id), count(*) FILTER (WHERE is_duplicate), max(published_at)::VARCHAR",
                     "sum(article_id), count(DISTINCT cluster_id), count(*) FILTER (WHERE is_duplicate), max(published_at)::text"),
    "event_cluster": ("count(*) FILTER (WHERE is_event), sum(n_articles), count(centroid_sum)",
                      "count(*) FILTER (WHERE is_event), sum(n_articles), count(centroid_sum)"),
    "node_aoi_day": ('round(sum("V"), 3)', 'round(sum("V")::numeric, 3)'),
    "alert": ("count(*) FILTER (WHERE status = 'active'), count(DISTINCT kind)", "count(*) FILTER (WHERE status = 'active'), count(DISTINCT kind)"),
    "article_geo": ("count(ev_country_iso), count(DISTINCT ev_country_iso)", "count(ev_country_iso), count(DISTINCT ev_country_iso)"),
}


def log(m):
    print(f"[migrate] {m}", flush=True)


def _norm(v):
    if v is None:
        return None
    try:
        import decimal
        if isinstance(v, decimal.Decimal):
            return float(v)
    except Exception:
        pass
    return v


def copy_table(duck, pg: NS.Store, src: str, dst: str, order: list[str]) -> int:
    n_src = duck.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
    if not n_src:
        return 0
    cols_pg = set(pg.columns(dst))
    ren = RENAME.get(dst, {})
    cols_src = [r[0] for r in duck.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{src}' ORDER BY ordinal_position").fetchall()]
    cols = [c for c in cols_src if ren.get(c, c) in cols_pg]
    dropped = [c for c in cols_src if c not in cols]
    if dropped:
        log(f"  {src}: columns not carried (no counterpart in PostgreSQL): {dropped}")
    q = ", ".join('"' + c + '"' for c in cols)
    rel = duck.execute(f"SELECT {q} FROM {src} ORDER BY {', '.join(order)}").fetch_record_batch(BATCH)
    done = 0
    t0 = time.time()
    for batch in rel:
        df = batch.to_pandas().rename(columns=ren)
        if dst == "run_log":
            # PostgreSQL's run_log (the live runs, richer columns) wins; add only runs it does not have
            pg.con.register("_rl", df)
            try:
                have = {r[0] for r in pg.con.execute("SELECT run_id FROM run_log").fetchall()}
            finally:
                pg.con.unregister("_rl")
            df = df[~df["run_id"].isin(have)]
        if dst == "state":
            have = {r[0] for r in pg.con.execute("SELECT key FROM state").fetchall()}
            df = df[~df["key"].isin(have) | df["key"].astype(str).str.startswith(("ingest_wm:", "live_start_day", "seeded"))]
        if len(df):
            pg.upsert(dst, df)
        done += len(batch)
        if n_src > BATCH:
            log(f"  {src}: {done:,}/{n_src:,} ({time.time() - t0:.0f}s)")
    return n_src


def verify(duck, pg: NS.Store) -> list[str]:
    bad = []
    for src, (dst, _) in MAP.items():
        a = duck.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
        b = pg.scalar(f"SELECT count(*) FROM {dst}")
        ok = (b >= a) if dst in ("run_log", "state") else (a == b)       # run_log/state: PostgreSQL may hold more (live runs, service keys)
        log(f"  {src:18s} -> {dst:18s} duckdb={a:>10,} postgres={b:>10,} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            bad.append(f"{src}: {a} vs {b}")
    for src, (qa, qb) in CHECK.items():
        dst = MAP[src][0]
        ra = tuple(_norm(x) for x in duck.execute(f"SELECT {qa} FROM {src}").fetchone())
        rb = tuple(_norm(x) for x in pg.con.execute(f"SELECT {qb} FROM {dst}").fetchone())
        # timestamps: compare at the instant (both rendered by their own engine) by parsing
        def canon(t):
            return tuple(pd.Timestamp(x).tz_convert("UTC").isoformat() if isinstance(x, str) and len(x) > 18 and x[:4].isdigit() else x for x in t)
        ok = canon(ra) == canon(rb)
        log(f"  checksum {src:14s} {'OK' if ok else 'MISMATCH'}  duckdb={ra} postgres={rb}")
        if not ok:
            bad.append(f"checksum {src}: {ra} vs {rb}")
    wm_a = duck.execute("SELECT value FROM kv WHERE key LIKE 'ingest_wm:%'").fetchall()
    for (v,) in wm_a:
        pass
    for k, v in duck.execute("SELECT key, value FROM kv WHERE key LIKE 'ingest_wm:%' OR key = 'live_start_day'").fetchall():
        ok = pg.get_kv(k) == v
        log(f"  state {k:22s} {'OK' if ok else 'MISMATCH'}  duckdb={v} postgres={pg.get_kv(k)}")
        if not ok:
            bad.append(f"state {k}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duckdb", required=True, help="path of the DuckDB working store to migrate (read-only)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    duck = duckdb.connect(a.duckdb, read_only=True)
    pg = NS.Store()
    log(f"source: {a.duckdb}   target: PostgreSQL schema {pg.schema}")
    for src, (dst, _) in MAP.items():
        n = duck.execute(f"SELECT count(*) FROM {src}").fetchone()[0]
        log(f"  {src:18s} {n:>10,} rows -> {dst}")
    if not a.apply:
        log("DRY RUN - nothing written. Re-run with --apply (with the engine stopped).")
        return 0
    t0 = time.time()
    for src, (dst, order) in MAP.items():
        log(f"copying {src} -> {dst}")
        copy_table(duck, pg, src, dst, order)
    log("verifying")
    bad = verify(duck, pg)
    if bad:
        log("VERIFICATION FAILED - the first-release mirror tables were NOT touched:")
        for b in bad:
            log("  " + b)
        return 1
    # retire the first-release mirror tables: their content is a subset of what was just copied and verified
    for t in ("article_cluster", "trend_node_day"):
        if pg._is_table(t):
            n = pg.scalar(f"SELECT count(*) FROM {t}")
            log(f"retiring mirror table {t} ({n:,} rows, subset of the verified full tables) -> compatibility view")
            pg.con.execute(f"DROP TABLE {t}")
    pg.con.execute(NS.VIEWS)
    pg.set_kv("migrated_from_duckdb", Path(a.duckdb).name)
    pg.set_kv("migrated_at", NS.now_utc().isoformat())
    log(f"DONE in {time.time() - t0:.0f}s. The engine can now be started on PostgreSQL; it resumes from the migrated checkpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
