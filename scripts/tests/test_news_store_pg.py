"""The PostgreSQL store: schema, SQL translation, upserts, read-only snapshots, run log, summary, stories/rankings, migration.
Runs on a throwaway Docker PostgreSQL (conftest.py)."""
import datetime as dt
import json
import secrets

import numpy as np
import pandas as pd
import pytest

import news_testutil  # noqa: F401
from news_testutil import UTC, fact_row, geo_row, populate
from lib import news_live as LV
from lib import news_store as S
from lib import news_stories as NST
from lib import news_trending as T

T0 = dt.datetime(2026, 10, 20, 12, tzinfo=UTC)


@pytest.fixture
def st(tmp_path):
    s = S.Store(tmp_path / "x.db")
    yield s
    s.close()


def seed(st, n=10, cluster="ev_a", start=1):
    facts = [fact_row(start + i, 0, cluster=cluster, publisher=f"p{i % 3}.com") for i in range(n)]
    for f in facts:
        f["processed_at"] = T0
    populate(st, facts, [geo_row(start + i) for i in range(n)])
    st.upsert("event_cluster", pd.DataFrame([{"event_id": cluster, "first_seen": T0, "last_seen": T0, "n_articles": n, "n_publishers": 3,
                                              "is_event": True, "loc_country_iso": "IN", "updated_at": T0}]))


# ---- schema ------------------------------------------------------------------------------------------------------------------
def test_schema_has_every_table_and_the_compatibility_views(st):
    tables = {r[0] for r in st.con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = ? AND table_type = 'BASE TABLE'", [st.schema]).fetchall()}
    assert set(S.TABLES) <= tables
    views = {r[0] for r in st.con.execute("SELECT table_name FROM information_schema.views WHERE table_schema = ?", [st.schema]).fetchall()}
    assert {"article_cluster", "trend_node_day"} <= views
    assert not tables & {"day_stage", "kv"}                      # renamed: checkpoint / state


def test_article_cluster_view_and_trend_view_are_live_projections(st):
    seed(st, 4)
    st.upsert("node_aoi_day", pd.DataFrame([{"aoi_id": "country:IN:event", "level": "topic", "node_code": "D01.T01", "day": T0.date(), "V": 7.0,
                                              "baseline_mu": 3.0, "surprise_q": 0.01, "alert_flag": True}]))
    assert st.scalar("SELECT count(*) FROM article_cluster") == 4
    assert st.scalar("SELECT event_id FROM article_cluster WHERE article_id = 1") == "ev_a"
    assert st.scalar("SELECT v FROM trend_node_day") == 7.0 and st.scalar('SELECT "V" FROM node_aoi_day') == 7.0


# ---- SQL translation (the analytics modules were written against DuckDB) ------------------------------------------------------
def test_translation_of_placeholders_percent_signs_casts_and_cased_columns():
    t = S._translate
    assert t("SELECT a FROM t WHERE x = ? AND y LIKE 'a%b?' AND n > ?") == "SELECT a FROM t WHERE x = %s AND y LIKE 'a%%b?' AND n > %s"
    assert t("SELECT V, V_rate, share_V FROM node_aoi_day WHERE n.peak_V > 1") == 'SELECT "V", "V_rate", "share_V" FROM node_aoi_day WHERE n."peak_V" > 1'
    assert t("SELECT x::DOUBLE, CAST(y AS DOUBLE), z::INT FROM t") == "SELECT x::double precision, CAST(y AS DOUBLE PRECISION), z::integer FROM t"
    assert t("""SELECT 'V' AS lab, "V" FROM t""") == """SELECT 'V' AS lab, "V" FROM t"""      # literals and quoted names untouched


def test_registered_dataframes_are_queryable_and_all_null_columns_keep_their_type(st):
    seed(st, 3)
    st.con.register("_k", pd.DataFrame({"source_table": ["scraped_articles"] * 2, "article_id": [1, 2], "loc_share": [None, None]}))
    assert st.scalar("SELECT count(*) FROM article_fact f JOIN _k USING (source_table, article_id)") == 2
    st.con.register("_u", pd.DataFrame({"event_id": ["ev_a"], "loc_share": [None]}))                    # all-NULL: must stay double precision
    st.con.execute("UPDATE event_cluster SET loc_share = u.loc_share FROM _u u WHERE event_cluster.event_id = u.event_id")
    st.con.unregister("_u")
    st.con.register("_k", pd.DataFrame({"x": [1]}))                                                    # re-registering replaces
    assert st.scalar("SELECT count(*) FROM _k") == 1


def test_results_look_like_the_analytics_code_expects(st):
    seed(st, 3)
    df = st.df("SELECT published_at, day, article_id, cluster_sim, is_duplicate FROM article_fact ORDER BY article_id")
    assert str(df["published_at"].dtype).startswith("datetime64") and str(df["day"].dtype).startswith("datetime64")     # DATE -> datetime64
    assert df["published_at"].dt.tz is not None and str(df["article_id"].dtype) == "int64"
    assert df["cluster_sim"].isna().all()                                                               # NULL -> NaN
    assert st.df("SELECT sum(n_articles) AS s FROM event_cluster")["s"].iloc[0] == 3                   # numeric sums are plain numbers
    assert isinstance(st.con.execute("SELECT centroid_sum FROM event_cluster").fetchone()[0], (bytes, type(None)))


# ---- upserts -------------------------------------------------------------------------------------------------------------------
def test_upsert_is_idempotent_rejects_duplicates_and_unknown_columns(st):
    df = pd.DataFrame({"source_table": ["a", "a"], "article_id": [1, 2], "title": ["x", "y"], "is_duplicate": [False] * 2})
    st.upsert("article_fact", df)
    st.upsert("article_fact", df.assign(title=["x2", "y2"]))
    assert st.scalar("SELECT count(*) FROM article_fact") == 2 and st.scalar("SELECT title FROM article_fact WHERE article_id = 1") == "x2"
    with pytest.raises(ValueError):
        st.upsert("article_fact", pd.concat([df, df]))
    with pytest.raises(KeyError):
        st.upsert("article_fact", df.assign(bogus=1))


def test_upsert_replace_all_nulls_omitted_columns_but_partial_update_keeps_them(st):
    st.upsert("event_cluster", pd.DataFrame([{"event_id": "e", "n_articles": 5, "rep_title": "Title", "is_event": True}]))
    st.upsert("event_cluster", pd.DataFrame([{"event_id": "e", "n_articles": 6}]), replace_all=False)
    r = st.df("SELECT n_articles, rep_title, is_event FROM event_cluster").iloc[0]
    assert (r.n_articles, r.rep_title, bool(r.is_event)) == (6, "Title", True)                          # untouched columns kept
    st.upsert("event_cluster", pd.DataFrame([{"event_id": "e", "n_articles": 7}]))                      # the old INSERT OR REPLACE
    r = st.df("SELECT n_articles, rep_title FROM event_cluster").iloc[0]
    assert r.n_articles == 7 and pd.isna(r.rep_title)


def test_transaction_rolls_back_everything(st):
    with pytest.raises(RuntimeError):
        with st.tx():
            st.upsert("article_fact", pd.DataFrame({"source_table": ["a"], "article_id": [1], "is_duplicate": [False]}))
            raise RuntimeError("boom")
    assert st.scalar("SELECT count(*) FROM article_fact") == 0


# ---- read-only store ---------------------------------------------------------------------------------------------------------------
def test_read_only_store_reads_a_stable_snapshot_and_refuses_writes(st, tmp_path):
    seed(st, 5)
    ro = S.Store(tmp_path / "x.db", read_only=True)
    try:
        assert ro.scalar("SELECT count(*) FROM article_fact") == 5
        seed(st, 3, cluster="ev_b", start=100)                                                          # a later commit by the engine
        assert ro.scalar("SELECT count(*) FROM article_fact") == 5                                      # same snapshot for the whole request
        for sql in ("DELETE FROM article_fact", "UPDATE article_fact SET title = 'x'", "DROP TABLE article_fact", "INSERT INTO state VALUES ('a','b',now())"):
            with pytest.raises(PermissionError):
                ro.con.execute(sql)
    finally:
        ro.close()
    with S.Store(tmp_path / "x.db", read_only=True) as ro2:
        assert ro2.scalar("SELECT count(*) FROM article_fact") == 8


# ---- runs, checkpoints, locks ------------------------------------------------------------------------------------------------------
def test_run_lifecycle_and_abandoned_runs(st):
    r1 = st.start_run("incremental", "all", None, None, {"trigger": "scheduled"})
    r2 = st.start_run("incremental", "all", None, None, {"trigger": "manual"})                         # r1 never finished
    assert st.scalar("SELECT status FROM run_log WHERE run_id = ?", [r1]) == "abandoned"
    st.finish_run(r2, "success", {"x": 1}, ingested=7, unique_new=6, touched=2, totals={"pending_labels": 3})
    row = st.df("SELECT status, articles_ingested, trigger, totals, summary FROM run_log WHERE run_id = ?", [r2]).iloc[0]
    assert row.status == "success" and row.articles_ingested == 7 and row.trigger == "manual" and row.totals["pending_labels"] == 3 and row.summary["x"] == 1


def test_checkpoints_are_idempotent_and_state_is_a_key_value_store(st):
    d = dt.date(2026, 10, 10)
    assert not st.stage_done("ingest", "all", d)
    st.mark_stage("ingest", "all", d, "failed", "r", error="x")
    assert not st.stage_done("ingest", "all", d)
    st.mark_stage("ingest", "all", d, "done", "r", 5, 4)
    st.mark_stage("ingest", "all", d, "done", "r2", 5, 4)
    assert st.stage_done("ingest", "all", d) and st.scalar("SELECT count(*) FROM checkpoint") == 1
    st.set_kv("k", "1"); st.set_kv("k", "2")
    assert st.get_kv("k") == "2" and st.get_kv("missing") is None


def test_single_writer_advisory_lock(st, tmp_path):
    other = S.Store(tmp_path / "x.db")
    try:
        assert st.try_lock()
        assert not other.try_lock()                                                                     # a second service (any host) is refused
        st.unlock()
        assert other.try_lock()
    finally:
        other.close()


# ---- live summary from stored records -----------------------------------------------------------------------------------------------
def test_live_summary_comes_from_stored_records(st):
    seed(st, 10)
    st.con.execute("UPDATE article_fact SET is_duplicate = TRUE WHERE article_id = 10")
    st.upsert("pending_label", pd.DataFrame([{"source_table": "scraped_articles", "article_id": 99, "attempts": 0}]))
    st.upsert("excluded_article", pd.DataFrame([{"source_table": "scraped_articles", "article_id": 98, "reason": "date_conflict"},
                                                {"source_table": "scraped_articles", "article_id": 97, "reason": "date_conflict"}]))
    rid = st.start_run("incremental", "all", None, None, {"trigger": "scheduled"})
    st.finish_run(rid, "success", {}, ingested=7, unique_new=6, touched=2)
    st.set_kv("next_run_at", (dt.datetime.now(UTC) + dt.timedelta(minutes=9)).isoformat())
    st.set_kv("interval_seconds", "3600"); st.set_kv("scope", "all")
    st.set_kv("service_heartbeat", dt.datetime.now(UTC).isoformat())
    s = LV.summary(st)
    assert s["status"] == "successful" and s["source"] == "postgresql" and s["last_successful_run"]["run_id"] == rid
    assert s["articles"] == {"processed_total": 10, "unique_total": 9, "duplicates_total": 1, "processed_in_latest_run": 7}
    assert s["clusters"] == {"total": 1, "qualifying_events": 1} and s["pending_labels"] == 1
    assert s["excluded"] == {"total": 2, "by_reason": {"date_conflict": 2}}
    assert s["latest_publication_date"].startswith("2026-10-01")
    assert s["schedule"]["interval_minutes"] == 60 and s["schedule"]["scheduler_alive"] is True and s["schedule"]["next_run_at"]
    assert s["schedule"]["phase"] == "idle"


def test_failed_interrupted_and_running_runs_are_reported_honestly(st):
    ok = st.start_run("incremental", "all", None, None, {}); st.finish_run(ok, "success", {}, ingested=1)
    bad = st.start_run("incremental", "all", None, None, {}); st.finish_run(bad, "failed", {}, error="OpenSearchError: timeout")
    cut = st.start_run("incremental", "all", None, None, {})
    run = st.start_run("range", "all", None, None, {})                                                # starting a run abandons `cut`
    assert st.scalar("SELECT status FROM run_log WHERE run_id = ?", [cut]) == "abandoned"
    st.set_kv("next_run_at", (dt.datetime.now(UTC) - dt.timedelta(minutes=20)).isoformat())          # stale: stored by the previous idle tick
    st.set_kv("service_heartbeat", dt.datetime.now(UTC).isoformat())
    st.set_kv("catchup_missing_days", "12")
    s = LV.summary(st)
    assert s["status"] == "running" and s["schedule"]["phase"] == "running" and s["schedule"]["next_run_at"] is None
    assert s["last_successful_run"]["run_id"] == ok and s["schedule"]["catching_up_days"] == 12
    st.finish_run(run, "failed", {}, error="boom")
    s = LV.summary(st)
    assert s["status"] == "failed" and s["latest_run"]["error"] == "boom"
    assert s["errors"]["consecutive_failures"] == 3 and s["errors"]["failed_runs_24h"] == 3
    assert {e["message"] for e in s["errors"]["recent"]} >= {"OpenSearchError: timeout", "boom"}
    st.set_kv("service_heartbeat", (dt.datetime.now(UTC) - dt.timedelta(hours=2)).isoformat())          # scheduler died mid-run
    r3 = st.start_run("range", "all", None, None, {})
    st.con.execute("UPDATE run_log SET heartbeat_at = now() - interval '2 hours' WHERE run_id = ?", [r3])
    s = LV.summary(st)
    assert s["status"] == "failed" and "interrupted" in s["latest_run"]["error"] and s["schedule"]["scheduler_alive"] is False


# ---- stories and rankings ----------------------------------------------------------------------------------------------------------------
def test_stories_and_rankings_are_materialised_exactly_as_ranked(tmp_path):
    from test_news_trending import build, run
    st = build(tmp_path)
    out = NST.materialise(st, "run1", top_n=30)
    assert "error" not in out and out["stories"] > 0 and out["rankings"] > 0
    data = T.load(st)
    assert st.scalar("SELECT count(*) FROM story") == len(data.stories)
    assert st.scalar("SELECT count(*) FROM story_event") == sum(len(e) for e in data.stories["event_ids"])
    ranked = run(data, country="IN", window="3d", top_n=30)["stories"]
    stored = st.df("SELECT rank, story_id, score FROM trending_ranking WHERE window_key = '3d' AND country_iso = 'IN' AND tax_code = '*' ORDER BY rank")
    assert stored["story_id"].tolist() == [x["story_id"] for x in ranked]
    assert np.allclose(stored["score"], [x["score"] for x in ranked])
    # a story that is not about one agreed domain stores no taxonomy path (same rule as the page)
    assert st.scalar("SELECT count(*) FROM story WHERE tax_status <> 'assigned' AND domain_id IS NOT NULL") == 0
    again = NST.materialise(st, "run2", top_n=30)                                                        # replaced wholesale, not appended
    assert again["stories"] == out["stories"] and st.scalar("SELECT count(*) FROM story") == out["stories"]
    st.close()


# ---- the one-time DuckDB -> PostgreSQL migration -----------------------------------------------------------------------------------------
def test_migration_copies_everything_verifies_and_retires_the_mirror_tables(tmp_path, monkeypatch):
    import importlib.util
    import duckdb
    spec = importlib.util.spec_from_file_location("migrate", news_testutil.ROOT / "deploy" / "migrate_duckdb_to_postgres.py")
    mig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mig)
    path = tmp_path / "old.duckdb"
    d = duckdb.connect(str(path))
    d.execute("""CREATE TABLE kv (key VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMPTZ);
                 CREATE TABLE run_log (run_id VARCHAR PRIMARY KEY, mode VARCHAR, scope VARCHAR, date_from DATE, date_to DATE, started_at TIMESTAMPTZ,
                                       finished_at TIMESTAMPTZ, status VARCHAR, params_json VARCHAR, summary_json VARCHAR, error VARCHAR);
                 CREATE TABLE day_stage (stage VARCHAR, scope VARCHAR, day DATE, status VARCHAR, n_in BIGINT, n_out BIGINT, run_id VARCHAR,
                                         updated_at TIMESTAMPTZ, error VARCHAR, PRIMARY KEY (stage, scope, day));
                 CREATE TABLE article_fact (source_table VARCHAR, article_id BIGINT, published_at TIMESTAMPTZ, day DATE, is_duplicate BOOLEAN,
                                            cluster_id VARCHAR, title VARCHAR, PRIMARY KEY (source_table, article_id));
                 CREATE TABLE article_geo (source_table VARCHAR, article_id BIGINT, ev_country_iso VARCHAR, mention_countries_iso VARCHAR[],
                                           PRIMARY KEY (source_table, article_id));
                 CREATE TABLE event_cluster (event_id VARCHAR PRIMARY KEY, n_articles INTEGER, is_event BOOLEAN, centroid_sum BLOB);
                 CREATE TABLE node_aoi_day (aoi_id VARCHAR, level VARCHAR, node_code VARCHAR, day DATE, V DOUBLE,
                                            PRIMARY KEY (aoi_id, level, node_code, day));
                 CREATE TABLE alert (alert_id VARCHAR PRIMARY KEY, kind VARCHAR, status VARCHAR);
                 CREATE TABLE taxonomy_node (code VARCHAR PRIMARY KEY, name VARCHAR);
                 CREATE TABLE aoi_def (aoi_id VARCHAR PRIMARY KEY, kind VARCHAR);
                 CREATE TABLE corpus_day (day DATE PRIMARY KEY, n_dated_total BIGINT);
                 CREATE TABLE metric_meta (metric_key VARCHAR PRIMARY KEY);
                 CREATE TABLE data_quality (run_id VARCHAR, key VARCHAR, value VARCHAR, PRIMARY KEY (run_id, key));
                 CREATE TABLE excluded_article (source_table VARCHAR, article_id BIGINT, reason VARCHAR, PRIMARY KEY (source_table, article_id));
                 CREATE TABLE pending_label (source_table VARCHAR, article_id BIGINT, attempts INTEGER, PRIMARY KEY (source_table, article_id));
                 CREATE TABLE event_metrics (event_id VARCHAR PRIMARY KEY, n_articles INTEGER);
                 CREATE TABLE event_aoi_test (aoi_id VARCHAR, event_id VARCHAR, kind VARCHAR, p DOUBLE, PRIMARY KEY (aoi_id, event_id, kind));
                 CREATE TABLE event_development (event_id VARCHAR, dev_id VARCHAR, aoi_id VARCHAR, PRIMARY KEY (dev_id, aoi_id));
                 CREATE TABLE aoi_day (aoi_id VARCHAR, day DATE, n_articles INTEGER, PRIMARY KEY (aoi_id, day));
                 CREATE TABLE node_aoi_series (aoi_id VARCHAR, level VARCHAR, node_code VARCHAR, total_v DOUBLE, PRIMARY KEY (aoi_id, level, node_code));
                 CREATE TABLE burst_episode (episode_id VARCHAR PRIMARY KEY, aoi_id VARCHAR);""")
    d.execute("INSERT INTO kv VALUES ('ingest_wm:all', '2026-10-23T09:00:00+00:00', now()), ('live_start_day', '2026-09-23', now()), ('schema_version', '1', now())")
    d.execute("INSERT INTO run_log VALUES ('r_old', 'pilot', 'india_bangladesh', DATE '2026-10-06', DATE '2026-10-19', now(), now(), 'success', '{\"a\": 1}', '{\"b\": 2}', NULL)")
    d.execute("INSERT INTO day_stage VALUES ('ingest', 'all', DATE '2026-10-20', 'done', 10, 9, 'r_old', now(), NULL)")
    d.execute("INSERT INTO article_fact SELECT 'scraped_articles', i, TIMESTAMPTZ '2026-10-20 12:00:00+00' + i * INTERVAL 1 MINUTE, DATE '2026-10-20', i % 5 = 0, 'ev_' || (i % 4), 't' || i FROM range(1, 301) t(i)")
    d.execute("INSERT INTO article_geo SELECT 'scraped_articles', i, CASE WHEN i % 3 = 0 THEN NULL ELSE 'IN' END, ['IN', 'US'] FROM range(1, 301) t(i)")
    d.execute("INSERT INTO event_cluster SELECT 'ev_' || i, 50 + i, i % 2 = 0, CASE WHEN i % 2 = 0 THEN '\\x01\\x02'::BLOB ELSE NULL END FROM range(0, 4) t(i)")
    d.execute("INSERT INTO node_aoi_day SELECT 'country:IN:event', 'topic', 'D01.T01', DATE '2026-10-01' + to_days(CAST(i AS INTEGER)), i * 1.5 FROM range(0, 20) t(i)")
    d.execute("INSERT INTO alert VALUES ('a1', 'node_trend', 'active'), ('a2', 'emerging', 'retracted')")
    d.close()

    schema = "ne_mig_" + secrets.token_hex(3)
    monkeypatch.setenv("NEWS_DERIVED_SCHEMA", schema)
    pg = S.Store()                                                                                      # creates the schema's tables
    # the first-release MIRROR tables exist as tables (as in production before the migration): they must be retired at the end
    pg.con.execute("DROP VIEW article_cluster"); pg.con.execute("DROP VIEW trend_node_day")
    pg.con.execute("CREATE TABLE article_cluster (source_table text, article_id bigint, event_id text)")
    pg.con.execute("CREATE TABLE trend_node_day (aoi_id text, v double precision)")
    pg.con.execute("INSERT INTO article_cluster VALUES ('scraped_articles', 1, 'ev_1')")
    pg.set_kv("service_heartbeat", "keep-me")
    pg.close()
    monkeypatch.setattr("sys.argv", ["migrate", "--duckdb", str(path), "--apply"])
    assert mig.main() == 0
    with S.Store(read_only=True) as r:
        assert r.scalar("SELECT count(*) FROM article_fact") == 300 and r.scalar("SELECT count(*) FROM article_geo") == 300
        assert r.scalar("SELECT count(*) FROM article_fact WHERE is_duplicate") == 60
        assert r.scalar("SELECT count(*) FROM article_geo WHERE 'US' = ANY(mention_countries_iso)") == 300          # list column carried
        assert r.scalar("SELECT count(centroid_sum) FROM event_cluster") == 2
        assert r.scalar('SELECT sum("V") FROM node_aoi_day') == sum(i * 1.5 for i in range(20))
        assert r.stage_done("ingest", "all", dt.date(2026, 10, 20)) and r.scalar("SELECT count(*) FROM checkpoint") == 1     # day_stage -> checkpoint
        assert r.get_kv("ingest_wm:all") == "2026-10-23T09:00:00+00:00" and r.get_kv("live_start_day") == "2026-09-23"        # kv -> state
        assert r.get_kv("service_heartbeat") == "keep-me"                                                    # service keys kept
        assert r.scalar("SELECT params->>'a' FROM run_log WHERE run_id = 'r_old'") == "1"                      # json text -> jsonb
        assert r.scalar("SELECT count(*) FROM information_schema.views WHERE table_schema = ? AND table_name IN ('article_cluster','trend_node_day')", [schema]) == 2
        assert r.scalar("SELECT count(*) FROM article_cluster") == 300                                          # the view reads the full table
    assert mig.main() == 0                                                                                    # idempotent: a second run changes nothing
    with S.Store(read_only=True) as r:
        assert r.scalar("SELECT count(*) FROM article_fact") == 300
