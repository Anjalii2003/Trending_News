"""The live service end to end with in-memory sources: all-country scope, resume-from-checkpoint, catch-up then
incremental, de-duplication, failure + recovery, no overlapping runs, scheduling cadence, health, derived PostgreSQL."""
import datetime as dt
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest

import news_testutil  # noqa: F401  (puts scripts/ on sys.path)
import news_service as SVC
from news_fakes import FakeOS, FakePG, near, topic_vector
from news_testutil import UTC, fact_row, geo_row, populate
from lib import news_config as C
from lib import news_live as LV
from lib import news_pgstore as PGS
from lib import news_sources as NS
from lib import news_store as NSt

NOW = dt.datetime.now(UTC)
TODAY = NOW.date()
D = lambda back: TODAY - dt.timedelta(days=back)
TAX = [{"code": "D01", "tier": "domain", "name": "Conflict", "parent": None, "sort_order": 1, "is_catch_all": False, "in_scope": True, "taxonomy_version": "v4.1"},
       {"code": "D01.T01", "tier": "topic", "name": "Protest", "parent": "D01", "sort_order": 1, "is_catch_all": False, "in_scope": True, "taxonomy_version": "v4.1"},
       {"code": "D01.T01.C01", "tier": "category", "name": "Cat", "parent": "D01.T01", "sort_order": 1, "is_catch_all": False, "in_scope": True, "taxonomy_version": "v4.1"}]
LAB = {"domain_id": "D01", "topic_id": "D01.T01", "category_id": "D01.T01.C01", "subtopic_id": "D01.T01.C01.S01", "category_confidence": 0.8,
       "subtopic_confidence": 0.6, "taxonomy_version": "v4.1", "published_at": None, "content_hash": None, "url": None}


def at(day, hour):
    return dt.datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


class Sim:
    """Adds groups of articles (each group = one event: same topic vector, 4 publishers) to the fake OpenSearch."""

    def __init__(self, fos):
        self.fos, self.n, self.rng = fos, 1000, np.random.default_rng(3)

    def group(self, seed, day=None, hour=10, country="IN", size=4, ingested=None, when=None):
        base = topic_vector(seed)
        ids = []
        for k in range(size):
            self.n += 1
            pub = when + dt.timedelta(minutes=k) if when else at(day, hour + k % 3)
            ing = ingested or (pub + dt.timedelta(minutes=30))
            ids.append(self.fos.add(self.n, published=pub, ingested=ing, vec=near(base, self.rng), country=country, domain=f"p{k}.com"))
        return ids


@pytest.fixture
def live(tmp_path, monkeypatch):
    """The live engine on a throw-away PostgreSQL schema, with in-memory sources."""
    import secrets
    monkeypatch.setenv("NEWS_DERIVED_SCHEMA", "ne_svc_" + secrets.token_hex(4))
    monkeypatch.setattr(C, "NEWS_LOCK", tmp_path / "live.lock")
    monkeypatch.setattr(SVC, "HEARTBEAT_FILE", tmp_path / "hb.json")
    monkeypatch.delenv("NEWS_START_DATE", raising=False)
    monkeypatch.delenv("NEWS_CATCHUP_DAYS_PER_CYCLE", raising=False)
    fos = FakeOS()
    monkeypatch.setattr(NS, "ReadOnlyPG", FakePG)
    monkeypatch.setattr(NS, "ReadOnlyOpenSearch", lambda: fos)
    monkeypatch.setattr(NS, "fetch_taxonomy", lambda pg: TAX)
    monkeypatch.setattr(NS, "fetch_labels", lambda pg, keys: {k: dict(LAB) for k in keys})
    # already-processed data (what the migration carries over): another scope's checkpoint up to D(3)
    with NSt.Store() as st:
        facts = [fact_row(i, D(3), cluster="ev_old", publisher=f"s{i % 3}.com") for i in range(1, 7)]
        for f in facts:
            f["processed_at"] = at(D(3), 12)
        populate(st, facts, [geo_row(i) for i in range(1, 7)])
        st.upsert("event_cluster", pd.DataFrame([{"event_id": "ev_old", "first_seen": at(D(3), 12), "last_seen": at(D(3), 12), "n_articles": 6,
                                                  "n_publishers": 3, "is_event": True, "loc_country_iso": "IN", "updated_at": at(D(3), 12)}]))
        st.mark_stage("ingest", "india_bangladesh", D(3), "done", "seed")
    return fos, Sim(fos)


def duck(q, *a):
    """(historic name) a scalar from the live schema."""
    with NSt.Store(read_only=True) as st:
        return st.scalar(q, list(a))


# ---------------------------------------------------------------------------------------------------------------------------
def test_all_countries_are_processed_and_nothing_is_filtered_by_country(live):
    fos, sim = live
    for seed, c in ((1, "IN"), (2, "US"), (3, "BR"), (4, "NG")):
        sim.group(seed, D(2), country=c)
    sim.group(5, D(2), country=None)                                       # unlocated articles are processed too
    assert SVC.run_cycle("manual")["status"] == "success"
    with NSt.Store(read_only=True) as st:
        countries = set(st.df("SELECT DISTINCT ev_country_iso c FROM article_geo WHERE ev_country_iso IS NOT NULL")["c"])
        aois = set(st.df("SELECT aoi_id FROM aoi_def")["aoi_id"])
    assert {"IN", "US", "BR", "NG"} <= countries
    assert {"country:IN:event", "country:US:event", "country:BR:event", "country:NG:event"} <= aois
    assert duck("SELECT count(*) FROM article_fact WHERE source_table = 'scraped_articles' AND article_id > 1000") == 20
    assert not any("action_country" in json.dumps(q) for q in fos.queries), "a country filter leaked into the all-country scan"


def test_resumes_after_the_last_checkpoint_then_catches_up_then_goes_incremental(live):
    fos, sim = live
    sim.group(1, D(2)); sim.group(2, D(1))
    r1 = SVC.run_cycle("scheduled")
    assert r1["status"] == "success" and r1["catching_up"] is True                  # one missing day per cycle; D(1) still to do
    assert duck("SELECT count(*) FROM checkpoint WHERE scope = 'all' AND stage = 'ingest' AND status = 'done'") == 1
    r2 = SVC.run_cycle("scheduled")
    assert r2["status"] == "success" and r2["catching_up"] is False
    days = {str(r[0]) for r in NSt.Store(read_only=True).con.execute("SELECT day FROM checkpoint WHERE scope = 'all'").fetchall()}
    assert days == {str(D(2)), str(D(1))}
    # the seeded day (D3) and anything earlier was NEVER scanned again: every published-day scan starts at D(2) or later
    lows = [f["range"]["published_at"]["gte"] for q in fos.queries for f in q["bool"]["filter"] if "published_at" in f.get("range", {})]
    assert lows and min(lows) >= at(D(2), 0).strftime("%Y-%m-%dT%H:%M:%SZ")
    sim.group(3, when=NOW - dt.timedelta(minutes=20), ingested=NOW - dt.timedelta(minutes=10))
    r3 = SVC.run_cycle("scheduled")
    assert r3["status"] == "success" and r3["articles_ingested"] == 4
    assert duck("SELECT mode FROM run_log ORDER BY started_at DESC LIMIT 1") == "incremental"


def test_start_day_never_defaults_to_the_beginning_of_the_corpus(monkeypatch):
    import secrets
    monkeypatch.setenv("NEWS_DERIVED_SCHEMA", "ne_empty_" + secrets.token_hex(4))
    monkeypatch.delenv("NEWS_START_DATE", raising=False)
    with NSt.Store() as st:
        with pytest.raises(SystemExit, match="refusing to guess"):
            SVC.start_day(st)
    monkeypatch.setenv("NEWS_START_DATE", "2026-10-20")
    with NSt.Store() as st:
        assert SVC.start_day(st) == dt.date(2026, 10, 20)
        assert st.get_kv("live_start_day") == "2026-10-20"                  # remembered for restarts without the variable
    monkeypatch.delenv("NEWS_START_DATE")
    with NSt.Store() as st:
        assert SVC.start_day(st) == dt.date(2026, 10, 20)


def test_new_data_is_processed_once_and_duplicates_are_not_counted(live):
    fos, sim = live
    sim.group(1, D(2))
    sim.group(2, D(1))
    for _ in range(2):
        SVC.run_cycle("scheduled")
    n0 = duck("SELECT count(*) FROM article_fact")
    # an exact copy of an existing article (same content_hash) arrives later under another id
    sim.n += 1
    fos.add(sim.n, published=NOW - dt.timedelta(minutes=5), ingested=NOW - dt.timedelta(minutes=2), vec=fos.vectors[fos.hits[0]["_id"]],
            content_hash=fos.hits[0]["_source"]["content_hash"], domain="copycat.com")
    r = SVC.run_cycle("scheduled")
    assert r["articles_ingested"] == 1 and r["articles_unique_new"] == 0
    assert duck("SELECT count(*) FROM article_fact") == n0 + 1
    assert duck("SELECT count(*) FROM article_fact WHERE NOT is_duplicate") == n0
    again = SVC.run_cycle("scheduled")                                   # the overlap window re-reads it: nothing changes
    assert again["articles_ingested"] == 0
    assert duck("SELECT count(*) FROM article_fact") == n0 + 1
    assert duck("SELECT count(*) FROM (SELECT source_table, article_id FROM article_fact GROUP BY 1, 2 HAVING count(*) > 1) x") == 0


def test_failure_is_recorded_and_the_next_cycle_recovers_the_same_day(live):
    fos, sim = live
    sim.group(1, D(2))
    fos.fail_scan = 1
    bad = SVC.run_cycle("scheduled")
    assert bad["status"] == "failed" and "simulated outage" in bad["error"]
    row = NSt.Store(read_only=True).df("SELECT status, error FROM run_log ORDER BY started_at DESC LIMIT 1").iloc[0]
    assert row.status == "failed" and "simulated outage" in row.error
    assert duck("SELECT count(*) FROM checkpoint WHERE scope = 'all' AND status = 'done'") == 0     # the failed day is not 'done'
    assert duck("SELECT count(*) FROM article_fact WHERE article_id > 1000") == 0                  # and left no half-written data
    ok = SVC.run_cycle("scheduled")
    assert ok["status"] == "success" and duck("SELECT count(*) FROM article_fact WHERE article_id > 1000") == 4
    s = LV.summary(NSt.Store(read_only=True))
    assert s["errors"]["recent"] and "simulated outage" in s["errors"]["recent"][0]["message"] and s["status"] == "successful"


def test_a_second_run_never_overlaps_the_first(live):
    from lib.news_store import LockBusy, run_lock
    fos, sim = live
    sim.group(1, D(2))
    with run_lock(C.NEWS_LOCK):
        with pytest.raises(LockBusy):
            SVC.run_cycle("manual")
    assert duck("SELECT count(*) FROM run_log") == 0
    assert SVC.run_cycle("manual")["status"] == "success"


def test_interrupted_run_is_marked_abandoned_and_work_resumes(live):
    fos, sim = live
    sim.group(1, D(2))
    with NSt.Store() as st:
        st.start_run("incremental", "all", None, None, {})             # a run that crashed leaves 'running' behind
    assert SVC.run_cycle("scheduled")["status"] == "success"
    statuses = NSt.Store(read_only=True).df("SELECT status FROM run_log ORDER BY started_at")["status"].tolist()
    assert statuses == ["abandoned", "success"]


def test_scheduler_runs_at_the_configured_interval_and_stops_cleanly(live, monkeypatch):
    calls = []
    monkeypatch.setattr(SVC, "interval_seconds", lambda: 2)
    monkeypatch.setattr(SVC, "run_cycle", lambda trigger="scheduled", hb=None, interval=None: calls.append(time.monotonic()) or
                        {"status": "success", "catching_up": False})
    SVC.STOP.clear()
    t = threading.Thread(target=SVC.serve, daemon=True)
    t.start()
    time.sleep(6.5)
    SVC.STOP.set()
    t.join(10)
    assert not t.is_alive()
    assert 3 <= len(calls) <= 4, calls
    gaps = np.diff(calls)
    assert all(1.5 <= g <= 3.0 for g in gaps), gaps
    hb = json.loads(SVC.HEARTBEAT_FILE.read_text())
    assert hb["state"] == "stopped" and hb["interval_seconds"] == 2
    SVC.STOP.clear()


def test_scheduler_continues_immediately_while_catching_up_and_backs_off_after_failures(live, monkeypatch):
    seq = iter([{"status": "success", "catching_up": True}, {"status": "failed", "catching_up": False, "error": "x"}, {"status": "success", "catching_up": False}])
    stamps = []
    monkeypatch.setattr(SVC, "interval_seconds", lambda: 600)
    monkeypatch.setattr(SVC, "run_cycle", lambda trigger="scheduled", hb=None, interval=None: (stamps.append(time.monotonic()), next(seq))[1])
    SVC.STOP.clear()
    t = threading.Thread(target=SVC.serve, daemon=True)
    t.start()
    deadline = time.time() + 25
    while len(stamps) < 2 and time.time() < deadline:
        time.sleep(0.2)
    SVC.STOP.set()
    t.join(40)
    SVC.STOP.clear()
    assert len(stamps) >= 2 and stamps[1] - stamps[0] < 15            # catching up: the next cycle starts at once, not after 10 minutes


def test_health_check(live):
    SVC.HEARTBEAT_FILE.write_text("not json")
    assert SVC.health() == 1
    fresh = lambda **k: SVC.HEARTBEAT_FILE.write_text(json.dumps({"ts": NOW.isoformat() if False else dt.datetime.now(UTC).isoformat(), "interval_seconds": 600, **k}))
    fresh(state="idle", next_run_at=(dt.datetime.now(UTC) + dt.timedelta(minutes=5)).isoformat())
    assert SVC.health() == 0
    fresh(state="idle", next_run_at=(dt.datetime.now(UTC) - dt.timedelta(hours=3)).isoformat())
    assert SVC.health() == 1                                           # the loop is wedged: next run overdue
    fresh(state="running", cycle_started_at=(dt.datetime.now(UTC) - dt.timedelta(hours=5)).isoformat())
    assert SVC.health() == 1                                           # one cycle running far too long
    SVC.HEARTBEAT_FILE.write_text(json.dumps({"ts": (dt.datetime.now(UTC) - dt.timedelta(minutes=10)).isoformat(), "state": "idle"}))
    assert SVC.health() == 1                                           # stale heartbeat


def test_interval_is_configurable(monkeypatch):
    monkeypatch.setenv("NEWS_INTERVAL_MINUTES", "7")
    assert SVC.interval_seconds() == 420
    monkeypatch.setenv("NEWS_INTERVAL_MINUTES", "0.1")
    assert SVC.interval_seconds() == 30                                 # floor: never a busy loop
    monkeypatch.delenv("NEWS_INTERVAL_MINUTES")
    assert SVC.interval_seconds() == 1800


# ---- PostgreSQL is the store ---------------------------------------------------------------------------------------------
def test_cycle_materialises_stories_and_rankings_and_the_summary_matches_the_tables(live):
    fos, sim = live
    sim.group(1, D(2)); sim.group(2, D(2), country="US"); sim.group(3, D(1))
    for _ in range(2):
        assert SVC.run_cycle("scheduled")["status"] == "success"
    with NSt.Store(read_only=True) as st:
        s = LV.summary(st)
        n_art = st.scalar("SELECT count(*) FROM article_fact")
        assert st.scalar("SELECT count(*) FROM article_cluster") == n_art                      # compatibility view over article_fact
        assert st.scalar("SELECT count(*) FROM trend_node_day") == st.scalar("SELECT count(*) FROM node_aoi_day")
        n_story = st.scalar("SELECT count(*) FROM story")
        assert n_story > 0 and st.scalar("SELECT count(*) FROM story_event") >= n_story
        assert st.scalar("SELECT count(*) FROM checkpoint WHERE scope = 'all' AND stage = 'ingest' AND status = 'done'") == 2
    assert s["status"] == "successful" and s["source"] == "postgresql"
    assert s["articles"]["processed_total"] == n_art
    assert s["articles"]["unique_total"] == duck("SELECT count(*) FROM article_fact WHERE NOT is_duplicate")
    assert s["clusters"]["total"] == duck("SELECT count(*) FROM event_cluster")
    assert s["articles"]["processed_in_latest_run"] == 4                                        # the latest run ingested D(1)'s single group
    assert s["latest_publication_date"].startswith(str(D(1))) and s["last_successful_update"]
    assert s["schedule"]["interval_minutes"] == 30 and s["schedule"]["scope"] == "all"


def test_repeated_cycles_never_duplicate_rows(live):
    fos, sim = live
    sim.group(1, D(2)); sim.group(2, D(1))
    for _ in range(2):
        SVC.run_cycle("scheduled")
    n = duck("SELECT count(*) FROM article_fact")
    SVC.run_cycle("scheduled")
    SVC.run_cycle("scheduled")
    assert duck("SELECT count(*) FROM article_fact") == n
    assert duck("SELECT count(*) FROM run_log WHERE status = 'success'") == 4
    assert duck("SELECT count(*) FROM (SELECT source_table, article_id FROM article_fact GROUP BY 1, 2 HAVING count(*) > 1) x") == 0


def test_second_service_is_refused_by_the_postgres_lock(live):
    from lib.news_store import LockBusy
    other = NSt.Store()
    assert other.try_lock()
    try:
        with pytest.raises(LockBusy):
            SVC.run_cycle("manual")
    finally:
        other.close()
    assert duck("SELECT count(*) FROM run_log") == 0


def test_daily_schedule_runs_once_a_day_at_the_configured_utc_time(monkeypatch):
    monkeypatch.setenv("NEWS_RUN_AT_UTC", "02:30")
    at_ = SVC.run_at_utc()
    assert (at_.hour, at_.minute) == (2, 30)
    now = dt.datetime(2026, 10, 23, 1, 0, tzinfo=UTC)
    assert SVC.next_daily(now, at_) == dt.datetime(2026, 10, 23, 2, 30, tzinfo=UTC)          # later today
    assert SVC.next_daily(now.replace(hour=2, minute=30), at_) == dt.datetime(2026, 10, 24, 2, 30, tzinfo=UTC)   # exactly now -> tomorrow
    assert SVC.next_daily(now.replace(hour=23), at_) == dt.datetime(2026, 10, 24, 2, 30, tzinfo=UTC)
    monkeypatch.delenv("NEWS_RUN_AT_UTC")
    assert SVC.run_at_utc() is None
    monkeypatch.setenv("NEWS_RUN_AT_UTC", "25:99")
    with pytest.raises(SystemExit):
        SVC.run_at_utc()
