"""Dashboard API: read-only, UTC timestamps, ad-hoc AOIs computed with the same code and persisted nowhere."""
import datetime as dt
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from news_testutil import ROOT, UTC
import test_news_metrics as TM
from lib import news_geo as G
from lib import news_metrics as M
from lib import news_store as NSt

sys.path.insert(0, str(ROOT / "UI"))
import news_api as NA  # noqa: E402

pytestmark = pytest.mark.skipif(not G.COUNTRIES_FILE.exists(), reason="boundary data missing")


@pytest.fixture
def api(tmp_path):
    db = tmp_path / "api.duckdb"
    st = NSt.Store(db)
    TM.build(st, spike_day=17)
    aoi = TM.AOI_X
    M.seed_metric_meta(st)
    from lib import news_aoi as A
    A.register_aoi(st, aoi)
    M.compute_aoi_panel(st, aoi, TM.P, "r1")
    st.upsert("aoi_day", M.compute_aoi_day(st, aoi, TM.P, None))
    st.close()
    return NA.NewsAPI(db)


def test_store_session_is_utc(tmp_path):
    with NSt.Store(tmp_path / "t.duckdb") as st:
        assert st.scalar("SELECT current_setting('TimeZone')") == "UTC"
    with NSt.Store(tmp_path / "t.duckdb", read_only=True) as st:
        assert st.scalar("SELECT current_setting('TimeZone')") == "UTC"


def test_clean_serialises_aware_times_as_utc():
    ist = dt.timezone(dt.timedelta(hours=5, minutes=30))
    t = dt.datetime(2026, 10, 20, 0, 2, 40, tzinfo=ist)            # = 2026-10-19 18:32:40Z
    assert NA.clean(t) == "2026-10-19T18:32:40+00:00"
    assert NA.clean(pd.Timestamp(t)) == "2026-10-19T18:32:40+00:00"
    assert NA.clean({"x": float("nan"), "y": np.float64("inf"), "z": np.int64(3)}) == {"x": None, "y": None, "z": 3}


def test_registered_bundle_shape_and_alerts(api):
    b = api.bundle(aoi_id=TM.AOI_X.aoi_id)
    assert b["aoi"]["registered"] and b["support"]["supported"] and len(b["coverage"]) == 20
    assert any(n["any_alert"] for n in b["nodes"]) and b["alert_counts"].get("node_trend", 0) >= 1
    assert all(a["q"] < 0.05 for a in b["alerts"] if a["status"] == "active")
    assert b["level"] == "topic" and b["series"]


def test_metric_meta_falls_back_to_code_definitions(tmp_path):
    db = tmp_path / "empty.duckdb"
    NSt.Store(db).close()
    m = NA.NewsAPI(db).meta()
    assert len(m["metric_meta"]) == len(M.METRIC_META) and any(u["metric"].startswith("likes") for u in m["unsupported"])


def test_adhoc_aoi_computes_but_writes_nothing(api):
    before = api.store().scalar("SELECT count(*) FROM node_aoi_day")
    spec = {"kind": "bbox", "bbox": [60, 5, 90, 30], "label": "box", "min_precision": "city"}
    b = api.bundle(spec=spec)
    assert b["aoi"]["adhoc"] and not b["aoi"]["registered"] and b["notes"]
    assert api.store().scalar("SELECT count(*) FROM node_aoi_day") == before                  # nothing persisted
    assert api.store().scalar("SELECT count(*) FROM aoi_def WHERE adhoc") == 0
    # the DB is opened read-only: a write attempt through the API's store would raise
    with api.store() as st:
        with pytest.raises(Exception):
            st.con.execute("DELETE FROM article_fact")


def test_adhoc_country_matches_registered_baseline(api):
    reg = api.bundle(aoi_id=TM.AOI_X.aoi_id)
    adh = api.bundle(spec={"kind": "country", "key": "IN", "mode": "event"})
    r = {n["node_code"]: n for n in reg["nodes"]}
    a = {n["node_code"]: n for n in adh["nodes"]}
    assert set(r) == set(a)
    for k in r:                                    # same code path -> same numbers
        assert r[k]["V"] == a[k]["V"] and (r[k]["baseline_mu"] == a[k]["baseline_mu"] or
                                           (r[k]["baseline_mu"] is None and a[k]["baseline_mu"] is None) or
                                           abs(r[k]["baseline_mu"] - a[k]["baseline_mu"]) < 1e-9)


@pytest.mark.parametrize("spec", [
    {"kind": "polygon", "coords": [[0, 0], [1, 1]]}, {"kind": "polygon", "coords": [[0, 0], [999, 1], [2, 2]]},
    {"kind": "polygon", "coords": [[0, 0]] * 600}, {"kind": "radius", "lon": 0, "lat": 0, "radius_km": -5},
    {"kind": "bbox", "bbox": [10, 0, 5, 1]}, {"kind": "mentioned"}, {"kind": "country", "key": "ZZ"}, "x", None])
def test_bad_adhoc_specs_rejected(api, spec):
    with pytest.raises(NA.ApiError):
        api.bundle(spec=spec)


def test_unregistered_aoi_id_is_404_and_event_ids_are_validated(api):
    with pytest.raises(NA.ApiError) as e:
        api.bundle(aoi_id="country:XX:event")
    assert e.value.status == 404
    for bad in ["ev_1' OR 1=1--", "x" * 40, "notanevent", None]:
        with pytest.raises(NA.ApiError):
            api.event_detail(bad)


def test_dashboard_reads_a_consistent_snapshot_while_the_engine_is_mid_write(api):
    """The engine holds an open write transaction (a run in progress); the dashboard neither blocks nor sees half of it."""
    from lib.news_store import Store
    a, db = api, api._path
    with a.store() as st:
        before = st.scalar("SELECT count(*) FROM article_fact")
    w = Store(db)
    try:
        w.con.execute("BEGIN")
        w.con.execute("DELETE FROM article_fact WHERE article_id IN (SELECT article_id FROM article_fact LIMIT 5)")     # uncommitted
        with a.store() as st:
            assert st.scalar("SELECT count(*) FROM article_fact") == before                 # not blocked, uncommitted change invisible
            with pytest.raises(PermissionError):
                st.con.execute("DELETE FROM article_fact")                                   # and it can never write
    finally:
        w.con.execute("ROLLBACK")
        w.close()


# ---------------------------------------------------------------------------------------------------
# Domain > Topic > Subtopic filter, article lists and citations (real taxonomy relationships, no headline matching)
# ---------------------------------------------------------------------------------------------------
def _tax_rows():
    r = []
    def add(code, tier, name, parent, scope=True):
        r.append({"code": code, "tier": tier, "name": name, "parent": parent, "in_scope": scope, "sort_order": 0,
                  "is_catch_all": False, "taxonomy_version": "v4.1", "loaded_at": pd.Timestamp.now(tz="UTC")})
    add("D00", "domain", "Non-relevant", None, False)
    add("D01", "domain", "Conflict", None)
    for t in ("T01", "T02"):
        add(f"D01.{t}", "topic", f"Topic {t}", "D01")
        add(f"D01.{t}.C01", "category", f"Category {t}", f"D01.{t}")
        for s in ("S01", "S02"):
            add(f"D01.{t}.C01.{s}", "subtopic", f"Sub {t}{s}", f"D01.{t}.C01")
    return pd.DataFrame(r)


@pytest.fixture
def tapi(tmp_path):
    db = tmp_path / "tax.duckdb"
    st = NSt.Store(db)
    TM.build(st, spike_day=17)
    st.upsert("taxonomy_node", _tax_rows())
    # real subtopic labels, alternating; urls; and a title that merely MENTIONS another topic's name
    st.con.execute("UPDATE article_fact SET subtopic_id = category_id || CASE WHEN article_id % 2 = 0 THEN '.S01' ELSE '.S02' END, "
                   "url = 'https://www.pub' || (article_id % 7) || '.example/a/' || article_id, title = 'Story ' || article_id")
    st.con.execute("UPDATE article_fact SET title = 'Topic T01 Category T01 Sub T01S01 everywhere' WHERE topic_id = 'D01.T02' AND article_id % 5 = 0")
    st.con.execute("UPDATE article_fact SET url = 'javascript:alert(1)' WHERE article_id = (SELECT min(article_id) FROM article_fact)")
    # three events with mixed labels (primary label by plurality, ties broken by code)
    ids = [r[0] for r in st.con.execute("SELECT article_id FROM article_fact f JOIN article_geo g USING (source_table, article_id) "
                                        "WHERE g.ev_country_iso = 'IN' ORDER BY article_id LIMIT 4000").fetchall()]
    t1 = [i for i in ids if st.scalar("SELECT topic_id FROM article_fact WHERE article_id = ?", [i]) == "D01.T01"]
    t2 = [i for i in ids if st.scalar("SELECT topic_id FROM article_fact WHERE article_id = ?", [i]) == "D01.T02"]
    groups = {"ev_pure1": t1[:5], "ev_major2": t2[:3] + t1[5:6], "ev_tie": t1[6:8] + t2[3:5]}
    rows = []
    for eid, arts in groups.items():
        for k, a in enumerate(arts):
            st.con.execute("UPDATE article_fact SET cluster_id = ?, source_domain = ? WHERE article_id = ?", [eid, f"p{k}.example", a])
        rows.append({"event_id": eid, "is_event": True, "n_articles": len(arts), "n_publishers": len(arts), "loc_status": "located",
                     "loc_country_iso": "IN", "loc_confidence": "high", "first_seen": pd.Timestamp("2026-10-10", tz="UTC"),
                     "last_seen": pd.Timestamp("2026-10-12", tz="UTC"), "rep_title": eid, "domain_id": "D01", "topic_id": "D01.T01"})
    st.upsert("event_cluster", pd.DataFrame(rows))
    from lib import news_aoi as A
    A.register_aoi(st, TM.AOI_X)
    M.compute_aoi_panel(st, TM.AOI_X, TM.P, "r1")
    st.upsert("aoi_day", M.compute_aoi_day(st, TM.AOI_X, TM.P, None))
    st.close()
    return NA.NewsAPI(db), db


def _sql(db, q, args=()):
    with NSt.Store(db, read_only=True) as st:
        return st.scalar(q, list(args))


@pytest.mark.parametrize("code,col", [("D01", "domain_id"), ("D01.T01", "topic_id"), ("D01.T02", "topic_id"),
                                       ("D01.T01.C01.S01", "subtopic_id"), ("D01.T02.C01.S02", "subtopic_id")])
def test_taxonomy_filter_matches_stored_labels_exactly(tapi, code, col):
    api, db = tapi
    b = api.bundle(aoi_id=TM.AOI_X.aoi_id, tax=code, level="subtopic" if col == "subtopic_id" else "topic")
    want = _sql(db, f"SELECT count(*) FROM article_fact f JOIN article_geo g USING (source_table, article_id) "
                    f"WHERE NOT f.is_duplicate AND g.ev_country_iso = 'IN' AND f.{col} = ?", [code])
    assert b["tax"]["n_match"] == want == b["articles"]["total"] > 0
    assert b["tax"]["code"] == code and [p["code"] for p in b["tax"]["path"]][-1] == code
    assert b["tax"]["path"][0]["tier"] == "domain"                                  # the real parent chain, from taxonomy_node


def test_never_matches_on_headline_text(tapi):
    api, db = tapi
    t1 = api.articles(aoi_id=TM.AOI_X.aoi_id, tax="D01.T01", limit=200)
    decoys = _sql(db, "SELECT count(*) FROM article_fact WHERE title LIKE 'Topic T01 Category T01%' AND topic_id = 'D01.T02'")
    assert decoys > 0
    assert not any(i["title"].startswith("Topic T01 Category T01") and i["tax_topic"] != "Topic T01" for i in t1["items"])
    assert all(i["tax_topic"] == "Topic T01" for i in t1["items"])


def test_event_matches_only_on_majority_agreement(tapi):
    api, _ = tapi
    ids = lambda code: {e["event_id"] for e in api.bundle(aoi_id=TM.AOI_X.aoi_id, tax=code)["events"]}
    assert ids("D01.T01") == {"ev_pure1"}                      # ev_tie is 2 vs 2: NO topic has a majority, so neither topic claims it
    assert ids("D01.T02") == {"ev_major2"}                     # 3 of 4 members
    assert ids("D01") == {"ev_pure1", "ev_major2", "ev_tie"}   # but all of its articles agree on the DOMAIN
    share = {e["event_id"]: e["tax_share"] for e in api.bundle(aoi_id=TM.AOI_X.aoi_id, tax="D01.T02")["events"]}
    assert share["ev_major2"] == pytest.approx(0.75)           # shown to the user, never hidden


def test_event_rows_show_the_agreed_path_or_an_honest_status(tapi):
    api, _ = tapi
    ev = {e["event_id"]: e for e in api.bundle(aoi_id=TM.AOI_X.aoi_id, tax="D01")["events"]}
    assert ev["ev_pure1"]["topic_name"] == "Topic T01" and ev["ev_pure1"]["tax_status"] == "assigned"
    assert ev["ev_tie"]["domain_name"] and ev["ev_tie"]["topic_name"] is None and ev["ev_tie"]["tax_status"] == "assigned"
    d = api.event_detail("ev_tie")["event"]
    assert d["domain_name"] and d["topic_name"] is None            # the drawer agrees with the table


def test_alerts_and_nodes_follow_the_filter(tapi):
    api, _ = tapi
    b = api.bundle(aoi_id=TM.AOI_X.aoi_id, tax="D01.T01")
    assert b["nodes"] and all(n["node_code"] == "D01.T01" for n in b["nodes"])
    assert all(a["entity_id"] == "D01.T01" or a["entity_id"].startswith("D01.T01.") for a in b["alerts"] if a["kind"] == "node_trend")
    assert b["alert_counts"].get("node_trend", 0) == len([a for a in b["alerts"] if a["kind"] == "node_trend"])
    other = api.bundle(aoi_id=TM.AOI_X.aoi_id, tax="D01.T02")
    assert not ({a["alert_id"] for a in b["alerts"]} & {a["alert_id"] for a in other["alerts"]})     # disjoint


def test_subtopic_level_is_computed_in_memory_and_stored_nothing(tapi):
    api, db = tapi
    before = _sql(db, "SELECT count(*) FROM node_aoi_day")
    b = api.bundle(aoi_id=TM.AOI_X.aoi_id, level="subtopic")
    assert b["nodes"] and all(n["node_code"].count(".") == 3 for n in b["nodes"])
    assert _sql(db, "SELECT count(*) FROM node_aoi_day") == before and _sql(db, "SELECT count(*) FROM node_aoi_day WHERE level = 'subtopic'") == 0
    assert any("Subtopic series are computed on demand" in n for n in b["notes"])
    f = api.bundle(aoi_id=TM.AOI_X.aoi_id, level="subtopic", tax="D01.T01.C01.S01")
    assert [n["node_code"] for n in f["nodes"]] == ["D01.T01.C01.S01"]


@pytest.mark.parametrize("bad", ["D00", "D99", "D01' OR 1=1 --", "", 5, ["D01"], "d01", "D01.T01.C01.S09"])
def test_bad_taxonomy_values_rejected_or_ignored(tapi, bad):
    api, _ = tapi
    if bad == "":
        assert api.bundle(aoi_id=TM.AOI_X.aoi_id, tax=bad)["tax"] is None          # empty = All Topics
        return
    with pytest.raises(NA.ApiError):
        api.bundle(aoi_id=TM.AOI_X.aoi_id, tax=bad)


def test_articles_endpoint_paging_order_and_node_day_filters(tapi):
    api, db = tapi
    a1 = api.articles(aoi_id=TM.AOI_X.aoi_id, limit=50)
    a2 = api.articles(aoi_id=TM.AOI_X.aoi_id, offset=50, limit=50)
    assert a1["total"] == a2["total"] and len(a1["items"]) == 50
    key = lambda i: i["url"] or i["title"]                                            # url is unique per article (titles can repeat)
    k1 = {key(i) for i in a1["items"]}
    assert len(k1) == 50 and not (k1 & {key(i) for i in a2["items"]})                 # pages are disjoint
    dates = [i["published_at"] for i in a1["items"]]
    assert dates == sorted(dates, reverse=True)                                        # newest first
    day = "2026-10-18"
    r = api.articles(aoi_id=TM.AOI_X.aoi_id, node={"level": "topic", "code": "D01.T01"}, day=day, limit=200)
    want = _sql(db, "SELECT count(*) FROM article_fact f JOIN article_geo g USING (source_table, article_id) WHERE NOT f.is_duplicate "
                    "AND g.ev_country_iso = 'IN' AND f.topic_id = 'D01.T01' AND f.day = ?", [day])
    assert r["total"] == want > 100                                                    # the spike day
    for bad in ({"level": "bogus", "code": "x"}, {"code": "D01"}, "x"):
        with pytest.raises(NA.ApiError):
            api.articles(aoi_id=TM.AOI_X.aoi_id, node=bad)
    with pytest.raises(NA.ApiError):
        api.articles(aoi_id=TM.AOI_X.aoi_id, day="not-a-day")
    assert len(api.articles(aoi_id=TM.AOI_X.aoi_id, limit=10 ** 6)["items"]) <= 200  # page size is capped


def test_citations_have_title_publisher_date_link_and_unsafe_urls_are_dropped(tapi):
    api, db = tapi
    b = api.bundle(aoi_id=TM.AOI_X.aoi_id)
    ev = {e["event_id"]: e for e in b["events"]}["ev_pure1"]
    assert 1 <= len(ev["sources"]) <= 3
    for c in ev["sources"]:
        assert c["title"] and c["publisher"] and c["published_at"] and (c["url"] is None or c["url"].startswith("https://"))
        assert not c["publisher"].startswith("www.")                                   # publisher name, www. stripped
    assert any(c["url"] for c in ev["sources"])
    alerts = [a for a in b["alerts"] if a["kind"] == "node_trend"]
    assert alerts and all(a["sources"] and a["n_sources"] >= len(a["sources"]) for a in alerts)
    for a in alerts:
        for c in a["sources"]:
            assert c["url"] is None or c["url"].startswith(("http://", "https://"))
    bad = _sql(db, "SELECT article_id FROM article_fact WHERE url LIKE 'javascript:%'")
    items = api.articles(aoi_id=TM.AOI_X.aoi_id, limit=200, offset=0)["items"]
    allitems = items
    off = 200
    tot = api.articles(aoi_id=TM.AOI_X.aoi_id, limit=1)["total"]
    while off < tot:
        allitems += api.articles(aoi_id=TM.AOI_X.aoi_id, limit=200, offset=off)["items"]
        off += 200
    assert not any((i["url"] or "").startswith("javascript:") for i in allitems)       # never emitted as a link
    assert any(i["url"] is None for i in allitems) and bad is not None


def test_taxonomy_filter_works_for_adhoc_areas_too(tapi):
    api, db = tapi
    b = api.bundle(spec={"kind": "country", "key": "IN", "mode": "event"}, tax="D01.T01")
    r = api.bundle(aoi_id=TM.AOI_X.aoi_id, tax="D01.T01")
    assert b["tax"]["n_match"] == r["tax"]["n_match"] and {e["event_id"] for e in b["events"]} == {e["event_id"] for e in r["events"]}


def test_dashboard_events_under_a_topic_filter_respect_the_relevance_gate(tapi):
    api, db = tapi
    with NSt.Store(db) as st:
        st.con.execute("UPDATE event_cluster SET ref_sim = 0.40, ref_votes = 'D00:10' WHERE event_id = 'ev_pure1'")        # off-topic
        st.con.execute("UPDATE event_cluster SET ref_sim = 0.85, ref_votes = 'D01:9,D02:1' WHERE event_id = 'ev_major2'")  # on-topic
    api._live_cache = None
    ids = lambda code: {e["event_id"] for e in api.bundle(aoi_id=TM.AOI_X.aoi_id, tax=code)["events"]}
    assert ids("D01.T01") == set()                                  # ev_pure1 is gone; the rest of D01.T01 never matched
    assert ids("D01.T02") == {"ev_major2"}                          # a scored, relevant event stays
    assert "ev_pure1" not in ids("D01") and "ev_tie" in ids("D01")  # unscored events are not filtered
