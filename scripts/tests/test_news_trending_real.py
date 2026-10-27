"""Top trending stories on REAL data: every displayed number is recomputed with independent SQL, and the four scope cases are
checked against the stored articles. Read-only. Runs only with NEWS_REAL_DB_TESTS=1 against the PostgreSQL schema configured
by the environment (the live `news_engine` schema, or a rehearsal copy of it); skipped otherwise."""
import os

import pandas as pd
import pytest

from news_testutil import ROOT
from lib import news_geo as G
from lib import news_store as NSt
import sys
sys.path.insert(0, str(ROOT / "UI"))
import news_api as NA  # noqa: E402

DB = None        # the configured schema
pytestmark = pytest.mark.skipif(os.environ.get("NEWS_REAL_DB_TESTS") != "1" or not G.COUNTRIES_FILE.exists(),
                                reason="set NEWS_REAL_DB_TESTS=1 (and the NEWS_DERIVED_PG_* environment) to run against real data")


@pytest.fixture(scope="module")
def api():
    return NA.NewsAPI(DB)


@pytest.fixture(scope="module")
def ro():
    s = NSt.Store(DB, read_only=True)
    yield s
    s.close()


def q(ro, sql, args=()):
    return ro.df(sql, list(args))


def window_numbers(ro, eids, as_of, hours):
    t1, t0, tp = pd.Timestamp(as_of), pd.Timestamp(as_of) - pd.Timedelta(hours=hours), pd.Timestamp(as_of) - pd.Timedelta(hours=2 * hours)
    r = q(ro, """SELECT count(*) FILTER (WHERE published_at > ? AND published_at <= ?) a0,
                        count(DISTINCT source_domain) FILTER (WHERE published_at > ? AND published_at <= ?) p0,
                        count(*) FILTER (WHERE published_at > ? AND published_at <= ?) a1,
                        count(*) total, max(published_at) latest_pub, min(published_at) first_pub
                 FROM article_fact WHERE NOT is_duplicate AND cluster_id IN (SELECT UNNEST(?))""", [t0, t1, t0, t1, tp, t0, eids]).iloc[0]
    return r


@pytest.mark.parametrize("window,hours", [("24h", 24), ("3d", 72), ("7d", 168)])
@pytest.mark.parametrize("country,tax", [("IN", None), (None, "D02"), ("IN", "D03"), ("BD", None), ("US", None)])
def test_every_number_equals_independent_sql(api, ro, window, hours, country, tax):
    r = api.trending(country=country, tax=tax, window=window, limit=10)
    assert r["needs_selection"] is False
    for s in r["stories"]:
        eids = s["event_ids"]
        n = window_numbers(ro, eids, r["as_of"], hours)
        assert s["in_window"]["articles"] == n.a0 and s["in_window"]["publishers"] == n.p0 and s["in_window"]["previous_articles"] == n.a1, s["title"]
        assert s["n_articles"] == n.total
        assert pd.Timestamp(s["latest_activity"]) == pd.Timestamp(n.latest_pub) and pd.Timestamp(s["first_seen"]) == pd.Timestamp(n.first_pub)
        assert s["in_window"]["articles"] >= 3 and s["in_window"]["publishers"] >= 2


@pytest.mark.parametrize("country,tax", [("IN", None), (None, "D01"), ("IN", "D02.T01"), (None, "D03.T01.C01.S01")])
def test_ranking_invariants_and_no_story_listed_twice(api, country, tax):
    r = api.trending(country=country, tax=tax, window="3d", limit=30)
    st = r["stories"]
    assert [s["rank"] for s in st] == list(range(1, len(st) + 1))
    assert [s["score"] for s in st] == sorted((s["score"] for s in st), reverse=True)
    seen = [e for s in st for e in s["event_ids"]] + [e for s in st for x in s["related"] for e in x["event_ids"]]
    assert len(seen) == len(set(seen)), "a cluster appears twice on the page"
    assert len({s["title"] for s in st}) == len(st)                                   # no repeated headline either
    for s in st:
        assert s["reasons"] and s["sources"] and s["title"] and s["n_publishers"] >= 2
        assert all(c["url"] and c["publisher"] and c["title"] and c["published_at"] for c in s["sources"])


def test_scope_cases_against_stored_labels_and_locations(api, ro):
    tm = {r.code: r for r in ro.df("SELECT code, tier, parent FROM taxonomy_node").itertuples()}
    # taxonomy only: the story's top-down path starts with the selected node, any country
    t = api.trending(tax="D02", window="3d", limit=30)
    assert t["stories"] and all(s["taxonomy"]["domain"]["code"] == "D02" for s in t["stories"])
    assert len({s["location"]["country"] for s in t["stories"]}) > 1                  # genuinely across countries
    # every listed story's path is a real branch of the stored taxonomy
    for s in t["stories"]:
        path = [s["taxonomy"][k]["code"] for k in ("domain", "topic", "category", "subtopic") if s["taxonomy"][k]]
        for child, parent in zip(path[1:], path[:-1]):
            assert tm[child].parent == parent
    # country only: located plurality is that country, from the stored article locations
    c = api.trending(country="IN", window="3d", limit=30)
    for s in c["stories"]:
        loc = q(ro, """SELECT g.ev_country_iso c, count(*) n FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                       WHERE NOT f.is_duplicate AND g.ev_country_iso IS NOT NULL AND f.cluster_id IN (SELECT UNNEST(?)) GROUP BY 1 ORDER BY 2 DESC""", [s["event_ids"]])
        assert loc.iloc[0].c == "IN" and loc.iloc[0].n / loc.n.sum() >= 0.5, s["title"]
    # both = stories that satisfy both, and are a subset of each single filter (large limit to compare sets)
    both = {s["story_id"] for s in api.trending(country="IN", tax="D02", window="3d", limit=30)["stories"]}
    all_c = {s["story_id"] for s in api.trending(country="IN", window="3d", limit=30)["stories"]}
    assert both and all(s["taxonomy"]["domain"]["code"] == "D02" and s["location"]["country"] == "IN"
                        for s in api.trending(country="IN", tax="D02", window="3d", limit=30)["stories"])
    assert len(all_c) == len(c["stories"])


def test_citations_and_related_articles_are_the_stored_records(api, ro):
    r = api.trending(country="IN", window="3d", limit=10)
    s = r["stories"][0]
    for c in s["sources"]:
        rec = q(ro, "SELECT title, source_domain, published_at FROM article_fact WHERE url = ? LIMIT 1", [c["url"]])
        assert len(rec) == 1 and rec.title[0] == c["title"] and c["publisher"] == (rec.source_domain[0] or "").removeprefix("www.")
    ids = s["event_ids"] + [e for x in s["related"] for e in x["event_ids"]]
    arts = api.story_articles(ids, limit=200)
    want = q(ro, "SELECT count(*) n FROM article_fact WHERE NOT is_duplicate AND cluster_id IN (SELECT UNNEST(?))", [ids]).n[0]
    assert arts["total"] == want
    for it in arts["items"][:50]:
        assert q(ro, "SELECT count(*) n FROM article_fact WHERE url = ?", [it["url"]]).n[0] >= 1


def test_nothing_selected_and_database_untouched(api, ro):
    tables = ["article_fact", "article_geo", "event_cluster", "node_aoi_day", "alert", "checkpoint", "story", "trending_ranking"]
    fp = lambda: [q(ro, f"SELECT count(*) n FROM {t}").n[0] for t in tables]
    before = fp()
    assert api.trending()["needs_selection"] is True
    api.trending(country="IN"); api.trending(tax="D01"); api.trending_meta()
    with NSt.Store(DB, read_only=True) as again:
        assert [again.scalar(f"SELECT count(*) FROM {t}") for t in tables] == before


def test_known_split_story_is_merged_on_real_data(api, ro):
    """Real example found while tuning: the Houthi attack on Saudi airports was reported in 4 separate clusters (the attack, India's
    reaction, Air India cancellations, the advisory). They must appear as ONE story whose articles are the union."""
    r = api.trending(country="SA", window="7d", limit=10)
    top = r["stories"][0]
    assert "Houthi" in top["title"] and top["n_events"] == 4
    union = q(ro, "SELECT count(*) n FROM article_fact WHERE NOT is_duplicate AND cluster_id IN (SELECT UNNEST(?))", [top["event_ids"]]).n[0]
    assert top["n_articles"] == union == 616
    assert sum(1 for s in r["stories"] if "Houthi" in s["title"]) == 1           # not listed again under another cluster


@pytest.mark.parametrize("country,tax", [(None, "D01"), (None, "D09"), ("IN", "D01.T01"), ("GB", None), (None, "D05")])
def test_listed_taxonomy_is_backed_by_a_majority_of_the_stored_labels(api, ro, country, tax):
    """Regression for irrelevant stories under a topic: re-count the stored labels of every listed story independently. Each
    displayed level must be held by MORE THAN HALF of the story's labelled articles (never a mere plurality), and a story
    listed under a taxonomy selection must itself carry that node."""
    r = api.trending(country=country, tax=tax, window="7d", limit=30)
    for s in r["stories"]:
        arts = q(ro, """SELECT domain_id, topic_id, category_id, subtopic_id FROM article_fact
                        WHERE NOT is_duplicate AND has_label AND domain_id IS NOT NULL AND cluster_id IN (SELECT UNNEST(?))""", [s["event_ids"]])
        for tier, col in (("domain", "domain_id"), ("topic", "topic_id"), ("category", "category_id"), ("subtopic", "subtopic_id")):
            node = s["taxonomy"][tier]
            if node:
                assert 2 * (arts[col] == node["code"]).sum() > len(arts), (s["title"], tier, node["code"])
        if tax:
            assert any(n and n["code"] == tax for n in s["taxonomy"].values()), (s["title"], tax)
        if s["taxonomy_status"] != "assigned":
            assert all(v is None for v in s["taxonomy"].values())


def test_reported_film_story_is_not_listed_under_terrorism_or_the_uk(api):
    """The 'Ranabaali Trailer' film story (68 articles, labels scattered over many domains, 6 of 9 located articles in the UK
    by a GDELT quirk) must appear under no single domain, must not claim the topic it used to get from a 31 % plurality, and
    must not be a United Kingdom story."""
    seen = []
    for country, tax in ((None, "D09"), (None, "D01"), (None, "D03"), ("GB", "D09"), ("GB", "D01"), ("GB", None), ("IN", "D01")):
        r = api.trending(country=country, tax=tax, window="7d", limit=30)
        seen += [(country, tax)] * sum("Ranabaali" in s["title"] for s in r["stories"])
    assert not seen, seen
    r = api.trending(country="IN", window="7d", limit=30)         # still a story in the data; if listed it is honestly 'mixed'
    for s in r["stories"]:
        if "Ranabaali" in s["title"]:
            assert s["taxonomy_status"] == "mixed" and all(v is None for v in s["taxonomy"].values())
