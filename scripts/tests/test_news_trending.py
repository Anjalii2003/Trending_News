"""Top trending stories: merging, window statistics, ranking, the four scope cases, evidence and the API.
Synthetic world with planted, known answers. No network, no production data."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from news_testutil import ROOT, UTC
from lib import news_config as C
from lib import news_geo as G
from lib import news_store as NSt
from lib import news_trending as T

import sys
sys.path.insert(0, str(ROOT / "UI"))
import news_api as NA  # noqa: E402

P = dict(C.PARAMS)
AS_OF = dt.datetime(2026, 10, 19, 23, 0, tzinfo=UTC)
START = dt.datetime(2026, 10, 6, 0, 0, tzinfo=UTC)
D = 1024
rng = np.random.default_rng(5)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def rand_unit():
    return unit(rng.normal(size=D))


def at_cos(base, c):
    """A unit vector with cosine exactly `c` to `base`."""
    o = rand_unit()
    o = unit(o - (o @ base) * base)
    return unit(c * base + np.sqrt(1 - c * c) * o)


class World:
    """Writes events + their articles straight into a derived store."""

    def __init__(self, path):
        self.st = NSt.Store(path)
        self.n = 0
        self.facts, self.geos, self.events = [], [], []
        tax = []
        for code, tier, name, parent in [("D01", "domain", "Conflict", None), ("D02", "domain", "Politics", None), ("D03", "domain", "Economy", None),
                                         ("D01.T01", "topic", "Protest", "D01"), ("D02.T01", "topic", "Bilateral", "D02"),
                                         ("D03.T05", "topic", "Trade", "D03"), ("D03.T06", "topic", "Energy", "D03"),
                                         ("D01.T01.C01", "category", "Protest cat", "D01.T01"), ("D02.T01.C01", "category", "Bilat cat", "D02.T01"),
                                         ("D03.T05.C01", "category", "Trade cat", "D03.T05"), ("D03.T06.C01", "category", "Energy cat", "D03.T06"),
                                         ("D01.T01.C01.S01", "subtopic", "Sub protest", "D01.T01.C01"), ("D02.T01.C01.S01", "subtopic", "Sub bilat", "D02.T01.C01"),
                                         ("D03.T05.C01.S01", "subtopic", "Sub trade", "D03.T05.C01"), ("D03.T06.C01.S01", "subtopic", "Sub energy", "D03.T06.C01")]:
            tax.append({"code": code, "tier": tier, "name": name, "parent": parent, "in_scope": True, "sort_order": 0, "is_catch_all": False,
                        "taxonomy_version": "v4.1", "loaded_at": pd.Timestamp.now(tz="UTC")})
        self.st.upsert("taxonomy_node", pd.DataFrame(tax))

    def event(self, eid, vec, arts, *, country="IN", labels=("D01", "D01.T01", "D01.T01.C01", "D01.T01.C01.S01"), title=None, place="Mumbai, Maharashtra, India"):
        """arts: list of (hours_before_AS_OF, publisher) ; labels may be a list per article."""
        rows = []
        for k, (h, pub) in enumerate(arts):
            self.n += 1
            ts = AS_OF - dt.timedelta(hours=h)
            lab = labels[k % len(labels)] if isinstance(labels, list) else labels
            self.facts.append({"source_table": "scraped_articles", "article_id": self.n, "os_id": f"scraped_articles-{self.n}", "published_at": ts, "day": ts.date(),
                               "has_label": True, "domain_id": lab[0], "topic_id": lab[1], "category_id": lab[2], "subtopic_id": lab[3], "is_duplicate": False,
                               "source_domain": pub, "title": f"{title or eid} #{self.n}", "url": f"https://{pub}/a/{self.n}", "cluster_id": eid})
            self.geos.append({"source_table": "scraped_articles", "article_id": self.n, "ev_country_iso": country, "ev_place": place if country else None,
                              "ev_precision": "city" if country else None, "loc_status": "located" if country else "unlocated",
                              "geo_inconsistent": False, "mention_countries_iso": []})
            rows.append(ts)
        v = np.asarray(vec, dtype=np.float32) * len(arts)
        self.events.append({"event_id": eid, "rep_title": title or eid, "n_articles": len(arts), "n_publishers": len({p for _, p in arts}), "is_event": True,
                            "first_seen": min(rows), "last_seen": max(rows), "centroid_sum": v.tobytes(), "loc_status": "located" if country else "unlocated",
                            "loc_country_iso": country, "loc_confidence": "high"})

    def commit(self):
        self.st.upsert("article_fact", pd.DataFrame(self.facts))
        self.st.upsert("article_geo", pd.DataFrame(self.geos))
        self.st.upsert("event_cluster", pd.DataFrame(self.events))
        return self.st


def pubs(n, prefix="p"):
    return [f"{prefix}{i}.example" for i in range(n)]


def build(tmp_path, with_filler=True):
    w = World(tmp_path / "w.duckdb")
    big_v = rand_unit()
    # BIG: 30 articles from 15 publishers in the last 20 hours (a NEW story: nothing before)
    w.event("ev_big", big_v, [(1 + i * 0.6, pubs(15)[i % 15]) for i in range(30)], title="Big new story")
    # STEADY: 8 articles in the current window and 8 in the previous one
    sv = rand_unit()
    w.event("ev_steady", sv, [(5 + i * 8, pubs(8, "s")[i % 8]) for i in range(8)] + [(80 + i * 8, pubs(8, "s")[i % 8]) for i in range(8)], title="Steady story")
    # OLD: active only 8-9 days ago: not a candidate in a 3-day window
    w.event("ev_old", rand_unit(), [(200 + i, pubs(6, "o")[i % 6]) for i in range(10)], title="Old story")
    # SMALL: 3 articles from 2 publishers, very recent
    w.event("ev_small", rand_unit(), [(0.5, "x0.example"), (1.0, "x1.example"), (1.5, "x0.example")], title="Small recent story")
    # other country / unlocated, different taxonomy where noted
    w.event("ev_us", rand_unit(), [(2 + i, pubs(10, "u")[i % 10]) for i in range(20)], country="US", labels=("D02", "D02.T01", "D02.T01.C01", "D02.T01.C01.S01"), title="US story")
    w.event("ev_unloc", rand_unit(), [(3 + i * 1.5, pubs(8, "n")[i % 8]) for i in range(12)], country=None, title="Unlocated story")
    # the SAME story split into two clusters (centroid cos 0.99, overlapping) -> must be ONE story
    dv = rand_unit()
    w.event("ev_dup_a", dv, [(4 + i, pubs(5, "da")[i % 5]) for i in range(10)], title="Split story part A")
    w.event("ev_dup_b", at_cos(dv, 0.99), [(6 + i, pubs(4, "db")[i % 4]) for i in range(6)], title="Split story part B")
    # a SIBLING of BIG: cosine 0.89 (below the 0.90 merge, above the 0.88 display fold)
    w.event("ev_sib", at_cos(big_v, 0.89), [(2 + i, pubs(6, "sb")[i % 6]) for i in range(8)], title="Sibling of big story")
    # labels that DISAGREE: 4 of 10 articles in D02 (a plurality, not a majority), the rest split over D03 and D04 -> no domain
    mixed = [("D02", "D02.T01", "D02.T01.C01", "D02.T01.C01.S01")] * 4 + [("D03", "D03.T05", "D03.T05.C01", "D03.T05.C01.S01")] * 3 + \
            [("D04", "D04.T01", "D04.T01.C01", "D04.T01.C01.S01")] * 3
    w.event("ev_mixed", rand_unit(), [(7 + i, pubs(5, "m")[i % 5]) for i in range(10)], labels=mixed, title="Mixed-label story")
    # domain has a majority (7 of 10 in D03) but its topics split 4/3 -> domain assigned, topic NOT (no majority)
    part = [("D03", "D03.T05", "D03.T05.C01", "D03.T05.C01.S01")] * 4 + [("D03", "D03.T06", "D03.T06.C01", "D03.T06.C01.S01")] * 3 + \
           [("D02", "D02.T01", "D02.T01.C01", "D02.T01.C01.S01")] * 3
    w.event("ev_partial", rand_unit(), [(8 + i, pubs(5, "q")[i % 5]) for i in range(10)], labels=part, title="Partly-agreeing story")
    # the classifier's own "Non-relevant / Other" is the majority -> outside the taxonomy
    w.event("ev_outside", rand_unit(), [(9 + i, pubs(5, "z")[i % 5]) for i in range(10)],
            labels=[("D00", "D00.T00", "D00.T00.C00", "D00.T00.C00.S00")] * 6 + [("D01", "D01.T01", "D01.T01.C01", "D01.T01.C01.S01")] * 4,
            title="Non-relevant story")
    # weak location evidence: 10 articles, ONE located in India (a stray GDELT hit) -> not an Indian story
    w.event("ev_weakloc", rand_unit(), [(10 + i, pubs(5, "w")[i % 5]) for i in range(10)], title="Barely-located story")
    for g in w.geos[-9:]:
        g.update(ev_country_iso=None, ev_place=None, ev_precision=None, loc_status="unlocated")
    # sparse location: 20 articles, 4 located and all in India (share 1.0) but only 20 % of the story -> not an Indian story
    w.event("ev_sparse", rand_unit(), [(12 + i * 0.5, pubs(5, "sp")[i % 5]) for i in range(20)], title="Sparsely-located story")
    for g in w.geos[-16:]:
        g.update(ev_country_iso=None, ev_place=None, ev_precision=None, loc_status="unlocated")
    # contested location: 6 located articles, 3 in India and 3 in the US (share 0.5) -> not confidently either
    w.event("ev_contested", rand_unit(), [(11 + i, pubs(5, "c")[i % 5]) for i in range(6)], title="Contested-location story")
    for g in w.geos[-3:]:
        g.update(ev_country_iso="US")
    if with_filler:     # 80 comparable story-windows in EARLIER, fully observed windows (D01, India)
        for i in range(80):
            hrs = float(rng.uniform(80, 290))
            w.event(f"ev_f{i:03d}", rand_unit(), [(hrs + j * 2, f"f{(i + j) % 4}.example") for j in range(int(rng.integers(3, 7)))], title=f"Filler {i}")
    return w.commit()


@pytest.fixture
def world(tmp_path):
    st = build(tmp_path)
    data = T.load(st)
    yield st, data
    st.close()


NAMES = {"D01": "Conflict", "D02": "Politics", "D03": "Economy", "D01.T01": "Protest", "D02.T01": "Bilateral", "D03.T05": "Trade", "D03.T06": "Energy"}


def run(data, **kw):
    kw.setdefault("names", NAMES)
    kw.setdefault("scope_name", "India" if kw.get("country") == "IN" else "")
    return T.trending(data, P=P, **kw)


def ids(res):
    return [s["story_id"] for s in res["stories"]]


# ---------------------------------------------------------------------------------------------------
# pure pieces
# ---------------------------------------------------------------------------------------------------
def test_epoch_seconds_is_resolution_independent():
    """Regression: pandas 3 stores datetimes in microseconds; astype('int64')/1e9 was off by 1000x and nothing ranked."""
    t = pd.Series(pd.to_datetime(["2026-10-19T23:35:56+00:00"], utc=True))
    assert T.epoch_seconds(t)[0] == pytest.approx(dt.datetime(2026, 10, 19, 23, 35, 56, tzinfo=UTC).timestamp())
    assert T.epoch_seconds(t.dt.as_unit("ns"))[0] == pytest.approx(T.epoch_seconds(t.dt.as_unit("us"))[0])


def test_merge_events_rules():
    base = rand_unit()
    U = np.vstack([base, at_cos(base, 0.95), at_cos(base, 0.95), rand_unit(), base])
    t0 = 1_800_000_000.0
    first = np.array([t0, t0 + 3600, t0 + 10 * 86400, t0, t0 + 3600])
    last = first + 7200
    lab = T.merge_events(U, first, last, 0.90, 72)
    assert lab[0] == lab[1] == lab[4]                       # similar AND overlapping/adjacent in time -> one story
    assert lab[2] != lab[0]                                 # same subject but 10 days later -> a different story
    assert lab[3] not in (lab[0], lab[2])                   # dissimilar
    has = np.array([True, True, True, True, False])
    lab2 = T.merge_events(U, first, last, 0.90, 72, has)
    assert lab2[4] != lab2[0]                               # an event without a centroid is never merged
    assert list(T.merge_events(U, first, last, 0.90, 72, block=1)) == list(lab)     # block size does not change the answer


def test_window_table_counts_articles_and_distinct_publishers():
    t = AS_OF.timestamp()
    mem = pd.DataFrame({"story_id": ["a"] * 6, "ts": [t - 3600, t - 7200, t - 7300, t - 80 * 3600, t - 81 * 3600, t + 3600],
                        "source_domain": ["x", "x", "y", "x", "z", "q"]})
    tab = T.window_table(mem, t, 72).set_index(["story_id", "k"])
    assert tab.loc[("a", 0), "n"] == 3 and tab.loc[("a", 0), "p"] == 2        # k=0: 3 articles from {x, y}
    assert tab.loc[("a", 1), "n"] == 2 and tab.loc[("a", 1), "p"] == 2        # k=1 (previous window)
    assert tab["n"].sum() == 5                                               # an article AFTER the data end is not counted


# ---------------------------------------------------------------------------------------------------
# stories
# ---------------------------------------------------------------------------------------------------
def test_same_story_in_two_clusters_is_one_story(world):
    _, data = world
    s = data.stories.set_index("story_id")
    split = [i for i in s.index if "ev_dup_a" in s.loc[i, "event_ids"]]
    assert len(split) == 1 and sorted(s.loc[split[0], "event_ids"]) == ["ev_dup_a", "ev_dup_b"]
    assert s.loc[split[0], "n_articles"] == 16                                  # 10 + 6 articles, counted once
    assert "ev_sib" not in s.loc[[i for i in s.index if "ev_big" in s.loc[i, "event_ids"]][0], "event_ids"]   # 0.89 < 0.90: not merged globally


def _story(data, ev):
    r = data.stories.set_index("story_id")
    return r[[ev in e for e in r["event_ids"]]].iloc[0]


def test_taxonomy_needs_majority_agreement_not_a_plurality(world):
    _, data = world
    row = _story(data, "ev_mixed")                       # 4 / 3 / 3 over three branches: D02 has the plurality but only 40 %
    assert row["tax_status"] == "mixed" and pd.isna(row["domain"]) and pd.isna(row["topic"])
    for tier, code in (("domain", "D02"), ("domain", "D03"), ("domain", "D04"), ("topic", "D02.T01"), ("topic", "D03.T05")):
        out = run(data, tax={"tier": tier, "code": code, "name": code})
        assert not any("ev_mixed" in s["event_ids"] for s in out["stories"]), (tier, code)


def test_taxonomy_path_stops_where_agreement_stops(world):
    _, data = world
    row = _story(data, "ev_partial")                     # D03 = 7/10 (majority); D03.T05 = 4/10 (not)
    assert row["tax_status"] == "assigned" and row["domain"] == "D03" and pd.isna(row["topic"]) and pd.isna(row["subtopic"])
    assert any("ev_partial" in s["event_ids"] for s in run(data, tax={"tier": "domain", "code": "D03", "name": "x"})["stories"])
    assert not any("ev_partial" in s["event_ids"] for s in run(data, tax={"tier": "topic", "code": "D03.T05", "name": "x"})["stories"])
    full = _story(data, "ev_big")                        # every article agrees -> the full 4-level path
    assert (full["domain"], full["topic"], full["category"], full["subtopic"]) == ("D01", "D01.T01", "D01.T01.C01", "D01.T01.C01.S01")


def test_outside_taxonomy_story_is_reported_as_such_and_never_listed_under_a_topic(world):
    _, data = world
    row = _story(data, "ev_outside")
    assert row["tax_status"] == "outside"
    assert not any("ev_outside" in s["event_ids"] for s in run(data, tax={"tier": "domain", "code": "D01", "name": "x"})["stories"])
    out = run(data, country="IN", top_n=30)               # still reachable by country alone, and labelled honestly
    s = [x for x in out["stories"] if "ev_outside" in x["event_ids"]]
    assert len(s) == 1 and s[0]["taxonomy_status"] == "outside" and all(v is None for v in s[0]["taxonomy"].values())


def test_exact_duplicates_are_not_counted(tmp_path):
    w = World(tmp_path / "d.duckdb")
    w.event("ev_x", rand_unit(), [(1, "a.example"), (2, "b.example"), (3, "c.example")], title="X")
    st = w.commit()
    st.con.execute("UPDATE article_fact SET is_duplicate = TRUE WHERE article_id = 1")
    d = T.load(st)
    assert d.stories.iloc[0]["n_articles"] == 2 and len(d.mem) == 2
    st.close()


# ---------------------------------------------------------------------------------------------------
# the four cases
# ---------------------------------------------------------------------------------------------------
def test_nothing_selected_is_not_a_ranking(world):
    _, data = world
    assert run(data) == {"needs_selection": True}


def test_country_only_means_that_country_across_all_taxonomy(world):
    _, data = world
    r = run(data, country="IN", window="3d")
    titles = [s["title"] for s in r["stories"]]
    assert titles[0] == "Big new story"                                        # biggest + broadest + newest
    assert "US story" not in titles and "Unlocated story" not in titles        # other country / no location: not IN
    assert "Old story" not in titles                                            # nothing in the window: not a candidate
    domains = {s["taxonomy"]["domain"]["code"] for s in r["stories"] if s["taxonomy"]["domain"]}
    assert len(domains) >= 2                                                    # "across all taxonomy": D01 and D03 stories both present
    assert all(s["location"]["country"] == "IN" for s in r["stories"])


def test_country_view_needs_real_location_evidence(world):
    _, data = world
    titles = lambda r: [s["title"] for s in r["stories"]]
    t = titles(run(data, country="IN", top_n=30))
    assert "Barely-located story" not in t              # 1 located article: not evidence
    assert "Sparsely-located story" not in t            # all 4 located articles are IN, but that is 20 % of the story
    assert "Contested-location story" not in t          # 3 IN vs 3 US: no consensus
    assert "Big new story" in t
    # a taxonomy-only view has no country test, so these stories remain reachable there (and are flagged uncertain)
    allt = run(data, tax={"tier": "domain", "code": "D01", "name": "Conflict"}, top_n=30)
    weak = [s for s in allt["stories"] if s["title"] == "Barely-located story"]
    assert weak and weak[0]["location"]["uncertain"]


def test_taxonomy_only_means_all_countries_including_unlocated(world):
    _, data = world
    r = run(data, tax={"tier": "domain", "code": "D01", "name": "Conflict"}, window="3d")
    titles = [s["title"] for s in r["stories"]]
    assert "Unlocated story" in titles and "Big new story" in titles           # unlocated stories belong to a topic-only view
    assert "US story" not in titles                                              # US story is D02
    assert all(s["taxonomy"]["domain"]["code"] == "D01" for s in r["stories"])
    r2 = run(data, tax={"tier": "domain", "code": "D02", "name": "Politics"}, window="3d")
    assert [s["title"] for s in r2["stories"]][0] == "US story"                 # a foreign country appears when no country is chosen


def test_both_is_the_intersection(world):
    _, data = world
    both = run(data, country="IN", tax={"tier": "domain", "code": "D01", "name": "Conflict"}, window="3d")
    c_only = run(data, country="IN", window="3d", top_n=30)
    t_only = run(data, tax={"tier": "domain", "code": "D01", "name": "Conflict"}, window="3d", top_n=30)
    b = set(ids(both))
    assert b and b <= set(ids(c_only)) and b <= set(ids(t_only))
    assert "Unlocated story" not in [s["title"] for s in both["stories"]]


def test_ranking_prefers_strength_over_recency(world):
    """The newest story (3 articles, 30 min ago) must not outrank a 30-article, 15-publisher story."""
    _, data = world
    r = run(data, country="IN", window="3d", top_n=30)
    pos = {s["title"]: s["rank"] for s in r["stories"]}
    assert pos["Big new story"] < pos["Steady story"] < pos["Small recent story"]
    scores = [s["score"] for s in r["stories"]]
    assert scores == sorted(scores, reverse=True) and [s["rank"] for s in r["stories"]] == list(range(1, len(scores) + 1))


def test_exact_window_numbers_match_the_data(world):
    st, data = world
    r = run(data, country="IN", window="3d", top_n=30)
    s = {x["title"]: x for x in r["stories"]}
    big = s["Big new story"]["in_window"]
    assert (big["articles"], big["publishers"], big["previous_articles"]) == (30, 15, 0)
    steady = s["Steady story"]["in_window"]
    assert (steady["articles"], steady["publishers"], steady["previous_articles"]) == (8, 8, 8)
    assert steady["growth"] == pytest.approx(0.0)                                # 9/9
    sp = s["Split story part A"]["in_window"]
    assert sp["articles"] == 16                                                  # both clusters counted once


def test_reasons_use_real_counts_and_distinguish_new_from_rising(world):
    _, data = world
    s = {x["title"]: x for x in run(data, country="IN", window="3d", top_n=30)["stories"]}
    big = " ".join(s["Big new story"]["reasons"])
    assert "New story: nothing in the previous 3 days" in big and "×" not in big.split("New story")[1].split(".")[0]     # no fake ratio for a new story
    steady = " ".join(s["Steady story"]["reasons"])
    assert "Steady: 8 articles vs 8 in the previous 3 days" in steady
    assert "30 articles from 15 publishers in the last 3 days" in big


def test_growth_is_null_below_support_never_zero(world):
    _, data = world
    s = {x["title"]: x for x in run(data, country="IN", window="3d", top_n=30)["stories"]}
    assert s["Small recent story"]["in_window"]["growth"] is None                # 3 + 0 < 5 articles


def test_baseline_needs_enough_comparables_else_omitted_with_a_note(world, tmp_path):
    st, data = world
    r = run(data, country="IN", window="3d")
    assert r["n_reference"] >= P["trend_min_ref"] and r["stories"][0]["baseline"]["pct_beaten"] >= 0.99
    assert any("typical" in x for x in r["stories"][0]["reasons"])
    (tmp_path / "thin").mkdir()
    thin = build(tmp_path / "thin", with_filler=False)                              # no comparable story-windows at all
    d2 = T.load(thin)
    r2 = run(d2, country="IN", window="3d")
    assert r2["stories"] and r2["stories"][0]["baseline"] is None and any("comparable story-windows" in n for n in r2["notes"])
    assert not any("typical" in x for x in r2["stories"][0]["reasons"])
    thin.close()


def test_evidence_only_where_it_exists_and_missing_is_not_zero(world):
    _, data = world

    def ev(cands, as_of, w):
        out = {}
        for c in cands:
            if c["country"] == "IN":
                out[c["story_id"]] = {"available": True, "items": [{"kind": "emerging", "q": 0.02, "text": "Flagged as an emerging event (FDR q = 0.020)."}]
                                      if "ev_big" in c["event_ids"] else []}
        return out
    r = run(data, tax={"tier": "domain", "code": "D01", "name": "Conflict"}, window="3d", evidence_fn=ev, top_n=30)
    by = {s["title"]: s for s in r["stories"]}
    assert by["Big new story"]["evidence"] == {"available": True, "items": [{"kind": "emerging", "q": 0.02, "text": "Flagged as an emerging event (FDR q = 0.020)."}]}
    assert any("Flagged as an emerging event" in x for x in by["Big new story"]["reasons"])
    assert by["Unlocated story"]["evidence"]["available"] is False
    assert any("no significance claim is made" in x for x in by["Unlocated story"]["reasons"])
    assert any("None of the pipeline's statistical tests" in x for x in by["Steady story"]["reasons"])


def test_near_duplicate_is_folded_under_the_listed_story_not_listed_twice(world):
    _, data = world
    r = run(data, country="IN", window="3d", top_n=30)
    titles = [s["title"] for s in r["stories"]]
    assert "Sibling of big story" not in titles                                  # folded, takes no slot of its own
    big = next(s for s in r["stories"] if s["title"] == "Big new story")
    assert [x["title"] for x in big["related"]] == ["Sibling of big story"]
    assert big["related"][0]["event_ids"] == ["ev_sib"]
    listed = [e for s in r["stories"] for e in s["event_ids"]] + [e for s in r["stories"] for x in s["related"] for e in x["event_ids"]]
    assert len(listed) == len(set(listed))                                       # no cluster appears twice anywhere on the page


def test_window_choice_changes_the_candidates(world):
    _, data = world
    a = run(data, country="IN", window="24h", top_n=30)
    b = run(data, country="IN", window="7d", top_n=30)
    assert a["window"]["hours"] == 24 and b["window"]["hours"] == 168
    assert "Steady story" in [s["title"] for s in b["stories"]]
    assert "Old story" not in [s["title"] for s in a["stories"]]


def test_custom_date_range_is_inclusive_and_compares_with_previous_equal_period(world):
    _, data = world
    day = (AS_OF + dt.timedelta(hours=5, minutes=30)).date()
    r = run(data, country="IN", date_from=day, date_to=day, top_n=30)
    assert r["window"] == {"key": "custom", "hours": 24, "days": 1, "label": f"{day} to {day} (IST)",
                           "date_from": day.isoformat(), "date_to": day.isoformat(), "timezone": "Asia/Kolkata"}
    big = next(s for s in r["stories"] if s["title"] == "Big new story")
    assert big["in_window"]["articles"] == 6  # The custom day starts at midnight IST, not midnight UTC.
    expected_end = pd.Timestamp(day + dt.timedelta(days=1), tz="Asia/Kolkata").tz_convert("UTC").isoformat()
    assert r["as_of"] == expected_end


def test_custom_window_table_includes_start_and_excludes_end():
    start = pd.Timestamp("2026-10-10 00:00", tz="Asia/Kolkata").tz_convert("UTC").timestamp()
    end = start + 24 * 3600
    mem = pd.DataFrame({"story_id": ["s"] * 4, "ts": [start - 24 * 3600, start - 1, start, end - 1],
                        "source_domain": ["a", "b", "c", "d"]})
    tab = T.window_table(mem, end, 24, start)
    counts = dict(zip(tab["k"], tab["n"]))
    assert counts == {0: 2, 1: 2}


def test_deterministic(world):
    _, data = world
    a = run(data, country="IN", window="3d")
    b = run(data, country="IN", window="3d")
    assert [(s["story_id"], s["score"]) for s in a["stories"]] == [(s["story_id"], s["score"]) for s in b["stories"]]


@pytest.mark.parametrize("kw", [dict(window="9d", country="IN"), dict(window="3d", country="IN", top_n=0), dict(window="3d", country="IN", top_n=999)])
def test_bad_arguments_rejected(world, kw):
    _, data = world
    with pytest.raises(T.TrendError):
        run(data, **kw)


# ---------------------------------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------------------------------
@pytest.fixture
def api(tmp_path):
    if not G.COUNTRIES_FILE.exists():
        pytest.skip("boundary data missing")
    st = build(tmp_path)
    st.close()
    return NA.NewsAPI(tmp_path / "w.duckdb"), tmp_path / "w.duckdb"


def test_api_four_cases(api):
    a, _ = api
    assert a.trending()["needs_selection"] is True and "Choose" in a.trending()["message"]
    c = a.trending(country="IN")
    t = a.trending(tax="D02")
    b = a.trending(country="IN", tax="D01")
    assert c["stories"][0]["title"] == "Big new story" and c["scope"]["label"] == "India"
    assert [s["title"] for s in t["stories"]][0] == "US story" and t["scope"]["label"] == "all countries"
    assert b["stories"] and all(s["taxonomy"]["domain"]["code"] == "D01" and s["location"]["country"] == "IN" for s in b["stories"])


@pytest.mark.parametrize("kw", [dict(country="ZZ"), dict(country="IN", tax="D99"), dict(country="IN", tax="D01' OR 1=1"), dict(country="IN", window="1y"),
                                dict(country="IN", limit="abc"), dict(country="IN", limit=500), dict(country="IN", limit=0),
                                dict(country="IN", date_from="2026-10-19"), dict(country="IN", date_from="2026-10-20", date_to="2026-10-19"),
                                dict(country="IN", date_from="2026-10-01", date_to="2026-10-02")])
def test_api_rejects_bad_input(api, kw):
    a, _ = api
    with pytest.raises(NA.ApiError):
        a.trending(**kw)


def test_api_meta_lists_countries_with_stories_and_the_taxonomy(api):
    a, _ = api
    m = a.trending_meta()
    assert m["n_stories"] > 0 and [c["iso2"] for c in m["countries"]][0] == "IN" and {c["iso2"] for c in m["countries"]} == {"IN", "US"}
    assert {t["tier"] for t in m["taxonomy"]} == {"domain", "topic", "category", "subtopic"} and m["windows"][1]["key"] == "3d"
    assert m["date_start"] <= m["date_end"]
    assert m["date_end"] == "2026-10-20"
    assert m["methodology"]["trend_weights"] == {"volume": 0.30, "breadth": 0.30, "growth": 0.20, "recency": 0.10, "evidence": 0.10}
    assert m["methodology"]["trend_min_window_articles"] == 3
    assert "event_metrics" not in m["methodology"] and "publisher_pie" not in m["methodology"]


def test_trending_methodology_is_lightweight_and_only_includes_page_settings(api):
    a, _ = api
    result = a.trending_methodology()
    assert result["methodology"]["trend_weights"]["evidence"] == 0.10
    assert result["methodology"]["emerging_min_ref"] == 50
    assert "trend_min_window_articles" in result["methodology"]
    assert "publisher_pie" not in result["methodology"]


def test_api_custom_date_range_uses_ist_calendar_dates(api):
    a, _ = api
    day = (AS_OF + dt.timedelta(hours=5, minutes=30)).date().isoformat()
    r = a.trending(country="IN", window="custom", date_from=day, date_to=day)
    assert r["window"]["key"] == "custom"
    assert r["window"]["date_from"] == r["window"]["date_to"] == day
    assert r["window"]["timezone"] == "Asia/Kolkata"
    assert r["stories"]


def test_api_story_articles_newest_first_with_links_and_validation(api):
    a, _ = api
    s = a.trending(country="IN")["stories"][0]
    r = a.story_articles(s["event_ids"] + [e for x in s["related"] for e in x["event_ids"]], limit=200)
    assert r["total"] == 30 + 8                                                  # BIG + folded sibling, duplicates excluded
    d = [i["published_at"] for i in r["items"]]
    assert d == sorted(d, reverse=True)
    assert all(i["url"].startswith("https://") and i["publisher"] and i["title"] for i in r["items"])
    page2 = a.story_articles(s["event_ids"] + [e for x in s["related"] for e in x["event_ids"]], offset=30, limit=30)
    assert len(page2["items"]) == 8 and not ({i["url"] for i in page2["items"]} & {i["url"] for i in r["items"][:30]})
    for bad in (["x"], [], ["ev_" + "a" * 40], ["ev_1' OR 1=1"], "ev_a", ["ev_a"] * 101):
        with pytest.raises(NA.ApiError):
            a.story_articles(bad)


def test_api_evidence_comes_from_the_pipelines_stored_tests_where_they_exist(api):
    a, db = api
    with NSt.Store(db) as st:
        st.upsert("aoi_def", pd.DataFrame([{"aoi_id": "country:IN:event", "kind": "country", "key": "IN", "name": "India", "mode": "event", "min_precision": "city",
                                            "adhoc": False, "iso2": "IN", "registered_at": pd.Timestamp.now(tz="UTC")}]))
        st.upsert("event_aoi_test", pd.DataFrame([{"aoi_id": "country:IN:event", "event_id": "ev_big", "kind": "emerging", "as_of": pd.Timestamp(AS_OF) - pd.Timedelta(hours=10),
                                                   "q": 0.012, "alert_flag": True, "support_ok": True},
                                                  {"aoi_id": "country:IN:event", "event_id": "ev_steady", "kind": "emerging", "as_of": pd.Timestamp(AS_OF) - pd.Timedelta(days=9),
                                                   "q": 0.01, "alert_flag": True, "support_ok": True}]))      # flagged long ago: outside the window, must not count
        st.upsert("event_development", pd.DataFrame([{"event_id": "ev_big", "dev_id": "d1", "aoi_id": "country:IN:event", "detected_asof": pd.Timestamp(AS_OF) - pd.Timedelta(hours=5),
                                                      "shift_q": 0.004, "alert_flag": True}]))
        st.upsert("node_aoi_day", pd.DataFrame([{"aoi_id": "country:IN:event", "level": "topic", "node_code": "D01.T01", "day": AS_OF.date(), "V": 40.0, "baseline_mu": 10.0,
                                                 "surprise_q": 0.003, "alert_flag": True}]))
    r = a.trending(country="IN", window="3d", limit=30)
    by = {s["title"]: s for s in r["stories"]}
    kinds = {i["kind"] for i in by["Big new story"]["evidence"]["items"]}
    assert kinds == {"emerging", "development", "topic_surge"}
    assert any("×4.0 above this area's expected level" in t for t in by["Big new story"]["reasons"])
    assert by["Steady story"]["evidence"]["available"] and not any(i["kind"] == "emerging" for i in by["Steady story"]["evidence"]["items"])
    us = a.trending(tax="D02")["stories"][0]
    assert us["evidence"]["available"] is False                                   # no test exists for the US: no claim either way


def test_api_never_writes(api):
    a, db = api
    from lib import news_store as S
    def fingerprint():
        with S.Store(db, read_only=True) as st:
            return {t: st.scalar(f"SELECT count(*) FROM {t}") for t in S.TABLES}, st.scalar("SELECT max(updated_at) FROM event_cluster")
    before = fingerprint()
    a.trending(country="IN"); a.trending(tax="D01"); a.trending_meta(); a.live()
    assert fingerprint() == before


def test_events_with_pruned_centroids_load_and_stay_unmerged(tmp_path):
    """Centroids of clusters older than the retention horizon are NULL (pandas gives pd.NA); loading must not crash on them."""
    w = World(tmp_path / "n.duckdb")
    w.event("ev_a", rand_unit(), [(1 + i, pubs(4)[i % 4]) for i in range(5)], title="Has centroid")
    w.event("ev_b", rand_unit(), [(2 + i, pubs(4, "b")[i % 4]) for i in range(5)], title="Pruned centroid")
    st = w.commit()
    st.con.execute("UPDATE event_cluster SET centroid_sum = NULL WHERE event_id = 'ev_b'")
    data = T.load(st)
    assert set(data.stories["title"]) == {"Has centroid", "Pruned centroid"}
    st.close()
