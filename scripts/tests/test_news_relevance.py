"""Relevance gate: nearest-ground-truth scoring, the pass/fail rule, and its use when ranking stories / listing events."""
import numpy as np
import pandas as pd
import pytest

import news_testutil  # noqa: F401
from news_fakes import FakeOS, near, topic_vector
from lib import news_config as C
from lib import news_relevance as NR
from lib import news_store as NSt
from lib import news_trending as T

P = dict(C.PARAMS)
DIM = C.EMBEDDING_DIM


def make_ref(per=60, domains=("D01", "D05", "D09"), seed=0):
    rng = np.random.default_rng(seed)
    vecs, doms = [], []
    for k, d in enumerate(domains):
        base = topic_vector(100 + k)
        for _ in range(per):
            vecs.append(near(base, rng, noise=0.2))
            doms.append(d)
    return NR.Reference(np.asarray(vecs, dtype=np.float32), np.asarray(doms))


def test_scoring_separates_on_topic_from_off_topic_and_votes_name_the_domain():
    ref = make_ref()
    on = topic_vector(101)                      # the D05 cluster's centre
    off = topic_vector(999)                     # unrelated direction
    sim, votes = NR.score_vectors(ref, np.stack([on, off]))
    assert sim[0] > 0.65 > sim[1]
    assert NR.parse_votes(votes[0]) == {"D05": 10}
    assert sum(NR.parse_votes(votes[1]).values()) == 10          # always 10 neighbours, however far


def test_scoring_is_independent_of_block_size():
    ref = make_ref()
    U = np.stack([topic_vector(i) for i in range(100, 160)])
    a, va = NR.score_vectors(ref, U, block=7)
    b, vb = NR.score_vectors(ref, U, block=256)
    assert np.allclose(a, b, atol=1e-6) and va == vb


@pytest.mark.parametrize("sim,votes,domain,expected", [
    (0.80, "D05:9,D01:1", "D05", True),
    (0.80, "D05:9,D01:1", "D01", False),      # close to the taxonomy, but its neighbours are Military, not Conflict
    (0.80, "D05:9,D01:1", None, True),        # no domain selected: similarity only
    (0.60, "D05:10", "D05", False),           # far from everything the taxonomy covers
    (0.65, "D05:3,D01:7", "D05", True),       # boundary values pass (>=)
    (0.65, "D05:2,D01:8", "D05", False),
    (None, None, "D05", True),                # unscored: never treated as irrelevant
    (float("nan"), None, "D05", True),
])
def test_pass_rule(sim, votes, domain, expected):
    assert NR.passes(sim, votes, domain, P) is expected


def test_score_events_writes_scores_only_for_events_with_a_centroid(tmp_path):
    ref = make_ref()
    st = NSt.Store(tmp_path / "s.duckdb")
    c = lambda v: (np.asarray(v, dtype=np.float32) * 5).tobytes()
    t0 = pd.Timestamp("2026-10-20", tz="UTC")
    st.upsert("event_cluster", pd.DataFrame([
        {"event_id": "ev_on", "is_event": True, "centroid_sum": c(topic_vector(101)), "first_seen": t0, "last_seen": t0},
        {"event_id": "ev_off", "is_event": True, "centroid_sum": c(topic_vector(999)), "first_seen": t0, "last_seen": t0},
        {"event_id": "ev_pruned", "is_event": True, "centroid_sum": None, "first_seen": t0, "last_seen": t0},
        {"event_id": "ev_single", "is_event": False, "centroid_sum": c(topic_vector(101)), "first_seen": t0, "last_seen": t0}]))
    assert NR.score_events(st, ref) == 2
    r = st.df("SELECT event_id, ref_sim, ref_votes FROM event_cluster").set_index("event_id")
    assert r.loc["ev_on", "ref_sim"] > 0.65 > r.loc["ev_off", "ref_sim"]
    assert pd.isna(r.loc["ev_pruned", "ref_sim"]) and pd.isna(r.loc["ev_single", "ref_sim"])
    st.con.execute("UPDATE event_cluster SET ref_sim = NULL WHERE event_id = 'ev_off'")
    assert NR.score_events(st, ref, only_missing=True) == 1               # only the one that lacks a score
    st.close()


def test_reference_roundtrip_and_build_from_fake_sources(tmp_path):
    class PG:
        def query(self, sql, params=None):
            assert sql.lstrip().upper().startswith("SELECT") and "content" not in sql.lower()      # ids + domain only
            return [{"source_table": "raw.scraped_articles", "source_id": str(i), "gt_domain_id": "D05" if i % 2 else "D01"} for i in range(1, 41)]

    fos = FakeOS()
    import datetime as dt
    rng = np.random.default_rng(1)
    for i in range(1, 38):                                                    # 3 ids missing in the index: skipped, not fatal
        fos.add(i, published=dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc), ingested=dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc),
                vec=near(topic_vector(i % 2), rng))
    path = tmp_path / "ref.npz"
    n = NR.build_reference(PG(), fos, path, page=10, log=lambda *_: None)
    assert n == 37
    ref = NR.load_reference(path)
    assert ref.n == 37 and ref.R.shape == (37, DIM) and set(ref.dom) == {0 + 1, 0 + 5}
    assert NR.load_reference(tmp_path / "missing.npz") is None


# ---- ranking ----------------------------------------------------------------------------------------------------------
from test_news_trending import World, build, run, pubs, rand_unit   # noqa: E402


@pytest.fixture
def scored_world(tmp_path):
    st = build(tmp_path)
    return st


def set_scores(st, event_id, sim, votes):
    st.con.execute("UPDATE event_cluster SET ref_sim = ?, ref_votes = ? WHERE event_id = ?", [sim, votes, event_id])


def titles(res):
    return [s["title"] for s in res["stories"]]


def test_off_topic_story_is_not_listed_and_the_page_says_why(scored_world):
    st = scored_world
    set_scores(st, "ev_big", 0.90, "D01:10")
    set_scores(st, "ev_steady", 0.55, "D01:4,D00:6")        # far from the taxonomy: a film trailer, a hospital admission ...
    data = T.load(st)
    res = run(data, country="IN", top_n=30)
    assert "Big new story" in titles(res) and "Steady story" not in titles(res)
    assert res["n_hidden_offtopic"] >= 1 and any("not about the subject" in n for n in res["notes"])
    assert res["n_candidates"] == len(res["stories"]) or res["n_candidates"] >= len(res["stories"])


def test_unscored_stories_are_never_filtered(scored_world):
    data = T.load(scored_world)                              # nothing scored (no reference): behaves exactly as before
    res = run(data, country="IN", top_n=30)
    assert "Big new story" in titles(res) and res.get("n_hidden_offtopic", 0) == 0


def test_domain_selection_needs_the_neighbours_to_agree_with_the_domain(scored_world):
    st = scored_world
    set_scores(st, "ev_big", 0.90, "D02:9,D01:1")            # about politics, but labelled D01 (Protest) by the classifier
    data = T.load(st)
    d01 = run(data, tax={"tier": "domain", "code": "D01", "name": "Conflict"}, top_n=30)
    assert "Big new story" not in titles(d01)                # wrong domain for this view ...
    assert "Big new story" in titles(run(data, country="IN", top_n=30))   # ... but a fine story when no domain is selected
    set_scores(st, "ev_big", 0.90, "D01:6,D02:4")
    assert "Big new story" in titles(run(T.load(st), tax={"tier": "domain", "code": "D01", "name": "Conflict"}, top_n=30))


def test_story_relevance_combines_its_events_weighted_by_size(tmp_path):
    w = World(tmp_path / "m.duckdb")
    base = rand_unit()
    w.event("ev_a", base, [(1 + i, pubs(5)[i % 5]) for i in range(10)], title="Split story A")
    from test_news_trending import at_cos
    w.event("ev_b", at_cos(base, 0.99), [(2 + i, pubs(4, "b")[i % 4]) for i in range(5)], title="Split story B")
    st = w.commit()
    set_scores(st, "ev_a", 0.80, "D01:10")                  # 10 articles
    set_scores(st, "ev_b", 0.50, "D00:10")                  # 5 articles
    s = T.load(st).stories
    assert len(s) == 1
    assert s["ref_sim"].iloc[0] == pytest.approx((0.80 * 10 + 0.50 * 5) / 15)
    assert s["ref_votes"].iloc[0] == {"D01": 10, "D00": 10}
    st.close()
