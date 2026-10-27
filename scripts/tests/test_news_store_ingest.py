"""Derived store + ingest reliability: idempotency, atomic commits, resumability, no silent drops."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from news_testutil import FakeCtxParts, UTC, doc, label
from lib import news_geo as G
from lib import news_ingest as I
from lib import news_sources as NS
from lib import news_store as NSt


@pytest.fixture
def st(tmp_path):
    s = NSt.Store(tmp_path / "t.duckdb")
    yield s
    s.close()


@pytest.fixture(scope="module")
def geo():
    if not G.COUNTRIES_FILE.exists():
        pytest.skip("boundary data missing")
    return G.GeoIndex()


def mk_ctx(st, geo, monkeypatch, labelled=()):
    """ctx whose label lookup returns a label only for article ids in `labelled`."""
    monkeypatch.setattr(NS, "fetch_labels", lambda pg, keys: {k: label() for k in keys if k[1] in labelled})
    return I.Ctx(st, FakeCtxParts.PG(), FakeCtxParts.OS(), geo, I.Scope("t", ["IN"]), "run1",
                 now=dt.datetime(2026, 10, 20, tzinfo=UTC))


# ---- store primitives ---------------------------------------------------------
def test_upsert_is_idempotent_and_rejects_batch_duplicates(st):
    df = pd.DataFrame({"source_table": ["a", "a"], "article_id": [1, 2], "title": ["x", "y"], "is_duplicate": [False] * 2})
    st.upsert("article_fact", df)
    st.upsert("article_fact", df.assign(title=["x2", "y2"]))
    assert st.scalar("SELECT count(*) FROM article_fact") == 2
    assert st.scalar("SELECT title FROM article_fact WHERE article_id = 1") == "x2"
    with pytest.raises(ValueError):
        st.upsert("article_fact", pd.concat([df, df]))
    with pytest.raises(KeyError):
        st.upsert("article_fact", df.assign(bogus=1))


def test_transaction_rolls_back_everything(st):
    df = pd.DataFrame({"source_table": ["a"], "article_id": [1], "is_duplicate": [False]})
    with pytest.raises(RuntimeError):
        with st.tx():
            st.upsert("article_fact", df)
            st.mark_stage("ingest", "s", dt.date(2026, 10, 1), "done", "r")
            raise RuntimeError("crash mid-day")
    assert st.scalar("SELECT count(*) FROM article_fact") == 0
    assert not st.stage_done("ingest", "s", dt.date(2026, 10, 1))      # status and data move together


def test_run_lock_is_exclusive(tmp_path):
    with NSt.run_lock(tmp_path / "l"):
        with pytest.raises(NSt.LockBusy):
            with NSt.run_lock(tmp_path / "l"):
                pass
    with NSt.run_lock(tmp_path / "l"):      # released afterwards
        pass


def test_crashed_run_is_marked_abandoned(st):
    r1 = st.start_run("range", "s", None, None, {})
    r2 = st.start_run("range", "s", None, None, {})
    assert st.scalar("SELECT status FROM run_log WHERE run_id = ?", [r1]) == "abandoned"
    assert st.scalar("SELECT status FROM run_log WHERE run_id = ?", [r2]) == "running"


# ---- ingest -------------------------------------------------------------------
def test_missing_and_conflicting_dates_excluded_never_substituted(st, geo, monkeypatch):
    ctx = mk_ctx(st, geo, monkeypatch, labelled={1, 2, 3, 4})
    docs = [doc(1), doc(2, published=None), doc(3, conflict=True), doc(4, published="2026-12-01T00:00:00+00:00")]
    I.ingest_docs(ctx, docs)
    assert st.df("SELECT article_id FROM article_fact").article_id.tolist() == [1]
    ex = dict(st.df("SELECT article_id, reason FROM excluded_article").values)
    assert ex == {2: "missing_published_at", 3: "date_conflict", 4: "future_published_at"}
    assert ctx.counters["excluded_missing_published_at"] == 1


def test_ingest_is_idempotent_no_duplicate_records(st, geo, monkeypatch):
    ctx = mk_ctx(st, geo, monkeypatch, labelled={1, 2})
    docs = [doc(1), doc(2)]
    I.ingest_docs(ctx, docs)
    r = I.ingest_docs(ctx, docs)
    assert r["new_keys"] == [] and ctx.counters["skipped_already_processed"] == 2
    assert st.scalar("SELECT count(*) FROM article_fact") == 2
    assert st.scalar("SELECT count(*) FROM article_geo") == 2


def test_unlabelled_kept_flagged_and_enriched_later(st, geo, monkeypatch):
    ctx = mk_ctx(st, geo, monkeypatch, labelled={1})
    I.ingest_docs(ctx, [doc(1), doc(2)])
    row = st.df("SELECT article_id, has_label, topic_id FROM article_fact ORDER BY 1")
    assert row.has_label.tolist() == [True, False] and pd.isna(row.topic_id[1])   # NULL, not a made-up label
    assert st.scalar("SELECT count(*) FROM pending_label") == 1
    # the classifier catches up
    monkeypatch.setattr(NS, "fetch_labels", lambda pg, keys: {k: label(topic="D02.T01") for k in keys})
    res = I.resolve_pending(ctx)
    assert res["resolved"] == 1
    assert st.scalar("SELECT topic_id FROM article_fact WHERE article_id = 2") == "D02.T01"
    assert st.scalar("SELECT count(*) FROM pending_label") == 0
    assert st.scalar("SELECT count(*) FROM article_fact") == 2         # enriched in place, not duplicated


def test_content_hash_duplicates_flagged_earliest_wins(st, geo, monkeypatch):
    ctx = mk_ctx(st, geo, monkeypatch, labelled={1, 2, 3})
    I.ingest_docs(ctx, [doc(1, published="2026-10-10T10:00:00+00:00", content_hash="H"),
                        doc(2, published="2026-10-10T09:00:00+00:00", content_hash="H", domain="b.com"),
                        doc(3, content_hash="other")])
    d = st.df("SELECT article_id, is_duplicate, dup_of FROM article_fact ORDER BY 1")
    assert d.is_duplicate.tolist() == [True, False, False]
    assert d.dup_of[0] == "scraped_articles-2"                    # the earlier article is canonical
    # a later batch with the same hash is also recognised across runs
    ctx.counters.clear()
    I.ingest_docs(ctx, [doc(9, published="2026-10-11T10:00:00+00:00", content_hash="H")])
    assert st.scalar("SELECT is_duplicate FROM article_fact WHERE article_id = 9") is True


def test_day_commit_is_atomic_and_failed_day_is_retried(st, geo, monkeypatch):
    ctx = mk_ctx(st, geo, monkeypatch, labelled={1})
    day = dt.date(2026, 10, 10)
    calls = {"n": 0}

    def flaky_scan(c, d):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("OpenSearch hiccup")
        return [doc(1)]
    monkeypatch.setattr(I, "scan_day", flaky_scan)
    with pytest.raises(RuntimeError):
        I.ingest_day(ctx, day)
    assert st.scalar("SELECT status FROM checkpoint WHERE stage='ingest' AND day = ?", [day]) == "failed"
    assert st.scalar("SELECT count(*) FROM article_fact") == 0        # nothing half-written
    I.ingest_day(ctx, day)                                            # resume: retried
    assert st.stage_done("ingest", "t", day) and st.scalar("SELECT count(*) FROM article_fact") == 1
    r = I.ingest_day(ctx, day)                                        # done days are skipped
    assert r.get("skipped") and calls["n"] == 2


def test_scope_filter_uses_fips_for_action_or_mention(geo):
    f = I.Scope("x", ["IN", "BD"]).os_filter(geo)["bool"]
    codes = {c for s in f["should"] for c in list(s["terms"].values())[0]}
    assert codes == {"IN", "BG"} and f["minimum_should_match"] == 1
    assert I.Scope("all").os_filter(geo) is None


def test_day_bucketing_respects_timezone():
    s, e = I.day_bounds(dt.date(2026, 10, 10), "Asia/Kolkata")
    assert s == dt.datetime(2026, 10, 9, 18, 30, tzinfo=UTC) and e - s == dt.timedelta(days=1)
    assert I.to_day(dt.datetime(2026, 10, 9, 19, 0, tzinfo=UTC), "Asia/Kolkata") == dt.date(2026, 10, 10)
