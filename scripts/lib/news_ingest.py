"""Stage: ingest articles into the derived store (metadata only, no vectors).

The SAME `ingest_docs()` is used by the pilot, a historical backfill and incremental updates;
only *discovery* differs (by published day range vs by OpenSearch ingest watermark).

Rules (each is counted in `data_quality`, nothing is silently dropped)
  * no usable `published_at`        -> excluded_article('missing_published_at' | ...)
                                       `scraped_at` is NEVER substituted
  * conflicting date flag           -> excluded ('date_conflict') when configured
  * not yet labelled in Postgres    -> pending_label (retried every run), not excluded
  * same content_hash seen before   -> kept, flagged is_duplicate, excluded from counts/clusters
  * already in article_fact         -> skipped (idempotent)
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from . import news_config as C
from . import news_geo as G
from . import news_sources as NS
from .news_store import Store, now_utc


# --------------------------------------------------------------------------
# scope + helpers
# --------------------------------------------------------------------------
@dataclass
class Scope:
    """Which articles a run processes. `countries` are ISO alpha-2; None = everything."""
    name: str
    countries: Optional[list[str]] = None

    def os_filter(self, geo: G.GeoIndex) -> Optional[dict]:
        if not self.countries:
            return None
        fips = [geo.iso_to_fips(c) for c in self.countries]
        if any(f is None for f in fips):
            raise ValueError(f"no FIPS code for one of {self.countries}")
        return {"bool": {"should": [{"terms": {"gdelt.action_country": fips}},
                                    {"terms": {"gkg.locations_country_code": fips}}],
                         "minimum_should_match": 1}}


def day_bounds(day: dt.date, tz: str = "UTC") -> tuple[dt.datetime, dt.datetime]:
    z = ZoneInfo(tz)
    s = dt.datetime(day.year, day.month, day.day, tzinfo=z)
    e = s + dt.timedelta(days=1)
    return s.astimezone(dt.timezone.utc), e.astimezone(dt.timezone.utc)


def to_day(ts: dt.datetime, tz: str = "UTC") -> dt.date:
    return ts.astimezone(ZoneInfo(tz)).date()


def iso(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Ctx:
    store: Store
    pg: NS.ReadOnlyPG
    osc: NS.ReadOnlyOpenSearch
    geo: G.GeoIndex
    scope: Scope
    run_id: str
    P: dict = field(default_factory=lambda: C.PARAMS)
    now: dt.datetime = field(default_factory=now_utc)
    counters: dict = field(default_factory=dict)
    # what THIS run changed - the live service publishes exactly these rows to the derived PostgreSQL schema
    new_keys: set = field(default_factory=set)         # (source_table, article_id) first ingested by this run
    touched: set = field(default_factory=set)          # event ids that gained articles in this run
    prune_floor: Optional[dt.datetime] = None          # centroids newer than this are kept (days still waiting to be processed)
    recomputed: set = field(default_factory=set)       # AOI ids whose metrics were recomputed in this run

    def bump(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n


# --------------------------------------------------------------------------
# taxonomy
# --------------------------------------------------------------------------
def load_taxonomy(ctx: Ctx) -> dict[str, dict]:
    rows = NS.fetch_taxonomy(ctx.pg)
    df = pd.DataFrame(rows)
    df["loaded_at"] = now_utc()
    ctx.store.upsert("taxonomy_node", df[["code", "tier", "name", "parent", "sort_order", "is_catch_all",
                                          "in_scope", "taxonomy_version", "loaded_at"]])
    return {r["code"]: r for r in rows}


# --------------------------------------------------------------------------
# core: docs -> facts
# --------------------------------------------------------------------------
def _existing_keys(store: Store, keys: list[tuple[str, int]]) -> set[tuple[str, int]]:
    if not keys:
        return set()
    k = pd.DataFrame(keys, columns=["source_table", "article_id"])
    store.con.register("_k", k)
    try:
        rows = store.con.execute("SELECT f.source_table, f.article_id FROM article_fact f "
                                 "JOIN _k USING (source_table, article_id)").fetchall()
    finally:
        store.con.unregister("_k")
    return {(a, int(b)) for a, b in rows}


def _existing_hashes(store: Store, hashes: list[str]) -> dict[str, str]:
    if not hashes:
        return {}
    h = pd.DataFrame({"content_hash": sorted(set(hashes))})
    store.con.register("_h", h)
    try:
        rows = store.con.execute(
            "SELECT f.content_hash, (array_agg(f.source_table || '-' || CAST(f.article_id AS VARCHAR) ORDER BY f.published_at))[1] "
            "FROM article_fact f JOIN _h USING (content_hash) WHERE NOT f.is_duplicate GROUP BY 1").fetchall()
    finally:
        store.con.unregister("_h")
    return dict(rows)


def ingest_docs(ctx: Ctx, docs: list[dict]) -> dict:
    """Normalised OS docs -> article_fact / article_geo / excluded_article / pending_label.

    Returns {'new_keys': [...], 'new_days': {...}} describing what changed (for downstream
    stages that only need to touch affected data). Must be called inside a transaction by the
    caller if atomic day commits are wanted (ingest_day does).
    """
    P, st = ctx.P, ctx.store
    out = {"new_keys": [], "affected_days": set()}
    if not docs:
        return out
    ctx.bump("docs_seen", len(docs))

    # 1. key sanity + already-processed (idempotent skip)
    good = []
    for d in docs:
        if d["article_id"] is None or d["source_table"] not in C.SOURCE_TABLES:
            ctx.bump("excluded_bad_key")
            continue
        good.append(d)
    have = _existing_keys(st, [(d["source_table"], d["article_id"]) for d in good])
    ctx.bump("skipped_already_processed", len(have))
    todo = [d for d in good if (d["source_table"], d["article_id"]) not in have]

    # 2. date reliability (never falls back to scrape time)
    dated, excl = [], []
    for d in todo:
        ok, reason = NS.validate_published_at(d, ctx.now, P)
        if ok:
            dated.append(d)
        else:
            ctx.bump(f"excluded_{reason}")
            excl.append({"source_table": d["source_table"], "article_id": d["article_id"], "reason": reason,
                         "published_at": d["published_at"], "run_id": ctx.run_id, "seen_at": ctx.now})
    st.upsert("excluded_article", pd.DataFrame(excl))
    if excl:  # a doc that is now excluded must not linger as pending
        e = pd.DataFrame(excl)[["source_table", "article_id"]]
        st.con.register("_e", e)
        st.con.execute("DELETE FROM pending_label WHERE (source_table, article_id) IN "
                       "(SELECT source_table, article_id FROM _e)")
        st.con.unregister("_e")

    # 3. labels (Postgres, read-only). Unlabelled articles are STILL ingested (events do not
    #    need a taxonomy label) with has_label=false, and tracked in pending_label so the label
    #    is attached when it appears. They are excluded from taxonomy-node series.
    keys = [(d["source_table"], d["article_id"]) for d in dated]
    labels = NS.fetch_labels(ctx.pg, keys)
    labelled = [(d, labels.get((d["source_table"], d["article_id"]))) for d in dated]
    pend = [{"source_table": d["source_table"], "article_id": d["article_id"], "first_seen_at": ctx.now,
             "last_checked_at": ctx.now, "attempts": 1, "published_at": d["published_at"]}
            for d, lab in labelled if lab is None]
    ctx.bump("unlabelled", len(pend))
    if pend:
        st.upsert("pending_label", pd.DataFrame(pend))

    # 4. dedupe by content_hash (earliest published wins; stable tiebreak on key)
    labelled.sort(key=lambda x: (x[0]["published_at"], x[0]["source_table"], x[0]["article_id"]))
    seen = _existing_hashes(st, [d["content_hash"] for d, _ in labelled if d.get("content_hash")])
    fact_rows, geo_rows = [], []
    for d, lab in labelled:
        key_s = f"{d['source_table']}-{d['article_id']}"
        lab = lab or {}
        h = d.get("content_hash") or lab.get("content_hash")
        dup_of = None
        if P["dedupe_by_content_hash"] and h:
            if h in seen:
                dup_of = seen[h]
            else:
                seen[h] = key_s
        if d.get("dup_indices"):
            ctx.bump("doc_in_two_indices")
        pa = d["published_at"]
        if lab.get("published_at") is not None and abs((lab["published_at"] - pa).total_seconds()) > 3600:
            ctx.bump("date_mismatch_pg_vs_os")
        lag = None
        if d.get("ingested_at") is not None:
            lag = (d["ingested_at"] - pa).total_seconds() / 3600.0
        fact_rows.append({
            "source_table": d["source_table"], "article_id": d["article_id"], "os_id": d["os_id"],
            "os_index": d["os_index"], "published_at": pa, "day": to_day(pa, P["day_tz"]),
            "ingested_at": d.get("ingested_at"), "lag_hours": lag,
            "date_confidence": d.get("date_confidence"), "date_source": d.get("date_source"),
            "date_precision": d.get("date_precision"),
            "has_label": bool(lab), "domain_id": lab.get("domain_id"), "topic_id": lab.get("topic_id"),
            "category_id": lab.get("category_id"), "subtopic_id": lab.get("subtopic_id"),
            "category_conf": lab.get("category_confidence"), "subtopic_conf": lab.get("subtopic_confidence"),
            "taxonomy_version": lab.get("taxonomy_version"),
            "content_hash": h, "is_duplicate": dup_of is not None, "dup_of": dup_of,
            "source_domain": d.get("source_domain"), "url": d.get("url") or lab.get("url"),
            "title": d.get("title"), "gdelt_event_id": d.get("gdelt_event_id"),
            "cluster_id": None, "cluster_sim": None, "cluster_note": ("duplicate" if dup_of else None), "run_id": ctx.run_id, "processed_at": ctx.now})
        loc = ctx.geo.article_location(d)
        geo_rows.append({"source_table": d["source_table"], "article_id": d["article_id"], **loc})
        if dup_of:
            ctx.bump("duplicates_flagged")
        ctx.bump("located" if loc["loc_status"] == "located" else "unlocated")
        if loc["geo_inconsistent"]:
            ctx.bump("geo_inconsistent")
        if loc["country_code_unmapped"]:
            ctx.bump("country_code_unmapped")
    if fact_rows:
        st.upsert("article_fact", pd.DataFrame(fact_rows))
        gdf = pd.DataFrame(geo_rows)
        gdf["mention_countries_iso"] = gdf["mention_countries_iso"].map(lambda l: list(l))
        st.upsert("article_geo", gdf[[c for c in st.columns("article_geo")]])
        ctx.bump("facts_written", len(fact_rows))
        fk = pd.DataFrame([(r["source_table"], r["article_id"]) for r in fact_rows],
                          columns=["source_table", "article_id"])
        st.con.register("_f", fk)   # an article that is now usable must not stay listed as excluded
        st.con.execute("DELETE FROM excluded_article WHERE (source_table, article_id) IN "
                       "(SELECT source_table, article_id FROM _f)")
        st.con.unregister("_f")
        out["new_keys"] = [(r["source_table"], r["article_id"]) for r in fact_rows]
        out["affected_days"] = {r["day"] for r in fact_rows}
    return out


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def _hits_to_docs(hits: list[dict]) -> list[dict]:
    return [NS.normalise_hit(h) for h in NS.dedupe_hits(hits).values()]


def scan_day(ctx: Ctx, day: dt.date) -> list[dict]:
    """All in-scope OpenSearch docs whose published_at falls on `day` (metadata only)."""
    s, e = day_bounds(day, ctx.P["day_tz"])
    filt = [{"range": {"published_at": {"gte": iso(s), "lt": iso(e)}}}]
    sf = ctx.scope.os_filter(ctx.geo)
    q = {"bool": {"filter": filt + ([sf] if sf else [])}}
    hits = []
    for page in ctx.osc.scan(q):
        hits.extend(page)
    docs = _hits_to_docs(hits)
    ctx.bump("os_hits_raw", len(hits))
    return docs


def ingest_day(ctx: Ctx, day: dt.date, force: bool = False) -> dict:
    """Discover + ingest one published day, atomically with its checkpoint."""
    st = ctx.store
    if not force and st.stage_done("ingest", ctx.scope.name, day):
        ctx.bump("days_skipped_done")
        return {"new_keys": [], "affected_days": set(), "skipped": True}
    try:
        docs = scan_day(ctx, day)
        with st.tx():
            res = ingest_docs(ctx, docs)
            st.mark_stage("ingest", ctx.scope.name, day, "done", ctx.run_id, len(docs), len(res["new_keys"]))
        ctx.new_keys.update(map(tuple, res["new_keys"]))
        return res
    except Exception as ex:  # record and re-raise: the day stays not-done and is retried
        try:
            st.mark_stage("ingest", ctx.scope.name, day, "failed", ctx.run_id, error=str(ex)[:500])
        except Exception:
            pass
        raise


def discover_incremental(ctx: Ctx, since: dt.datetime, until: Optional[dt.datetime] = None) -> list[dict]:
    """In-scope docs whose OpenSearch `ingested_at` >= since (catches newly ingested articles,
    whatever their published date). Late-dated docs are handled by the metrics recompute."""
    flt = [{"range": {"ingested_at": {"gte": iso(since)}}},
           {"exists": {"field": "published_at"}}]
    if until:
        flt[0]["range"]["ingested_at"]["lt"] = iso(until)
    sf = ctx.scope.os_filter(ctx.geo)
    q = {"bool": {"filter": flt + ([sf] if sf else [])}}
    hits = []
    for page in ctx.osc.scan(q):
        hits.extend(page)
    ctx.bump("os_hits_raw", len(hits))
    return _hits_to_docs(hits)


def resolve_pending(ctx: Ctx, max_rows: int = 50000) -> dict:
    """Attach labels to articles that were ingested before their label existed.

    Only the taxonomy columns of the existing fact row change; the article keeps its cluster.
    Returns the affected days so taxonomy-node series for those days are recomputed.
    """
    st = ctx.store
    pend = st.df("SELECT source_table, article_id FROM pending_label ORDER BY first_seen_at LIMIT ?", [max_rows])
    out = {"new_keys": [], "affected_days": set(), "resolved": 0}
    if not len(pend):
        return out
    keys = [(r.source_table, int(r.article_id)) for r in pend.itertuples()]
    labels = NS.fetch_labels(ctx.pg, keys)
    rows = [{"source_table": k[0], "article_id": k[1], "has_label": True,
             "domain_id": v.get("domain_id"), "topic_id": v.get("topic_id"), "category_id": v.get("category_id"),
             "subtopic_id": v.get("subtopic_id"), "category_conf": v.get("category_confidence"),
             "subtopic_conf": v.get("subtopic_confidence"), "taxonomy_version": v.get("taxonomy_version")}
            for k, v in labels.items()]
    st.con.execute("UPDATE pending_label SET last_checked_at = ?, attempts = attempts + 1", [ctx.now])
    if not rows:
        return out
    df = pd.DataFrame(rows)
    with st.tx():
        st.con.register("_lab", df)
        st.con.execute("""UPDATE article_fact SET has_label = l.has_label, domain_id = l.domain_id,
            topic_id = l.topic_id, category_id = l.category_id, subtopic_id = l.subtopic_id,
            category_conf = l.category_conf, subtopic_conf = l.subtopic_conf,
            taxonomy_version = l.taxonomy_version
            FROM _lab l WHERE article_fact.source_table = l.source_table AND article_fact.article_id = l.article_id""")
        days = st.df("SELECT DISTINCT f.day FROM article_fact f JOIN _lab l USING (source_table, article_id)")
        st.con.execute("DELETE FROM pending_label WHERE (source_table, article_id) IN "
                       "(SELECT source_table, article_id FROM _lab)")
        st.con.unregister("_lab")
    out["affected_days"] = set(days["day"].tolist())
    out["resolved"] = len(rows)
    ctx.bump("pending_resolved", len(rows))
    return out


# --------------------------------------------------------------------------
# corpus-level coverage (context for "what share is located / dated")
# --------------------------------------------------------------------------
def measure_corpus_day(ctx: Ctx, day: dt.date) -> dict:
    """Whole-corpus per-day counts via OpenSearch _count (no documents pulled)."""
    s, e = day_bounds(day, ctx.P["day_tz"])
    rng = {"range": {"published_at": {"gte": iso(s), "lt": iso(e)}}}

    def cnt(extra=None):
        return ctx.osc.count({"bool": {"filter": [rng] + ([extra] if extra else [])}})
    row = {"day": day, "n_dated_total": cnt(),
           "n_action_country": cnt({"exists": {"field": "gdelt.action_country"}}),
           "n_action_geo": cnt({"exists": {"field": "gdelt.action_geo"}}),
           "n_mention_country": cnt({"exists": {"field": "gkg.locations_country_code"}}),
           "source": "opensearch_count", "measured_at": now_utc()}
    ctx.store.upsert("corpus_day", pd.DataFrame([row]))
    return row
