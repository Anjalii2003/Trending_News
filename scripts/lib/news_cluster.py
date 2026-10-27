"""Event clustering: group articles that report the same event.

Method (deliberately simple, incremental, and the same for pilot / backfill / incremental):
  * articles are processed in publication order
  * an article joins the most similar OPEN cluster if cosine(article, cluster centroid) >=
    `sim_threshold` AND its time is within `max_gap_hours` of the cluster's span; otherwise it
    founds a new cluster
  * the taxonomy is NOT a hard partition: subtopic confidence is low (median 0.38), so a hard
    partition would split one event across labels. Taxonomy is attached to the cluster
    afterwards (modal label of its members).
  * exact duplicates (same content_hash) are not clustered; they are counted as `n_copies`
  * a cluster is an *event* only with >= event_min_articles articles from >= event_min_publishers
    distinct publishers; everything else stays a candidate (mostly singletons)

Evidence base: Trending_news/outputs (agglomerative / hybrid / HDBSCAN comparison on the same
bge-m3 vectors) found a 0.80 cosine + 24-72 h gate competitive, with most articles remaining
singletons. Cluster quality on real data is reported by scripts/23_news_pilot_report.py and
needs human review - silver-label F1 is weak evidence.

Event ids are deterministic (hash of the founding article), so reruns reproduce the same ids.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Optional

import numpy as np
import pandas as pd

from . import news_config as C
from .news_store import Store, now_utc

DIM = C.EMBEDDING_DIM


def _cluster_compute_backend(device: Optional[str] = None) -> tuple[str, Optional[ModuleType]]:
    """Choose CPU or optional CUDA/Torch backend for the batched centroid dot product."""
    setting = os.environ.get("NEWS_CLUSTER_DEVICE", "auto") if device is None else device
    requested = setting.strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("NEWS_CLUSTER_DEVICE must be 'auto', 'cpu', or 'cuda'")
    if requested == "cpu":
        return "cpu", None
    try:
        import torch
    except ImportError as exc:
        if exc.name != "torch":
            raise
        if requested == "cuda":
            raise RuntimeError("CUDA clustering requested but PyTorch is not installed; build with NEWS_GPU=1") from exc
        return "cpu", None
    if torch.cuda.is_available():
        return "cuda", torch
    if requested == "cuda":
        raise RuntimeError("CUDA clustering requested but CUDA is not available to this process")
    return "cpu", None


def event_id_for(source_table: str, article_id: int) -> str:
    return "ev_" + hashlib.sha1(f"{source_table}-{article_id}".encode()).hexdigest()[:16]


def valid_vector(v) -> tuple[Optional[np.ndarray], Optional[str]]:
    """-> (unit float32 vector, None) or (None, reason)."""
    if v is None:
        return None, "missing_embedding"
    a = np.asarray(v, dtype=np.float32)
    if a.shape != (DIM,):
        return None, "invalid_vector"
    if not np.isfinite(a).all():
        return None, "invalid_vector"
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return None, "invalid_vector"
    return a / n, None


@dataclass
class _Cl:
    event_id: str
    first_ts: float
    last_ts: float
    n: int


class OnlineClusterer:
    """Centroid-based online clusterer over unit vectors. Pure numpy; no I/O."""

    def __init__(
        self,
        sim_threshold: float,
        max_gap_hours: float,
        capacity: int = 4096,
        device: Optional[str] = None,
    ):
        self.thr = float(sim_threshold)
        self.gap = float(max_gap_hours) * 3600.0
        self.device, self._torch = _cluster_compute_backend(device)
        self._sum = np.zeros((capacity, DIM), dtype=np.float32)   # sum of member unit vectors
        self._unit = np.zeros((capacity, DIM), dtype=np.float32)  # normalised centroid
        self._first = np.zeros(capacity, dtype=np.float64)
        self._last = np.zeros(capacity, dtype=np.float64)
        self._n = np.zeros(capacity, dtype=np.int32)
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        self.touched: set[str] = set()

    # ----- state ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    def _grow(self) -> None:
        cap = self._sum.shape[0] * 2
        for name in ("_sum", "_unit"):
            old = getattr(self, name)
            new = np.zeros((cap, DIM), dtype=np.float32)
            new[:old.shape[0]] = old
            setattr(self, name, new)
        for name in ("_first", "_last", "_n"):
            old = getattr(self, name)
            new = np.zeros(cap, dtype=old.dtype)
            new[:old.shape[0]] = old
            setattr(self, name, new)

    def add_existing(self, event_id: str, centroid_sum: np.ndarray, first_ts: float, last_ts: float, n: int) -> None:
        """Load a persisted open cluster."""
        if event_id in self.index:
            return
        if len(self.ids) >= self._sum.shape[0]:
            self._grow()
        i = len(self.ids)
        self.ids.append(event_id)
        self.index[event_id] = i
        self._sum[i] = centroid_sum
        nrm = np.linalg.norm(centroid_sum)
        self._unit[i] = centroid_sum / nrm if nrm > 0 else centroid_sum
        self._first[i], self._last[i], self._n[i] = first_ts, last_ts, n

    def centroid_sum(self, event_id: str) -> np.ndarray:
        return self._sum[self.index[event_id]].copy()

    def span(self, event_id: str) -> tuple[float, float, int]:
        i = self.index[event_id]
        return float(self._first[i]), float(self._last[i]), int(self._n[i])

    # ----- assignment -------------------------------------------------------
    def assign(self, vec: np.ndarray, ts: float, founder_event_id: str) -> tuple[str, float, bool]:
        """Join the best matching cluster or found `founder_event_id`.
        Returns (event_id, similarity_to_centroid_before_joining, created)."""
        m = len(self.ids)
        best_i, best_s = -1, -1.0
        if m:
            cand = np.flatnonzero((self._last[:m] >= ts - self.gap) & (self._first[:m] <= ts + self.gap))
            if cand.size:
                sims = self._unit[cand] @ vec
                j = int(np.argmax(sims))
                best_i, best_s = int(cand[j]), float(sims[j])
        if best_i >= 0 and best_s >= self.thr:
            self._join(best_i, vec, ts)
            self.touched.add(self.ids[best_i])
            return self.ids[best_i], best_s, False
        if founder_event_id in self.index:   # rerun safety: founder already exists
            i = self.index[founder_event_id]
            self._join(i, vec, ts)
            self.touched.add(founder_event_id)
            return founder_event_id, 1.0, False
        if m >= self._sum.shape[0]:
            self._grow()
        i = m
        self.ids.append(founder_event_id)
        self.index[founder_event_id] = i
        self._sum[i] = vec
        self._unit[i] = vec
        self._first[i] = self._last[i] = ts
        self._n[i] = 1
        self.touched.add(founder_event_id)
        return founder_event_id, 1.0, True

    def _join(self, i: int, vec: np.ndarray, ts: float) -> None:
        self._sum[i] += vec
        nrm = float(np.linalg.norm(self._sum[i]))
        self._unit[i] = self._sum[i] / nrm if nrm > 0 else self._sum[i]
        self._first[i] = min(self._first[i], ts)
        self._last[i] = max(self._last[i], ts)
        self._n[i] += 1

    def assign_many(self, vecs: np.ndarray, tss: list, founders: list) -> list:
        """Exactly `[self.assign(v, t, f) for ...]`, computed block-wise.

        Sequential assignment is a matrix-vector product against every open cluster, per article, and copies the candidate
        centroids each time: at all-country volume (~85 k open clusters, ~60 k articles/day) that is what makes a day take
        half an hour. Here the similarities of the whole block to the clusters that exist at its start are ONE matrix product;
        only clusters changed or created INSIDE the block (at most len(vecs)) are re-scored per article against their current
        centroids, so every article sees exactly the centroids it would have seen sequentially. Same candidates (time gate), same
        best cluster, same lowest-index tie-break, same threshold; results are identical to `assign` up to float rounding of
        the dot products (tests compare them on randomised data)."""
        n = len(founders)
        if n == 0:
            return []
        m0 = len(self.ids)
        K = 0
        if m0:
            keep = np.flatnonzero(self._last[:m0] >= min(tss) - self.gap)
            K = keep.size
        if K:
            U0 = self._unit[keep]
            if self.device == "cuda":
                with self._torch.inference_mode():
                    vectors_gpu = self._torch.as_tensor(np.ascontiguousarray(vecs), device="cuda")
                    centroids_gpu = self._torch.as_tensor(np.ascontiguousarray(U0), device="cuda")
                    S = self._torch.matmul(vectors_gpu, centroids_gpu.T).cpu().numpy()
            else:
                S = np.asarray(vecs, dtype=np.float32) @ U0.T                 # (n, K): ONE matrix product for the whole block
            last0, first0 = self._last[keep].copy(), self._first[keep].copy()
            pos = np.full(m0, -1, dtype=np.int64)
            pos[keep] = np.arange(K)
            dirty_base = np.zeros(K, dtype=bool)
        dirty: set[int] = set()                                              # indices (in self) changed/created in this block
        out = []
        for j in range(n):
            vec, ts = vecs[j], tss[j]
            best_i, best_s = -1, -1.0
            if K:
                ok = (last0 >= ts - self.gap) & (first0 <= ts + self.gap) & ~dirty_base
                if ok.any():
                    row = np.where(ok, S[j], -np.inf)
                    c = int(np.argmax(row))
                    best_i, best_s = int(keep[c]), float(row[c])
            if dirty:
                d = np.asarray(sorted(dirty))
                dm = (self._last[d] >= ts - self.gap) & (self._first[d] <= ts + self.gap)
                if dm.any():
                    dc = d[dm]
                    sims = self._unit[dc] @ vec
                    k = int(np.argmax(sims))
                    ds, di = float(sims[k]), int(dc[k])
                    if best_i < 0 or ds > best_s or (ds == best_s and di < best_i):
                        best_i, best_s = di, ds
            if best_i >= 0 and best_s >= self.thr:
                self._join(best_i, vec, ts)
                self.touched.add(self.ids[best_i])
                dirty.add(best_i)
                if K and best_i < m0 and pos[best_i] >= 0:
                    dirty_base[pos[best_i]] = True
                out.append((self.ids[best_i], best_s, False))
                continue
            f = founders[j]
            if f in self.index:                                              # rerun safety: founder already exists
                i = self.index[f]
                self._join(i, vec, ts)
                self.touched.add(f)
                dirty.add(i)
                if K and i < m0 and pos[i] >= 0:
                    dirty_base[pos[i]] = True
                out.append((f, 1.0, False))
                continue
            if len(self.ids) >= self._sum.shape[0]:
                self._grow()
            i = len(self.ids)
            self.ids.append(f)
            self.index[f] = i
            self._sum[i] = vec
            self._unit[i] = vec
            self._first[i] = self._last[i] = ts
            self._n[i] = 1
            self.touched.add(f)
            dirty.add(i)                                                     # created in this block: always scored fresh
            out.append((f, 1.0, True))
        return out


# --------------------------------------------------------------------------
# DB glue
# --------------------------------------------------------------------------
def _ts(x) -> float:
    return pd.Timestamp(x).timestamp()


def load_open_clusters(store: Store, clusterer: OnlineClusterer, since: Optional[dt.datetime]) -> int:
    q = "SELECT event_id, centroid_sum, first_seen, last_seen, coalesce(n_articles, 1) FROM event_cluster WHERE centroid_sum IS NOT NULL"
    args = []
    if since is not None:
        q += " AND last_seen >= ?"
        args.append(since)
    n = 0
    for r in store.con.execute(q, args).fetchall():
        clusterer.add_existing(r[0], np.frombuffer(r[1], dtype=np.float32), _ts(r[2]), _ts(r[3]), int(r[4]))
        n += 1
    return n


def cluster_new_articles(ctx, batch_vectors: int = 400) -> dict:
    """Cluster every article in article_fact that has no cluster yet. Resumable: the
    assignment, cluster rows and checkpoint for each chunk commit in one transaction, and the
    selection is simply 'cluster_id IS NULL AND cluster_note IS NULL', so a crash re-selects
    only unfinished work."""
    P, st = ctx.P, ctx.store
    todo = st.df("""SELECT source_table, article_id, os_id, published_at, title
                    FROM article_fact WHERE cluster_id IS NULL AND cluster_note IS NULL
                    ORDER BY published_at, source_table, article_id""")
    res = {"clustered": 0, "new_clusters": 0, "joined": 0, "no_vector": 0, "touched": set()}
    # recovery: a run killed between the chunk commits and the final statistics refresh leaves clusters without counts -
    # finish their statistics now (also when there is nothing new to cluster)
    unfinished = {r[0] for r in st.con.execute("SELECT event_id FROM event_cluster WHERE n_articles IS NULL").fetchall()}
    if unfinished:
        ctx.bump("clusters_recovered", len(unfinished))
        res["touched"] |= unfinished
    if not len(todo):
        if unfinished:
            refresh_cluster_stats(ctx, unfinished)
        return res
    cl = OnlineClusterer(P["sim_threshold"], P["max_gap_hours"])
    print(f"[news-cluster] batched similarity device={cl.device}", flush=True)
    # a cluster whose last article is older than `max_gap_hours` before the EARLIEST article to place can never accept any of
    # them (OnlineClusterer.assign), so loading older centroids changes nothing but memory - at all-country volume it is GBs
    horizon = todo["published_at"].min().to_pydatetime() - dt.timedelta(hours=P["max_gap_hours"])
    load_open_clusters(st, cl, horizon)
    ctx.bump("open_clusters_loaded", len(cl))

    for i in range(0, len(todo), batch_vectors):
        chunk = todo.iloc[i:i + batch_vectors]
        hits = ctx.osc.fetch_by_ids(chunk["os_id"].tolist(), include_embedding=True, batch=200)
        facts, notes = [], []
        new_clusters_before = len(cl)
        cl.touched.clear()
        rows, vecs, tss, founders = [], [], [], []
        for r in chunk.itertuples():
            h = hits.get(r.os_id)
            vec, reason = valid_vector((h or {}).get("_source", {}).get("embedding") if h else None)
            if vec is None:
                notes.append({"source_table": r.source_table, "article_id": r.article_id,
                              "cluster_note": reason if h else "missing_embedding"})
                res["no_vector"] += 1
                continue
            rows.append(r); vecs.append(vec); tss.append(_ts(r.published_at)); founders.append(event_id_for(r.source_table, r.article_id))
        for r, (eid, sim, created) in zip(rows, cl.assign_many(np.asarray(vecs, dtype=np.float32).reshape(-1, DIM), tss, founders)):
            facts.append({"source_table": r.source_table, "article_id": r.article_id,
                          "cluster_id": eid, "cluster_sim": None if created else sim})
            res["new_clusters" if created else "joined"] += 1
        touched = set(cl.touched)
        with st.tx():
            _apply_assignments(st, facts, notes)
            _write_cluster_rows(st, cl, touched, ctx.run_id)
        res["clustered"] += len(facts)
        res["touched"] |= touched
    _prune_centroids(st, P, todo["published_at"].max().to_pydatetime(), getattr(ctx, "prune_floor", None))
    refresh_cluster_stats(ctx, res["touched"])
    ctx.bump("clustered", res["clustered"])
    ctx.bump("new_clusters", res["new_clusters"])
    ctx.bump("clustered_joined", res["joined"])
    ctx.bump("cluster_no_vector", res["no_vector"])
    return res


def _apply_assignments(st: Store, facts: list[dict], notes: list[dict]) -> None:
    if facts:
        df = pd.DataFrame(facts)
        st.con.register("_a", df)
        st.con.execute("""UPDATE article_fact SET cluster_id = a.cluster_id, cluster_sim = a.cluster_sim
                          FROM _a a WHERE article_fact.source_table = a.source_table
                          AND article_fact.article_id = a.article_id""")
        st.con.unregister("_a")
    if notes:
        df = pd.DataFrame(notes)
        st.con.register("_n", df)
        st.con.execute("""UPDATE article_fact SET cluster_note = n.cluster_note FROM _n n
                          WHERE article_fact.source_table = n.source_table AND article_fact.article_id = n.article_id""")
        st.con.unregister("_n")


def _write_cluster_rows(st: Store, cl: OnlineClusterer, touched: set[str], run_id: str) -> None:
    if not touched:
        return
    now = now_utc()
    rows = []
    for eid in touched:
        f, l, n = cl.span(eid)
        rows.append({"event_id": eid, "first_seen": dt.datetime.fromtimestamp(f, dt.timezone.utc),
                     "last_seen": dt.datetime.fromtimestamp(l, dt.timezone.utc),
                     "centroid_sum": cl.centroid_sum(eid).astype(np.float32).tobytes(), "is_open": True,
                     "updated_run": run_id, "updated_at": now})
    # one batched upsert; existing rows keep every column not listed here (counts, location, labels ...)
    st.upsert("event_cluster", pd.DataFrame(rows), replace_all=False)
    st.con.execute("UPDATE event_cluster SET created_run = ? WHERE created_run IS NULL AND event_id = ANY(?)", [run_id, sorted(touched)])


def _prune_centroids(st: Store, P: dict, newest: Optional[dt.datetime] = None, floor: Optional[dt.datetime] = None) -> None:
    """Drop centroid blobs of clusters that can no longer accept articles (storage bound).
    The cluster row, its membership and its metrics are kept.

    The retention horizon is measured from the NEWEST article this run placed, not from the newest cluster in the store: when
    older days are processed after newer ones (catch-up / month back-fill) a store-wide maximum would delete the centroids of
    the clusters just created, and the next day could not join them. `floor` (set by the live service while days are still
    waiting to be processed) additionally protects every cluster that one of those days could still join."""
    mx = newest if newest is not None else st.scalar("SELECT max(last_seen) FROM event_cluster")
    if mx is None:
        return
    cutoff = mx - dt.timedelta(days=P["centroid_retention_days"])
    if floor is not None and floor < cutoff:
        cutoff = floor
    st.con.execute("UPDATE event_cluster SET centroid_sum = NULL, is_open = FALSE "
                   "WHERE centroid_sum IS NOT NULL AND last_seen < ?", [cutoff])


def refresh_cluster_stats(ctx, event_ids: Optional[set[str]] = None) -> None:
    """Recompute counts, taxonomy label, location consensus and the is_event flag for the given
    clusters (None = all). Pure SQL over article_fact/article_geo, so it is exact and idempotent."""
    P, st = ctx.P, ctx.store
    where = ""
    if event_ids is not None:
        if not event_ids:
            return
        ids = pd.DataFrame({"event_id": sorted(event_ids)})
        st.con.register("_ev", ids)
        where = "AND f.cluster_id IN (SELECT event_id FROM _ev)"
    share = P["location_consensus_min_share"]
    min_loc = P["location_min_located"]
    sql = f"""
    WITH m AS (
      SELECT f.cluster_id, f.source_table, f.article_id, f.published_at, f.source_domain, f.title, f.url,
             f.domain_id, f.topic_id, f.category_id, g.ev_country_iso, g.ev_adm1_code, g.ev_precision,
             g.geo_inconsistent
      FROM article_fact f LEFT JOIN article_geo g USING (source_table, article_id)
      WHERE f.cluster_id IS NOT NULL AND NOT f.is_duplicate {where}),
    agg AS (
      SELECT cluster_id, count(*) n_articles, count(DISTINCT source_domain) n_publishers,
             min(published_at) first_seen, max(published_at) last_seen,
             (array_agg(title ORDER BY published_at))[1] rep_title, (array_agg(url ORDER BY published_at))[1] rep_url,
             (array_agg(source_table ORDER BY published_at))[1] rep_st, (array_agg(article_id ORDER BY published_at))[1] rep_aid,
             mode() WITHIN GROUP (ORDER BY domain_id) domain_id, mode() WITHIN GROUP (ORDER BY topic_id) topic_id,
             mode() WITHIN GROUP (ORDER BY category_id) category_id,
             count(ev_country_iso) n_located
      FROM m GROUP BY 1),
    ctry AS (
      SELECT cluster_id, ev_country_iso c, count(*) k,
             row_number() OVER (PARTITION BY cluster_id ORDER BY count(*) DESC, ev_country_iso) rn
      FROM m WHERE ev_country_iso IS NOT NULL GROUP BY cluster_id, ev_country_iso),
    adm AS (
      SELECT m.cluster_id, m.ev_adm1_code a, count(*) k,
             row_number() OVER (PARTITION BY m.cluster_id ORDER BY count(*) DESC, m.ev_adm1_code) rn
      FROM m JOIN ctry c ON c.cluster_id = m.cluster_id AND c.rn = 1 AND m.ev_country_iso = c.c
      WHERE m.ev_adm1_code IS NOT NULL AND NOT coalesce(m.geo_inconsistent, FALSE) GROUP BY m.cluster_id, m.ev_adm1_code),
    admn AS (SELECT m.cluster_id, count(*) n FROM m JOIN ctry c ON c.cluster_id = m.cluster_id AND c.rn = 1
             AND m.ev_country_iso = c.c WHERE m.ev_adm1_code IS NOT NULL GROUP BY 1),
    cp AS (
      SELECT m.cluster_id, count(*) n_copies FROM article_fact d
      JOIN m ON d.dup_of = m.source_table || '-' || CAST(m.article_id AS VARCHAR)
      WHERE d.is_duplicate GROUP BY 1)
    SELECT a.cluster_id event_id, a.n_articles, a.n_publishers, a.first_seen, a.last_seen, a.rep_title, a.rep_url,
           a.rep_st rep_source_table, a.rep_aid rep_article_id, a.domain_id, a.topic_id, a.category_id, a.n_located,
           c.c ctry, c.k::DOUBLE / a.n_located loc_share_c,
           ad.a adm, ad.k::DOUBLE / nullif(an.n, 0) loc_share_a,
           coalesce(cp.n_copies, 0) n_copies
    FROM agg a LEFT JOIN cp ON cp.cluster_id = a.cluster_id LEFT JOIN ctry c ON c.cluster_id = a.cluster_id AND c.rn = 1
    LEFT JOIN adm ad ON ad.cluster_id = a.cluster_id AND ad.rn = 1
    LEFT JOIN admn an ON an.cluster_id = a.cluster_id
    """
    df = st.df(sql)
    if event_ids is not None:
        st.con.unregister("_ev")
    if not len(df):
        return
    out = []
    for r in df.itertuples():
        if r.n_located >= min_loc and r.ctry is not None and r.loc_share_c >= share:
            status, ctry = "located", r.ctry
            adm, prec, sh = None, "country", float(r.loc_share_c)
            if r.adm is not None and pd.notna(r.loc_share_a) and r.loc_share_a >= share:
                adm, prec = r.adm, "adm1"
        elif r.n_located >= min_loc and r.ctry is not None:
            status, ctry, adm, prec, sh = "ambiguous", None, None, None, float(r.loc_share_c)
        else:
            status, ctry, adm, prec, sh = "unlocated", None, None, None, None
        # confidence in the GEOGRAPHIC claim (GDELT ActionGeo is noisy: ~20% of events with >=2 located
        # members have members in >1 country, and a 3-of-4 majority can be wrong - measured in the pilot)
        if status != "located":
            conf = None
        elif r.n_located >= P["loc_high_min_located"] and sh >= P["loc_high_min_share"]:
            conf = "high"
        elif r.n_located >= 2:
            conf = "medium"
        else:
            conf = "low"
        is_event = bool(r.n_articles >= P["event_min_articles"] and r.n_publishers >= P["event_min_publishers"])
        out.append({"event_id": r.event_id, "n_articles": int(r.n_articles), "n_publishers": int(r.n_publishers),
                    "n_copies": int(r.n_copies), "first_seen": r.first_seen, "last_seen": r.last_seen,
                    "domain_id": r.domain_id, "topic_id": r.topic_id, "category_id": r.category_id,
                    "rep_title": r.rep_title, "rep_url": r.rep_url, "rep_source_table": r.rep_source_table,
                    "rep_article_id": int(r.rep_article_id), "loc_country_iso": ctry, "loc_adm1_code": adm,
                    "loc_precision": prec, "loc_share": sh, "n_located": int(r.n_located),
                    "loc_status": status, "loc_confidence": conf, "is_event": is_event})
    upd = pd.DataFrame(out)
    st.con.register("_u", upd)
    try:
        with st.tx():
            st.con.execute("""UPDATE event_cluster SET n_articles=u.n_articles, n_publishers=u.n_publishers,
              n_copies=u.n_copies, first_seen=u.first_seen, last_seen=u.last_seen, domain_id=u.domain_id,
              topic_id=u.topic_id, category_id=u.category_id, rep_title=u.rep_title, rep_url=u.rep_url,
              rep_source_table=u.rep_source_table, rep_article_id=u.rep_article_id,
              loc_country_iso=u.loc_country_iso, loc_adm1_code=u.loc_adm1_code, loc_precision=u.loc_precision,
              loc_share=u.loc_share, n_located=u.n_located, loc_status=u.loc_status, loc_confidence=u.loc_confidence,
              is_event=u.is_event
              FROM _u u WHERE event_cluster.event_id = u.event_id""")
    finally:
        st.con.unregister("_u")
