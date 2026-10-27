"""Event-level metrics, emerging-event tests (AOI-specific) and new-development detection.

EMERGING-EVENT TEST (why it is not a per-event time-series test)
  A new event has no history of its own, so a per-event baseline is meaningless. Instead each
  candidate is compared with the AOI's own recent population of events AT THE SAME AGE:

      candidate : an event first seen within `emerging_window_hours` of the evaluation time T,
                  with >= 3 articles from >= 2 publishers so far
      reference : every cluster located in the same AOI whose first article is in
                  [T - emerging_ref_days, T - emerging_window_hours]  (singletons included - they
                  are the normal outcome of a new cluster)
      statistic : number of distinct publishers within the first `age` hours
      p-value   : (1 + #{reference clusters with >= k publishers at that age}) / (n_ref + 1)
                  - an exact empirical upper tail, no distributional assumption

  Reference clusters are stratified by taxonomy domain when the stratum has >= `emerging_min_ref`
  members, otherwise the AOI-wide pool is used; below the minimum the result is NULL with the
  reason (never a number on thin evidence). BH-FDR is applied across all candidates of one
  (AOI, evaluation time). An alert additionally needs >= `emerging_min_publishers` publishers.
  Only data up to T is used for the candidate; reference clusters are complete at that age.
  Known leakage: AOI membership of reference clusters uses final consensus location.

NEW DEVELOPMENTS inside an existing event
  For an established event (>= 8 articles, >= 3 publishers) the articles of the last
  `dev_recent_hours` are compared with the earlier ones: shift = 1 - cos(centroid_new, centroid_old).
  A development is reported only if the new articles form a coherent sub-group (>= 3 articles from
  >= 2 publishers) and the new articles is outside the envelope of the event's earlier coverage (rank test vs leave-one-out similarity of old articles + a 10th-percentile effect floor).
  BH-FDR per (AOI, evaluation time). Vectors are re-fetched from OpenSearch for these events only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from typing import Optional

import numpy as np
import pandas as pd

from . import news_aoi as A
from . import news_cluster as K
from . import news_geo as G
from . import news_metrics as M
from . import stats_v2 as S
from .news_store import Store, now_utc

OUT_OF_SCOPE_DOMAIN = "D00"


# --------------------------------------------------------------------------
# event metrics
# --------------------------------------------------------------------------
def compute_event_metrics(store: Store, P: dict, as_of: dt.datetime,
                          event_ids: Optional[set[str]] = None) -> int:
    """Volume / growth / acceleration / persistence / publisher diversity for event clusters.

    Metrics are computed from member articles published <= as_of. Below minimum support the
    growth/acceleration/half-life fields are NULL (reason in support_note)."""
    where = "AND e.is_event"
    reg = False
    if event_ids is not None:
        if not event_ids:
            return 0
        store.con.register("_evm", pd.DataFrame({"event_id": sorted(event_ids)}))
        reg = True
        where += " AND e.event_id IN (SELECT event_id FROM _evm)"
    try:
        mem = store.df(f"""SELECT f.cluster_id event_id, f.published_at, f.day, f.source_domain
                           FROM article_fact f JOIN event_cluster e ON e.event_id = f.cluster_id
                           WHERE NOT f.is_duplicate AND f.published_at <= ? {where}""", [as_of])
        ev = store.df(f"SELECT event_id, n_copies FROM event_cluster e WHERE TRUE {where}")
    finally:
        if reg:
            store.con.unregister("_evm")
    if not len(mem):
        return 0
    mem["published_at"] = pd.to_datetime(mem["published_at"], utc=True)
    copies = dict(zip(ev["event_id"], ev["n_copies"]))
    asof = pd.Timestamp(as_of)
    rows = []
    for eid, g in mem.groupby("event_id"):
        n, k = len(g), g["source_domain"].nunique()
        first, last = g["published_at"].min(), g["published_at"].max()
        t = (asof - g["published_at"]).dt.total_seconds() / 3600.0
        a24 = int((t < 24).sum())
        p24 = int(((t >= 24) & (t < 48)).sum())
        note = []
        growth = np.nan
        if a24 + p24 >= 5:
            growth = float(math.log((a24 + 1.0) / (p24 + 1.0)))
        else:
            note.append("growth: <5 articles in last 48h")
        accel = np.nan
        buckets = np.floor(t / 6.0).astype(int)
        last48 = g[(t < 48).to_numpy()]
        if len(last48) >= 8:
            cnt = np.bincount(buckets[(t < 48).to_numpy()], minlength=8)[:8][::-1].astype(float)  # old -> new
            lr = np.log(cnt + 1.0)
            d2 = S.causal_savgol_deriv(lr, window=5, order=2, deriv=2)
            accel = float(d2[-1]) if np.isfinite(d2[-1]) else np.nan
        else:
            note.append("accel: <8 articles in last 48h")
        daily = g.groupby("day").size().sort_index()
        peak_day = daily.idxmax()
        post = daily.loc[peak_day:].to_numpy(dtype=float)
        hl, hl_ok = np.nan, False
        if len(post) >= 4:
            fit = S.fit_exp_decay(post)
            hl, hl_ok = fit["half_life_days"], bool(fit["ok"])
            if not hl_ok:
                note.append("half-life: decay fit not reliable")
        else:
            note.append("half-life: <4 days after peak")
        nc = copies.get(eid)
        nc = 0 if nc is None or pd.isna(nc) else int(nc)
        rows.append({
            "event_id": eid, "computed_at": now_utc(), "as_of": as_of, "n_articles": int(n), "n_publishers": int(k),
            "n_copies": nc, "syndication_share": nc / (n + nc) if (n + nc) else np.nan,
            "first_seen": first.to_pydatetime(), "last_seen": last.to_pydatetime(),
            "duration_hours": float((last - first).total_seconds() / 3600.0), "active_days": int(daily.size),
            "peak_day": peak_day, "articles_24h": a24, "articles_prev_24h": p24, "growth_24h": growth,
            "accel_6h": accel, "publisher_pie": M.hurlbert_pie(g["source_domain"].value_counts().to_numpy()),
            "cross_source_ratio": k / n, "half_life_days": hl if np.isfinite(hl) else np.nan, "half_life_ok": hl_ok,
            "persistence_note": "; ".join(n_ for n_ in note if n_.startswith("half-life")) or None,
            "support_ok": True, "support_note": "; ".join(note) or None})
    df = pd.DataFrame(rows)
    store.upsert("event_metrics", df)
    return len(df)


# --------------------------------------------------------------------------
# emerging-event test
# --------------------------------------------------------------------------
class _RefTable:
    """Per-cluster distinct-publisher counts at integer ages 0..48h (one matrix per AOI)."""

    def __init__(self, members: pd.DataFrame, ev: pd.DataFrame, max_h: int = 48):
        self.max_h = max_h
        m = members.copy()
        m["published_at"] = pd.to_datetime(m["published_at"], utc=True)
        first = m.groupby("cluster_id")["published_at"].min().rename("first")
        m = m.join(first, on="cluster_id")
        m["dt_h"] = (m["published_at"] - m["first"]).dt.total_seconds() / 3600.0
        dom_first = m.groupby(["cluster_id", "source_domain"], dropna=False)["dt_h"].min().reset_index()
        ids = sorted(m["cluster_id"].unique())
        idx = {c: i for i, c in enumerate(ids)}
        mat = np.zeros((len(ids), max_h + 1), dtype=np.int16)
        for cid, g in dom_first.groupby("cluster_id"):
            t = np.sort(g["dt_h"].to_numpy())
            mat[idx[cid]] = np.searchsorted(t, np.arange(max_h + 1), side="right")
        self.ids, self.idx, self.mat = ids, idx, mat
        self.first = first.to_dict()
        meta = ev.set_index("event_id")
        self.domain = meta["domain_id"].to_dict()

    def pubs_at(self, cid: str, age_h: int) -> int:
        return int(self.mat[self.idx[cid], min(max(age_h, 0), self.max_h)])


_CONF_RANK = {None: 0, "low": 1, "medium": 2, "high": 3}


def _loc_ok(conf, P: dict) -> bool:
    """Geographic alerts need a trustworthy event location (see news_cluster.refresh_cluster_stats)."""
    return _CONF_RANK.get(conf, 0) >= _CONF_RANK[P["alert_min_loc_confidence"]]


def emerging_tests(store: Store, aoi: G.AOI, P: dict, as_of_list: list[dt.datetime], run_id: str,
                   persist: bool = True) -> dict:
    """Run the emerging-event test for an event-mode AOI at each evaluation time. Idempotent:
    an event that already has a passing test keeps it (point-in-time alert is never overwritten).
    persist=False writes nothing (read-only store OK) and returns res['rows'] / res['alert_rows'] - used
    for ad-hoc AOIs from the API."""
    res = {"aoi_id": aoi.aoi_id, "evaluations": 0, "candidates": 0, "alerts": 0}
    if aoi.mode != "event":
        return res
    ev = A.aoi_events(store, aoi, P)
    if not len(ev):
        return res
    ids = pd.DataFrame({"event_id": ev["event_id"].tolist()})
    store.con.register("_aev", ids)
    try:
        members = store.df("""SELECT f.cluster_id, f.published_at, f.source_domain FROM article_fact f
                              WHERE NOT f.is_duplicate AND f.cluster_id IN (SELECT event_id FROM _aev)""")
    finally:
        store.con.unregister("_aev")
    ref = _RefTable(members, ev)
    ev = ev.set_index("event_id")
    members["published_at"] = pd.to_datetime(members["published_at"], utc=True)
    first_seen = pd.Series(ref.first)
    existing = store.df("SELECT event_id, alert_flag FROM event_aoi_test WHERE aoi_id = ? AND kind = 'emerging'", [aoi.aoi_id])
    already = set(existing.loc[existing["alert_flag"].fillna(False), "event_id"])
    win = dt.timedelta(hours=P["emerging_window_hours"])
    new_rows, new_alerts = [], []
    for T in as_of_list:
        Tt = pd.Timestamp(T)
        res["evaluations"] += 1
        cand_ids = first_seen[(first_seen > Tt - win) & (first_seen <= Tt)].index
        ref_ids = first_seen[(first_seen >= Tt - pd.Timedelta(days=P["emerging_ref_days"])) &
                             (first_seen <= Tt - win)].index
        ref_dom = pd.Series({c: ref.domain.get(c) for c in ref_ids})
        cand = []
        for cid in cand_ids:
            if cid in already or ev.loc[cid, "domain_id"] == OUT_OF_SCOPE_DOMAIN:
                continue
            if not _loc_ok(ev.loc[cid, "loc_confidence"], P):
                continue
            g = members[(members["cluster_id"] == cid) & (members["published_at"] <= Tt)]
            n_c, k_c = len(g), g["source_domain"].nunique()
            if n_c < P["event_min_articles"] or k_c < P["event_min_publishers"]:
                continue
            age = max(1.0, (Tt - first_seen[cid]).total_seconds() / 3600.0)
            h = int(min(math.ceil(age), ref.max_h))
            dom = ev.loc[cid, "domain_id"]
            pool = ref_dom[ref_dom == dom].index if dom is not None else ref_dom.index
            stratum = f"domain:{dom}"
            if len(pool) < P["emerging_min_ref"]:
                pool, stratum = ref_ids, "aoi_pool"
            n_ref = len(pool)
            if n_ref < P["emerging_min_ref"]:
                cand.append({"event_id": cid, "n_c": n_c, "k_c": k_c, "age": age, "n_ref": n_ref, "stratum": stratum,
                             "p": np.nan, "ok": False, "note": f"only {n_ref} reference clusters (< {P['emerging_min_ref']})"})
                continue
            cols = np.array([ref.idx[c] for c in pool])
            ge = int((ref.mat[cols, h] >= k_c).sum())
            cand.append({"event_id": cid, "n_c": n_c, "k_c": k_c, "age": age, "n_ref": n_ref, "stratum": stratum,
                         "p": (1.0 + ge) / (n_ref + 1.0), "ok": True, "note": None})
        if not cand:
            continue
        res["candidates"] += len(cand)
        cdf = pd.DataFrame(cand)
        cdf["q"] = np.nan
        okm = cdf["ok"]
        if okm.any():
            cdf.loc[okm, "q"] = S.bh_qvalues(cdf.loc[okm, "p"].to_numpy(), method=P["fdr_method"])
        for r in cdf.itertuples():
            flag = bool(r.ok and r.q < P["fdr_alpha"] and r.k_c >= P["emerging_min_publishers"])
            new_rows.append({"aoi_id": aoi.aoi_id, "event_id": r.event_id, "kind": "emerging", "as_of": T,
                             "age_hours": float(r.age), "n_articles_at_age": int(r.n_c),
                             "n_publishers_at_age": int(r.k_c), "n_ref": int(r.n_ref), "stratum": r.stratum,
                             "p": None if not r.ok else float(r.p), "q": None if not r.ok else float(r.q),
                             "effect": float(r.k_c), "support_ok": bool(r.ok), "support_note": r.note,
                             "alert_flag": flag})
            if flag:
                already.add(r.event_id)
                res["alerts"] += 1
                e = ev.loc[r.event_id]
                new_alerts.append({
                    "alert_id": M.alert_id("emerging_event", aoi.aoi_id, r.event_id, None), "kind": "emerging_event",
                    "aoi_id": aoi.aoi_id, "entity_type": "event", "entity_id": r.event_id, "day": Tt.date(),
                    "as_of": T, "p": float(r.p), "q": float(r.q), "effect": float(r.k_c),
                    "severity": "high" if (r.q < P["severity_q_high"] and r.k_c >= 2 * P["emerging_min_publishers"]) else "medium",
                    "rationale": (f"{r.k_c} publishers / {r.n_c} articles within {r.age:.0f}h; exceeded by "
                                  f"{(r.p * (r.n_ref + 1) - 1):.0f} of {r.n_ref} reference clusters ({r.stratum}); "
                                  f"BH q={r.q:.4f}: {e['rep_title']}"),
                    "support_json": json.dumps({"n_ref": int(r.n_ref), "stratum": r.stratum, "age_hours": float(r.age)})})
    if not persist:
        res["rows"] = pd.DataFrame(new_rows).drop_duplicates(["aoi_id", "event_id", "kind"], keep="last") if new_rows else pd.DataFrame()
        res["alert_rows"] = new_alerts
        return res
    with store.tx():
        if new_rows:
            df = pd.DataFrame(new_rows).drop_duplicates(["aoi_id", "event_id", "kind"], keep="last")
            # keep an earlier passing evaluation if the same event reappears later in this batch
            store.upsert("event_aoi_test", df)
        if new_alerts:
            M.sync_alerts(store, "emerging_event", aoi.aoi_id, new_alerts, run_id, retract_missing=False)
    return res


# --------------------------------------------------------------------------
# new developments
# --------------------------------------------------------------------------
def development_stats(old: np.ndarray, new: np.ndarray, envelope_pct: float) -> dict:
    """Do the NEW articles sit outside the envelope of the event's earlier coverage?

    s_j  = cosine(old article j, centroid of the OTHER old articles)   (leave-one-out envelope)
    t_i  = cosine(new article i, centroid of all old articles)
    p    = one-sided Mann-Whitney U, H1: new articles are less similar to the old core than old
           articles are to theirs  (rank-based, so it respects the event's own dispersion)
    gate = median(t) below the `envelope_pct` percentile of s  (an effect-size floor: outside the
           envelope, not merely 'a bit different')

    A plain permutation of member labels was tried first and REJECTED: articles published close in
    time are more alike, so it flagged 56% of events on real data (shift ~0.05 cosine).
    """
    from scipy import stats as sps
    so = old.sum(axis=0)
    c_old = so / np.linalg.norm(so)
    loo = so[None, :] - old
    loo /= np.linalg.norm(loo, axis=1, keepdims=True)
    s = np.einsum("ij,ij->i", old, loo)
    t = new @ c_old
    c_new = new.mean(axis=0)
    c_new /= np.linalg.norm(c_new)
    p = float(sps.mannwhitneyu(t, s, alternative="less").pvalue)
    return {"p": p, "shift": 1.0 - float(c_new @ c_old), "median_new": float(np.median(t)),
            "envelope_floor": float(np.percentile(s, envelope_pct)),
            "outside_envelope": bool(np.median(t) < np.percentile(s, envelope_pct)),
            "coherence": float((new @ c_new).mean())}


def detect_developments(ctx, aois: list[G.AOI], as_of_list: list[dt.datetime]) -> dict:
    """New-development detection for events in each event-mode AOI. Needs OpenSearch for vectors."""
    P, st = ctx.P, ctx.store
    res = {"tested": 0, "alerts": 0, "capped": 0}
    rng = np.random.default_rng(P["seed"])
    ev_cache: dict[str, pd.DataFrame] = {}
    vec_cache: dict[str, dict] = {}
    for aoi in aois:
        if aoi.mode != "event":
            continue
        ev = A.aoi_events(st, aoi, P)
        if not len(ev):
            continue
        ev = ev[ev["loc_confidence"].map(lambda c: _loc_ok(c, P))]
        ev = ev[(ev["n_articles"] >= P["dev_min_event_articles"]) & (ev["n_publishers"] >= P["dev_min_event_publishers"])
                & (ev["domain_id"] != OUT_OF_SCOPE_DOMAIN)]
        for T in as_of_list:
            Tt = pd.Timestamp(T)
            cut = Tt - pd.Timedelta(hours=P["dev_recent_hours"])
            tests = []
            for eid in ev["event_id"]:
                if eid not in ev_cache:
                    ev_cache[eid] = st.df("""SELECT source_table, article_id, os_id, published_at, source_domain, title
                                             FROM article_fact WHERE cluster_id = ? AND NOT is_duplicate""", [eid])
                m = ev_cache[eid]
                pa = pd.to_datetime(m["published_at"], utc=True)
                m = m[pa <= Tt]
                pa = pd.to_datetime(m["published_at"], utc=True)
                new, old = m[pa > cut], m[pa <= cut]
                if len(new) < P["dev_min_new_articles"] or new["source_domain"].nunique() < P["dev_min_new_publishers"] \
                        or len(old) < P["dev_min_event_articles"] - P["dev_min_new_articles"]:
                    continue
                dev_id = hashlib.sha1(f"{eid}|{Tt.date()}".encode()).hexdigest()[:20]
                if st.scalar("SELECT count(*) FROM event_development WHERE dev_id=? AND aoi_id=?", [dev_id, aoi.aoi_id]):
                    continue
                if len(tests) >= P["dev_max_events_per_run"]:
                    res["capped"] += 1
                    continue
                vc = vec_cache.get(eid)
                if vc is None:   # one OpenSearch fetch per event per run, reused for every evaluation time
                    full = ev_cache[eid]
                    hits = ctx.osc.fetch_by_ids(full["os_id"].tolist(), include_embedding=True)
                    vc = {}
                    for oid in full["os_id"]:
                        v, _why = K.valid_vector(((hits.get(oid) or {}).get("_source") or {}).get("embedding"))
                        if v is not None:
                            vc[oid] = v
                    vec_cache[eid] = vc
                keep_mask = [o in vc for o in m["os_id"]]
                if sum(keep_mask) < len(m) * 0.8:
                    continue
                m = m[keep_mask]
                V = np.vstack([vc[o] for o in m["os_id"]])
                is_new = (pd.to_datetime(m["published_at"], utc=True) > cut).to_numpy()
                if is_new.sum() < P["dev_min_new_articles"] or (~is_new).sum() < 2:
                    continue
                ds = development_stats(V[~is_new], V[is_new], P["dev_envelope_pct"])
                shift, coherence, p = ds["shift"], ds["coherence"], ds["p"]
                outside = ds["outside_envelope"]
                nn = m[is_new]
                tests.append({"event_id": eid, "dev_id": dev_id, "n_new": int(is_new.sum()),
                              "n_new_pub": int(nn["source_domain"].nunique()), "shift": shift, "p": p,
                              "coherence": coherence, "outside": outside, "title": nn.sort_values("published_at")["title"].iloc[0],
                              "members": ",".join(f"{a}-{b}" for a, b in zip(nn["source_table"], nn["article_id"]))})
            if not tests:
                continue
            tdf = pd.DataFrame(tests)
            tdf["q"] = S.bh_qvalues(tdf["p"].to_numpy(), method=P["fdr_method"])
            res["tested"] += len(tdf)
            rows, alerts = [], []
            for r in tdf.itertuples():
                flag = bool(r.q < P["fdr_alpha"] and r.coherence >= P["dev_min_coherence"] and r.outside)
                rows.append({"event_id": r.event_id, "dev_id": r.dev_id, "aoi_id": aoi.aoi_id, "detected_asof": T,
                             "n_new": r.n_new, "n_new_publishers": r.n_new_pub, "shift": r.shift, "shift_p": r.p,
                             "shift_q": r.q, "coherence": r.coherence, "rep_title": r.title, "alert_flag": flag,
                             "member_keys": r.members})
                if flag:
                    res["alerts"] += 1
                    alerts.append({"alert_id": M.alert_id("new_development", aoi.aoi_id, r.dev_id, None),
                                   "kind": "new_development", "aoi_id": aoi.aoi_id, "entity_type": "event",
                                   "entity_id": r.event_id, "day": Tt.date(), "as_of": T, "p": float(r.p),
                                   "q": float(r.q), "effect": float(r.shift), "severity": "medium",
                                   "rationale": f"{r.n_new} new articles ({r.n_new_pub} publishers) shifted away from the "
                                                f"event's earlier coverage (outside the earlier-coverage envelope; "
                                                f"coherence {r.coherence:.2f}, rank-test q={r.q:.4f}): {r.title}",
                                   "support_json": json.dumps({"n_new": r.n_new, "n_new_publishers": r.n_new_pub})})
            with st.tx():
                st.upsert("event_development", pd.DataFrame(rows))
                if alerts:
                    M.sync_alerts(st, "new_development", aoi.aoi_id, alerts, ctx.run_id, retract_missing=False)
    return res
