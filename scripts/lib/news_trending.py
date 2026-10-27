"""Top trending STORIES for an area and/or a topic. Pure read-time analysis over the derived store.

Nothing here writes anything: it reads event clusters, their articles, geography and taxonomy labels that the pipeline
already stored, and ranks them. It reuses `stats_v2` (BH-FDR, percentile ranks) and the rank-composite design of the
existing `emergence_score`.

WHAT IS A STORY
  An event cluster (>= 3 articles from >= 2 publishers). Clusters that are really the same story split in two - centroid
  cosine >= 0.90 AND overlapping/adjacent in time (<= 72 h apart) - are merged, so one story is listed once. 0.90 was
  chosen by reading headline pairs: every sampled pair >= 0.90 was the same story, ~half of 0.85-0.90 were, and 0.85
  chains unrelated coverage into one 113-event blob. Exact duplicate articles are already counted once.

WHAT "TRENDING" MEANS (all numbers are shown to the user)
  For a preset window W ending at the latest data time T (24 h / 3 d / 7 d), or a custom date range bounded by
  midnight IST on `date_from` and midnight IST after `date_to` (converted to UTC for stored timestamps):
    volume    articles in (T-W, T]          breadth   distinct publishers in the window
    growth    log((a_W+1)/(a_prevW+1))      NULL unless a_W + a_prev >= 5 (never a made-up number)
    recency   hours since the latest article
    baseline  DESCRIPTIVE: how this story's window volume compares with comparable story-windows (same taxonomy domain
              when >= 50 exist, otherwise all) in the same geographic scope, in earlier fully-observed windows, defined
              like the candidates (>= 3 articles from >= 2 publishers): "more than X % of comparable stories; a typical
              one gets N". It is NOT a hypothesis test: the candidates are drawn from the same population, so the
              biggest story is by construction an extreme of it (with n comparables and m candidates the smallest
              attainable p-value, 1/(n+1), times m exceeds 0.05 - a first version that reported FDR q here gave q = 1 to
              a 306-article story and was REJECTED).
    evidence  statistical significance is taken only from tests with a proper null that the pipeline already ran, and only
              WHERE THEY EXIST (registered areas): the emerging-event test, the new-development test and the
              negative-binomial topic surge, each FDR-controlled. A story outside those areas has no evidence component
              (not "zero evidence"): the component is dropped and the weights renormalised.
  Score = weighted percentile ranks (volume .30, breadth .30, growth .20, recency .10, evidence .10; the same rank-composite
  design as the existing `emergence_score`). It orders the stories relative to each other in the view; it is not a
  probability.

SCOPE (the four cases)
  country only   stories located in that country (the country dominates the located articles AND is supported by enough of the whole story: trend_country_*), any taxonomy
  taxonomy only  stories whose majority-supported label at that level is the selection (lib/news_taxonomy.py), any country (including unlocated)
  both           the intersection
  neither        not defined: the caller must ask the user to choose
  The comparison population is the GEOGRAPHIC scope (the country, or everything), not the taxonomy slice, so a narrow
  topic is judged against how stories behave in that place rather than against a handful of peers.
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import news_config as C
from . import stats_v2 as S
from .news_store import Store
from . import news_relevance as NR
from .news_taxonomy import assign_paths

WINDOWS = {"24h": (24, "last 24 hours"), "3d": (72, "last 3 days"), "7d": (168, "last 7 days")}
TIER_COL = {"domain": "domain_id", "topic": "topic_id", "category": "category_id", "subtopic": "subtopic_id"}


class TrendError(ValueError):
    pass


# --------------------------------------------------------------------------
# story merging (pure)
# --------------------------------------------------------------------------
def merge_events(unit: np.ndarray, first_ts: np.ndarray, last_ts: np.ndarray, cos_thr: float, gap_h: float,
                 has_vec: Optional[np.ndarray] = None, block: int = 800) -> np.ndarray:
    """Connected components of 'same story' links -> a story label per event. Events without a centroid stay alone.
    Computed in row blocks so memory stays bounded for large stores."""
    n = len(first_ts)
    if n == 0:
        return np.zeros(0, dtype=int)
    has_vec = np.ones(n, dtype=bool) if has_vec is None else has_vec
    idx = np.flatnonzero(has_vec)
    rows, cols = [], []
    U = unit[idx]
    for s in range(0, len(idx), block):
        sim = U[s:s + block] @ U.T
        a, b = idx[s:s + block], idx
        gap = np.maximum(np.maximum(first_ts[a][:, None] - last_ts[b][None, :], first_ts[b][None, :] - last_ts[a][:, None]), 0) / 3600.0
        ii, jj = np.nonzero((sim >= cos_thr) & (gap <= gap_h))
        keep = a[ii] != b[jj]
        rows.append(a[ii][keep]); cols.append(b[jj][keep])
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=int)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=int)
    adj = coo_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(n, n))
    _, lab = connected_components(adj, directed=False)
    return lab


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
@dataclass
class TrendData:
    stories: pd.DataFrame       # one row per story
    mem: pd.DataFrame           # one row per (non-duplicate) article in a story, with story_id and epoch seconds `ts`
    as_of: pd.Timestamp         # latest article time in the store
    data_start: pd.Timestamp
    n_events: int
    vec: Optional[np.ndarray] = None      # unit centroid per story (row order = vec_ids)
    vec_ids: Optional[list] = None


_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def epoch_seconds(x) -> np.ndarray:
    """Seconds since the epoch, independent of the datetime resolution (pandas 3 uses microseconds, older ns)."""
    return ((pd.to_datetime(x, utc=True) - _EPOCH).dt.total_seconds()).to_numpy(dtype=float)


def _plural(df: pd.DataFrame, key: str, col: str) -> tuple[pd.Series, pd.Series]:
    """Most frequent value of `col` per `key` (ties -> smallest value), and its count. Deterministic."""
    g = df.dropna(subset=[col]).groupby([key, col]).size().rename("n").reset_index()
    if not len(g):
        return pd.Series(dtype=object), pd.Series(dtype=int)
    g = g.sort_values([key, "n", col], ascending=[True, False, True]).drop_duplicates(key).set_index(key)
    return g[col], g["n"]


def load(st: Store, P: Optional[dict] = None) -> TrendData:
    P = P or C.PARAMS
    ev = st.df("SELECT event_id, rep_title, n_articles, first_seen, last_seen, centroid_sum, ref_sim, ref_votes FROM event_cluster WHERE is_event ORDER BY event_id")
    mem = st.df("""SELECT f.cluster_id AS event_id, f.source_table, f.article_id, f.published_at, f.source_domain, f.title, f.url,
                          f.has_label, f.domain_id, f.topic_id, f.category_id, f.subtopic_id,
                          g.ev_country_iso, g.ev_place, g.loc_status, g.geo_inconsistent
                   FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                   WHERE NOT f.is_duplicate AND f.cluster_id IN (SELECT event_id FROM event_cluster WHERE is_event)""")
    empty = TrendData(pd.DataFrame(), mem, pd.NaT, pd.NaT, 0)
    if not len(ev) or not len(mem):
        return empty
    mem["published_at"] = pd.to_datetime(mem["published_at"], utc=True)
    mem["ts"] = epoch_seconds(mem["published_at"])
    ev["first_ts"] = epoch_seconds(ev["first_seen"])
    ev["last_ts"] = epoch_seconds(ev["last_seen"])
    d = 1024
    has = ev["centroid_sum"].map(lambda b: isinstance(b, (bytes, bytearray, memoryview)) and len(b) == d * 4).to_numpy()   # pruned centroids are NULL (pd.NA)
    U = np.zeros((len(ev), d), dtype=np.float32)
    for i in np.flatnonzero(has):
        v = np.frombuffer(ev["centroid_sum"].iat[i], dtype=np.float32)
        nrm = float(np.linalg.norm(v))
        if nrm > 0:
            U[i] = v / nrm
        else:
            has[i] = False
    lab = merge_events(U, ev["first_ts"].to_numpy(), ev["last_ts"].to_numpy(), P["trend_merge_cos"],
                       P["trend_merge_gap_hours"], has)
    ev["story_no"] = lab
    lead = ev.sort_values(["story_no", "n_articles", "event_id"], ascending=[True, False, True]).drop_duplicates("story_no")
    story_of = dict(zip(ev["event_id"], ev["story_no"]))
    lead_of = dict(zip(lead["story_no"], lead["event_id"]))
    mem["story_id"] = mem["event_id"].map(story_of).map(lead_of)       # a story is identified by its lead event id
    members_of = ev.assign(story_id=ev["story_no"].map(lead_of)).groupby("story_id")["event_id"].apply(list)

    g = mem.groupby("story_id")
    st_df = pd.DataFrame({"n_articles": g.size(), "n_publishers": g["source_domain"].nunique(),
                          "first_ts": g["ts"].min(), "last_ts": g["ts"].max()})
    st_df["event_ids"] = members_of.reindex(st_df.index)
    st_df["title"] = lead.set_index(lead["story_no"].map(lead_of))["rep_title"].reindex(st_df.index)
    # relevance of the story to the taxonomy's subject (lib/news_relevance.py): event scores combined, weighted by article count
    ev["story_id"] = ev["story_no"].map(lead_of)
    scored = ev.dropna(subset=["ref_sim"])
    st_df["ref_sim"] = np.nan
    st_df["ref_votes"] = [{} for _ in range(len(st_df))]
    if len(scored):
        w = scored["n_articles"].astype(float)
        num = (scored["ref_sim"] * w).groupby(scored["story_id"]).sum()
        den = w.groupby(scored["story_id"]).sum()
        st_df["ref_sim"] = (num / den).reindex(st_df.index)
        vt = {}
        for sid, v in zip(scored["story_id"], scored["ref_votes"]):
            acc = vt.setdefault(sid, {})
            for k, n in NR.parse_votes(v).items():
                acc[k] = acc.get(k, 0) + n
        st_df["ref_votes"] = [vt.get(i, {}) for i in st_df.index]
    # taxonomy path: only what a strict majority of the story's labelled articles agree on (see lib/news_taxonomy.py)
    paths = assign_paths(mem, "story_id", min_agreement=P["taxonomy_min_agreement"])
    for c in list(TIER_COL) + ["tax_status", "n_labelled", "domain_share"]:
        st_df[c] = paths[c].reindex(st_df.index)
    st_df["tax_status"] = st_df["tax_status"].fillna("unlabelled")
    # geography of the story: plurality country among LOCATED members (unlocated stories have none)
    loc = mem[mem["ev_country_iso"].notna()]
    cty, cn = _plural(loc, "story_id", "ev_country_iso")
    n_loc = loc.groupby("story_id").size()
    st_df["country"] = cty.reindex(st_df.index)
    st_df["n_in_country"] = cn.reindex(st_df.index).fillna(0).astype(int)
    st_df["n_located"] = n_loc.reindex(st_df.index).fillna(0).astype(int)
    st_df["loc_share"] = (cn.reindex(st_df.index) / n_loc.reindex(st_df.index)).astype(float)
    placed = loc.merge(st_df[["country"]], left_on="story_id", right_index=True)
    placed = placed[placed["ev_country_iso"] == placed["country"]]
    pl, _ = _plural(placed.assign(ev_place=placed["ev_place"].replace("", np.nan)), "story_id", "ev_place")
    st_df["place"] = pl.reindex(st_df.index)
    st_df = st_df.reset_index()
    # story centroid = normalised sum of its events' centroids (each already a sum of article vectors -> size-weighted)
    row_of = {e: i for i, e in enumerate(ev["event_id"])}
    V = np.zeros((len(st_df), d), dtype=np.float32)
    for j, eids in enumerate(st_df["event_ids"]):
        for e in eids:
            i = row_of[e]
            if has[i]:
                V[j] += np.frombuffer(ev["centroid_sum"].iat[i], dtype=np.float32)
    nrm = np.linalg.norm(V, axis=1, keepdims=True)
    V = np.divide(V, nrm, out=np.zeros_like(V), where=nrm > 0)
    return TrendData(st_df, mem, mem["published_at"].max(), mem["published_at"].min(), len(ev), V, list(st_df["story_id"]))


# --------------------------------------------------------------------------
# window statistics (pure)
# --------------------------------------------------------------------------
def in_country(st: pd.DataFrame, country: str, P: Optional[dict] = None) -> pd.DataFrame:
    """Stories that belong to `country` on real evidence (see the trend_country_* parameters)."""
    P = P or C.PARAMS
    ok = ((st["country"] == country) & (st["loc_share"] >= P["trend_country_min_share"])
          & (st["n_in_country"] >= P["trend_country_min_located"])
          & (st["n_in_country"] >= P["trend_country_min_coverage"] * st["n_articles"]))
    return st[ok]


def window_table(mem: pd.DataFrame, as_of_ts: float, W_h: float,
                 period_start_ts: Optional[float] = None) -> pd.DataFrame:
    """Articles and distinct publishers per story/window. Custom periods are [start, end); k=1 is the preceding window."""
    if period_start_ts is None:
        m = mem[mem["ts"] <= as_of_ts]
        k = np.floor((as_of_ts - m["ts"].to_numpy()) / (W_h * 3600.0)).astype(int)
    else:
        m = mem[mem["ts"] < as_of_ts]
        ts = m["ts"].to_numpy()
        width_s = W_h * 3600.0
        k = np.where(ts >= period_start_ts, 0, np.ceil((period_start_ts - ts) / width_s)).astype(int)
    t = pd.DataFrame({"story_id": m["story_id"].to_numpy(), "k": k, "src": m["source_domain"].to_numpy()})
    g = t.groupby(["story_id", "k"])
    return pd.DataFrame({"n": g.size(), "p": g["src"].nunique()}).reset_index()


def _iso(ts: float) -> str:
    """Epoch seconds -> ISO time at the stored (microsecond) precision; the float round-trip leaves sub-microsecond noise."""
    return pd.Timestamp(ts, unit="s", tz="UTC").round("us").isoformat()


def cite(title, domain, published_at, url) -> dict:
    d = (domain or "").lower()
    ok = isinstance(url, str) and url.startswith(("http://", "https://"))
    return {"title": title or "(no title)", "publisher": d[4:] if d.startswith("www.") else (d or "unknown publisher"),
            "published_at": None if published_at is None or pd.isna(published_at) else pd.Timestamp(published_at).isoformat(),
            "url": url if ok else None}


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------
def _fmt_when(h: float) -> str:
    if h < 1:
        return "less than an hour"
    if h < 48:
        return f"{h:.0f} h"
    return f"{h / 24:.1f} days"


def trending(data: TrendData, *, country: Optional[str] = None, tax: Optional[dict] = None, window: str = "3d",
             date_from: Optional[dt.date] = None, date_to: Optional[dt.date] = None,
             top_n: Optional[int] = None, scope_name: str = "", names: Optional[dict] = None, P: Optional[dict] = None,
             evidence_fn: Optional[Callable[[list, pd.Timestamp, float], dict]] = None,
             enrich: Optional[Callable[[dict], None]] = None) -> dict:
    """Rank stories. `tax` = {"tier": ..., "code": ..., "name": ...} or None. See the module docstring."""
    P = P or C.PARAMS
    custom = date_from is not None or date_to is not None
    if custom and (date_from is None or date_to is None):
        raise TrendError("custom date range requires both date_from and date_to")
    if custom and (not isinstance(date_from, dt.date) or not isinstance(date_to, dt.date)):
        raise TrendError("custom date range must use valid dates")
    if custom and date_from > date_to:
        raise TrendError("date_from must be on or before date_to")
    if not custom and window not in WINDOWS:
        raise TrendError("window must be 24h, 3d or 7d")
    if not country and not tax:
        return {"needs_selection": True}
    top_n = P["trend_top_n"] if top_n is None else int(top_n)      # 0 is an invalid limit, not "use the default"
    if not (1 <= top_n <= P["trend_max_n"]):
        raise TrendError(f"limit must be 1..{P['trend_max_n']}")
    if custom:
        days = (date_to - date_from).days + 1
        W_h = days * 24
        wlabel = f"{date_from.isoformat()} to {date_to.isoformat()} (IST)"
        period_start = pd.Timestamp(date_from, tz="Asia/Kolkata").tz_convert("UTC")
        as_of_override = pd.Timestamp(date_to + dt.timedelta(days=1), tz="Asia/Kolkata").tz_convert("UTC")
        window_info = {"key": "custom", "hours": W_h, "days": days, "label": wlabel,
                       "date_from": date_from.isoformat(), "date_to": date_to.isoformat(), "timezone": "Asia/Kolkata"}
    else:
        W_h, wlabel = WINDOWS[window]
        period_start = None
        as_of_override = None
        window_info = {"key": window, "hours": W_h, "label": wlabel}
    out = {"needs_selection": False, "window": window_info, "country": country,
           "tax": tax, "stories": [], "n_scope_stories": 0, "n_candidates": 0, "n_reference": 0, "notes": []}
    st = data.stories
    if not len(st):
        out["notes"].append("There are no story clusters in the database.")
        return out
    as_of = as_of_override if custom else data.as_of
    as_of_ts = as_of.timestamp()
    start_ts = data.data_start.timestamp()
    out.update(as_of=as_of.isoformat(), data_start=data.data_start.isoformat())
    if custom:
        available_start = data.data_start.tz_convert("Asia/Kolkata").date()
        available_end = data.as_of.tz_convert("Asia/Kolkata").date()
        if date_from < available_start:
            raise TrendError(f"date_from is before available data ({available_start.isoformat()})")
        if date_to > available_end:
            raise TrendError(f"date_to is after available data ({available_end.isoformat()})")

    # ---- geographic scope = the comparison population
    if country:
        ref_st = in_country(st, country, P)
    else:
        ref_st = st
    out["n_scope_stories"] = int(len(ref_st))
    if not len(ref_st):
        out["notes"].append("No stories are located in this country in the processed data.")
        return out
    cand_st = ref_st[ref_st[tax["tier"]] == tax["code"]] if tax else ref_st
    if tax and not len(cand_st):
        where = "in this country " if country else ""
        out["notes"].append(f"None of the {len(ref_st):,} stories {where}has a majority of its labelled articles in this {tax['tier']}. "
                            "Stories whose articles disagree are not listed under any single topic.")
        return out

    # ---- window statistics for every story in the comparison population
    ids = set(ref_st["story_id"])
    tab = window_table(data.mem[data.mem["story_id"].isin(ids)], as_of_ts, W_h,
                       period_start.timestamp() if custom else None)
    by = {sid: g.set_index("k") for sid, g in tab.groupby("story_id")}
    tol = P["trend_window_tolerance_hours"] * 3600.0
    complete = lambda k: (as_of_ts - (k + 1) * W_h * 3600.0) >= start_ts - tol
    # reference cells: earlier, fully observed windows, defined exactly like the candidates
    ma, mp = P["trend_min_window_articles"], P["trend_min_window_publishers"]
    ref_rows = []
    if complete(1):
        t = tab[(tab["k"] >= 1) & (tab["k"].map(complete)) & (tab["n"] >= ma) & (tab["p"] >= mp)]
        dom = ref_st.set_index("story_id")["domain"]
        ref_rows = t.assign(domain=t["story_id"].map(dom))
    pooled = np.sort(ref_rows["n"].to_numpy()) if len(ref_rows) else np.zeros(0)
    pooled_p = np.sort(ref_rows["p"].to_numpy()) if len(ref_rows) else np.zeros(0)
    strata = {d: (np.sort(g["n"].to_numpy()), np.sort(g["p"].to_numpy())) for d, g in ref_rows.groupby("domain")} if len(ref_rows) else {}
    out["n_reference"] = int(len(pooled))

    # ---- candidates: enough activity in the window
    rows = []
    mem_by = data.mem[data.mem["story_id"].isin(set(cand_st["story_id"]))].groupby("story_id")
    for r in cand_st.itertuples():
        w = by.get(r.story_id)
        a0 = int(w.loc[0, "n"]) if w is not None and 0 in w.index else 0
        p0 = int(w.loc[0, "p"]) if w is not None and 0 in w.index else 0
        if a0 < ma or p0 < mp:
            continue
        a1 = int(w.loc[1, "n"]) if (w is not None and 1 in w.index) else 0
        growth = None
        if complete(1) and a0 + a1 >= P["trend_growth_min_total"]:
            growth = math.log((a0 + 1.0) / (a1 + 1.0))
        stratum, ref_a = "pooled", pooled
        if r.domain in strata and len(strata[r.domain][0]) >= P["trend_min_ref"]:
            stratum, ref_a = f"domain {r.domain}", strata[r.domain][0]
        stats = None
        if len(ref_a) >= P["trend_min_ref"]:
            stats = {"n_ref": int(len(ref_a)), "stratum": stratum, "typical": float(np.median(ref_a)),
                     "p90": float(np.percentile(ref_a, 90)), "pct_beaten": float(np.mean(ref_a < a0))}
        rows.append({"story": r, "a0": a0, "p0": p0, "a1": a1, "growth": growth, "ref": stats,
                     "age_h": max(0.0, (as_of_ts - r.last_ts) / 3600.0)})
    # relevance gate: hide stories that are not about the taxonomy's subject (or, under a selection, not about that domain)
    sel_dom = tax["code"].split(".")[0] if tax else None
    n_active = len(rows)
    rows = [x for x in rows if NR.passes(x["story"].ref_sim, x["story"].ref_votes, sel_dom, P)]
    out["n_hidden_offtopic"] = n_active - len(rows)
    if out["n_hidden_offtopic"]:
        out["notes"].append(f"{out['n_hidden_offtopic']} of {n_active} active stories were not listed because they are not about the subject "
                            "area of the taxonomy (security, politics, defence, economy, humanitarian) or not about the selected domain.")
    out["n_candidates"] = len(rows)
    if not rows:
        if not n_active:
            out["notes"].append(f"No story in this view has at least {ma} articles from {mp} publishers in the {wlabel}.")
        return out

    # ---- evidence from the pipeline's own FDR-controlled tests (only where they exist), then the rank-composite score
    ev = {}
    if evidence_fn:
        ev = evidence_fn([{"story_id": x["story"].story_id, "event_ids": list(x["story"].event_ids), "country": x["story"].country,
                           "topic": x["story"].topic} for x in rows], as_of, W_h)
    for x in rows:
        e = ev.get(x["story"].story_id) or {"available": False, "items": []}
        x["evidence"] = e
    comps = pd.DataFrame({
        "volume": S.rank_pct([x["a0"] for x in rows]),
        "breadth": S.rank_pct([x["p0"] for x in rows]),
        "growth": S.rank_pct([np.nan if x["growth"] is None else x["growth"] for x in rows]),
        "recency": S.rank_pct([-x["age_h"] for x in rows]),
        "evidence": S.rank_pct([float(len(x["evidence"]["items"])) if x["evidence"]["available"] else np.nan for x in rows])})
    w = pd.Series(P["trend_weights"])[comps.columns]
    avail = comps.notna()
    score = (comps.fillna(0) * w).sum(axis=1) / (avail * w).sum(axis=1)
    for i, x in enumerate(rows):
        x["score"] = float(100 * score.iat[i])
    rows.sort(key=lambda x: (-x["score"], -x["a0"], x["story"].story_id))

    names = names or {}
    kept, folded = _fold_duplicates(rows, data, top_n, P)
    stories = []
    for rank, x in enumerate(kept, 1):
        r = x["story"]
        mrows = mem_by.get_group(r.story_id)
        if custom:
            in_period = (mrows["ts"] >= period_start.timestamp()) & (mrows["ts"] < as_of_ts)
        else:
            in_period = mrows["ts"] > as_of_ts - W_h * 3600.0
        win = mrows[in_period].sort_values(["ts", "article_id"], ascending=[False, True])
        pubs, seen = [], set()
        for m in win.itertuples():
            if m.source_domain not in seen:
                seen.add(m.source_domain); pubs.append(cite(m.title, m.source_domain, m.published_at, m.url))
            if len(pubs) == 3:
                break
        loc_high = r.n_located >= P["loc_high_min_located"] and (r.loc_share or 0) >= P["loc_high_min_share"]
        place = ", ".join(str(r.place).split(", ")[:-1]) if isinstance(r.place, str) and ", " in r.place else ""
        story = {
            "rank": rank, "story_id": r.story_id, "event_ids": list(r.event_ids), "title": r.title,
            "n_events": len(r.event_ids), "n_articles": int(r.n_articles), "n_publishers": int(r.n_publishers),
            "first_seen": _iso(r.first_ts),
            "latest_activity": _iso(r.last_ts),
            "hours_since_latest": x["age_h"], "hours_since_first": max(0.0, (as_of_ts - r.first_ts) / 3600.0),
            "in_window": {"articles": x["a0"], "publishers": x["p0"], "previous_articles": x["a1"], "growth": x["growth"]},
            "taxonomy": {t: ({"code": getattr(r, t), "name": names.get(getattr(r, t))}
                             if r.tax_status == "assigned" and isinstance(getattr(r, t), str) else None) for t in TIER_COL},
            "taxonomy_status": r.tax_status,
            "location": {"country": r.country if isinstance(r.country, str) else None, "place": place or None,
                         "n_located": int(r.n_located), "share": None if pd.isna(r.loc_share) else float(r.loc_share),
                         "uncertain": bool(r.n_located and not loc_high)},
            "score": x["score"], "baseline": x["ref"],
            "evidence": {"available": x["evidence"]["available"], "items": x["evidence"]["items"]},
            "sources": pubs, "reasons": [],
            "related": [{"story_id": f["story"].story_id, "title": f["story"].title, "n_articles": int(f["story"].n_articles),
                         "articles_in_window": f["a0"], "latest_activity": _iso(f["story"].last_ts),
                         "event_ids": list(f["story"].event_ids)} for f in folded.get(r.story_id, [])]}
        story["reasons"] = reasons(story, wlabel, W_h, scope_name, P)
        if enrich:
            enrich(story)
        stories.append(story)
    out["stories"] = stories
    if len(pooled) < P["trend_min_ref"]:
        out["notes"].append(f"Only {len(pooled)} comparable story-windows exist in this scope (need {P['trend_min_ref']}), so the 'compared with typical stories' line is omitted.")
    return out


def _fold_duplicates(rows: list, data: TrendData, top_n: int, P: dict):
    """Walk the ranked candidates top-down. A story whose centroid is >= trend_fold_cos to an ALREADY LISTED story, and that
    overlaps it in time (<= trend_merge_gap_hours apart), is folded under it as 'also covered' instead of taking a slot.
    No transitive chaining (it only folds into a story that is itself listed), and nothing is hidden: the folded story's
    articles remain reachable from the listed story."""
    if data.vec is None or not len(rows):
        return rows[:top_n], {}
    pos = {sid: i for i, sid in enumerate(data.vec_ids)}
    kept: list = []
    folded: dict = {}
    gap = P["trend_merge_gap_hours"] * 3600.0
    for x in rows:
        r = x["story"]
        target = None
        i = pos.get(r.story_id)
        if i is not None and np.any(data.vec[i]):
            for k in kept:
                j = pos.get(k["story"].story_id)
                if j is None or not np.any(data.vec[j]):
                    continue
                g = max(r.first_ts - k["story"].last_ts, k["story"].first_ts - r.last_ts, 0.0)
                if g <= gap and float(data.vec[i] @ data.vec[j]) >= P["trend_fold_cos"]:
                    target = k
                    break
        if target is not None:
            folded.setdefault(target["story"].story_id, []).append(x)
        elif len(kept) < top_n:
            kept.append(x)
        elif not folded:       # list full: stop once nothing more needs folding
            pass
    return kept, folded


_WSHORT = {24: "24-hour", 72: "3-day", 168: "7-day"}


def reasons(s: dict, wlabel: str, W_h: float, scope_name: str, P: dict) -> list[str]:
    """Plain-language explanation built only from the numbers shown on the card (real counts, never a smoothed ratio)."""
    w, sg = s["in_window"], s["baseline"] or {}
    prev = wlabel.replace("last ", "")
    out = [f"{w['articles']} articles from {w['publishers']} publishers in the {wlabel}"
           + (f" ({s['n_articles']} in total since the first report)" if s["n_articles"] > w["articles"] else "") + "."]
    if sg.get("n_ref"):
        pb = sg["pct_beaten"]
        more = ("More coverage than every comparable story" if pb >= 0.999 else
                f"More coverage than {100 * pb:.0f}% of comparable stories" if pb >= 0.5 else "About typical coverage for a story")
        dom = s["taxonomy"]["domain"]["name"] if (sg.get("stratum", "").startswith("domain") and s["taxonomy"]["domain"] and s["taxonomy"]["domain"]["name"]) else None
        where = (f" in {scope_name}" if scope_name else "") + (f" ({dom})" if dom else "")
        out.append(f"{more}{where}: a typical story gets {sg['typical']:.0f} articles in a {_WSHORT.get(int(W_h), 'same-length')} window; "
                   f"the top 10% get {sg['p90']:.0f} or more.")
    a0, a1 = w["articles"], w["previous_articles"]
    if a1 == 0 and s["hours_since_first"] <= W_h:
        out.append(f"New story: nothing in the previous {prev}; first reported {_fmt_when(s['hours_since_first'])} before the latest data.")
    elif w["growth"] is not None:                         # supported only when the two windows hold >= 5 articles in total
        if a0 >= 1.2 * a1:
            out.append(f"Rising: {a0} articles vs {a1} in the previous {prev}" + (f" (×{a0 / a1:.1f})." if a1 else "."))
        elif a0 <= a1 / 1.2:
            out.append(f"Slowing: {a0} articles vs {a1} in the previous {prev} (×{a0 / a1:.1f}).")
        else:
            out.append(f"Steady: {a0} articles vs {a1} in the previous {prev}.")
    ev = s["evidence"]
    for it in ev["items"]:
        out.append(it["text"])
    if ev["available"] and not ev["items"]:
        out.append("None of the pipeline's statistical tests (emerging event, new development, topic surge) flagged it.")
    elif not ev["available"]:
        out.append("No statistical test results exist for this location, so no significance claim is made.")
    return out


def pipeline_evidence(st: Store, cands: list, as_of: pd.Timestamp, W_h: float, names: dict) -> dict:
    """Statistical evidence from the pipeline's own FDR-controlled tests, ONLY where those tests exist (registered
    country areas). Returns {story_id: {available, items:[{kind,text,q}]}}. Never invents a test."""
    reg = {r[0] for r in st.con.execute("SELECT iso2 FROM aoi_def WHERE kind = 'country' AND mode = 'event'").fetchall()}
    eids = sorted({e for c in cands for e in c["event_ids"]})
    start = as_of - pd.Timedelta(hours=W_h)
    out = {c["story_id"]: {"available": c["country"] in reg, "items": []} for c in cands}
    if not eids or not reg:
        return out
    st.con.register("_evd", pd.DataFrame({"event_id": eids}))
    try:
        em = st.df("SELECT event_id, min(q) q FROM event_aoi_test WHERE kind = 'emerging' AND alert_flag AND as_of > ? AND as_of <= ? "
                   "AND event_id IN (SELECT event_id FROM _evd) GROUP BY 1", [start, as_of + pd.Timedelta(days=1)])
        dv = st.df("SELECT event_id, min(shift_q) q, count(*) n FROM event_development WHERE alert_flag AND detected_asof > ? AND detected_asof <= ? "
                   "AND event_id IN (SELECT event_id FROM _evd) GROUP BY 1", [start, as_of + pd.Timedelta(days=1)])
    finally:
        st.con.unregister("_evd")
    sg = st.df("""SELECT replace(replace(aoi_id, 'country:', ''), ':event', '') AS iso, node_code, day, V, baseline_mu, surprise_q
                  FROM node_aoi_day WHERE level = 'topic' AND alert_flag AND aoi_id LIKE 'country:%:event' AND day >= ? AND day <= ?""",
               [start.date(), as_of.date()])
    emq, dvq = dict(zip(em["event_id"], em["q"])), dict(zip(dv["event_id"], dv["q"]))
    surge = {}
    for r in sg.itertuples():
        ratio = r.V / r.baseline_mu if r.baseline_mu else None
        cur = surge.get((r.iso, r.node_code))
        if ratio and (cur is None or ratio > cur[0]):
            surge[(r.iso, r.node_code)] = (ratio, r.day, r.surprise_q, r.V, r.baseline_mu)
    for c in cands:
        o = out[c["story_id"]]
        if not o["available"]:
            continue
        qs = [emq[e] for e in c["event_ids"] if e in emq]
        if qs:
            o["items"].append({"kind": "emerging", "q": float(min(qs)),
                               "text": f"Flagged as an emerging event by the pipeline's test (FDR q = {min(qs):.3f})."})
        qd = [dvq[e] for e in c["event_ids"] if e in dvq]
        if qd:
            o["items"].append({"kind": "development", "q": float(min(qd)),
                               "text": f"A new development inside this story was detected in the window (FDR q = {min(qd):.3f})."})
        hit = surge.get((c["country"], c["topic"]))
        if hit:
            ratio, day, q, v, mu = hit
            o["items"].append({"kind": "topic_surge", "q": float(q),
                               "text": f"Its topic “{names.get(c['topic'], c['topic'])}” ran ×{ratio:.1f} above this area's expected level on "
                                       f"{pd.Timestamp(day).date()} ({int(v)} articles vs {mu:.0f} expected; FDR q = {q:.3f})."})
    return out
