#!/usr/bin/env python3
"""Stage 23 - pilot validation report (read-only over the derived store; optional read-only PG check).

  python scripts/23_news_pilot_report.py                       # pilot database
  python scripts/23_news_pilot_report.py --validate-geo-live   # also compare precision proxy with GDELT ActionGeo_Type

Writes reports/news/pilot_report.md plus human-review samples (CSV):
  cluster_inspection_sample.csv   stratified clusters - fill `analyst_verdict`
  location_inspection_sample.csv  stratified located articles - fill `analyst_verdict`
These files contain headlines, so reports/news/ is git-ignored.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import news_config as C  # noqa: E402
from lib import news_geo as G  # noqa: E402
from lib.news_store import Store  # noqa: E402

IS_VOLUME, CONTAMINATED = 0.7, 0.3          # thresholds of scripts/audit_restatement.py (GDELT project's test)
DECLARED = {"V_rate": "by construction: V / exposure", "share_V": "by construction: V / (V + exposure)",
            "burst_intensity": "level of the Kleinberg/CUSUM burst state, which is a function of the counts: measured "
                               "Spearman ~0.8 vs V in the pilot. CONTEXT ONLY - never ranked or alerted on; the "
                               "binary burst_state / episode is the published signal"}
CANDIDATES = ["V_rate", "share_V", "growth", "growth_7d", "momentum", "accel", "accel_smooth", "surprise",
              "burst_intensity", "publisher_pie"]


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or not len(df):
        return "_(none)_\n"
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "|".join("---" for _ in d.columns) + "|\n"
    return head + "\n".join("| " + " | ".join("" if pd.isna(x) else str(x).replace("|", "/") for x in r) + " |"
                            for r in d.itertuples(index=False)) + "\n"


def restatement_audit(panel: pd.DataFrame) -> pd.DataFrame:
    """Spearman (leading) and Pearson of each published metric against V, support-gated rows only."""
    g = panel[panel["support_ok"] & panel["V"].notna()]
    rows = []
    for m in CANDIDATES:
        x = g[["V", m]].dropna()
        if len(x) < 30 or x[m].nunique() < 3:
            rows.append({"metric": m, "n": len(x), "spearman": np.nan, "pearson": np.nan, "verdict": "insufficient n"})
            continue
        sp = float(spearmanr(x["V"], x[m]).statistic)
        pe = float(np.corrcoef(x["V"], x[m])[0, 1])
        v = "volume (declared)" if (abs(sp) > IS_VOLUME and m in DECLARED) else \
            "UNDECLARED VOLUME RESTATEMENT" if abs(sp) > IS_VOLUME else \
            "contaminated" if abs(sp) > CONTAMINATED else "clean"
        rows.append({"metric": m, "n": len(x), "spearman": sp, "pearson": pe, "verdict": v})
    return pd.DataFrame(rows)


def pairs(n):
    return n * (n - 1) / 2


def silver_check(st: Store) -> dict:
    """GDELT globaleventid shared by >=2 articles is a WEAK label for 'same event'."""
    df = st.df("SELECT cluster_id, gdelt_event_id gid FROM article_fact WHERE NOT is_duplicate "
               "AND cluster_id IS NOT NULL AND gdelt_event_id IS NOT NULL")
    gsz = df.groupby("gid").size()
    gsz = gsz[gsz >= 2]
    df = df[df["gid"].isin(gsz.index)]
    if not len(df):
        return {}
    same_id = float(sum(pairs(n) for n in gsz))
    csz = df.groupby("cluster_id").size()
    same_cluster = float(sum(pairs(n) for n in csz))
    both = float(sum(pairs(n) for n in df.groupby(["cluster_id", "gid"]).size()))
    mid = float(df.groupby("cluster_id")["gid"].nunique().loc[lambda x: df.groupby("cluster_id").size() >= 2].median())
    return {"median_ids_per_cluster": mid, "silver_articles": int(len(df)), "silver_events": int(len(gsz)),
            "recall_pairs": both / same_id if same_id else np.nan,
            "strict_precision_pairs": both / same_cluster if same_cluster else np.nan,
            "same_id_pairs": same_id, "same_cluster_pairs": same_cluster, "both": both}


def cluster_sample(st: Store, n_per_bin: int = 14, seed: int = 7) -> pd.DataFrame:
    ev = st.df("SELECT event_id, n_articles, n_publishers, n_copies, loc_status, loc_country_iso, loc_adm1_code, "
               "rep_title FROM event_cluster WHERE is_event")
    ev["size_bin"] = pd.cut(ev["n_articles"], [2, 5, 15, 10 ** 6], labels=["3-5", "6-15", "16+"])
    parts = [g.sample(min(n_per_bin, len(g)), random_state=seed) for _, g in ev.groupby("size_bin", observed=True)]
    s = pd.concat(parts).reset_index(drop=True)
    rows = []
    for r in s.itertuples():
        m = st.df("SELECT title, source_domain, cluster_sim FROM article_fact WHERE cluster_id = ? AND NOT is_duplicate "
                  "ORDER BY published_at", [r.event_id])
        low = m.sort_values("cluster_sim").dropna(subset=["cluster_sim"]).head(3)
        rows.append({"event_id": r.event_id, "size_bin": str(r.size_bin), "n_articles": r.n_articles,
                     "n_publishers": r.n_publishers, "location": f"{r.loc_status}:{r.loc_country_iso or ''}",
                     "first_headline": m["title"].iloc[0],
                     "least_similar_headlines": " || ".join(low["title"].str.slice(0, 110)),
                     "sample_headlines": " || ".join(m["title"].sample(min(4, len(m)), random_state=1).str.slice(0, 110)),
                     "analyst_verdict": ""})
    return pd.DataFrame(rows)


def location_sample(st: Store, geo: G.GeoIndex, seed: int = 7) -> pd.DataFrame:
    df = st.df("""SELECT f.title, f.source_domain, g.ev_place, g.ev_country_iso, g.ev_adm1_code, g.ev_precision,
                         g.geo_inconsistent, g.ev_lat, g.ev_lon, f.url
                  FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                  WHERE NOT f.is_duplicate AND g.loc_status='located'""")
    parts = []
    for prec, n in (("city", 30), ("adm1", 15), ("country", 15)):
        g = df[df.ev_precision == prec]
        parts.append(g.sample(min(n, len(g)), random_state=seed))
    s = pd.concat(parts).reset_index(drop=True)
    s["adm1_name"] = s["ev_adm1_code"].map(lambda c: geo.admin1_by_code(c).name if c and geo.admin1_by_code(c) else None)
    s["analyst_verdict"] = ""
    return s[["title", "source_domain", "ev_place", "ev_country_iso", "adm1_name", "ev_precision", "geo_inconsistent",
              "url", "analyst_verdict"]]


def geo_live_validation(st: Store, n: int = 400, seed: int = 3) -> dict:
    """Compare (a) my name-based precision proxy and (b) OS vs PG country with PostgreSQL's raw GDELT
    ActionGeo_Type / ActionGeo_CountryCode. Read-only SELECT by globaleventid."""
    from lib import news_sources as NS
    ids = st.df("SELECT DISTINCT gdelt_event_id gid, ev_precision prec, ev_country_iso iso, ev_place FROM article_fact f "
                "JOIN article_geo g USING (source_table, article_id) WHERE gdelt_event_id IS NOT NULL "
                "AND g.loc_status='located'")
    ids = ids.sample(min(n, len(ids)), random_state=seed)
    out = {"requested": int(len(ids))}
    try:
        with NS.ReadOnlyPG(statement_timeout_s=90) as pg:
            rows = pg.query("SELECT globaleventid, actiongeo_type, actiongeo_countrycode, actiongeo_fullname "
                            "FROM raw.gdelt_events WHERE globaleventid = ANY(%s)", ([str(x) for x in ids["gid"]],))
    except Exception as e:  # a timeout or missing index is a finding, not a crash
        out["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        return out
    pgdf = pd.DataFrame(rows)
    out["found_in_pg"] = int(len(pgdf))
    if not len(pgdf):
        return out
    pgdf["gid"] = pgdf["globaleventid"].astype(str)
    m = ids.assign(gid=ids["gid"].astype(str)).merge(pgdf, on="gid")
    tmap = {"1": "country", "2": "adm1", "3": "city", "4": "city", "5": "adm1"}
    m["pg_prec"] = m["actiongeo_type"].astype(str).map(tmap)
    m = m[m["pg_prec"].notna()]
    out["precision_confusion"] = pd.crosstab(m["pg_prec"].rename("PG ActionGeo_Type"), m["prec"].rename("proxy")).to_dict()
    out["precision_agreement"] = float((m["pg_prec"] == m["prec"]).mean()) if len(m) else np.nan
    out["fullname_equal_os_pg"] = float((m["ev_place"].fillna("") == m["actiongeo_fullname"].fillna("")).mean())
    out["n_compared"] = int(len(m))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(C.NEWS_REPORT_DIR))
    ap.add_argument("--validate-geo-live", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    st = Store(read_only=True)
    geo = G.GeoIndex()
    L: list[str] = []
    w = L.append
    run = st.df("SELECT * FROM run_log ORDER BY started_at DESC").iloc[0]
    d0, d1 = st.scalar("SELECT min(day) FROM article_fact"), st.scalar("SELECT max(day) FROM article_fact")
    w(f"# News engine - pilot validation report\n\nSchema `{st.schema}` - published days {d0} .. {d1} - "
      f"generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}. Source PostgreSQL / OpenSearch were only read.\n")

    # ---- 1 coverage -------------------------------------------------------------------------------------------
    q = {r.key: r.value for r in st.df("SELECT key, value FROM data_quality WHERE run_id IN (SELECT run_id FROM run_log "
                                       "WHERE mode='pilot' ORDER BY started_at LIMIT 1)").itertuples()}
    A_ = st.df("SELECT count(*) n, sum(is_duplicate::INT) dup, sum((NOT has_label)::INT) unl FROM article_fact").iloc[0]
    G_ = st.df("SELECT count(*) n, sum((loc_status='located')::INT) n_loc, sum(geo_inconsistent::INT) n_inc, "
               "sum(country_code_unmapped::INT) n_unm FROM article_geo g JOIN article_fact f USING (source_table, article_id) "
               "WHERE NOT f.is_duplicate").iloc[0]
    n_bad_date = int(st.scalar("SELECT count(*) FROM excluded_article WHERE reason <> 'date_conflict'") or 0)
    w("## 1. Data quality and coverage\n")
    w(f"* documents seen in scope: **{q.get('docs_seen', '?')}**; ingested **{int(A_.n):,}**; excluded for date "
      f"conflict: **{q.get('excluded_date_conflict', 0)}**; missing/invalid date: "
      f"**{n_bad_date}** "
      "(`scraped_at` is never substituted)\n"
      f"* exact duplicates (same `content_hash`): **{int(A_.dup):,}** ({A_.dup / A_.n:.1%}) - kept, flagged, not counted\n"
      f"* not yet labelled in PostgreSQL: **{int(A_.unl):,}** ({A_.unl / A_.n:.1%}) - used for events, excluded from taxonomy series\n"
      f"* event location available (non-duplicate articles): **{int(G_.n_loc):,} of {int(G_.n):,} ({G_.n_loc / G_.n:.1%})** - "
      f"the rest are `unlocated` and shown as such; geo-inconsistent (FIPS vs polygon disagree): {int(G_.n_inc)}; "
      f"unmapped country code: {int(G_.n_unm)}\n")
    w("Corpus-wide event-location coverage per day (OpenSearch counts, whole corpus - NOT only this scope):\n")
    cd = st.df("SELECT day, n_dated_total, n_action_country, n_action_geo FROM corpus_day ORDER BY day")
    if len(cd):
        cd["action_country_%"] = (100 * cd.n_action_country / cd.n_dated_total).round(1)
        cd["action_geo_%"] = (100 * cd.n_action_geo / cd.n_dated_total).round(1)
        w(md(cd))
    w("Date provenance of ingested articles:\n")
    w(md(st.df("SELECT coalesce(date_confidence,'(none)') confidence, coalesce(date_source,'(none)') AS date_src, count(*) n "
               "FROM article_fact GROUP BY 1,2 ORDER BY 3 DESC")))
    lag = st.df("SELECT quantile_cont(lag_hours,[0.5,0.9,0.99]) q FROM article_fact WHERE lag_hours IS NOT NULL")["q"][0]
    w(f"Ingest lag (OpenSearch `ingested_at` - `published_at`): median {lag[0]:.1f} h, p90 {lag[1]:.1f} h, p99 {lag[2]:.1f} h. "
      f"PostgreSQL-vs-OpenSearch publication-date mismatches >1 h: {q.get('date_mismatch_pg_vs_os', 0)}. "
      f"Documents present in both OpenSearch backing indices: {q.get('doc_in_two_indices', 0)}.\n")

    # ---- 2 clustering -----------------------------------------------------------------------------------------
    w("## 2. Event clustering\n")
    sz = st.df("SELECT n_articles, count(*) c FROM event_cluster GROUP BY 1 ORDER BY 1")
    tot_cl, tot_ev = int(sz.c.sum()), int(st.scalar("SELECT count(*) FROM event_cluster WHERE is_event"))
    single = int(sz[sz.n_articles == 1].c.sum())
    w(f"* clusters: **{tot_cl:,}**; singletons **{single:,} ({single / tot_cl:.1%})**; events (>= {C.PARAMS['event_min_articles']} "
      f"articles from >= {C.PARAMS['event_min_publishers']} publishers): **{tot_ev:,}**\n")
    bins = st.df("""SELECT CASE WHEN n_articles=1 THEN '1' WHEN n_articles=2 THEN '2' WHEN n_articles<=5 THEN '3-5'
                    WHEN n_articles<=15 THEN '6-15' WHEN n_articles<=50 THEN '16-50' ELSE '>50' END AS size_bin, count(*) clusters,
                    sum(n_articles) articles, round(avg(n_publishers),1) avg_publishers FROM event_cluster GROUP BY 1""")
    bins["o"] = bins["size_bin"].map({"1": 0, "2": 1, "3-5": 2, "6-15": 3, "16-50": 4, ">50": 5})
    w(md(bins.sort_values("o").drop(columns="o")))
    mx = st.df("SELECT n_articles, n_publishers, n_copies, left(rep_title,90) first_headline FROM event_cluster "
               "ORDER BY n_articles DESC LIMIT 5")
    w("Largest clusters (a very large cluster may merge related sub-stories - check these first):\n")
    w(md(mx))
    sv = silver_check(st)
    if sv:
        w(f"**Silver-label check** (GDELT `globaleventid` shared by >=2 articles = weak 'same event' label; "
          f"{sv['silver_articles']:,} articles / {sv['silver_events']:,} GDELT events): pairwise recall "
          f"**{sv['recall_pairs']:.3f}**, strict precision **{sv['strict_precision_pairs']:.3f}** (a lower bound: GDELT splits one "
          "real event into several ids). This is NOT comparable with the earlier experiment's "
          "numbers (different pair definition and sample). Low strict precision is expected when one real story is "
          "reported under many GDELT ids (median distinct GDELT ids per multi-article silver cluster: "
          f"{sv['median_ids_per_cluster']:.0f}). Silver labels are weak evidence - **human review of "
          "`cluster_inspection_sample.csv` is the real validation and has NOT been done by you yet.**\n")
    cs = cluster_sample(st)
    cs.to_csv(out / "cluster_inspection_sample.csv", index=False)
    w(f"Wrote `cluster_inspection_sample.csv` ({len(cs)} clusters, stratified by size) with an empty `analyst_verdict` "
      "column (SAME_EVENT / MIXED / DIFFERENT_EVENTS).\n")

    # ---- 3 location -------------------------------------------------------------------------------------------
    w("## 3. Location accuracy (event location, GDELT ActionGeo)\n")
    pr = st.df("SELECT coalesce(ev_precision,'unlocated') AS loc_precision, count(*) n FROM article_geo g JOIN article_fact f "
               "USING (source_table, article_id) WHERE NOT f.is_duplicate GROUP BY 1 ORDER BY 2 DESC")
    w(md(pr))
    ev = st.df("SELECT loc_status, count(*) n FROM event_cluster WHERE is_event GROUP BY 1 ORDER BY 2 DESC")
    w("Event clusters by location status (consensus of located members; `ambiguous` = members disagree):\n")
    w(md(ev))
    conf = st.df("SELECT coalesce(loc_confidence,'(not located)') AS confidence, count(*) events FROM event_cluster WHERE is_event "
                 "GROUP BY 1 ORDER BY 2 DESC")
    mc = st.df("""WITH c AS (SELECT cluster_id, count(DISTINCT ev_country_iso) k, count(ev_country_iso) n FROM article_fact f
                   JOIN article_geo g USING (source_table, article_id) WHERE NOT f.is_duplicate AND cluster_id IS NOT NULL GROUP BY 1)
                 SELECT count(*) FILTER (WHERE n >= 2) multi_located, count(*) FILTER (WHERE n >= 2 AND k >= 2) disagree FROM c""").iloc[0]
    w("Event-location **confidence** (high = >=3 located members and >=80% agree; only high-confidence locations can raise "
      "AOI alerts):\n")
    w(md(conf))
    w(f"Of {int(mc.multi_located):,} clusters with >=2 located articles, **{int(mc.disagree):,} ({mc.disagree / max(mc.multi_located, 1):.1%}) "
      "have members geocoded to more than one country** - GDELT ActionGeo is noisy. Example found in the pilot: a Delhi University "
      "story whose members were geocoded to Dhaka (3 of 4 members). No cheap automatic validator exists: mention-support is circular "
      "(99.8% agreement, same geocoder) and a headline place-name check was measured and REJECTED (8.3% of located articles flagged, "
      "mostly legitimate cross-border stories).\n")
    ls = location_sample(st, geo)
    ls.to_csv(out / "location_inspection_sample.csv", index=False)
    w(f"Wrote `location_inspection_sample.csv` ({len(ls)} located articles: 30 city, 15 adm1, 15 country) with an empty "
      "`analyst_verdict` column.\n")
    if a.validate_geo_live:
        gv = geo_live_validation(st)
        w("Cross-check against PostgreSQL `raw.gdelt_events` (read-only):\n")
        if "error" in gv:
            w(f"* could not run: `{gv['error']}`\n")
        else:
            w(f"* {gv.get('found_in_pg', 0)} of {gv['requested']} sampled GDELT events found; precision-proxy agreement "
              f"with `ActionGeo_Type` **{gv.get('precision_agreement', float('nan')):.1%}** (n={gv.get('n_compared', 0)}); "
              f"OpenSearch `action_location` == PG `ActionGeo_FullName`: {gv.get('fullname_equal_os_pg', float('nan')):.1%}\n"
              f"* confusion (rows = PG type, cols = proxy): `{gv.get('precision_confusion')}`\n")
    # AOI coverage
    w("AOI coverage on the last pilot day (scope = everything ingested; unlocated shown, never hidden):\n")
    w(md(st.df("""SELECT d.aoi_id, d.n_articles, d.n_located_event, d.n_mention_only, d.n_scope_unlocated, d.n_scope_total,
                  round(100.0*d.n_scope_unlocated/nullif(d.n_scope_total,0),1) pct_scope_unlocated
                  FROM aoi_day d WHERE d.day=(SELECT max(day) FROM aoi_day) ORDER BY d.n_articles DESC LIMIT 8""")))

    # ---- 4 metrics ----------------------------------------------------------------------------------------------
    w("## 4. Trending metrics\n")
    ps = st.df("""SELECT a.aoi_id, a.kind, a.mode, count(DISTINCT s.node_code||s.level) series,
                  sum(CASE WHEN s.support_ok THEN 1 ELSE 0 END) supported_days,
                  count(*) FILTER (WHERE s.V IS NULL) gap_rows
                  FROM aoi_def a LEFT JOIN node_aoi_day s USING (aoi_id) GROUP BY 1,2,3 ORDER BY series DESC""")
    sup = ps[ps.series > 0]
    w(f"{len(sup)} of {len(ps)} registered AOIs reach the minimum evidence for node series "
      f"(>= {C.PARAMS['aoi_min_total_articles']} labelled in-scope articles); the others report NULL with a reason, never zeros. "
      f"Baseline needs {C.PARAMS['baseline_min_days']} days, so with a 14-day pilot only days 8-14 can be tested "
      "(**limitation**: production should use the configured 28-day window with >= 14 baseline days).\n")
    w(md(sup.head(12)))
    panel = st.df("SELECT * FROM node_aoi_day")
    w("**Anti-restatement audit** (Spearman vs V, support-gated rows; thresholds of `scripts/audit_restatement.py`):\n")
    au = restatement_audit(panel)
    w(md(au))
    bad = au[au.verdict == "UNDECLARED VOLUME RESTATEMENT"]
    w("Verdict: " + ("**FAIL - undeclared volume restatement(s): " + ", ".join(bad.metric) + "**\n" if len(bad)
                      else "no undeclared volume restatements.\n"))

    # ---- 5 alerts -----------------------------------------------------------------------------------------------
    w("## 5. Alerts (FDR-controlled)\n")
    w(md(st.df("SELECT kind, status, count(*) n, round(min(q),5) min_q, round(max(q),4) max_q FROM alert GROUP BY 1,2 ORDER BY 1,2")))
    et = st.df("SELECT support_ok, count(*) n FROM event_aoi_test WHERE kind='emerging' GROUP BY 1")
    w("Emerging-event tests by support (`support_ok=false` = fewer than "
      f"{C.PARAMS['emerging_min_ref']} reference clusters; result NULL, no alert):\n")
    w(md(et))
    dv = st.df("SELECT count(*) tested, sum(alert_flag::INT) flagged, count(DISTINCT event_id) events FROM event_development")
    w(f"New-development tests: {int(dv.tested[0])} (event, AOI, evaluation-time) tests over {int(dv.events[0])} events, "
      f"{int(dv.flagged[0])} flagged. The statistic was calibrated on synthetic data (false positive <4%, planted sub-story "
      "found >95%) **but the real-data flag rate is high because large events genuinely evolve; it needs your review.**\n")
    w("Top active alerts per kind (rationale is the audit trail):\n")
    for kind in ("emerging_event", "new_development", "node_trend"):
        t = st.df("SELECT aoi_id, round(q,5) q, severity, left(rationale,190) rationale FROM alert WHERE kind=? "
                  "AND status='active' ORDER BY q, effect DESC LIMIT 5", [kind])
        w(f"\n**{kind}**\n")
        w(md(t))
    w("\n## 6. Limitations (read before approving a backfill)\n")
    for s in [
        "Event location is GDELT ActionGeo for ~40% of articles; the rest are `unlocated` and cannot be in any event-location AOI. "
        "Bangladesh has very few event-located articles; its AOI mostly reports NULL with a reason.",
        "GDELT ActionGeo is itself machine-extracted and may be wrong; accuracy needs the human check of the location sample.",
        "Mentioned-in AOIs are country-level only and are a different population from event-location AOIs.",
        "Cluster quality is unvalidated by humans; singletons dominate (most articles are not part of a multi-publisher event).",
        "A 14-day window leaves only 7 testable days per node series; emerging-event references are limited to <= 14 days.",
        "Late-arriving data revises node series; revised alerts are marked `retracted`, never deleted.",
        "Natural Earth boundaries are de facto lines, not an authoritative position on disputed territory; GDELT points that fall "
        "in disputed areas may be flagged `geo_inconsistent` and demoted to country precision.",
        "No publisher-location filter exists yet (see the publisher registry review file).",
    ]:
        w(f"* {s}")
    (out / "pilot_report.md").write_text("\n".join(L))
    print(f"wrote {out / 'pilot_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
