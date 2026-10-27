"""News metrics at the taxonomy-node x AOI x day grain.

Statistical primitives come from `stats_v2.py` (negative-binomial baseline, drift factor,
exceedance probability, BH-FDR, causal smoothing, burst detection, AR(1), onset, and decay).
This module adapts them to News data and defines News-specific metrics.

Modeling choices
  * the baseline of an AOI series is fitted on that AOI's own history and exposure; a locally
    quiet topic is not judged against global volume
  * exposure ("effort") = labelled, in-scope articles located in the AOI that day that are NOT in
    the node being tested (leave-one-out, so a node's own spike cannot inflate its own baseline).
    Collector ramps still cancel. `share_V` = V / (V + exposure) = share of the AOI total
  * a day whose exposure is below `min_exposure_day` is a DATA GAP: its row is emitted with every
    metric NULL. It is never treated as zero volume, and it is skipped by baselines
  * out-of-scope taxonomy (domain D00) and unlabelled articles are not in node series
  * support gates: NULL + support_ok=false rather than a number below minimum evidence
  * FDR: one Benjamini-Hochberg family per (AOI, day), pooled across every node and level tested
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
import pandas as pd

from . import news_config as C
from . import news_geo as G
from . import news_aoi as A
from . import stats_v2 as S
from .news_store import Store, now_utc

NODE_COL = {"domain": "domain_id", "topic": "topic_id", "category": "category_id", "subtopic": "subtopic_id"}
OUT_OF_SCOPE_DOMAIN = "D00"

NODE_COLS = ["aoi_id", "level", "node_code", "day", "in_window", "V", "exposure", "V_rate", "share_V", "growth",
             "growth_7d", "accel", "accel_smooth", "momentum", "baseline_mu", "baseline_theta", "baseline_days",
             "baseline_mean", "baseline_dist", "baseline_method", "surprise", "surprise_p", "surprise_q",
             "alert_flag", "burst_state", "burst_intensity", "burst_method", "burst_support_ok",
             "burst_episode_id", "n_publishers", "publisher_pie", "support_n", "support_ok"]


def hurlbert_pie(counts) -> float:
    """Hurlbert's PIE = N/(N-1) * (1 - sum p_i^2): probability two articles drawn without
    replacement come from different publishers. NaN (never 0) when N < 2. It is size-unbiased."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    N = c.sum()
    if N < 2:
        return float("nan")
    return float(N / (N - 1.0) * (1.0 - np.sum((c / N) ** 2)))


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(x).ewm(span=span, adjust=False, min_periods=1).mean().to_numpy()


def series_panel(V: np.ndarray, E: np.ndarray, P: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Metrics for ONE series over consecutive VALID days. Pure function (unit-tested).

    Returns (frame with one row per day, burst episodes in index space). No look-ahead: the baseline
    for day t is fitted on days strictly before t.
    """
    V = np.asarray(V, dtype=float)
    E = np.clip(np.asarray(E, dtype=float), 1.0, None)
    n = len(V)
    a = P["alpha_count"]
    Es = pd.Series(E).rolling(max(1, int(P.get("exposure_smooth_days", 7))), min_periods=1).mean().to_numpy()
    out = pd.DataFrame({"V": V, "exposure": E})
    out["V_rate"] = V / E
    lr = np.log(V + a) - np.log(E)
    out["log_rate"] = lr
    growth = np.r_[np.nan, np.diff(lr)]
    out["growth"] = growth
    L = P["growth_7d_lag"]
    v7 = pd.Series(V).rolling(L, min_periods=1).sum().to_numpy()
    e7 = pd.Series(E).rolling(L, min_periods=1).sum().to_numpy()
    r7 = np.log(v7 + a) - np.log(e7)
    out["growth_7d"] = r7 - pd.Series(r7).shift(L).to_numpy()
    gz = np.nan_to_num(growth, nan=0.0)
    out["momentum"] = _ema(gz, P["ema_fast"]) - _ema(gz, P["ema_slow"])
    out["accel"] = np.r_[np.nan, np.diff(growth)]
    out["accel_smooth"] = S.causal_savgol_deriv(lr, window=P["savgol_window"], order=P["savgol_order"], deriv=2)

    mu_adj = np.full(n, np.nan); theta = np.full(n, np.nan); bdays = np.zeros(n, dtype=int)
    bmean = np.full(n, np.nan); p_ex = np.full(n, np.nan); p_mid = np.full(n, np.nan)
    dist: list[Optional[str]] = [None] * n
    method: list[Optional[str]] = [None] * n
    fits = S.trailing_baseline_fits(V, window=P["baseline_window"], min_days=P["baseline_min_days"],
                                    min_mean=P["baseline_min_mean"])
    for t, f in enumerate(fits):
        if f is None or not f.ok:
            continue
        lo = max(0, t - P["baseline_window"])
        scale = 1.0
        if P["nb_use_exposure_offset"] and t > lo:
            # SMOOTHED exposure (trailing mean incl. day t): the offset exists to cancel slow collector
            # ramps, and a raw day-to-day exposure adds its own sampling noise to the baseline -
            # measured: null rejection 9.2% at ~40 articles/day, 7.1% at ~100 (nominal 5%).
            e_base = float(np.nanmean(Es[lo:t]))
            if np.isfinite(e_base) and e_base > 0:
                scale = float(np.clip(Es[t] / e_base, 1.0 / P["exposure_scale_cap"], P["exposure_scale_cap"]))
        if P["nb_use_drift_adjust"]:
            scale *= S.drift_factor(V[lo:t])
        th = f.theta * scale if np.isfinite(f.theta) else f.theta
        fa = S.NBFit(f.mu * scale, th, f.dist, f.method, f.n, f.var)
        mu_adj[t], theta[t], bdays[t], bmean[t] = fa.mu, fa.theta, f.n, f.mu
        dist[t], method[t] = f.dist, f.method
        p_ex[t] = S.exceedance_p(V[t], fa, midp=False)
        p_mid[t] = S.exceedance_p(V[t], fa, midp=True)
    out["baseline_mu"], out["baseline_theta"], out["baseline_days"] = mu_adj, theta, bdays
    out["baseline_mean"], out["baseline_dist"], out["baseline_method"] = bmean, dist, method
    out["surprise_p"] = p_ex
    out["_p_fdr"] = p_mid if P["fdr_pvalue_kind"] == "midp" else p_ex
    out["surprise"] = S.surprise_from_p(p_ex)
    out["support_n"] = V
    out["support_ok"] = ((bdays >= P["baseline_min_days"]) & (np.nan_to_num(bmean) >= P["baseline_min_mean"])
                         & (V >= P["support_min_v"]))

    burst_ok = bool(n >= P["baseline_min_days"] and float(np.mean(V)) >= P["baseline_min_mean"])
    out["burst_support_ok"] = burst_ok
    out["burst_state"] = np.nan
    out["burst_intensity"] = np.nan
    out["burst_method"] = None
    episodes: list[dict] = []
    if burst_ok:
        br = S.detect_bursts(V, baseline_mu=mu_adj, min_days_kleinberg=P["burst_min_days_kleinberg"],
                             min_total_kleinberg=P["burst_min_total_kleinberg"],
                             min_mean_kleinberg=P["burst_min_mean_kleinberg"], s=P["burst_s"],
                             gamma=P["burst_gamma"], cusum_h=P["burst_cusum_h"])
        out["burst_state"] = br.state
        out["burst_intensity"] = br.intensity
        out["burst_method"] = br.method
        for ep in S.extract_episodes(br.state, V, br.intensity):
            rise_lo = max(0, ep["start_i"] - 1)
            onset = S.fit_logistic_onset(V[rise_lo: ep["peak_i"] + 1])
            decay = S.fit_exp_decay(V[ep["peak_i"]: min(n, ep["end_i"] + 8)])
            episodes.append({**ep, "burst_method": br.method, "onset_slope": onset["onset_slope"],
                             "half_life_days": decay["half_life_days"], "decay_ok": decay["ok"]})
    # support gate blanks the trend/evidence family (NULL, never a guess)
    blank = ["surprise", "surprise_p", "accel", "accel_smooth", "momentum", "growth", "growth_7d"]
    out.loc[~out["support_ok"].to_numpy(), blank] = np.nan
    return out, episodes


def _expand(valid_days: list, calendar: list, part: pd.DataFrame, V_all: np.ndarray, E_all: np.ndarray) -> pd.DataFrame:
    """Place metrics computed on valid days back onto the full calendar; data-gap days get NULLs
    (V and exposure are kept so the gap is visible and explainable)."""
    full = pd.DataFrame({"day": calendar})
    part = part.copy()
    part["day"] = valid_days
    full = full.merge(part, on="day", how="left", suffixes=("_x", ""))
    full["exposure"] = full["exposure"].where(full["exposure"].notna(), E_all)
    full["V"] = full["V"].where(full["V"].notna(), np.nan)  # gap day: V unknown, NOT zero
    full["support_ok"] = full["support_ok"].fillna(False).astype(bool)
    full["burst_support_ok"] = full["burst_support_ok"].fillna(False).astype(bool)
    return full


def compute_aoi_panel(store: Store, aoi: G.AOI, P: dict, run_id: str,
                      day_from=None, day_to=None, persist: bool = True) -> dict:
    """Build (and, if `persist`, store) the node series for one AOI. Idempotent: the AOI's previous
    rows are replaced in one transaction. With persist=False nothing is written (read-only store OK) and
    the results are returned under summary['_data'] - this is how ad-hoc AOIs from the API are computed
    with the SAME code as registered ones. Returns a summary dict (also used by the pilot report)."""
    fr = A.aoi_frame(store, aoi, day_from, day_to)
    summary = {"aoi_id": aoi.aoi_id, "n_frame": int(len(fr)), "supported": False, "reason": None,
               "n_series": 0, "n_series_ts": 0, "n_tests": 0, "n_alerts": 0, "gap_days": 0}
    lab = fr[fr["has_label"].fillna(False) & fr["topic_id"].notna() & (fr["domain_id"] != OUT_OF_SCOPE_DOMAIN)]
    summary["n_labelled_in_scope"] = int(len(lab))
    summary["n_unlabelled_or_out_of_scope"] = int(len(fr) - len(lab))
    if len(lab) < P["aoi_min_total_articles"]:
        summary["reason"] = f"only {len(lab)} labelled in-scope articles (< {P['aoi_min_total_articles']})"
        if persist:
            _clear_aoi(store, aoi.aoi_id, with_tx=True)
        return summary
    lab = lab.copy()
    lab["day"] = pd.to_datetime(lab["day"]).dt.date
    calendar = [d.date() for d in pd.date_range(lab["day"].min(), lab["day"].max(), freq="D")]
    E_all = lab.groupby("day").size().reindex(calendar).fillna(0).to_numpy(dtype=float)
    valid = E_all >= P["min_exposure_day"]
    summary["gap_days"] = int((~valid).sum())
    if valid.sum() < 2:
        summary["reason"] = "fewer than 2 days with sufficient exposure"
        if persist:
            _clear_aoi(store, aoi.aoi_id, with_tx=True)
        return summary
    vdays = [d for d, ok in zip(calendar, valid) if ok]
    Ev = E_all[valid]
    summary["supported"] = True

    frames, series_rows, ep_rows = [], [], []
    for level in P["node_levels"]:
        col = NODE_COL[level]
        counts = lab.groupby([col, "day"]).size().rename("V").reset_index()
        tot = counts.groupby(col)["V"].sum()
        pub = lab.groupby([col, "day", "source_domain"], dropna=False).size().rename("k").reset_index()
        for node, total in tot.items():
            summary["n_series"] += 1
            if total < P["node_min_total_v"]:
                continue
            c = counts[counts[col] == node].set_index("day")["V"]
            Vfull = c.reindex(calendar).fillna(0).to_numpy(dtype=float)   # valid days: a real zero
            Vv = Vfull[valid]
            # Leave this node out of its exposure; otherwise a spike inflates its own baseline.
            Eo = np.clip(Ev - Vv, 1.0, None)
            panel, eps = series_panel(Vv, Eo, P)
            summary["n_series_ts"] += 1
            full = _expand(vdays, calendar, panel, Vfull, np.clip(E_all - Vfull, 1.0, None))
            full.loc[~full["day"].isin(vdays), "V"] = np.nan
            pk = pub[pub[col] == node].groupby("day")["k"]
            npub = pk.size().reindex(calendar)
            pie = pk.apply(lambda s: hurlbert_pie(s.to_numpy())).reindex(calendar)
            full["n_publishers"] = npub.to_numpy()
            full["publisher_pie"] = pie.to_numpy()
            full.loc[~full["day"].isin(vdays), ["n_publishers", "publisher_pie"]] = np.nan
            full["aoi_id"], full["level"], full["node_code"] = aoi.aoi_id, level, node
            frames.append(full)
            fit = S.fit_ar1(panel["log_rate"].to_numpy(), min_obs=P["ar1_min_obs"], lb_lags=P["ljung_box_lags"])
            series_rows.append({"aoi_id": aoi.aoi_id, "level": level, "node_code": node, "total_v": float(total),
                                "n_days": int(len(vdays)), "ar1_phi": fit.phi, "ar1_half_life_days": fit.half_life_days,
                                "ar1_ljung_box_p": fit.ljung_box_p, "ar1_n_obs": int(fit.n_obs), "ar1_fit_ok": bool(fit.ok)})
            for k, ep in enumerate(eps):
                s_day, e_day, p_day = vdays[ep["start_i"]], vdays[ep["end_i"]], vdays[ep["peak_i"]]
                ep_rows.append({"episode_id": f"{aoi.aoi_id}|{level}|{node}|{s_day}", "aoi_id": aoi.aoi_id,
                                "level": level, "node_code": node, "start_day": s_day, "peak_day": p_day,
                                "end_day": e_day, "peak_V": ep["peak_V"], "total_V": ep["total_V"],
                                "n_days": ep["n_days"], "level_k": None if not np.isfinite(ep["level"]) else int(ep["level"]),
                                "burst_method": ep["burst_method"], "onset_slope": ep["onset_slope"],
                                "half_life_days": ep["half_life_days"], "decay_ok": ep["decay_ok"]})
                full.loc[(full["day"] >= s_day) & (full["day"] <= e_day), "burst_episode_id"] = ep_rows[-1]["episode_id"]
    if not frames:
        summary["reason"] = "no node reached node_min_total_v"
        if persist:
            _clear_aoi(store, aoi.aoi_id, with_tx=True)
        return summary
    panel_all = pd.concat(frames, ignore_index=True)
    if "burst_episode_id" not in panel_all.columns:
        panel_all["burst_episode_id"] = None
    # ---- FDR: one BH family per (AOI, day), pooled over every node/level tested that day
    panel_all["surprise_q"] = np.nan
    gated = panel_all["support_ok"] & panel_all["_p_fdr"].notna()
    for d, idx in panel_all[gated].groupby("day").groups.items():
        panel_all.loc[list(idx), "surprise_q"] = S.bh_qvalues(
            panel_all.loc[list(idx), "_p_fdr"].to_numpy(dtype=float), method=P["fdr_method"])
        summary["n_tests"] += len(idx)
    panel_all["alert_flag"] = (panel_all["surprise_q"] < P["fdr_alpha"]).fillna(False)
    panel_all["share_V"] = panel_all["V"] / (panel_all["V"] + panel_all["exposure"])   # share of the AOI total
    panel_all["in_window"] = True
    for c in NODE_COLS:
        if c not in panel_all.columns:
            panel_all[c] = np.nan
    out = panel_all[NODE_COLS].copy()
    out["day"] = pd.to_datetime(out["day"]).dt.date
    out["baseline_days"] = out["baseline_days"].fillna(0).astype(int)
    alerts = node_alerts(out, P)
    summary["n_alerts"] = len(alerts)
    if not persist:
        summary["_data"] = {"node": out, "series": pd.DataFrame(series_rows), "episodes": pd.DataFrame(ep_rows),
                            "alerts": alerts}
        return summary
    with store.tx():
        _clear_aoi(store, aoi.aoi_id)
        store.upsert("node_aoi_day", out)
        store.upsert("node_aoi_series", pd.DataFrame(series_rows))
        if ep_rows:
            store.upsert("burst_episode", pd.DataFrame(ep_rows))
        sync_alerts(store, "node_trend", aoi.aoi_id, alerts, run_id)
    return summary


def _clear_aoi(store: Store, aoi_id: str, with_tx: bool = False) -> None:
    def go():
        for t in ("node_aoi_day", "node_aoi_series", "burst_episode"):
            store.con.execute(f"DELETE FROM {t} WHERE aoi_id = ?", [aoi_id])
    if with_tx:
        with store.tx():
            go()
            sync_alerts(store, "node_trend", aoi_id, [], "clear")
    else:
        go()


# --------------------------------------------------------------------------
# alerts
# --------------------------------------------------------------------------
def alert_id(kind: str, aoi_id: str, entity: str, day) -> str:
    return hashlib.sha1(f"{kind}|{aoi_id}|{entity}|{day or ''}".encode()).hexdigest()[:20]


def node_alerts(panel: pd.DataFrame, P: dict) -> list[dict]:
    """Statistically justified node alerts: BH q < alpha AND support gate AND an effect-size floor
    (V >= alert_min_ratio x baseline mean). Hashtag-style raw thresholds are not used."""
    d = panel[panel["alert_flag"] & panel["support_ok"]].copy()
    if not len(d):
        return []
    d["ratio"] = d["V"] / d["baseline_mu"].replace(0, np.nan)
    d = d[d["ratio"] >= P["alert_min_ratio"]]
    out = []
    for r in d.itertuples():
        high = (r.surprise_q < P["severity_q_high"]) and (r.V >= P["severity_min_v_high"])
        ent = f"{r.level}:{r.node_code}"
        out.append({"alert_id": alert_id("node_trend", r.aoi_id, ent, r.day), "kind": "node_trend",
                    "aoi_id": r.aoi_id, "entity_type": f"node_{r.level}", "entity_id": r.node_code, "day": r.day,
                    "as_of": pd.Timestamp(r.day, tz="UTC") + pd.Timedelta(days=1), "p": float(r.surprise_p),
                    "q": float(r.surprise_q), "effect": float(r.ratio), "severity": "high" if high else "medium",
                    "rationale": f"{r.level} {r.node_code}: {int(r.V)} articles vs baseline {r.baseline_mu:.1f} "
                                 f"({r.ratio:.1f}x), BH q={r.surprise_q:.4f}",
                    "support_json": json.dumps({"V": float(r.V), "baseline_mu": float(r.baseline_mu),
                                                "baseline_days": int(r.baseline_days), "exposure": float(r.exposure)})})
    return out


def sync_alerts(store: Store, kind: str, aoi_id: str, current: list[dict], run_id: str,
                retract_missing: bool = True) -> dict:
    """Upsert `current` alerts for (kind, aoi). Re-running never duplicates (stable alert_id). If
    `retract_missing`, previously-active alerts of this kind/AOI that no longer qualify are marked
    'retracted' (a revision caused by late data), never deleted."""
    now = now_utc()
    existing = store.df("SELECT alert_id, status, first_run, first_seen_at FROM alert WHERE kind = ? AND aoi_id = ?",
                        [kind, aoi_id])
    ex = {r.alert_id: r for r in existing.itertuples()}
    rows = []
    for a in current:
        prev = ex.get(a["alert_id"])
        rows.append({**a, "status": "active", "first_run": prev.first_run if prev is not None else run_id,
                     "last_run": run_id, "first_seen_at": prev.first_seen_at if prev is not None else now,
                     "last_seen_at": now, "retracted_at": None})
    if rows:
        store.upsert("alert", pd.DataFrame(rows))
    n_retracted = 0
    if retract_missing:
        keep = {a["alert_id"] for a in current}
        for aid, r in ex.items():
            if aid not in keep and r.status == "active":
                store.con.execute("UPDATE alert SET status='retracted', retracted_at=?, last_run=? WHERE alert_id=?",
                                  [now, run_id, aid])
                n_retracted += 1
    return {"upserted": len(rows), "retracted": n_retracted}


# --------------------------------------------------------------------------
# AOI-day coverage / activity table
# --------------------------------------------------------------------------
def compute_aoi_day(store: Store, aoi: G.AOI, P: dict, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per-day AOI totals + coverage. Definitions (scope = everything in `article_fact`):
      n_articles         distinct (non-duplicate) articles matched to the AOI
      n_located_event    of those, articles whose EVENT location is in the AOI (== n_articles for event AOIs)
      n_mention_only     articles that mention the AOI country but are NOT matched to the AOI
      n_scope_total      all non-duplicate articles in the store that day
      n_scope_unlocated  of those, with no usable event location (shown so coverage is never hidden)
    """
    fr = A.aoi_frame(store, aoi)
    if len(fr):
        fr["day"] = pd.to_datetime(fr["day"]).dt.date
    scope = store.df("""SELECT f.day, count(*) n_scope_total,
                               sum((g.loc_status <> 'located')::INT) n_scope_unlocated
                        FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                        WHERE NOT f.is_duplicate GROUP BY 1""")
    scope["day"] = pd.to_datetime(scope["day"]).dt.date
    if aoi.iso2:
        ment = store.df("""SELECT f.day, f.source_table, f.article_id FROM article_fact f
                           JOIN article_geo g USING (source_table, article_id)
                           WHERE NOT f.is_duplicate AND g.mention_countries_iso @> ARRAY[?]::text[]""", [aoi.iso2])
        matched = set(zip(fr["source_table"], fr["article_id"])) if len(fr) else set()
        keep = np.array([(a, b) not in matched for a, b in zip(ment["source_table"], ment["article_id"])], dtype=bool)
        ment = ment[keep]            # explicit boolean mask: a bare [] on an empty frame would select COLUMNS
        ment["day"] = pd.to_datetime(ment["day"]).dt.date
        mo = ment.groupby("day").size()
    else:
        mo = pd.Series(dtype=float)
    days = sorted(set(scope["day"]) | (set(fr["day"]) if len(fr) else set()))
    rows = []
    sc = scope.set_index("day")
    for d in days:
        sub = fr[fr["day"] == d] if len(fr) else fr
        ev_started = ev_active = None
        if events is not None and len(events):
            ev = events[events["is_event"]]
            fs = pd.to_datetime(ev["first_seen"]).dt.date
            ls = pd.to_datetime(ev["last_seen"]).dt.date
            ev_started = int((fs == d).sum())
            ev_active = int(((fs <= d) & (ls >= d)).sum())
        rows.append({"aoi_id": aoi.aoi_id, "day": d, "n_articles": int(len(sub)),
                     "n_publishers": int(sub["source_domain"].nunique()) if len(sub) else 0,
                     "n_events_started": ev_started, "n_events_active": ev_active,
                     "n_located_event": int(len(sub)) if aoi.mode == "event" else int((sub["ev_country_iso"] == aoi.iso2).sum()) if len(sub) else 0,
                     "n_mention_only": int(mo.get(d, 0)) if aoi.iso2 else None,
                     "n_scope_unlocated": int(sc.loc[d, "n_scope_unlocated"]) if d in sc.index else None,
                     "n_scope_total": int(sc.loc[d, "n_scope_total"]) if d in sc.index else None,
                     "in_window": True})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# metric_meta: every metric ships its definition, units, direction and caveats in-band
# --------------------------------------------------------------------------
_M = lambda k, n, fam, f, u, g, d, c, gate: dict(metric_key=k, display_name=n, family=fam, formula=f, units=u,
                                                    grain=g, direction=d, caveats=c, support_gate=gate)
METRIC_META = [
    _M("V", "Articles", "volume", "distinct, de-duplicated, labelled in-scope articles located in the AOI on the day", "articles",
       "AOI x taxonomy node x day", "higher = more coverage", "NULL on data-gap days (exposure below min_exposure_day), never 0",
       "day exposure >= min_exposure_day"),
    _M("exposure", "Exposure (rest of AOI)", "volume", "articles of every OTHER node in the same AOI that day (leave-one-out)",
       "articles", "AOI x node x day", "context", "leave-one-out so a node's own spike cannot inflate its baseline", "-"),
    _M("share_V", "Share of AOI", "volume", "V / (V + exposure)", "share", "AOI x node x day", "context",
       "restates volume by construction (declared); context only, never ranked", "-"),
    _M("growth", "Growth", "growth", "day-over-day change of log((V+1)/exposure)", "log-rate", "AOI x node x day",
       "positive = growing relative to the AOI", "NULL below support", "baseline_days>=min, V>=3"),
    _M("accel_smooth", "Acceleration", "acceleration", "causal Savitzky-Golay 2nd derivative of the log-rate (window 5, order 2)",
       "log-rate/day^2", "AOI x node x day", "positive = speeding up", "causal: no look-ahead; NULL below support",
       "baseline_days>=min, V>=3"),
    _M("surprise", "Statistical surprise", "intensity", "-log10 P(X >= V | negative-binomial baseline of THIS AOI's own history, "
       "exposure-offset, drift-adjusted)", "-log10 p", "AOI x node x day", "higher = more unusual",
       "baseline excludes the day tested; Poisson fallback recorded in baseline_dist", "baseline_days>=min, baseline_mean>=0.5, V>=3"),
    _M("surprise_q", "FDR q-value", "intensity", "Benjamini-Hochberg q over every node tested in the AOI that day",
       "probability", "AOI x node x day", "lower = stronger", "alerts need q < 0.05 AND V >= 2x baseline", "as surprise"),
    _M("burst_state", "Burst state", "burst", "Kleinberg 2-state (long series) or Poisson CUSUM (short), method recorded per row",
       "0/1", "AOI x node x day", "1 = bursting", "short pilot series use CUSUM", "series >= baseline_min_days"),
    _M("burst_intensity", "Burst level (context)", "burst", "level of the burst detector's state", "level", "AOI x node x day",
       "context", "DECLARED volume restatement (Spearman ~0.8 vs V in the pilot): never ranked or alerted on", "-"),
    _M("publisher_pie", "Publisher diversity (PIE)", "diversity", "Hurlbert PIE = N/(N-1) * (1 - sum p_i^2) over source domains",
       "0-1", "AOI x node x day / event", "higher = more independent outlets",
       "size-unbiased; NULL (not 0) when N<2. HHI/Shannon rejected: volume proxies", "N >= 2"),
    _M("n_articles", "Event articles", "event", "non-duplicate articles in the event cluster", "articles", "event", "higher = bigger",
       "clusters are story-level: a very large cluster may merge related sub-stories", "is_event"),
    _M("n_publishers", "Event publishers", "event", "distinct source domains in the event", "publishers", "event",
       "higher = wider pickup", "syndicated copies are counted separately", "is_event"),
    _M("syndication_share", "Syndicated copy share", "event", "n_copies / (n_articles + n_copies); copies = exact content_hash duplicates",
       "share", "event", "context", "identical text on other domains", "-"),
    _M("growth_24h", "Event growth (24h)", "growth", "log((articles in last 24h + 1) / (articles in previous 24h + 1))", "log ratio",
       "event", "positive = still growing", "NULL if < 5 articles in the last 48h", ">=5 articles in 48h"),
    _M("accel_6h", "Event acceleration (6h)", "acceleration", "causal SavGol 2nd derivative of log 6-hour counts (last 48h)",
       "log/bucket^2", "event", "positive = speeding up", "NULL if < 8 articles in the last 48h", ">=8 articles in 48h"),
    _M("half_life_days", "Event half-life", "persistence", "exponential decay fit of daily counts after the peak", "days", "event",
       "longer = more persistent", "NULL unless >= 4 days after peak and the fit is reliable", ">=4 days post-peak"),
    _M("emerging_q", "Emerging-event q", "emergence", "empirical upper-tail p of distinct publishers at the event's age vs the AOI's own "
       "recent clusters of the same domain, BH-FDR per (AOI, time)", "probability", "event x AOI", "lower = more unusual",
       "reference needs >= 50 clusters else NULL; needs HIGH-confidence location to alert", "n_ref >= 50"),
    _M("development_q", "New-development q", "novelty", "new articles fall outside the envelope of the event's earlier coverage "
       "(rank test vs leave-one-out similarity, 10th-percentile effect floor)", "probability", "event x AOI",
       "lower = clearer new angle", "large evolving events develop often; needs coherent sub-group of >=3 articles / 2 publishers", "n_old>=5"),
    _M("loc_confidence", "Event-location confidence", "location", "high: >=3 located members and >=80% agree; medium: >=2; low: 1",
       "tier", "event", "context", "GDELT ActionGeo is machine-extracted; ~20% of multi-located events disagree on country", "-"),
]

# Social-media metrics that are deliberately NOT produced (no data source) - listed so their absence is explicit
UNSUPPORTED_METRICS = [
    ("likes / reactions", "no engagement data for news articles"), ("shares / reposts", "no engagement data"),
    ("followers / reach", "no audience data (publisher traffic is not available)"),
    ("engagement / virality", "no engagement data; never inferred from volume"),
    ("hate / toxicity", "uncalibrated classifier, not meaningful for news"), ("hashtag metrics", "news articles carry no hashtags"),
    ("author concentration", "author field is not reliably populated; publisher diversity (PIE) is used instead"),
]


def seed_metric_meta(store: Store) -> None:
    store.upsert("metric_meta", pd.DataFrame(METRIC_META))
