"""Read-only JSON API for the News Trending & Emerging Events dashboard (UI/news.html).

Mounted by api_server.py under /api/news/*. Never writes: the derived store is opened read-only (or a
snapshot copy if the pipeline holds the DuckDB write lock) and PostgreSQL / OpenSearch are not touched.

Registered AOIs are served from stored results. AD-HOC AOIs (a polygon / box / radius drawn in the UI)
are computed in memory by the SAME functions with persist=False - their baselines are fitted on the
AOI's own articles, exactly like registered ones - and flagged `adhoc` (multiplicity across arbitrary
user-drawn areas cannot be controlled, and the response says so).
"""
from __future__ import annotations

import json
import datetime as dt
import math
import re
import sys
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from lib import news_aoi as A  # noqa: E402
from lib import news_config as C  # noqa: E402
from lib import news_events as EV  # noqa: E402
from lib import news_geo as G  # noqa: E402
from lib import news_live as LV  # noqa: E402
from lib import news_relevance as NR  # noqa: E402
from lib import news_metrics as M  # noqa: E402
from lib import news_publishers as PB  # noqa: E402
from lib import news_taxonomy as TX  # noqa: E402
from lib import news_trending as TR  # noqa: E402
from lib.news_store import Store  # noqa: E402

MAX_BODY = 256 * 1024
_EVENT_ID = re.compile(r"^ev_[A-Za-z0-9_]{1,28}$")     # plain identifiers only
MAX_VERTICES = 500
MAX_EVENTS = 300
ARTICLE_PAGE = 50
TRENDING_METHODOLOGY_KEYS = (
    "sim_threshold", "max_gap_hours", "centroid_retention_days", "event_min_articles",
    "event_min_publishers", "location_consensus_min_share", "location_min_located",
    "loc_high_min_located", "loc_high_min_share", "alert_min_loc_confidence",
    "trend_merge_cos", "trend_merge_gap_hours", "trend_fold_cos", "trend_min_window_articles",
    "trend_min_window_publishers", "trend_min_ref", "trend_growth_min_total",
    "trend_window_tolerance_hours", "trend_weights", "taxonomy_min_agreement",
    "trend_country_min_share", "trend_country_min_located", "trend_country_min_coverage",
    "relevance_min_similarity", "relevance_min_agreement", "trend_top_n", "trend_max_n",
    "fdr_method", "fdr_alpha", "fdr_pvalue_kind", "emerging_window_hours", "emerging_ref_days",
    "emerging_min_ref", "emerging_min_publishers", "dev_min_event_articles",
    "dev_min_event_publishers", "dev_recent_hours", "dev_min_new_articles",
    "dev_min_new_publishers", "dev_min_coherence", "dev_envelope_pct", "dev_max_events_per_run",
    "aoi_min_total_articles", "node_min_total_v", "min_exposure_day", "baseline_window",
    "baseline_min_days", "baseline_min_mean", "support_min_v", "alert_min_ratio",
    "severity_q_high", "severity_min_v_high",
)


class ApiError(Exception):
    def __init__(self, msg: str, status: int = 400):
        super().__init__(msg)
        self.status = status


def clean(o: Any) -> Any:
    """JSON-safe: NaN/NaT/inf -> None, numpy -> python, timestamps -> ISO strings."""
    if o is None:
        return None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return clean(o.to_dict(orient="records"))
    if isinstance(o, (pd.Timestamp,)):
        if pd.isna(o):
            return None
        return (o.tz_convert("UTC") if o.tzinfo is not None else o).isoformat()
    if hasattr(o, "tzinfo") and getattr(o, "tzinfo", None) is not None and hasattr(o, "astimezone"):
        import datetime as _dt
        return o.astimezone(_dt.timezone.utc).isoformat()
    if hasattr(o, "isoformat") and not isinstance(o, str):
        try:
            return o.isoformat()
        except Exception:
            return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    return o


class NewsAPI:
    MAX_CONNECTIONS = 8          # the dashboard's role has a connection limit; requests beyond it wait instead of failing

    def __init__(self, db_path: Optional[Path] = None):
        # production: None -> the configured schema (news_engine). Tests pass a path, which the Store maps to its own schema.
        self._path = Path(db_path) if db_path else None
        self._sem = threading.BoundedSemaphore(self.MAX_CONNECTIONS)
        self._live_cache: Optional[tuple] = None
        self._geo: Optional[G.GeoIndex] = None
        self._lock = threading.Lock()
        self._geojson_cache: dict = {}
        self._registry: Optional[dict] = None
        self._sub_cache: dict = {}
        self._trend_cache: Optional[tuple] = None

    # ---- plumbing ----------------------------------------------------------------------
    @property
    def db_name(self) -> str:
        """The PostgreSQL schema this server reads (it is the live engine's own schema: no copy, no snapshot file)."""
        from lib import news_store as _S
        return _S.schema_for(self._path)

    @property
    def geo(self) -> G.GeoIndex:
        with self._lock:
            if self._geo is None:
                self._geo = G.GeoIndex()
            return self._geo

    def registry(self) -> dict:
        if self._registry is None:
            self._registry = PB.load_registry()
        return self._registry

    def store(self) -> Store:
        """A read-only store: one REPEATABLE READ snapshot of the live schema, never blocking (or blocked by) the engine."""
        self._sem.acquire()
        try:
            st = Store(self._path, read_only=True)
        except Exception as e:
            self._sem.release()
            raise ApiError(f"derived database unavailable: {type(e).__name__}: {str(e)[:120]}", 503)
        st._on_close = self._sem.release
        return st

    @staticmethod
    def _version(st: Store) -> int:
        """Data version = finish time of the newest successful run; it changes exactly when the engine publishes."""
        v = st.scalar("SELECT max(finished_at) FROM run_log WHERE status = 'success'")
        return 0 if v is None else int(v.timestamp())

    # ---- meta ---------------------------------------------------------------------------
    def meta(self) -> dict:
        with self.store() as st:
            run = st.df("SELECT run_id, mode, scope, date_from, date_to, status, finished_at FROM run_log "
                        "WHERE status='success' ORDER BY started_at DESC LIMIT 1")
            rng = st.df("SELECT min(day) d0, max(day) d1, count(*) n FROM article_fact").iloc[0]
            aois = st.df("""SELECT a.aoi_id, a.kind, a.key, a.name, a.mode, a.iso2, a.parent_aoi,
                                   coalesce(d.n, 0) n_articles, coalesce(s.n_series, 0) n_series
                            FROM aoi_def a
                            LEFT JOIN (SELECT aoi_id, sum(n_articles) n FROM aoi_day GROUP BY 1) d USING (aoi_id)
                            LEFT JOIN (SELECT aoi_id, count(*) n_series FROM node_aoi_series GROUP BY 1) s USING (aoi_id)
                            ORDER BY (a.kind='country') DESC, n_articles DESC""")
            meta = st.df("SELECT * FROM metric_meta ORDER BY family, metric_key")
            if not len(meta):   # database built before definitions were seeded: use the in-code contract (read-time only)
                meta = pd.DataFrame(M.METRIC_META).sort_values(["family", "metric_key"])
            n_active = st.scalar("SELECT count(*) FROM alert WHERE status = 'active'")
            cov = st.df("SELECT * FROM corpus_day ORDER BY day")
            tax = st.df("SELECT code, tier, name, parent, in_scope FROM taxonomy_node")
            unl = st.scalar("SELECT count(*) FROM pending_label")
        return clean({"database": self.db_name, "last_run": run.to_dict("records")[0] if len(run) else None,
                      "articles_from": rng.d0, "articles_to": rng.d1, "n_articles": rng.n, "aois": aois,
                      "metric_meta": meta, "unsupported": [{"metric": a, "reason": b} for a, b in M.UNSUPPORTED_METRICS],
                      "corpus_coverage": cov, "taxonomy": tax, "pending_labels": unl,
                      "alert_export": self.alert_export(n_active),
                      "params": {k: C.PARAMS[k] for k in ("baseline_min_days", "baseline_window", "fdr_alpha", "sim_threshold",
                                                          "max_gap_hours", "event_min_articles", "event_min_publishers",
                                                          "emerging_min_ref", "alert_min_loc_confidence", "min_exposure_day")}})

    def alert_export(self, n_active_db: int) -> dict:
        """Status of UI/data/news_alerts.json (written by the pipeline) vs the alerts the dashboard reads
        from the database. Read-only; a stale or missing export is reported, not hidden."""
        f = HERE / "data" / "news_alerts.json"
        if not f.exists():
            return {"present": False}
        try:
            j = json.loads(f.read_text())
            rows = j.get("alerts", [])
            return {"present": True, "generated_at": j.get("generated_at"), "n_alerts": len(rows),
                    "n_active_db": int(n_active_db), "in_sync": len(rows) == int(n_active_db)}
        except (OSError, ValueError):
            return {"present": True, "error": "unreadable"}

    # ---- gazetteer / boundaries ------------------------------------------------------------
    def gazetteer(self) -> dict:
        g = self.geo
        with self.store() as st:
            have = dict(st.df("SELECT ev_country_iso c, count(*) n FROM article_geo WHERE ev_country_iso IS NOT NULL GROUP BY 1").values)
        countries = [{"iso2": c.iso2, "name": c.name, "n_articles": int(have.get(c.iso2, 0))} for c in g.countries]
        countries.sort(key=lambda x: (-x["n_articles"], x["name"]))
        return {"countries": countries}

    def admin1_list(self, iso2: str) -> dict:
        iso2 = self._iso(iso2)
        with self.store() as st:
            have = dict(st.df("SELECT ev_adm1_code c, count(*) n FROM article_geo WHERE ev_adm1_code IS NOT NULL GROUP BY 1").values)
        return {"admin1": [{"code": a.code, "name": a.name, "n_articles": int(have.get(a.code, 0))}
                           for a in self.geo.admin1_of(iso2)]}

    def _iso(self, iso2: str) -> str:
        iso2 = (iso2 or "").upper()
        if len(iso2) != 2 or self.geo.country(iso2) is None:
            raise ApiError("unknown country")
        return iso2

    def boundaries(self, level: str, iso2: Optional[str] = None) -> dict:
        key = (level, iso2)
        if key in self._geojson_cache:
            return self._geojson_cache[key]
        g = self.geo
        feats = []
        if level == "country":
            for c in g.countries:
                geom = c.geom.simplify(0.08, preserve_topology=True)
                feats.append({"type": "Feature", "properties": {"id": c.iso2, "name": c.name},
                              "geometry": G.shapely.geometry.mapping(geom)})
        elif level == "adm1":
            iso2 = self._iso(iso2 or "")
            for a in g.admin1_of(iso2):
                feats.append({"type": "Feature", "properties": {"id": a.code, "name": a.name},
                              "geometry": G.shapely.geometry.mapping(a.geom.simplify(0.02, preserve_topology=True))})
        else:
            raise ApiError("level must be country or adm1")
        out = {"type": "FeatureCollection", "features": feats}
        self._geojson_cache[key] = out
        return out

    # ---- AOI resolution ---------------------------------------------------------------------
    def resolve(self, st: Store, aoi_id: str) -> G.AOI:
        r = st.df("SELECT * FROM aoi_def WHERE aoi_id = ?", [aoi_id])
        if not len(r):
            raise ApiError("AOI is not registered; draw an ad-hoc AOI instead", 404)
        r = r.iloc[0]
        if r.kind == "country":
            return G.country_aoi(self.geo, r.iso2, r["mode"])
        if r.kind == "adm1":
            return G.admin1_aoi(self.geo, r["key"])
        raise ApiError("unsupported registered AOI kind", 400)

    def adhoc_aoi(self, spec: dict) -> G.AOI:
        if not isinstance(spec, dict):
            raise ApiError("aoi must be an object")
        if spec.get("kind") == "polygon":
            co = spec.get("coords")
            if not isinstance(co, list) or not (3 <= len(co) <= MAX_VERTICES):
                raise ApiError(f"polygon needs 3..{MAX_VERTICES} vertices")
            for p in co:
                if (not isinstance(p, (list, tuple)) or len(p) != 2 or not all(isinstance(v, (int, float)) for v in p)
                        or not (-180 <= p[0] <= 180 and -90 <= p[1] <= 90)):
                    raise ApiError("polygon vertices must be [lon, lat] within range")
        try:
            a = G.aoi_from_request(self.geo, spec)
        except (ValueError, KeyError) as e:
            raise ApiError(f"invalid AOI: {e}")
        if a.kind in ("country", "adm1"):
            return a
        a.adhoc = True
        return a

    # ---- taxonomy (Domain > Topic > Category > Subtopic, from taxonomy_node) --------------------------------
    TIER_COL = {"domain": "domain_id", "topic": "topic_id", "category": "category_id", "subtopic": "subtopic_id"}

    @staticmethod
    def _taxmap(st: Store) -> dict:
        df = st.df("SELECT code, tier, name, parent, in_scope FROM taxonomy_node")
        return {r.code: {"tier": r.tier, "name": r.name, "parent": (None if pd.isna(r.parent) else r.parent),
                         "in_scope": bool(r.in_scope)} for r in df.itertuples()}

    def _tax_sel(self, taxmap: dict, code) -> Optional[dict]:
        """Validate a taxonomy selection against the stored hierarchy. Headline text is never consulted."""
        if code in (None, ""):
            return None
        if not isinstance(code, str) or code not in taxmap:
            raise ApiError("unknown taxonomy node")
        n = taxmap[code]
        if not n["in_scope"]:
            raise ApiError("taxonomy node is out of scope")
        path, c = [], code
        while c:
            path.append({"code": c, "name": taxmap[c]["name"], "tier": taxmap[c]["tier"]})
            c = taxmap[c]["parent"]
        return {"code": code, "tier": n["tier"], "col": self.TIER_COL[n["tier"]], "name": n["name"], "path": path[::-1]}

    @staticmethod
    def _within(code: Optional[str], sel_code: str) -> bool:
        return bool(code) and (code == sel_code or code.startswith(sel_code + "."))

    @staticmethod
    def _pub(domain: Optional[str]) -> str:
        d = (domain or "").lower()
        return d[4:] if d.startswith("www.") else (d or "unknown publisher")

    @staticmethod
    def _newest_first(df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic total order: many articles share an identical timestamp, and an unstable sort would let
        page 1 and page 2 order ties differently (repeating or skipping articles in 'Show more')."""
        if not len(df):
            return df
        return df.sort_values(["published_at", "source_table", "article_id"], ascending=[False, True, True], kind="mergesort")

    def _frame(self, st: Store, aoi: G.AOI) -> pd.DataFrame:
        fr = A.aoi_frame(st, aoi)
        if len(fr):
            fr["day"] = pd.to_datetime(fr["day"]).dt.date.astype(str)
        return fr

    def _cite(self, title, domain, published_at, url, **extra) -> dict:
        ok = isinstance(url, str) and url.startswith(("http://", "https://"))
        return {"title": title or "(no title)", "publisher": self._pub(domain), "published_at": published_at,
                "url": url if ok else None, **extra}

    def _items(self, rows: pd.DataFrame, taxmap: dict) -> list:
        nm = lambda c: taxmap.get(c, {}).get("name") if isinstance(c, str) else None
        return [self._cite(r.title, r.source_domain, r.published_at, r.url, tax_domain=nm(r.domain_id), tax_topic=nm(r.topic_id),
                           tax_subtopic=nm(r.subtopic_id), event_id=r.cluster_id, place=r.ev_place, country=r.ev_country_iso,
                           loc_status=r.loc_status) for r in rows.itertuples()]

    def _event_paths(self, st: Store, ids: list) -> pd.DataFrame:
        """Per event: the taxonomy path its labelled, non-duplicate articles AGREE on (strict majority, top-down; see
        lib/news_taxonomy.py - the same rule as the Trending page), plus its status (assigned / outside / mixed /
        unlabelled). Indexed by event_id."""
        st.con.register("_etx", pd.DataFrame({"event_id": ids}))
        try:
            lab = st.df("""SELECT cluster_id AS event_id, has_label, domain_id, topic_id, category_id, subtopic_id FROM article_fact
                           WHERE NOT is_duplicate AND cluster_id IN (SELECT event_id FROM _etx)""")
        finally:
            st.con.unregister("_etx")
        if not len(lab):
            return pd.DataFrame(columns=TX.PATH_COLUMNS).rename_axis("event_id")
        return TX.assign_paths(lab, "event_id", min_agreement=C.PARAMS["taxonomy_min_agreement"])

    def _event_tax(self, st: Store, ids: list, sel: dict) -> pd.DataFrame:
        """Per event: its AGREED label at the selected level (NaN when its articles do not agree on one), and how many
        labelled members carry the selected node."""
        paths = self._event_paths(st, ids)
        st.con.register("_etx", pd.DataFrame({"event_id": ids}))
        try:
            n = st.df(f"""SELECT cluster_id AS event_id, count(*) FILTER (WHERE {sel['col']} = ?) AS n_match
                          FROM article_fact WHERE NOT is_duplicate AND has_label AND cluster_id IN (SELECT event_id FROM _etx)
                          GROUP BY 1""", [sel["code"]])
        finally:
            st.con.unregister("_etx")
        out = paths[[sel["tier"], "n_labelled"]].rename(columns={sel["tier"]: "primary_label"}).reset_index()
        return out.merge(n, on="event_id", how="left").fillna({"n_match": 0})

    def _event_cites(self, st: Store, ids, sel: Optional[dict], k: int = 3) -> dict:
        """Up to k citations per event: the earliest article of each distinct publisher (articles that carry the
        selected taxonomy node come first when a filter is active)."""
        ids = sorted(set(ids))
        if not ids:
            return {}
        order = (f"(CASE WHEN {sel['col']} = ? THEN 0 ELSE 1 END), published_at, source_table, article_id" if sel else "published_at, source_table, article_id")
        args = [sel["code"]] * 2 if sel else []
        st.con.register("_cit", pd.DataFrame({"event_id": ids}))
        try:
            df = st.df(f"""SELECT cluster_id, title, url, source_domain, published_at FROM (
                             SELECT *, row_number() OVER (PARTITION BY cluster_id ORDER BY {order}) rn FROM (
                               SELECT *, row_number() OVER (PARTITION BY cluster_id, source_domain ORDER BY {order}) pr
                               FROM article_fact WHERE NOT is_duplicate AND cluster_id IN (SELECT event_id FROM _cit)) WHERE pr = 1)
                           WHERE rn <= {int(k)} ORDER BY cluster_id, rn""", args)
        finally:
            st.con.unregister("_cit")
        out: dict = {}
        for r in df.itertuples():
            out.setdefault(r.cluster_id, []).append(self._cite(r.title, r.source_domain, r.published_at, r.url))
        return out

    def _key_cites(self, st: Store, keys: list, k: int = 3) -> list:
        pairs = []
        for s in keys[:200]:
            st_, _, aid = str(s).rpartition("-")
            if st_ in C.SOURCE_TABLES and aid.isdigit():
                pairs.append((st_, int(aid)))
        if not pairs:
            return []
        st.con.register("_kc", pd.DataFrame(pairs, columns=["source_table", "article_id"]))
        try:
            df = st.df("""SELECT f.title, f.url, f.source_domain, f.published_at FROM article_fact f
                          JOIN _kc USING (source_table, article_id) ORDER BY f.published_at DESC LIMIT ?""", [k])
        finally:
            st.con.unregister("_kc")
        return [self._cite(r.title, r.source_domain, r.published_at, r.url) for r in df.itertuples()]

    def _panel_subtopic(self, st: Store, aoi: G.AOI, key: str) -> dict:
        """Subtopic-level series are not stored by the pipeline; compute them in memory (read-only) with the SAME
        function. Cached per database version. Its FDR family is the subtopic series of this AOI."""
        ck = (key, self._version(st))
        if ck in self._sub_cache:
            return self._sub_cache[ck]
        P2 = dict(C.PARAMS, node_levels=["subtopic"])
        summ = M.compute_aoi_panel(st, aoi, P2, "view", persist=False)
        d = summ.pop("_data", None)
        res = {"summary": summ, "node": d["node"] if d else pd.DataFrame(), "ser": d["series"] if d else pd.DataFrame(),
               "alerts": d["alerts"] if d else []}
        if len(self._sub_cache) >= 6:
            self._sub_cache.pop(next(iter(self._sub_cache)))
        self._sub_cache[ck] = res
        return res

    # ---- the AOI bundle -------------------------------------------------------------------------
    def bundle(self, aoi_id: Optional[str] = None, spec: Optional[dict] = None, level: str = "topic",
               tax: Optional[str] = None) -> dict:
        if level not in ("topic", "category", "subtopic"):
            raise ApiError("level must be topic, category or subtopic")
        with self.store() as st:
            if spec is not None:
                aoi = self.adhoc_aoi(spec)
                registered = False
            else:
                aoi = self.resolve(st, aoi_id or "")
                registered = True
            taxmap = self._taxmap(st)
            sel = self._tax_sel(taxmap, tax)
            tax_names = {c: v["name"] for c, v in taxmap.items()}
            P = C.PARAMS
            fr = self._frame(st, aoi)
            if registered:
                node = st.df("SELECT * FROM node_aoi_day WHERE aoi_id = ? ORDER BY day", [aoi.aoi_id])
                ser = st.df("SELECT * FROM node_aoi_series WHERE aoi_id = ?", [aoi.aoi_id])
                alerts = st.df("SELECT * FROM alert WHERE aoi_id = ? ORDER BY (status='active') DESC, as_of DESC, q LIMIT 1000", [aoi.aoi_id])
                cov = st.df("SELECT * FROM aoi_day WHERE aoi_id = ? ORDER BY day", [aoi.aoi_id])
                summary = {"supported": len(node) > 0, "reason": None if len(node) else "below minimum evidence (see coverage)"}
                tests = st.df("SELECT * FROM event_aoi_test WHERE aoi_id = ? AND kind='emerging'", [aoi.aoi_id])
                devs = st.df("SELECT * FROM event_development WHERE aoi_id = ?", [aoi.aoi_id])
            else:
                summary = M.compute_aoi_panel(st, aoi, P, "adhoc", persist=False)
                d = summary.pop("_data", None)
                node = d["node"] if d else pd.DataFrame()
                ser = d["series"] if d else pd.DataFrame()
                alerts = pd.DataFrame(d["alerts"]) if d and d["alerts"] else pd.DataFrame()
                events_for_cov = A.aoi_events(st, aoi, P) if aoi.mode == "event" else None
                cov = M.compute_aoi_day(st, aoi, P, events_for_cov)
                er = EV.emerging_tests(st, aoi, P, [pd.Timestamp(cov["day"].max()).tz_localize("UTC") + pd.Timedelta(days=1)]
                                       if len(cov) else [], "adhoc", persist=False) if aoi.mode == "event" and len(cov) else {}
                tests = er.get("rows", pd.DataFrame()) if er else pd.DataFrame()
                devs = pd.DataFrame()
                if len(alerts) and "as_of" in alerts:
                    alerts["status"] = "active"
            extra_alerts: list = []
            if level == "subtopic":
                sub = self._panel_subtopic(st, aoi, aoi.aoi_id if registered else json.dumps(spec, sort_keys=True))
                node, ser = sub["node"], sub["ser"]
                summary = sub["summary"] if not registered or len(sub["node"]) else summary
                extra_alerts = [{**a, "entity_type": "node_subtopic", "status": "active"} for a in sub["alerts"]]
            events, match_ids = self._events(st, aoi, P, tests, devs, tax_names, sel)
            nodes_latest, series = self._nodes(node, ser, level, tax_names, sel)
            alist = self._alerts(alerts, tax_names) + [{k: a.get(k) for k in ("alert_id", "kind", "entity_type", "entity_id", "day", "as_of", "p", "q",
                                                                              "effect", "severity", "status", "rationale")}
                                                       | {"entity_name": tax_names.get(a["entity_id"])} for a in extra_alerts]
            if sel:
                alist = [a for a in alist if (self._within(a["entity_id"], sel["code"]) if a["kind"] == "node_trend"
                                              else (match_ids is not None and a["entity_id"] in match_ids))]
            self._attach_sources(st, alist[:120], fr, devs, sel)
            acounts = {}
            acts = [a for a in alist if a.get("status", "active") == "active"]
            for a in acts:
                acounts[a["kind"]] = acounts.get(a["kind"], 0) + 1
            # ---- taxonomy view of the area's articles
            lab = fr[fr["has_label"].fillna(False) & fr["domain_id"].notna() & (fr["domain_id"] != "D00")] if len(fr) else fr
            counts = {}
            for col in self.TIER_COL.values():
                if len(lab):
                    for k, v in lab.groupby(col).size().items():
                        counts[k] = int(v)
            frt = fr[fr[sel["col"]] == sel["code"]] if (sel and len(fr)) else fr
            tax_days = ([{"day": d, "n": int(n)} for d, n in frt.groupby("day").size().items()] if sel and len(frt) else [])
            arts = self._newest_first(frt)
            out = {
                "aoi": {**aoi.to_dict(), "registered": registered, "adhoc": not registered and aoi.kind not in ("country", "adm1")},
                "support": {k: v for k, v in summary.items() if k in ("supported", "reason", "n_series", "n_series_ts", "gap_days",
                                                                      "n_labelled_in_scope", "n_unlabelled_or_out_of_scope", "n_tests")},
                "coverage": cov.to_dict("records") if len(cov) else [], "nodes": nodes_latest, "series": series, "events": events,
                "alerts": alist, "alert_counts": acounts, "level": level,
                "tax": None if not sel else {**sel, "n_match": int(len(frt)), "n_area": int(len(fr)), "days": tax_days},
                "taxonomy_counts": counts,
                "articles": {"total": int(len(arts)), "items": self._items(arts.head(ARTICLE_PAGE), taxmap) if len(arts) else [], "offset": 0},
                "notes": ([] if registered or aoi.kind in ("country", "adm1") else
                          ["Ad-hoc AOI: baselines are fitted on this AOI's own articles, but FDR control covers tests within this AOI only - "
                           "multiplicity across arbitrary areas you draw is not controlled.",
                           "Country-centroid points are excluded (min precision: %s)." % aoi.min_precision]) +
                         (["'Mentioned in' AOI: articles that merely name this country; not where events happened. Events are not defined for this mode."]
                          if aoi.mode == "mentioned" else []) +
                         (["Subtopic series are computed on demand for this view (not stored by the pipeline); their FDR family is the subtopic series of this area."]
                          if level == "subtopic" else []),
            }
        return clean(out)

    def articles(self, aoi_id: Optional[str] = None, spec: Optional[dict] = None, tax: Optional[str] = None,
                 node: Optional[dict] = None, day: Optional[str] = None, offset: int = 0, limit: int = ARTICLE_PAGE) -> dict:
        """Source articles of the current view (area + taxonomy filter [+ one node [+ one day]]), newest first.
        Used for 'show more', and for the citations of a trend or an alert."""
        try:
            offset, limit = max(0, int(offset)), max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            raise ApiError("bad paging")
        with self.store() as st:
            if spec is not None:
                aoi = self.adhoc_aoi(spec)
            else:
                aoi = self.resolve(st, aoi_id or "")
            taxmap = self._taxmap(st)
            sel = self._tax_sel(taxmap, tax)
            fr = self._frame(st, aoi)
            if len(fr) and sel:
                fr = fr[fr[sel["col"]] == sel["code"]]
            if node:
                if not isinstance(node, dict) or node.get("level") not in self.TIER_COL or not isinstance(node.get("code"), str):
                    raise ApiError("bad node")
                if len(fr):
                    fr = fr[fr[self.TIER_COL[node["level"]]] == node["code"]]
            if day:
                if not (isinstance(day, str) and len(day) == 10):
                    raise ApiError("bad day")
                if len(fr):
                    fr = fr[fr["day"] == day]
            fr = self._newest_first(fr)
            return clean({"total": int(len(fr)), "offset": offset, "items": self._items(fr.iloc[offset:offset + limit], taxmap) if len(fr) else []})

    def _attach_sources(self, st: Store, alerts: list, fr: pd.DataFrame, devs: pd.DataFrame, sel: Optional[dict]) -> None:
        """Citations for each alert: the articles it was computed from (title + publisher + original URL)."""
        ev_ids = {a["entity_id"] for a in alerts if a["kind"] in ("emerging_event", "new_development")}
        ecit = self._event_cites(st, ev_ids, None, k=3)
        dkeys = {}
        if len(devs):
            for r in devs.itertuples():
                dkeys[(r.event_id, pd.Timestamp(r.detected_asof).tz_convert("UTC") if pd.Timestamp(r.detected_asof).tzinfo else pd.Timestamp(r.detected_asof))] = r.member_keys
        col_of = {"node_topic": "topic_id", "node_category": "category_id", "node_subtopic": "subtopic_id"}
        for a in alerts:
            if a["kind"] == "node_trend" and len(fr) and a.get("entity_type") in col_of:
                day = str(a.get("day"))[:10]
                m = fr[(fr[col_of[a["entity_type"]]] == a["entity_id"]) & (fr["day"] == day)].sort_values("published_at", ascending=False)
                a["n_sources"] = int(len(m))
                a["sources"] = [self._cite(r.title, r.source_domain, r.published_at, r.url) for r in m.head(3).itertuples()]
            elif a["kind"] == "new_development":
                key = (a["entity_id"], pd.Timestamp(a["as_of"]).tz_convert("UTC") if pd.Timestamp(a["as_of"]).tzinfo else pd.Timestamp(a["as_of"]))
                mk = dkeys.get(key)
                src = self._key_cites(st, mk.split(","), 3) if mk else []
                a["n_sources"] = len(mk.split(",")) if mk else None
                a["sources"] = src or ecit.get(a["entity_id"], [])
            elif a["kind"] == "emerging_event":
                a["sources"] = ecit.get(a["entity_id"], [])
                a["n_sources"] = None
            else:
                a["sources"] = []

    def _nodes(self, node: pd.DataFrame, ser: pd.DataFrame, level: str, tax: dict, sel: Optional[dict] = None):
        if not len(node):
            return [], {}
        n = node[node["level"] == level].copy()
        if sel:
            n = n[n["node_code"].map(lambda c: self._within(c, sel["code"]))]
        if not len(n):
            return [], {}
        n["day"] = pd.to_datetime(n["day"]).dt.date
        series = {}
        rows = []
        sdict = {} if not len(ser) else {(r.level, r.node_code): r for r in ser.itertuples()}
        for code, g in n.groupby("node_code"):
            g = g.sort_values("day")
            series[code] = [[str(r.day), r.V, r.baseline_mu, r.surprise_q, bool(r.alert_flag), bool(r.support_ok)] for r in g.itertuples()]
            last = g[g["V"].notna()].tail(1)
            if not len(last):
                continue
            r = last.iloc[0]
            s = sdict.get((level, code))
            rows.append({"node_code": code, "name": tax.get(code, code), "day": str(r["day"]), "V": r["V"], "exposure": r["exposure"],
                         "total_V": float(g["V"].sum(skipna=True)), "support_ok": bool(r["support_ok"]), "growth": r["growth"],
                         "accel_smooth": r["accel_smooth"], "surprise": r["surprise"], "surprise_q": r["surprise_q"],
                         "alert_flag": bool(r["alert_flag"]), "baseline_mu": r["baseline_mu"], "burst_state": r["burst_state"],
                         "n_publishers": r["n_publishers"], "publisher_pie": r["publisher_pie"],
                         "any_alert": bool(g["alert_flag"].any()), "n_alert_days": int(g["alert_flag"].sum()),
                         "ar1_half_life_days": None if s is None else s.ar1_half_life_days,
                         "ar1_fit_ok": None if s is None else bool(s.ar1_fit_ok)})
        rows.sort(key=lambda x: (-(x["any_alert"]), -(x["total_V"])))
        return rows, series

    def _events(self, st: Store, aoi: G.AOI, P: dict, tests: pd.DataFrame, devs: pd.DataFrame, tax: dict,
                sel: Optional[dict] = None):
        """-> (event rows for the table, ids of ALL events matching the taxonomy filter | None when unfiltered)."""
        if aoi.mode != "event":
            return [], None
        ev = A.aoi_events(st, aoi, P)
        if not len(ev):
            return [], None
        ev = ev[ev["is_event"]].copy()
        if not len(ev):
            return [], None
        match_ids = None
        if sel:
            tx = self._event_tax(st, ev["event_id"].tolist(), sel)
            ev = ev.merge(tx, on="event_id", how="left")
            ev = ev[ev["primary_label"] == sel["code"]].copy()
            if "ref_sim" in ev:                      # relevance gate (unscored events pass): not about the selected domain -> not listed
                dom = sel["code"].split(".")[0]
                ev = ev[np.array([NR.passes(a, b, dom, C.PARAMS) for a, b in zip(ev["ref_sim"], ev["ref_votes"])], dtype=bool)].copy()   # (empty list would select columns)
            ev["tax_share"] = ev["n_match"] / ev["n_labelled"]
            match_ids = set(ev["event_id"])
            if not len(ev):
                return [], match_ids
        em = st.df("SELECT * FROM event_metrics")
        ev = ev.merge(em.drop(columns=["n_articles", "n_publishers", "n_copies", "first_seen", "last_seen"], errors="ignore"),
                      on="event_id", how="left")
        if len(tests):
            t = tests[["event_id", "p", "q", "alert_flag", "support_ok", "support_note", "n_ref", "stratum", "n_publishers_at_age"]]
            ev = ev.merge(t.rename(columns={"p": "emerging_p", "q": "emerging_q", "alert_flag": "emerging_alert",
                                            "support_ok": "emerging_support_ok", "support_note": "emerging_note"}),
                          on="event_id", how="left")
        if len(devs):
            dn = devs[devs["alert_flag"]].groupby("event_id").size().rename("n_developments")
            ev = ev.merge(dn, on="event_id", how="left")
        for c, d in (("emerging_alert", False), ("n_developments", 0)):
            if c not in ev:
                ev[c] = d
            ev[c] = ev[c].fillna(d)
        pth = self._event_paths(st, ev["event_id"].tolist())      # stored per-event modal labels are NOT used: no agreement test
        ev["tax_status"] = ev["event_id"].map(pth["tax_status"]).fillna("unlabelled")
        for t in ("domain", "topic", "category", "subtopic"):
            ev[f"{t}_name"] = ev["event_id"].map(pth[t]).map(lambda c: tax.get(c) if isinstance(c, str) else None)
        ev["last_seen"] = pd.to_datetime(ev["last_seen"], utc=True)
        ev = ev.sort_values(["emerging_alert", "last_seen", "n_publishers"], ascending=[False, False, False]).head(MAX_EVENTS)
        keep = ["event_id", "rep_title", "rep_url", "n_articles", "n_publishers", "n_copies", "syndication_share", "first_seen",
                "last_seen", "duration_hours", "active_days", "articles_24h", "growth_24h", "accel_6h", "half_life_days",
                "publisher_pie", "cross_source_ratio", "loc_country_iso", "loc_adm1_code", "loc_confidence", "loc_share",
                "n_located", "aoi_share", "topic_name", "domain_name", "category_name", "subtopic_name", "tax_status", "emerging_p", "emerging_q", "emerging_alert",
                "emerging_support_ok", "emerging_note", "emerging_n_ref", "n_developments", "support_note", "tax_share"]
        ev = ev.rename(columns={"n_ref": "emerging_n_ref"})
        cites = self._event_cites(st, ev["event_id"].tolist(), sel, k=3)
        rows = []
        for r in ev.to_dict("records"):
            d = {k: r.get(k) for k in keep}
            d["sources"] = cites.get(r["event_id"], [])
            rows.append(d)
        return rows, match_ids

    def _alerts(self, alerts: pd.DataFrame, tax: dict) -> list:
        if not len(alerts):
            return []
        a = alerts.copy()
        a["entity_name"] = [tax.get(e, None) if t.startswith("node") else None for e, t in zip(a["entity_id"], a["entity_type"])]
        cols = ["alert_id", "kind", "entity_type", "entity_id", "entity_name", "day", "as_of", "p", "q", "effect", "severity",
                "status", "rationale"]
        return a[[c for c in cols if c in a.columns]].to_dict("records")


    # ---- trending stories (read-only; computed on request from the stored clusters) --------------------------
    def _trend_data(self, st: Store) -> TR.TrendData:
        m = self._version(st)
        if self._trend_cache and self._trend_cache[0] == m:
            return self._trend_cache[1]
        data = TR.load(st)
        self._trend_cache = (m, data)
        return data

    # ---- live processing summary (stored records only) -------------------------------------------------------
    LIVE_TTL = 5.0

    def live(self) -> dict:
        """Run status, totals and schedule - read from the derived PostgreSQL schema (or the working store's run log when no
        PostgreSQL is configured). Never computed from constants; cached for a few seconds so polling is cheap."""
        import time
        c = self._live_cache
        if c and time.monotonic() - c[0] < self.LIVE_TTL:
            return c[1]
        with self.store() as st:
            out = LV.summary(st)
        out["database"] = self.db_name
        res = clean(out)
        self._live_cache = (time.monotonic(), res)
        return res

    def health(self) -> dict:
        """Liveness of the dashboard itself (used by the container health check): can it open its database read-only?"""
        try:
            with self.store() as st:
                n = st.scalar("SELECT count(*) FROM article_fact")
            return {"status": "ok", "database": self.db_name, "articles": int(n or 0)}
        except Exception as e:
            raise ApiError(f"unhealthy: {type(e).__name__}: {str(e)[:120]}", 503)

    def trending_meta(self) -> dict:
        """Everything the simple trending page needs to build its filters: countries that have stories, the taxonomy, data range."""
        with self.store() as st:
            data = self._trend_data(st)
            taxmap = self._taxmap(st)
            P = C.PARAMS
            s = data.stories
            countries = []
            if len(s):
                ok = pd.concat([TR.in_country(s, c, P) for c in s["country"].dropna().unique()])
                for iso, n in ok.groupby("country").size().sort_values(ascending=False).items():
                    c = self.geo.country(iso)
                    countries.append({"iso2": iso, "name": c.name if c else iso, "n_stories": int(n)})
            return clean({
                "database": self.db_name, "as_of": data.as_of if len(s) else None, "data_start": data.data_start if len(s) else None,
                "date_start": data.data_start.tz_convert("Asia/Kolkata").date().isoformat() if len(s) else None,
                "date_end": data.as_of.tz_convert("Asia/Kolkata").date().isoformat() if len(s) else None,
                "n_stories": int(len(s)), "n_events": data.n_events, "countries": countries,
                "taxonomy": [{"code": c, "tier": v["tier"], "name": v["name"], "parent": v["parent"]} for c, v in taxmap.items() if v["in_scope"]],
                "windows": [{"key": k, "label": lbl} for k, (_, lbl) in TR.WINDOWS.items()],
                "top_n": P["trend_top_n"], "max_n": P["trend_max_n"],
                "merge_cos": P["trend_merge_cos"], "fold_cos": P["trend_fold_cos"],
                "methodology": self.trending_methodology()["methodology"]})

    def trending_methodology(self) -> dict:
        """Fast, read-only thresholds/formulas needed by the Trending methodology panel."""
        return {"methodology": {key: C.PARAMS[key] for key in TRENDING_METHODOLOGY_KEYS}}

    def trending(self, country: Optional[str] = None, tax: Optional[str] = None, window: str = "3d", limit=None,
                 date_from: Optional[str] = None, date_to: Optional[str] = None) -> dict:
        """Top trending stories for: a country, a taxonomy node, or both. With neither, trending is undefined."""
        country = (country or "").strip().upper() or None
        if country:
            country = self._iso(country)
        if (date_from is None) != (date_to is None):
            raise ApiError("custom date range requires both date_from and date_to")

        def parse_date(name, value):
            if value is None:
                return None
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ApiError(f"{name} must be a date in YYYY-MM-DD format")
            try:
                return dt.date.fromisoformat(value)
            except ValueError:
                raise ApiError(f"{name} must be a valid date in YYYY-MM-DD format")

        date_from = parse_date("date_from", date_from)
        date_to = parse_date("date_to", date_to)
        if date_from and date_from > date_to:
            raise ApiError("date_from must be on or before date_to")
        try:
            limit = int(limit) if limit not in (None, "") else None
        except (TypeError, ValueError):
            raise ApiError("bad limit")
        with self.store() as st:
            taxmap = self._taxmap(st)
            sel = self._tax_sel(taxmap, tax)
            if not country and not sel:
                return {"needs_selection": True,
                        "message": "Trending depends on the area or topic you care about. Choose a country, a topic, or both."}
            data = self._trend_data(st)
            if date_from:
                available_start = data.data_start.tz_convert("Asia/Kolkata").date()
                available_end = data.as_of.tz_convert("Asia/Kolkata").date()
                if date_from < available_start:
                    raise ApiError(f"date_from is before available data ({available_start.isoformat()})")
                if date_to > available_end:
                    raise ApiError(f"date_to is after available data ({available_end.isoformat()})")
            names = {c: v["name"] for c, v in taxmap.items()}
            c = self.geo.country(country) if country else None
            scope = (c.name if c else "all countries")
            tax_arg = {"tier": sel["tier"], "code": sel["code"], "name": sel["name"], "path": sel["path"]} if sel else None
            try:
                res = TR.trending(data, country=country, tax=tax_arg, window=window, date_from=date_from, date_to=date_to,
                                  top_n=limit, scope_name=scope, names=names,
                                  P=C.PARAMS, evidence_fn=lambda cands, as_of, w: self._evidence(st, cands, as_of, w, names))
            except TR.TrendError as e:
                raise ApiError(str(e))
            res["scope"] = {"country": country, "country_name": c.name if c else None, "tax": tax_arg, "label": scope}
            res["pipeline_total_events"] = data.n_events
            return clean(res)

    def _evidence(self, st: Store, cands: list, as_of: pd.Timestamp, W_h: float, names: dict) -> dict:
        """Statistical evidence from the pipeline's own FDR-controlled tests (see news_trending.pipeline_evidence)."""
        return TR.pipeline_evidence(st, cands, as_of, W_h, names)

    def story_articles(self, ids: list, offset=0, limit: int = 50) -> dict:
        """Related articles of a story (all its event clusters), newest first, exact duplicates counted once."""
        if not isinstance(ids, list) or not (1 <= len(ids) <= 100) or not all(isinstance(i, str) and _EVENT_ID.match(i) for i in ids):
            raise ApiError("bad story ids")
        try:
            offset, limit = max(0, int(offset)), max(1, min(int(limit), 200))
        except (TypeError, ValueError):
            raise ApiError("bad paging")
        with self.store() as st:
            taxmap = self._taxmap(st)
            st.con.register("_sid", pd.DataFrame({"event_id": sorted(set(ids))}))
            try:
                df = st.df("""SELECT f.cluster_id, f.source_table, f.article_id, f.title, f.url, f.source_domain, f.published_at, f.domain_id, f.topic_id,
                                     f.subtopic_id, g.ev_place, g.ev_country_iso, g.loc_status
                              FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                              WHERE NOT f.is_duplicate AND f.cluster_id IN (SELECT event_id FROM _sid)""")
                copies = st.scalar("SELECT count(*) FROM article_fact WHERE is_duplicate AND dup_of IN "
                                   "(SELECT source_table || '-' || CAST(article_id AS VARCHAR) FROM article_fact WHERE cluster_id IN (SELECT event_id FROM _sid))")
            finally:
                st.con.unregister("_sid")
            df = self._newest_first(df)
            return clean({"total": int(len(df)), "offset": offset, "exact_copies": int(copies or 0),
                          "items": self._items(df.iloc[offset:offset + limit], taxmap) if len(df) else []})

    # ---- event detail ----------------------------------------------------------------------------------
    def event_detail(self, event_id: str, aoi_id: Optional[str] = None) -> dict:
        if not (isinstance(event_id, str) and _EVENT_ID.match(event_id)):
            raise ApiError("bad event id")
        with self.store() as st:
            ev = st.df("SELECT * FROM event_cluster WHERE event_id = ?", [event_id])
            if not len(ev):
                raise ApiError("unknown event", 404)
            e = ev.iloc[0].drop(labels=["centroid_sum"]).to_dict()
            m = st.df("""SELECT f.source_table, f.article_id, f.title, f.url, f.source_domain, f.published_at, f.is_duplicate,
                                f.cluster_sim, f.has_label, f.domain_id, f.topic_id, f.category_id, f.subtopic_id, g.ev_place, g.ev_country_iso, g.ev_adm1_code, g.ev_precision,
                                g.loc_status, g.geo_inconsistent
                         FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                         WHERE f.cluster_id = ? ORDER BY f.published_at LIMIT 600""", [event_id])
            dups = st.df("SELECT count(*) n FROM article_fact WHERE is_duplicate AND dup_of IN "
                         "(SELECT source_table || '-' || CAST(article_id AS VARCHAR) FROM article_fact WHERE cluster_id = ?)", [event_id]).iloc[0].n
            metrics = st.df("SELECT * FROM event_metrics WHERE event_id = ?", [event_id])
            tests = st.df("SELECT * FROM event_aoi_test WHERE event_id = ?", [event_id])
            devs = st.df("SELECT * FROM event_development WHERE event_id = ? ORDER BY detected_asof", [event_id])
            tax = {r.code: r.name for r in st.df("SELECT code, name FROM taxonomy_node").itertuples()}
        reg = self.registry()
        arts = []
        for r in m.to_dict("records"):
            pc = PB.publisher_country(r["source_domain"], reg)
            url = r["url"] if isinstance(r["url"], str) and r["url"].startswith(("http://", "https://")) else None
            arts.append({**{k: r[k] for k in ("title", "source_domain", "published_at", "is_duplicate", "cluster_sim",
                                              "ev_place", "ev_country_iso", "ev_precision", "loc_status", "geo_inconsistent")},
                         "url": url, "publisher_country": pc["country"], "publisher_country_status": pc["status"]})
        live = m[~m["is_duplicate"]].copy()
        live["published_at"] = pd.to_datetime(live["published_at"], utc=True)
        hourly = live.set_index("published_at").resample("3h").size() if len(live) else pd.Series(dtype=int)
        loc_counts = live.groupby(live["ev_country_iso"].fillna("unlocated")).size().sort_values(ascending=False).to_dict()
        pub_counts = live.groupby("source_domain").size().sort_values(ascending=False).head(15).to_dict()
        pth = TX.assign_paths(m[~m["is_duplicate"]].assign(event_id=event_id), "event_id", min_agreement=C.PARAMS["taxonomy_min_agreement"])
        agreed = {t: (pth[t].iat[0] if len(pth) else None) for t in ("domain", "topic", "category", "subtopic")}
        names = {f"{t}_name": (tax.get(c) if isinstance(c, str) else None) for t, c in agreed.items()}
        return clean({"event": {**e, **names, "tax_status": pth["tax_status"].iat[0] if len(pth) else "unlabelled"},
                      "metrics": metrics.to_dict("records")[0] if len(metrics) else None, "articles": arts,
                      "n_copies_exact": int(dups), "timeline_3h": [[str(k), int(v)] for k, v in hourly.items()],
                      "location_counts": loc_counts, "top_publishers": pub_counts, "tests": tests, "developments": devs,
                      "location_note": "Event location is GDELT ActionGeo (machine-extracted); publisher country comes from a separate "
                                       "registry and is shown only as reviewed/auto/unknown - never mixed with event location."})

    # ---- alerts feed ------------------------------------------------------------------------------------
    def alerts(self, aoi_id: Optional[str], kind: Optional[str], status: str = "active", limit: int = 200) -> dict:
        if status not in ("active", "retracted", "all"):
            raise ApiError("bad status")
        q, args = "SELECT * FROM alert WHERE 1=1", []
        if aoi_id:
            q += " AND aoi_id = ?"; args.append(aoi_id)
        if kind:
            if kind not in ("node_trend", "emerging_event", "new_development"):
                raise ApiError("bad kind")
            q += " AND kind = ?"; args.append(kind)
        if status != "all":
            q += " AND status = ?"; args.append(status)
        with self.store() as st:
            df = st.df(q + " ORDER BY as_of DESC, q LIMIT ?", args + [max(1, min(int(limit), 500))])
            tax = {r.code: r.name for r in st.df("SELECT code, name FROM taxonomy_node").itertuples()}
        return clean({"alerts": self._alerts(df, tax)})


# --------------------------------------------------------------------------
# routing (called by api_server.Handler)
# --------------------------------------------------------------------------
def route_get(api: NewsAPI, path: str, qs: dict) -> Any:
    one = lambda k, d=None: (qs.get(k) or [d])[0]
    if path == "/api/news/meta":
        return api.meta()
    if path == "/api/news/gazetteer":
        return api.gazetteer()
    if path == "/api/news/admin1":
        return api.admin1_list(one("iso2"))
    if path == "/api/news/boundaries":
        return api.boundaries(one("level", "country"), one("iso2"))
    if path == "/api/news/aoi":
        return api.bundle(aoi_id=one("id"), level=one("level", "topic"), tax=one("tax"))
    if path == "/api/news/event":
        return api.event_detail(one("id"), one("aoi"))
    if path == "/api/news/live":
        return api.live()
    if path == "/api/news/health":
        return api.health()
    if path == "/api/news/trending_meta":
        return api.trending_meta()
    if path == "/api/news/trending_methodology":
        return api.trending_methodology()
    if path == "/api/news/trending":
        return api.trending(one("country"), one("tax"), one("window", "3d"), one("limit"),
                            one("date_from"), one("date_to"))
    if path == "/api/news/story_articles":
        return api.story_articles([i for i in (one("ids") or "").split(",") if i], one("offset", 0), one("limit", 50))
    if path == "/api/news/alerts":
        return api.alerts(one("aoi"), one("kind"), one("status", "active"), one("limit", 200))
    raise ApiError("unknown endpoint", 404)


def route_post(api: NewsAPI, path: str, body: dict) -> Any:
    if path == "/api/news/adhoc":
        return api.bundle(spec=body.get("aoi"), level=body.get("level", "topic"), tax=body.get("tax"))
    if path == "/api/news/articles":
        return api.articles(aoi_id=body.get("aoi_id"), spec=body.get("aoi"), tax=body.get("tax"), node=body.get("node"),
                            day=body.get("day"), offset=body.get("offset", 0), limit=body.get("limit", ARTICLE_PAGE))
    raise ApiError("unknown endpoint", 404)
