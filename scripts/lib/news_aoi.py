"""AOI data layer: which articles / events belong to an Area of Interest.

The ONE membership implementation (`aoi_frame` + `aoi_events`) is used for pre-registered
AOIs (pilot, backfill, incremental) and ad-hoc AOIs from the API. Baselines and tests are then
computed on the AOI's own articles (see news_metrics.py) - never by filtering a global result.

Event membership: an event belongs to an AOI when at least `location_consensus_min_share` of
its LOCATED member articles fall inside the AOI. An event with no located member belongs to no
event-mode AOI (it is reported as 'unlocated', never guessed).
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Optional

import numpy as np
import pandas as pd

from . import news_config as C
from . import news_geo as G
from .news_store import Store, now_utc

FRAME_COLS = """f.source_table, f.article_id, f.day, f.published_at, f.has_label, f.domain_id, f.topic_id,
       f.category_id, f.subtopic_id, f.source_domain, f.cluster_id, f.title, f.url,
       g.ev_country_iso, g.ev_adm1_code, g.ev_precision, g.ev_lat, g.ev_lon, g.geo_inconsistent,
       g.mention_countries_iso, g.loc_status, g.ev_place"""


def _prefilter(aoi: G.AOI) -> tuple[str, list]:
    if aoi.mode == "mentioned":
        return "AND g.mention_countries_iso @> ARRAY[?]::text[]", [aoi.iso2]
    if aoi.kind == "country":
        return "AND g.ev_country_iso = ?", [aoi.iso2]
    if aoi.kind == "adm1":
        return "AND g.ev_adm1_code = ?", [aoi.key]
    w, s, e, n = aoi.geom.bounds
    return ("AND g.ev_lat BETWEEN ? AND ? AND g.ev_lon BETWEEN ? AND ?", [s, n, w, e])


def aoi_frame(store: Store, aoi: G.AOI, day_from: Optional[dt.date] = None,
              day_to: Optional[dt.date] = None, include_duplicates: bool = False) -> pd.DataFrame:
    """Article rows (with geo) that match the AOI. Duplicates (same content_hash) excluded by default."""
    where = "WHERE 1=1"
    args: list = []
    if not include_duplicates:
        where += " AND NOT f.is_duplicate"
    if day_from:
        where += " AND f.day >= ?"
        args.append(day_from)
    if day_to:
        where += " AND f.day <= ?"
        args.append(day_to)
    pf, pa = _prefilter(aoi)
    df = store.df(f"SELECT {FRAME_COLS} FROM article_fact f JOIN article_geo g USING (source_table, article_id) "
                  f"{where} {pf}", args + pa)
    if not len(df):
        return df
    return df[G.aoi_mask(aoi, df)].reset_index(drop=True)


def aoi_events(store: Store, aoi: G.AOI, P: dict, day_from: Optional[dt.date] = None,
               day_to: Optional[dt.date] = None) -> pd.DataFrame:
    """Event clusters located in the AOI (event-mode AOIs only).

    Returns event_cluster rows plus `aoi_share` (share of the event's located members inside the
    AOI) and `aoi_located` (n located members). Time window filters on the event's first_seen day.
    """
    if aoi.mode != "event":
        return pd.DataFrame()
    fr = aoi_frame(store, aoi, day_from, day_to)
    if not len(fr):
        return pd.DataFrame()
    cids = sorted({c for c in fr["cluster_id"].dropna().unique()})
    if not cids:
        return pd.DataFrame()
    ids = pd.DataFrame({"event_id": cids})
    store.con.register("_cids", ids)
    try:
        mem = store.df("""SELECT f.cluster_id, g.ev_country_iso, g.ev_adm1_code, g.ev_precision, g.ev_lat, g.ev_lon,
                                 g.geo_inconsistent, g.mention_countries_iso, g.loc_status
                          FROM article_fact f JOIN article_geo g USING (source_table, article_id)
                          WHERE NOT f.is_duplicate AND f.cluster_id IN (SELECT event_id FROM _cids)""")
        ev = store.df("SELECT * FROM event_cluster WHERE event_id IN (SELECT event_id FROM _cids)")
    finally:
        store.con.unregister("_cids")
    mem["_in"] = G.aoi_mask(aoi, mem)
    located = mem[mem["loc_status"] == "located"]
    agg = located.groupby("cluster_id").agg(aoi_located=("_in", "size"), aoi_hits=("_in", "sum")).reset_index()
    agg["aoi_share"] = agg["aoi_hits"] / agg["aoi_located"]
    out = ev.merge(agg, left_on="event_id", right_on="cluster_id", how="inner")
    out = out[out["aoi_share"] >= P["location_consensus_min_share"]].drop(columns=["cluster_id", "centroid_sum"])
    return out.reset_index(drop=True)


def register_aoi(store: Store, aoi: G.AOI, parent: Optional[str] = None) -> None:
    geom = None if aoi.geom is None else json.dumps(G.shapely.geometry.mapping(aoi.geom))
    store.upsert("aoi_def", pd.DataFrame([{
        "aoi_id": aoi.aoi_id, "kind": aoi.kind, "key": aoi.key, "name": aoi.name, "mode": aoi.mode,
        "min_precision": aoi.min_precision, "adhoc": aoi.adhoc, "iso2": aoi.iso2, "parent_aoi": parent,
        "geometry_json": geom, "registered_at": now_utc()}]))


def pilot_aois(geo: G.GeoIndex, countries: list[str], include_admin1: bool = True,
               include_mentioned: bool = True) -> list[tuple[G.AOI, Optional[str]]]:
    """Pre-registered AOI family for a scope: country (event) [+ its admin-1 units] [+ mentioned]."""
    out: list[tuple[G.AOI, Optional[str]]] = []
    for c in countries:
        parent = G.country_aoi(geo, c)
        out.append((parent, None))
        if include_admin1:
            for a1 in geo.admin1_of(c):
                out.append((G.admin1_aoi(geo, a1.code), parent.aoi_id))
        if include_mentioned:
            out.append((G.country_aoi(geo, c, "mentioned"), None))
    return out


def live_aois(store: Store, geo: G.GeoIndex, P: dict) -> list[tuple[G.AOI, Optional[str]]]:
    """AOI family of the all-country live scope, derived from the data itself: an event AOI for every country that has
    located articles, a mentioned-in AOI for every country mentioned in articles, and admin-1 units only for the
    countries in P['admin1_countries']. Countries the geography index does not know are skipped (never guessed)."""
    floor = int(P["live_min_country_articles"])
    ev = store.df("SELECT ev_country_iso AS c, count(*) AS n FROM article_geo WHERE ev_country_iso IS NOT NULL GROUP BY 1")
    mn = store.df("SELECT c, count(*) AS n FROM (SELECT unnest(mention_countries_iso) AS c FROM article_geo) m WHERE c IS NOT NULL GROUP BY 1")
    ev_c = {r.c for r in ev.itertuples() if r.n >= floor and geo.country(r.c)}
    mn_c = {r.c for r in mn.itertuples() if r.n >= floor and geo.country(r.c)}
    out: list[tuple[G.AOI, Optional[str]]] = []
    for c in sorted(ev_c):
        parent = G.country_aoi(geo, c)
        out.append((parent, None))
        if c in P["admin1_countries"]:
            for a1 in geo.admin1_of(c):
                out.append((G.admin1_aoi(geo, a1.code), parent.aoi_id))
    for c in sorted(mn_c):
        out.append((G.country_aoi(geo, c, "mentioned"), None))
    return out
