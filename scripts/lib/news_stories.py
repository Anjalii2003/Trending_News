"""Stories and rankings materialised in PostgreSQL after each successful run.

`story` / `story_event`   events of the same story merged (news_trending.merge_events), with the agreed taxonomy path, the country
                          verdict and the relevance score - what the Trending page ranks.
`trending_ranking`        the top stories per window for every country and every domain, exactly as the API ranks them
                          (same function, same parameters, same statistical evidence).
They are a materialised view of the committed clusters: replaced wholesale in one transaction, derived only from data already in
`news_engine`. The dashboard still ranks on request (it supports any country x topic x window); these tables are for consumers
that want the stored result.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np
import pandas as pd

from . import news_config as C
from . import news_trending as TR
from .news_store import Store

WINDOWS = ("24h", "3d", "7d")
ANY = "*"


def _ts(x) -> Optional[dt.datetime]:
    return None if x is None or (isinstance(x, float) and x != x) else dt.datetime.fromtimestamp(float(x), dt.timezone.utc)


def _votes(d) -> Optional[str]:
    return ",".join(f"{k}:{v}" for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))) if isinstance(d, dict) and d else None


def materialise(st: Store, run_id: str, top_n: int = 30, P: Optional[dict] = None) -> dict:
    """Replace the story and ranking tables from the current clusters. Never raises: a failure is returned (and logged by the
    caller) so one derived table cannot fail a run whose core data is already committed."""
    P = P or C.PARAMS
    out = {"stories": 0, "story_events": 0, "rankings": 0}
    try:
        data = TR.load(st, P)
        s = data.stories
        if not len(s):
            return out
        as_of = data.as_of.to_pydatetime()
        assigned = s["tax_status"] == "assigned"
        rows = pd.DataFrame({
            "story_id": s["story_id"], "as_of": as_of, "title": s["title"], "n_events": s["event_ids"].map(len),
            "n_articles": s["n_articles"], "n_publishers": s["n_publishers"],
            "first_seen": s["first_ts"].map(_ts), "last_seen": s["last_ts"].map(_ts),
            "domain_id": s["domain"].where(assigned), "topic_id": s["topic"].where(assigned),
            "category_id": s["category"].where(assigned), "subtopic_id": s["subtopic"].where(assigned),
            "tax_status": s["tax_status"], "country_iso": s["country"], "place": s["place"], "n_located": s["n_located"],
            "n_in_country": s["n_in_country"], "loc_share": s["loc_share"], "ref_sim": s["ref_sim"],
            "ref_votes": s["ref_votes"].map(_votes), "run_id": run_id})
        links = pd.DataFrame([(sid, e) for sid, eids in zip(s["story_id"], s["event_ids"]) for e in eids], columns=["story_id", "event_id"])
        names = {r[0]: r[1] for r in st.con.execute("SELECT code, name FROM taxonomy_node").fetchall()}
        rank_rows = []
        countries = sorted({c for c in s["country"].dropna().unique() if len(TR.in_country(s, c, P))})
        domains = sorted({d for d in s["domain"].dropna().unique() if d != "D00"})
        ev_fn = lambda cands, a, w: TR.pipeline_evidence(st, cands, a, w, names)
        for w in WINDOWS:
            jobs = [(c, None) for c in countries] + [(None, {"tier": "domain", "code": d, "name": names.get(d, d)}) for d in domains]
            for country, tax in jobs:
                res = TR.trending(data, country=country, tax=tax, window=w, top_n=top_n, names=names, P=P, evidence_fn=ev_fn)
                for x in res.get("stories", []):
                    rank_rows.append({"as_of": as_of, "window_key": w, "country_iso": country or ANY, "tax_code": (tax or {}).get("code", ANY),
                                      "rank": x["rank"], "story_id": x["story_id"], "score": x["score"],
                                      "n_articles_window": x["in_window"]["articles"], "n_publishers_window": x["in_window"]["publishers"],
                                      "run_id": run_id})
        with st.tx():
            st.con.execute("DELETE FROM story_event")
            st.con.execute("DELETE FROM story")
            st.con.execute("DELETE FROM trending_ranking")
            st.upsert("story", rows)
            st.upsert("story_event", links)
            st.upsert("trending_ranking", pd.DataFrame(rank_rows))
        out.update(stories=len(rows), story_events=len(links), rankings=len(rank_rows), countries=len(countries), domains=len(domains))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"[news-stories] materialisation failed: {out['error']}", flush=True)
    return out
