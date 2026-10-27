"""In-memory stand-ins for the read-only source systems, so the live service runs end-to-end with no network.
The fake OpenSearch evaluates the same range filters the engine sends (published_at day scans, ingested_at watermark)."""
from __future__ import annotations

import datetime as dt

import numpy as np

import news_testutil  # noqa: F401  (sys.path)
from lib import news_config as C

DIM = C.EMBEDDING_DIM
_GEO = None


def fips(iso):
    """GDELT carries FIPS country codes; tests speak ISO."""
    global _GEO
    from lib import news_geo as G
    _GEO = _GEO or G.GeoIndex()
    return _GEO.iso_to_fips(iso)


def topic_vector(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def near(base: np.ndarray, rng, noise: float = 0.05) -> list[float]:
    v = base + rng.normal(scale=noise / np.sqrt(DIM) * 3, size=DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _ts(v) -> dt.datetime:
    return dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))


class FakeOS:
    """Holds raw hits + vectors. `add()` can be called between cycles to simulate new articles arriving."""

    def __init__(self):
        self.hits: list[dict] = []
        self.vectors: dict[str, list] = {}
        self.queries: list[dict] = []
        self.fail_scan = 0                  # raise on the next N scans (outage simulation)

    def add(self, i: int, *, published: dt.datetime, ingested: dt.datetime, vec, country="IN", domain="a.com",
            content_hash=None, title=None, st="scraped_articles") -> str:
        oid = f"{st}-{i}"
        src = {"article_id": str(i), "source_table": st, "published_at": published.isoformat(), "published_at_conflict": False,
               "published_at_confidence": "high", "ingested_at": ingested.isoformat(), "source_domain": domain,
               "url": f"https://{domain}/{i}", "title": title or f"title {i}", "content_hash": content_hash or f"hash{i}",
               "gdelt": {"action_country": fips(country), "action_location": None, "action_geo": None} if country else {},
               "gkg": {"locations_country_code": [fips(country)] if country else []}}
        self.hits.append({"_id": oid, "_index": "full-articles-000002", "_source": src, "sort": [published.isoformat(), st, i]})
        self.vectors[oid] = list(vec)
        return oid

    def _filter(self, query):
        flt = query.get("bool", {}).get("filter", [])
        out = list(self.hits)
        for f in flt:
            if "range" in f:
                (field, cond), = f["range"].items()
                lo, hi = cond.get("gte"), cond.get("lt")
                out = [h for h in out if (lo is None or _ts(h["_source"][field]) >= _ts(lo)) and (hi is None or _ts(h["_source"][field]) < _ts(hi))]
        return out

    def scan(self, query, include_embedding=False, page_size=None):
        self.queries.append(query)
        if self.fail_scan > 0:
            self.fail_scan -= 1
            raise RuntimeError("OpenSearch request failed after retries: simulated outage")
        hits = self._filter(query)
        if hits:
            yield hits

    def count(self, query):
        return len(self._filter(query))

    def fetch_by_ids(self, ids, include_embedding=True, batch=200):
        return {i: {"_id": i, "_source": {"embedding": self.vectors[i]}} for i in ids if i in self.vectors}


class FakePG:
    def close(self):
        pass
