"""Is a story about the taxonomy's subject at all? A read-side second opinion that never touches a stored label.

Why it exists. The upstream classifier was trained on ~88 k *security / geopolitics* articles of which only 1.3 k are labelled
"Non-relevant / Other" (D00). It therefore forces almost everything it sees into one of D01-D09: a film trailer, a hospital
admission or a cricket result gets a confident "Terrorism" or "Military" label, and a majority of such labels inside a story
is still wrong (hand-judged sample: ~40 % of stories listed under a domain did not belong there).

The signal. The classifier's own ground truth (`tmc.a0_ground_truth`, bge-m3 embeddings fetched from OpenSearch, read-only) is
a map of what the taxonomy is about. For each event we take its centroid and look at its 10 nearest ground-truth articles:
  ref_sim    mean cosine to the 5 nearest        -> far from everything the taxonomy covers = off-topic
  ref_votes  the domains (D00-D09) of the 10 nearest, e.g. "D05:6,D01:3,D02:1"  -> do the neighbours agree with the assigned domain?
A story is shown only when ref_sim >= relevance_min_similarity and, for a domain/topic selection, at least
relevance_min_agreement of its neighbours belong to that domain. Thresholds were chosen on a hand-judged sample and then
validated on a second one (see docs/NEWS_ENGINE.md). Events with no score (no centroid / no reference file) are NOT filtered:
unknown is never treated as irrelevant.

Storage: one float16 matrix (~170 MB) + the domain of each row, built once from existing data; per-event results are two small
columns of `event_cluster`. Nothing is written to any source system.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import news_config as C

K_NEIGHBOURS = 10
K_SIM = 5
REFERENCE_FILE = Path(os.environ["NEWS_REFERENCE_FILE"]) if os.environ.get("NEWS_REFERENCE_FILE") else C.STATE_DIR / "relevance_reference.npz"
DOMAINS = [f"D{i:02d}" for i in range(10)]


# --------------------------------------------------------------------------
# reference set (built once from existing data)
# --------------------------------------------------------------------------
def build_reference(pg, osc, path: Path = REFERENCE_FILE, page: int = 200, log=print) -> int:
    """Fetch the ground-truth articles' embeddings (read-only) and save them with their domain. Returns the row count."""
    rows = pg.query("SELECT source_table, source_id, gt_domain_id FROM tmc.a0_ground_truth "
                    "WHERE source_table IN ('raw.scraped_articles', 'raw.scraped_articles_trusted') AND gt_domain_id IS NOT NULL")
    if not rows:
        raise RuntimeError("tmc.a0_ground_truth returned no rows")
    ids = [r["source_table"].replace("raw.", "", 1) + "-" + r["source_id"] for r in rows]
    dom = {i: r["gt_domain_id"] for i, r in zip(ids, rows)}
    vecs, keep = [], []
    for i in range(0, len(ids), page):
        chunk = ids[i:i + page]
        hits = osc.fetch_by_ids(chunk, include_embedding=True, batch=page)
        for k in chunk:
            h = hits.get(k)
            e = None if not h else h["_source"].get("embedding")
            if e is None:
                continue
            v = np.asarray(e, dtype=np.float32)
            n = float(np.linalg.norm(v))
            if v.shape == (C.EMBEDDING_DIM,) and np.isfinite(v).all() and n > 1e-6:
                vecs.append((v / n).astype(np.float16))
                keep.append(k)
        if (i // page) % 100 == 0:
            log(f"reference: {i}/{len(ids)} fetched, {len(keep)} usable")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, vec=np.stack(vecs), domain=np.array([dom[k] for k in keep]), ids=np.array(keep))
    os.replace(tmp, path)
    return len(keep)


class Reference:
    def __init__(self, vec: np.ndarray, domain: np.ndarray):
        self.R = np.ascontiguousarray(vec, dtype=np.float32)
        self.dom = np.array([DOMAINS.index(d) if d in DOMAINS else -1 for d in domain], dtype=np.int8)
        self.n = len(self.R)


def load_reference(path: Path = REFERENCE_FILE) -> Optional[Reference]:
    if not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=False)
    return Reference(z["vec"], z["domain"])


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score_vectors(ref: Reference, U: np.ndarray, block: int = 256) -> tuple[np.ndarray, list[str]]:
    """U: (n, d) unit vectors -> (ref_sim (n,), ref_votes strings). Block-wise matrix products; exact (no approximation)."""
    n = len(U)
    sim = np.zeros(n, dtype=np.float64)
    votes: list[str] = []
    for i in range(0, n, block):
        S = U[i:i + block].astype(np.float32) @ ref.R.T
        idx = np.argpartition(-S, K_NEIGHBOURS - 1, axis=1)[:, :K_NEIGHBOURS]
        top = np.take_along_axis(S, idx, axis=1)
        order = np.argsort(-top, axis=1)
        idx, top = np.take_along_axis(idx, order, axis=1), np.take_along_axis(top, order, axis=1)
        sim[i:i + block] = top[:, :K_SIM].mean(axis=1)
        for row in idx:
            c = np.bincount(ref.dom[row][ref.dom[row] >= 0], minlength=len(DOMAINS))
            votes.append(",".join(f"{DOMAINS[d]}:{int(c[d])}" for d in np.argsort(-c, kind="stable") if c[d] > 0))
    return sim, votes


def score_events(st, ref: Reference, event_ids: Optional[set] = None, only_missing: bool = False) -> int:
    """Score `is_event` clusters that still have a centroid (event_ids=None -> every one; only_missing -> those with no score yet).
    Writes ref_sim / ref_votes on event_cluster. Returns the number scored."""
    q = "SELECT event_id, centroid_sum FROM event_cluster WHERE is_event AND centroid_sum IS NOT NULL"
    if only_missing:
        q += " AND ref_sim IS NULL"
    ev = st.df(q)
    if event_ids is not None:
        ev = ev[ev["event_id"].isin(event_ids)]
    if not len(ev):
        return 0
    ok = ev["centroid_sum"].map(lambda b: isinstance(b, (bytes, bytearray, memoryview)) and len(b) == C.EMBEDDING_DIM * 4)
    ev = ev[ok.to_numpy()]
    if not len(ev):
        return 0
    U = np.stack([np.frombuffer(b, dtype=np.float32) for b in ev["centroid_sum"]])
    nrm = np.linalg.norm(U, axis=1, keepdims=True)
    good = (nrm[:, 0] > 0)
    U, ev = U[good] / nrm[good], ev[good]
    sim, votes = score_vectors(ref, U)
    out = pd.DataFrame({"event_id": ev["event_id"].to_numpy(), "ref_sim": sim, "ref_votes": votes})
    st.con.register("_rel", out)
    try:
        with st.tx():
            st.con.execute("UPDATE event_cluster SET ref_sim = r.ref_sim, ref_votes = r.ref_votes FROM _rel r "
                           "WHERE event_cluster.event_id = r.event_id")
    finally:
        st.con.unregister("_rel")
    return len(out)


# --------------------------------------------------------------------------
# the decision (pure)
# --------------------------------------------------------------------------
def parse_votes(s) -> dict:
    if not isinstance(s, str) or not s:
        return {}
    return {a: int(b) for a, b in (p.split(":") for p in s.split(","))}


def agreement(votes: dict, domain: Optional[str]) -> Optional[float]:
    """Share of the neighbours that belong to `domain` (None when unknown)."""
    tot = sum(votes.values())
    if not tot or not domain:
        return None
    return votes.get(domain, 0) / tot


def passes(ref_sim, votes, domain: Optional[str], P: dict) -> bool:
    """True when the story may be listed. Unknown scores pass (never treated as irrelevant)."""
    if ref_sim is None or (isinstance(ref_sim, float) and ref_sim != ref_sim):
        return True
    if ref_sim < P["relevance_min_similarity"]:
        return False
    if domain:
        a = agreement(votes if isinstance(votes, dict) else parse_votes(votes), domain)
        if a is not None and a < P["relevance_min_agreement"]:
            return False
    return True
