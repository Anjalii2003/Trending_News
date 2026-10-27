"""Configuration for the News Trending & Emerging Events engine.

Everything here is import-safe: nothing reads credentials or touches the network at
import time, so unit tests and the stats code can import it without a database.
Credentials are resolved lazily by `pg_settings()` / `os_settings()` from the
environment (optionally loaded from a .env file) and are NEVER hardcoded.

Resolution order for each source (first hit wins):
  PostgreSQL : NEWS_PG_URL | SOURCE_DATABASE_URL | NEWS_PG_{HOST,PORT,USER,PASSWORD,DB}
  OpenSearch : NEWS_OS_{URL,USER,PASSWORD,VERIFY_CERTS} | OPENSEARCH_{URL,USER,PASSWORD,VERIFY_CERTS}
The `SOURCE_DATABASE_URL` / `OPENSEARCH_*` aliases exist so the sibling project's existing
.env can be reused without copying secrets: set `NEWS_ENV_FILE=/path/to/that/.env`.

Both sources are strictly read-only; see `news_sources.py` for the enforcement.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

EDA_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = EDA_ROOT / "data"
GEO_DIR = DATA_DIR / "geo"

load_dotenv(EDA_ROOT / ".env", override=False)
if os.environ.get("NEWS_ENV_FILE"):
    load_dotenv(os.environ["NEWS_ENV_FILE"], override=False)

# State directory: the run lock and the service heartbeat live here (docker-compose sets NEWS_STATE_DIR=/state, a volume). All
# derived DATA lives in the PostgreSQL schema `news_engine` (lib/news_store.py) - there is no local database file.
STATE_DIR = Path(os.environ["NEWS_STATE_DIR"]) if os.environ.get("NEWS_STATE_DIR") else DATA_DIR
NEWS_LOCK = Path(os.environ["NEWS_LOCK_PATH"]) if os.environ.get("NEWS_LOCK_PATH") else STATE_DIR / "news_run.lock"
NEWS_REPORT_DIR = EDA_ROOT / "reports" / "news"

OPENSEARCH_INDEX = "full-articles"
EMBEDDING_DIM = 1024
SOURCE_TABLES = ("scraped_articles", "scraped_articles_trusted")


class ConfigError(RuntimeError):
    pass


def _first(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def pg_settings() -> dict:
    """libpq connection kwargs. Raises ConfigError (never returns defaults) if unset."""
    url = _first("NEWS_PG_URL", "SOURCE_DATABASE_URL")
    if url:
        # SQLAlchemy-style 'postgresql+psycopg://' -> plain scheme
        u = urlparse(url.replace("postgresql+psycopg2://", "postgresql://")
                        .replace("postgresql+psycopg://", "postgresql://"))
        if not (u.hostname and u.username and u.password and u.path.strip("/")):
            raise ConfigError("NEWS_PG_URL/SOURCE_DATABASE_URL is missing host/user/password/db")
        return {"host": u.hostname, "port": u.port or 5432, "user": unquote(u.username),
                "password": unquote(u.password), "dbname": u.path.strip("/")}
    need = {k: os.environ.get(f"NEWS_PG_{k.upper()}") for k in
            ("host", "port", "user", "password", "db")}
    missing = [f"NEWS_PG_{k.upper()}" for k, v in need.items() if not v]
    if missing:
        raise ConfigError(f"PostgreSQL not configured; set {', '.join(missing)} "
                          "(or NEWS_PG_URL, or NEWS_ENV_FILE). No defaults are applied.")
    return {"host": need["host"], "port": int(need["port"]), "user": need["user"],
            "password": need["password"], "dbname": need["db"]}


def os_settings() -> dict:
    url = _first("NEWS_OS_URL", "OPENSEARCH_URL")
    user = _first("NEWS_OS_USER", "OPENSEARCH_USER")
    pw = _first("NEWS_OS_PASSWORD", "OPENSEARCH_PASSWORD")
    if not (url and user and pw):
        raise ConfigError("OpenSearch not configured; set NEWS_OS_URL, NEWS_OS_USER, "
                          "NEWS_OS_PASSWORD (or the OPENSEARCH_* aliases / NEWS_ENV_FILE).")
    verify = (_first("NEWS_OS_VERIFY_CERTS", "OPENSEARCH_VERIFY_CERTS") or "true").lower()
    return {"url": url.rstrip("/"), "user": user, "password": pw,
            "verify": verify in ("1", "true", "yes", "on")}


# ==========================================================================
# PARAMS - the single place to retune. Spec: docs/NEWS_ENGINE.md.
# The same values apply to pilot, historical backfill and incremental runs.
# ==========================================================================
PARAMS: dict = {
    # --- day axis ----------------------------------------------------------
    "day_tz": "UTC",                 # timezone used to bucket published_at into days
    # --- date reliability --------------------------------------------------
    # Rows without a usable published_at are EXCLUDED and counted; scraped_at is
    # never substituted. A date flagged as conflicting is excluded from trend counts.
    "exclude_date_conflict": True,
    "max_future_days": 1,            # published_at beyond now + this is invalid
    "min_valid_year": 2000,
    # --- dedup -------------------------------------------------------------
    "dedupe_by_content_hash": True,  # count one article per content_hash (earliest published)
    # --- event clustering (online, centroid-based) ------------------------
    "sim_threshold": 0.80,           # cosine to cluster centroid (matches Trending_news B3w / tne default)
    "max_gap_hours": 72.0,           # a cluster stops accepting articles this long after its last article
    "centroid_retention_days": 14,   # centroid blobs of older clusters are dropped (cluster rows are kept)
    "event_min_articles": 3,         # a cluster is an "event" only with >= this many articles ...
    "event_min_publishers": 2,       # ... from >= this many distinct publishers
    # --- location ---------------------------------------------------------
    "location_consensus_min_share": 0.5,   # share of located members that must agree
    "location_min_located": 1,
    "loc_high_min_located": 3,        # an event location is HIGH confidence only with >= this many located members ...
    "loc_high_min_share": 0.8,        # ... and >= this share agreeing
    "alert_min_loc_confidence": "high",  # emerging-event / development ALERTS need this location confidence
    # --- node x AOI x day panel (NB baseline; mirrors Metrics v2 defaults) -
    "node_levels": ["topic", "category"],
    "baseline_window": 28,
    "baseline_min_days": 7,          # Metrics v2 uses 14; pilot window is 14 days. Reported as a limitation.
    "baseline_min_mean": 0.5,
    "support_min_v": 3,
    "nb_use_exposure_offset": True,
    "exposure_smooth_days": 7,       # trailing mean of exposure used by the offset (see news_metrics.series_panel)
    "exposure_scale_cap": 3.0,
    "nb_use_drift_adjust": True,
    "alpha_count": 1.0,
    "savgol_window": 5,
    "savgol_order": 2,
    "burst_s": 2.0,
    "burst_gamma": 1.0,
    "burst_cusum_h": 5.0,
    "burst_min_days_kleinberg": 21,
    "burst_min_total_kleinberg": 30,
    "burst_min_mean_kleinberg": 0.5,
    "ar1_min_obs": 12,
    "ljung_box_lags": 7,
    "node_min_total_v": 20,          # series with fewer articles in the whole window get no time series
    "ema_fast": 7,
    "ema_slow": 21,
    "growth_7d_lag": 7,
    "aoi_min_total_articles": 100,   # an AOI needs this many labelled articles before any node series is built
    "min_exposure_day": 30,          # a day with fewer in-AOI labelled articles is a DATA GAP, not zero volume
    "alert_min_ratio": 2.0,          # effect-size floor for node alerts: V >= ratio x baseline mean
    "severity_q_high": 0.01,
    "severity_min_v_high": 10,
    # --- multiple testing ---------------------------------------------------
    "fdr_method": "bh",
    "fdr_alpha": 0.05,
    "fdr_pvalue_kind": "midp",
    # --- emerging events ----------------------------------------------------
    "emerging_window_hours": 48,     # an event is "emerging" while first_seen is within this
    "emerging_ref_days": 28,         # reference = clusters first seen in the trailing N days ...
    "emerging_min_ref": 50,          # ... need at least this many reference clusters (else NULL)
    "emerging_min_publishers": 3,    # effect-size floor for an alert
    # --- new developments ---------------------------------------------------
    "dev_min_event_articles": 8,
    "dev_min_event_publishers": 3,
    "dev_recent_hours": 24,
    "dev_min_new_articles": 3,
    "dev_min_new_publishers": 2,
    "dev_min_coherence": 0.75,       # mean cosine of the new articles to their own centroid
    "dev_envelope_pct": 10,          # new articles' median similarity must fall below this percentile of old articles'
    "dev_max_events_per_run": 200,   # bound on OpenSearch vector re-fetches per run
    # --- batching / reliability --------------------------------------------
    "os_page_size": 500,
    "os_max_retries": 4,
    "pg_batch": 2000,
    "incremental_overlap_hours": 6,  # re-scan window to catch late arrivals (reruns are idempotent)
    # --- trending stories (read-time ranking; see news_trending.py) ----------------------------------------
    "trend_merge_cos": 0.90,          # events whose centroids are this similar AND overlap in time are ONE story
    "trend_merge_gap_hours": 72,
    "trend_fold_cos": 0.88,           # display-time: a lower-ranked story this close to a LISTED one is shown under it
    "trend_min_window_articles": 3,   # a story must have this much activity in the window to be ranked ...
    "trend_min_window_publishers": 2, # ... from this many distinct publishers
    "trend_min_ref": 50,              # comparable story-windows needed before a significance test is reported
    "trend_growth_min_total": 5,      # growth needs >= this many articles across the two windows (else NULL)
    "trend_window_tolerance_hours": 2,
    "trend_weights": {"volume": 0.30, "breadth": 0.30, "growth": 0.20, "recency": 0.10, "evidence": 0.10},
    # taxonomy of a story/event: a level is shown only when MORE THAN this share of its labelled articles carry the same value
    # (0.5 = strict majority; see lib/news_taxonomy.py). Read-time only; stored labels are never changed.
    "taxonomy_min_agreement": 0.5,
    # country membership of a story (read-time; the pipeline's own AOI membership above is unchanged). A story is "in" a
    # country only with real evidence: the country must clearly dominate the story's LOCATED articles, be supported by
    # several of them, and by a meaningful part of the WHOLE story - otherwise one stray GDELT hit puts a film story in the UK.
    "trend_country_min_share": 0.6,   # >= this share of the story's located articles are in the country ...
    "trend_country_min_located": 2,   # ... and at least this many articles are located in the country ...
    "trend_country_min_coverage": 0.25,  # ... and they are at least this share of ALL the story's articles
    # relevance gate (lib/news_relevance.py): a story is listed only if it lies close to the classifier's own ground truth
    # (the classifier cannot say "not about this" - only ~1.5 % of its training data is D00) and, under a domain/topic
    # selection, its nearest ground-truth articles belong to that domain. Thresholds chosen on a hand-judged sample and
    # validated on a second one (docs/NEWS_ENGINE.md). Unscored stories are never filtered.
    "relevance_min_similarity": 0.65,    # mean cosine of the 5 nearest ground-truth articles
    "relevance_min_agreement": 0.30,     # share of the 10 nearest that carry the selected/assigned domain
    "trend_top_n": 10,
    "trend_max_n": 30,
    # --- all-country live scope --------------------------------------------------------------------------------------
    "admin1_countries": ["IN", "BD"],   # admin-1 AOIs are registered only for these (every country still gets a country AOI)
    "live_min_country_articles": 1,     # a country gets AOIs once this many located articles exist (else it has no data to test)
    "seed": 20261020,
}

# AOI defaults used by the pilot (country level, ISO-3166 alpha-2)
PILOT_SCOPE = {"name": "india_bangladesh", "countries": ["IN", "BD"]}
