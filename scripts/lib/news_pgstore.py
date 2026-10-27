"""PostgreSQL connection settings and run totals for the news engine's own schema (`news_engine` by default).

The tables themselves, their DDL and every read/write live in `news_store.py` (the engine runs entirely on this schema). Access
model (deploy/sql/bootstrap_admin.sql): the engine's role has SELECT-only on the source tables it reads and write access to this
schema only; the dashboard's role has SELECT on this schema only. Nothing here writes to any source system.
"""
from __future__ import annotations

import os

from . import news_config as C


def schema_name() -> str:
    return os.environ.get("NEWS_DERIVED_SCHEMA", "news_engine")


SCHEMA = schema_name()        # default at import; Store re-reads the environment, so tests/containers can override it


def derived_settings() -> dict:
    """Connection kwargs for the derived schema. Defaults to the engine's own source credentials (same role)."""
    host = os.environ.get("NEWS_DERIVED_PG_HOST")
    if host:
        need = {k: os.environ.get(f"NEWS_DERIVED_PG_{k.upper()}") for k in ("port", "user", "password", "db")}
        missing = [f"NEWS_DERIVED_PG_{k.upper()}" for k, v in need.items() if not v]
        if missing:
            raise C.ConfigError(f"derived PostgreSQL not fully configured; set {', '.join(missing)}")
        return {"host": host, "port": int(need["port"]), "user": need["user"], "password": need["password"], "dbname": need["db"]}
    return C.pg_settings()


def configured() -> bool:
    """True when a derived PostgreSQL can be reached by configuration (does not connect)."""
    if os.environ.get("NEWS_DERIVED_PG_DISABLE", "").lower() in ("1", "true", "yes"):
        return False
    try:
        derived_settings()
        return True
    except C.ConfigError:
        return False


def run_totals(st) -> dict:
    """Counters taken from the store at the end of a run (recorded on the run row)."""
    q = st.scalar
    return {
        "articles_total": int(q("SELECT count(*) FROM article_fact") or 0),
        "articles_unique": int(q("SELECT count(*) FROM article_fact WHERE NOT is_duplicate") or 0),
        "clusters_total": int(q("SELECT count(*) FROM event_cluster") or 0),
        "events_total": int(q("SELECT count(*) FROM event_cluster WHERE is_event") or 0),
        "pending_labels": int(q("SELECT count(*) FROM pending_label") or 0),
        "excluded_total": int(q("SELECT count(*) FROM excluded_article") or 0),
        "excluded_by_reason": {r[0]: int(r[1]) for r in st.con.execute(
            "SELECT reason, count(*) FROM excluded_article GROUP BY 1 ORDER BY 2 DESC").fetchall()},
        "latest_published_at": (lambda v: None if v is None else v.isoformat())(q("SELECT max(published_at) FROM article_fact")),
        "active_alerts": int(q("SELECT count(*) FROM alert WHERE status = 'active'") or 0),
    }
