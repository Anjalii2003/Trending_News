"""Read-only PostgreSQL handle for the browser tests: `.execute(sql, params).fetchone()/fetchall()` like the old file database,
on the configured schema (NEWS_DERIVED_PG_* / NEWS_DERIVED_SCHEMA in the environment)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from lib import news_store as S  # noqa: E402


def connect():
    return S.Store(read_only=True).con
