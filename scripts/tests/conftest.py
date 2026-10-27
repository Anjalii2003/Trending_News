"""Shared fixtures. `docker_pg` starts a THROWAWAY PostgreSQL container (never the production database) for the tests of the
derived-schema store, and points the engine's derived-PostgreSQL settings at it. Skipped when Docker is not usable."""
import os
import secrets
import shutil
import subprocess
import time

import pytest


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def docker_pg():
    if not _docker_ok():
        pytest.skip("docker is not available")
    img = os.environ.get("NEWS_TEST_PG_IMAGE", "postgres:17-alpine")
    if subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode != 0:
        pytest.skip(f"docker image {img} is not present locally")
    pw = secrets.token_urlsafe(12)
    name = f"news-test-pg-{secrets.token_hex(3)}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-e", f"POSTGRES_PASSWORD={pw}", "-e", "POSTGRES_DB=newsdb",
                    "-p", "127.0.0.1::5432", img], check=True, capture_output=True)
    try:
        port = subprocess.run(["docker", "port", name, "5432/tcp"], capture_output=True, text=True, check=True).stdout.split(":")[-1].strip()
        import psycopg2
        for _ in range(60):
            try:
                psycopg2.connect(host="127.0.0.1", port=int(port), user="postgres", password=pw, dbname="newsdb", connect_timeout=2).close()
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("test postgres did not start")
        yield {"host": "127.0.0.1", "port": port, "user": "postgres", "password": pw, "db": "newsdb", "container": name}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture(autouse=True)
def news_pg(request, monkeypatch):
    """Every news-engine test runs on PostgreSQL: a throwaway container, one throw-away schema per Store path (the engine has no
    local database file any more). Non-news tests (legacy scripts) are untouched."""
    name = request.module.__name__
    if not name.startswith("test_news") or (name == "test_news_trending_real" and os.environ.get("NEWS_REAL_DB_TESTS") == "1"):
        yield None
        return
    pg = request.getfixturevalue("docker_pg")
    for k in ("host", "port", "user", "password", "db"):
        monkeypatch.setenv(f"NEWS_DERIVED_PG_{k.upper()}", str(pg[k]))
    monkeypatch.setenv("NEWS_STORE_SCHEMA_FROM_PATH", "1")
    monkeypatch.delenv("NEWS_DERIVED_PG_DISABLE", raising=False)
    yield pg
