# News Trending & Emerging Events

This repository runs a news ingestion and analytics service. It groups related articles into event clusters and stories, computes trending rankings, and serves a read-only dashboard backed by PostgreSQL.

## Start and monitor

```bash
./deploy/news-ctl.sh start
./deploy/news-ctl.sh status
```

The Trending UI is served at `http://127.0.0.1:8880/`.

To pause scheduled ingestion while keeping the dashboard available:

```bash
docker compose -f deploy/docker-compose.yml stop news-ingest
```

Resume ingestion with:

```bash
docker compose -f deploy/docker-compose.yml start news-ingest
```

## Documentation

- [News Engine guide](docs/NEWS_ENGINE.md) — architecture, processing lifecycle, database tables, Trending rules, and operations.
- [Project guide](docs/PROJECT_GUIDE.md) — plain-language overview and operator instructions.

## Main components

- `scripts/news_service.py` — scheduled, resumable ingestion service.
- `scripts/lib/` — ingestion, clustering, metrics, alerts, story assembly, and ranking.
- `UI/news_server.py` and `UI/news_api.py` — read-only dashboard server and API.
- `UI/trending.html` and `UI/assets/` — Trending user interface.
- `deploy/` — Docker image, Compose services, PostgreSQL setup and migration utilities.
- `data/geo/`, `data/taxonomy.json`, and `data/publisher_registry.csv` — runtime geography and classification inputs.

Deployment credentials are kept in ignored `deploy/news.env` and `deploy/dashboard.env` files. Do not commit or share them.
