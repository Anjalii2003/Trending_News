# News Trending & Emerging Events engine

Uses the repository's statistical machinery (`scripts/lib/stats_v2.py`) for **news articles**:
group articles that report the same event, compute trending metrics inside a selected **Area of Interest (AOI)**, and
raise FDR-controlled alerts. This checkout is maintained for the News Trending engine; the former social-media
dashboard and pipeline have been removed.

Status: pilot validated on 14 days (2026-10-06 → 2026-10-19), India + Bangladesh; **all-country live service in Docker** (see
"Live all-country service" below). No historical backfill has been run; the live service continues from the pilot's last checkpoint.
Read `reports/news/pilot_report.md` (generated, git-ignored) before approving a historical backfill.

## Safety invariants

1. PostgreSQL and OpenSearch are **read-only**, enforced in three layers: SELECT-only client guard, a read-only session
   with `default_transaction_read_only=on` (the server rejected a deliberate `CREATE TEMP TABLE`), and an OpenSearch
   whitelist that only allows `_search`/`_count`/`_mapping` (no scroll/PIT/bulk/delete). Tests: `scripts/tests/test_news_sources.py`.
2. ALL derived data lives in the PostgreSQL schema `news_engine` (no local database file). Source tables are never modified: the engine's PostgreSQL
   role has SELECT on exactly the source tables it reads and write access to its own schema only; the dashboard's role has SELECT on that schema only
   (`deploy/bootstrap_pg.py --verify` audits this).
3. Credentials come from the environment only (`scripts/lib/news_config.py`); nothing is hardcoded and nothing is defaulted.
4. Publication dates: `published_at` only. Rows without a usable date are **excluded and counted**; `scraped_at` is never substituted.
5. Missing data is `NULL`, never `0`. Days with too little exposure are data gaps, not zero volume.
6. The dashboard server binds to `127.0.0.1`, serves a static allowlist and `/api/news/*`, and opens the database read-only.

## Data model (what was verified, not assumed)

| Need | Source | Notes |
|---|---|---|
| Article identity | OpenSearch `_id = "{source_table}-{article_id}"` | 300/300 sample matched `tmc.article_label`; `_mget` fails on the 2-index alias → `ids` query |
| Publication date | `article_label.published_at` == OpenSearch `published_at` | 0 h difference; provenance/confidence/conflict flags in OpenSearch; conflicts excluded |
| Taxonomy | `tmc.article_label` + `tmc.taxonomy_node` (v4.1) | subtopic confidence is low (median 0.38) → taxonomy is a *label*, not a clustering partition. Domain `D00` (non-relevant) is excluded from topic series and alerts |
| Embeddings | OpenSearch `embedding` (bge-m3, 1024-d, unit norm) | reused unchanged |
| Event location | OpenSearch `gdelt.action_*` (GDELT ActionGeo) | ~35–40 % of articles; FIPS country codes → ISO; precision country/adm1/city |
| Mentioned-in | OpenSearch `gkg.locations_country_code` | country level only; separate AOI mode |
| Publisher location | `data/publisher_registry.csv` | separate registry; never feeds event location or metrics |
| Boundaries | Natural Earth 1:10m v5.1.2 (public domain), `data/geo/` | offline after `scripts/21_news_geo_data.py --download`; checksums pinned |

## Pipeline (one code path for pilot, backfill and incremental)

```
discover (published-day range | OpenSearch ingested_at watermark)
  → ingest_docs   date validation, label join, content-hash dedupe, geo, per-day atomic commit
  → resolve_pending  attach labels that arrive later (unlabelled articles still form events)
  → cluster       online centroid clustering (cosine ≥ 0.80, 72 h gap), deterministic event ids
  → recompute     AOI node panels, AOI-day coverage, event metrics, emerging tests, developments, alerts
```

* **Clustering**: story-level groups; an *event* needs ≥3 articles from ≥2 publishers. Re-running reproduces identical
  cluster ids (verified: 100 % identical on 56,635 articles across two independent runs).
* **Metrics** (`news_metrics.py`): negative-binomial baseline fitted on the **AOI's own** history, with a smoothed,
  *leave-one-out* exposure offset (a node's spike cannot inflate its own baseline; exposure noise would otherwise inflate
  false positives to 9 %), drift adjustment, support gates, BH-FDR per (AOI, day), causal acceleration, Kleinberg/CUSUM
  bursts, AR(1) persistence, Hurlbert PIE publisher diversity. Definitions ship in `metric_meta`.
* **Emerging events** (`news_events.py`): empirical upper-tail test of distinct publishers at the event's age against the
  same AOI's own recent clusters (stratified by taxonomy domain); NULL below 50 reference clusters; BH-FDR; high-confidence
  location required.
* **New developments**: new articles outside the envelope of the event's earlier coverage (rank test vs leave-one-out
  similarity + effect floor). A first version (label permutation) flagged 56 % of events and was **rejected**.
* **AOIs**: country, admin-1, polygon, box, radius; *event* vs *mentioned-in*. Baselines are computed per AOI. Ad-hoc AOIs
  (drawn in the UI) run the same functions in memory (`persist=False`) and are flagged ad-hoc.

## Run it

```bash
cd /home/sherya/anjali/MetricsCritique
export NEWS_ENV_FILE=/path/to/env-file-with-SOURCE_DATABASE_URL-and-OPENSEARCH_*   # or the NEWS_* variables
.venv/bin/python scripts/21_news_geo_data.py --verify                 # once: boundaries (use --download on a new machine)
.venv/bin/python scripts/22_news_publisher_registry.py --top 600       # optional refresh (read-only)
```

### 1. Pilot (scope `india_bangladesh`, same schema)

```bash
.venv/bin/python scripts/20_news_run.py plan  --countries IN,BD --from 2026-10-06 --to 2026-10-19   # read-only sizing
.venv/bin/python scripts/20_news_run.py pilot --from 2026-10-06 --to 2026-10-19                      # ~9 min
.venv/bin/python scripts/23_news_pilot_report.py --validate-geo-live                                 # reports/news/pilot_report.md
.venv/bin/python scripts/20_news_run.py status
```

Re-running a range skips days already ingested (`--force-days` re-scans them) and is idempotent: no duplicate rows.

### 2. Full historical backfill — **only after you have reviewed the pilot and approved**

`backfill` refuses to run without `--i-approve-backfill`. Before approving, review the pilot report, fill
`analyst_verdict` in `reports/news/*_inspection_sample.csv`, and decide on the open items below.

```bash
.venv/bin/python scripts/20_news_run.py plan --from 2026-08-17 --to 2026-10-19            # all corpus: size it first
.venv/bin/python scripts/20_news_run.py backfill --from 2026-08-17 --to 2026-10-19 \
     --countries IN,BD --max-days 7 --i-approve-backfill                                   # chunked, resumable
```

Repeat the last command until `days_remaining` is 0 (each run skips finished days). The dated, labelled corpus starts
2026-08-17; earlier data is negligible. A run for *all* countries needs an explicit AOI family (today the AOI family is
built from `--countries`); that is a deliberate extension to scope with you, not an accident waiting to happen.

### 3. Ongoing incremental processing

```bash
.venv/bin/python scripts/20_news_run.py incremental --since 2026-10-19T00:00:00Z  # first run only
.venv/bin/python scripts/20_news_run.py incremental
```

It processes articles whose OpenSearch `ingested_at` is newer than the stored watermark (6 h overlap, idempotent), attaches
late labels, updates only the affected clusters/AOIs and recomputes their metrics and alerts. Alerts revised by late data are
marked `retracted`, never deleted. **This is batch processing on a schedule, not real-time.**

**Scheduling.** No scheduler belonging to this project was found (`auto_scheduler.log` points at a missing script; the
scraper's APScheduler drives scraping only). The command is a one-shot, lock-protected process (a second invocation exits
with code 3 instead of overlapping), so any scheduler can call it. Example (not installed):

```cron
*/30 * * * *  cd /home/sherya/anjali/MetricsCritique && NEWS_ENV_FILE=/path/to/env .venv/bin/python scripts/20_news_run.py incremental >> logs/news_incremental.log 2>&1
```

Exit codes: 0 success · 1 failure (logged in `run_log`, resumable) · 2 refused (missing approval) · 3 another run holds the lock.
Failures never leave half-written days: each day commits data and its checkpoint in one transaction.

### 4. Dashboard (localhost only)

```bash
cd /home/sherya/anjali/MetricsCritique
NEWS_ENV_FILE=deploy/dashboard.env .venv/bin/python UI/news_server.py 8878     # reads the live PostgreSQL schema
# open http://127.0.0.1:8878/
```

The server reads one consistent PostgreSQL snapshot per request, so a running pipeline neither blocks it nor shows it half-written data.


## Live all-country service (Docker)

`scripts/news_service.py` is the one long-running process; `deploy/` holds the Docker deployment. It runs the **same code path** as the pilot
(ingest → cluster → metrics → alerts) for **every country** (scope `all`: no country filter on the source scan; unlocated articles are processed
too; an event AOI per country that has located articles, a mentioned-in AOI per mentioned country, admin-1 AOIs only for the countries in
`PARAMS["admin1_countries"]` = IN, BD).

**One cycle** (once a day at `NEWS_RUN_AT_UTC`, default 03:00 UTC; or every `NEWS_INTERVAL_MINUTES` when that is empty, minimum 0.5):
1. *Catch-up.* Complete days from the **day after the last processed one** (`live_start_day`, or `NEWS_START_DATE`) up to yesterday that have no `done`
   checkpoint are processed, `NEWS_CATCHUP_DAYS_PER_CYCLE` (default 1) per cycle, and the next cycle starts at once until none is left. The first start
   copies the pilot store as the seed, so its 14 days are **reused, not recomputed**. With `NEWS_START_DATE=2026-09-23` the service first processes the
   30 days before today in chronological order (5 per cycle, back to back), then switches to the daily update. A start day is never guessed:
   with neither a seed nor `NEWS_START_DATE` the service refuses to start (a full backfill still needs explicit approval).
2. *Incremental.* Otherwise articles whose OpenSearch `ingested_at` is newer than the stored watermark (6 h overlap) are processed; late labels are attached.
3. Clusters, trend metrics and alerts are recomputed for the affected areas; stories and rankings are materialised.
4. What changed is published to PostgreSQL in **one transaction**.

**No duplicate or overlapping processing.** Articles are keyed `(source_table, article_id)` and every write is an upsert; exact copies (same `content_hash`)
are flagged and counted once; each day commits data and checkpoint together; the 6 h overlap re-reads are idempotent. Two runs cannot overlap: a file
lock on the state volume **and** a PostgreSQL advisory lock (so a second host is refused too).

**Errors and recovery.** A failed cycle is recorded (PostgreSQL `run_log`, with the error text), the day stays un-checkpointed and is
retried on the next tick (back-off after repeated failures). A run killed mid-way is marked `abandoned` by the next run and resumes from the last
checkpoint. A failed PostgreSQL publication fails the run and the next publication catches up from the stored timestamps (`sync_pending` rewrites every
area). Docker restarts the container (`restart: unless-stopped`); `stop_grace_period` is 5 min so a running cycle can finish.

### Where the data lives (all in PostgreSQL schema `news_engine`)
| What | Tables | Notes |
|---|---|---|
| Articles, labels, taxonomy (source) | `tmc.article_label`, `tmc.taxonomy_node` (+ `tmc.a0_ground_truth`: ids and domain only), OpenSearch `full-articles` | read-only; **nothing is copied back** |
| Per-article derived facts | `article_fact`, `article_geo`, `excluded_article`, `pending_label` | metadata only: id, date, publisher, title, url, labels, location, cluster; **no article text, no embedding** |
| Clusters | `event_cluster` (centroid kept 14 days), `event_metrics`, `event_aoi_test`, `event_development` | |
| Areas and trend metrics | `aoi_def`, `aoi_day`, `node_aoi_day`, `node_aoi_series`, `burst_episode`, `alert`, `corpus_day`, `data_quality`, `metric_meta` | |
| Stories and rankings | `story`, `story_event`, `trending_ranking` | materialised after every successful run |
| Progress and operations | `checkpoint` (per stage/scope/day), `run_log`, `state` (watermark, schedule, heartbeat) | |
| Compatibility views | `article_cluster`, `trend_node_day` | the two table names of the first release, now views |
| Taxonomy of the labels | `taxonomy_node` | copy of `tmc.taxonomy_node` |
The run lock, heartbeat and the 170 MB relevance reference file are on the Docker volume `news_engine_state`.

PostgreSQL access (created once by `deploy/bootstrap_pg.py --apply`, SQL in `deploy/sql/bootstrap_admin.sql`): schema `news_engine`; role
`news_engine_app` (SELECT on `tmc.article_label` and `tmc.taxonomy_node`, full rights on `news_engine`, nothing else — it cannot write or even read any
other table, verified by the script's audit); role `news_engine_reader` (SELECT on `news_engine`, used by the dashboard). Passwords are generated and
written only to `deploy/news.env` / `deploy/dashboard.env` (mode 600, git-ignored). Undo: the `DROP` statements at the top of the SQL file.

### Commands (`deploy/news-ctl.sh`, from the repository root)
```bash
./deploy/news-ctl.sh start              # build + start both services (detached); NEWS_INTERVAL_MINUTES=15 ./deploy/news-ctl.sh start to change the interval
./deploy/news-ctl.sh status             # containers + health
./deploy/news-ctl.sh logs [news-ingest|news-dashboard]
./deploy/news-ctl.sh health             # ingest heartbeat + dashboard /api/news/health
./deploy/news-ctl.sh summary            # the live summary JSON (same data as the dashboard panel)
./deploy/news-ctl.sh once               # one extra cycle now (refused while a cycle is running)
./deploy/news-ctl.sh restart [service]  # restart
./deploy/news-ctl.sh stop               # graceful stop; `down` removes containers but KEEPS the data volume
```

### Optional CUDA acceleration for clustering

The default image and configuration use CPU. On an NVIDIA host with the NVIDIA Container Toolkit, build the image with the optional CUDA-enabled PyTorch dependency:

```bash
NEWS_GPU=1 docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml build news-ingest
```

This build can require several gigabytes of free disk space. It does not start a container or ingestion. Only when you are ready to resume ingestion, start the service with GPU access:

```bash
NEWS_CLUSTER_DEVICE=auto docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml up -d --no-deps news-ingest
```

`NEWS_CLUSTER_DEVICE=auto` uses CUDA when available and otherwise uses CPU. Set it to `cuda` to require GPU access (startup fails clearly if CUDA is unavailable), or `cpu` to force CPU. GPU acceleration applies to the batched article-to-centroid similarity matrix only; database work, embedding retrieval, and other metric calculations remain on CPU. The normal `news-ctl start` command does not use the GPU override. Check `docker logs news-ingest` for `[news-cluster] batched similarity device=cuda` after a clustering run.
Dashboard: `http://127.0.0.1:8880/` (localhost only; `NEWS_DASHBOARD_PORT` to change). Without Docker: `NEWS_ENV_FILE=deploy/news.env python scripts/news_service.py once|run`.
Configuration (compose environment): `NEWS_INTERVAL_MINUTES`, `NEWS_CATCHUP_DAYS_PER_CYCLE`, `NEWS_START_DATE`, `NEWS_SCOPE_NAME`, `NEWS_SEED_DB`,
`NEWS_MAX_CYCLE_MINUTES` (health limit for one cycle).

**Health.** The ingest container's check (`news_service.py health`) fails when the heartbeat file is older than 2 min, a cycle has run longer than
`NEWS_MAX_CYCLE_MINUTES`, or the next run is overdue by more than 2 intervals; the dashboard's check opens its database. Logs are JSON-file with rotation (20 MB × 5).

### Relevance gate: not listing wrong stories
Measured problem: the classifier only ever saw ~1.5 % "Non-relevant" examples in training, so it labels almost everything D01-D09 with a high-looking
confidence, and a majority of such labels inside a story is still wrong. On a hand-judged stratified sample of 126 listed stories **42 % belonged to
their domain, 17 % were borderline, 41 % were wrong** (film, health, cricket and local-crime stories under "Military" or "Terrorism"). The classifier's own
confidence barely separates them (AUC 0.66-0.74), so confidence thresholds do not fix it.

What the engine does instead (`lib/news_relevance.py`, read-only, no stored label changes): each event centroid is compared with the classifier's own
**ground-truth set** (83 k articles of `tmc.a0_ground_truth`, embeddings from OpenSearch, built once into a 170 MB float16 file on the state volume).
`ref_sim` = mean cosine to the 5 nearest; `ref_votes` = domains of the 10 nearest. A story is listed only if `ref_sim >= 0.65` and, under a
domain/topic selection, at least 30 % of its neighbours carry that domain (`relevance_min_similarity`, `relevance_min_agreement`). On the 126-story sample
this keeps 75 % of the relevant stories and cuts the wrong share from 41 % to 16 %. Stories without a score (no centroid / no reference) are never hidden.
The page states how many active stories were hidden. The ground-truth *content* column is never read (the database role has column-level SELECT on
`source_table, source_id, gt_domain_id` only).

### Live summary (Trending page header, `GET /api/news/live`)
Every value is read from stored records in PostgreSQL `news_engine` (run log, state and the tables above):
* **Articles processed** = rows in `article_cluster` (every article ingested); **unique** = not an exact content copy; **duplicates** = the difference.
* **Latest run** = articles first ingested by the most recent run (`run_log.articles_ingested`), its mode and duration.
* **Last successful update** = `finished_at` of the newest successful run; **latest publication date** = newest `published_at` among processed articles.
* **Event clusters / qualifying events** = rows / `is_event` rows in `event_cluster` (event = ≥3 articles from ≥2 publishers).
* **Pending labels** (articles waiting for the classifier), **excluded** (no usable publication date, by reason) come from the last successful run's totals.
* **Status** = running / successful / failed (an interrupted run, or one whose scheduler stopped reporting, is shown as failed); **errors** = failed runs in 24 h
  with the latest message; **next scheduled run** = the stored `next_run_at`.

### Limitations to know
* The corpus is ~60 k articles/day. The live window is the last 30 days (approved); anything older is not processed unless you approve it.
* This is batch processing every N minutes, not real time; latency ≈ source ingestion + labelling + one interval.
* The OpenSearch credential in use is an admin account; the client is restricted to read-only calls (whitelist) in code, but a dedicated read-only OpenSearch user is advisable.

## Trending news page (the default UI at `/`)

Simple page, no map: choose a **country**, a **topic** (Domain → Topic → Subtopic), or **both**, and a period (24 h / 3 days / 7 days or a custom date range);
it shows the **top 10 trending stories** ranked strongest first ("Show more" up to 30). Nothing selected → it asks the user to choose
(trending is defined relative to an area/topic). The advanced dashboard (map, alerts, baselines, methodology) is still at `/news.html`.

* **Custom dates** are inclusive Indian Standard Time (IST, `Asia/Kolkata`) calendar dates. The API converts their midnight boundaries to UTC
  for comparison with stored timestamps. Growth and typical-coverage comparisons use an equally long period immediately before the selected range.
  A message saying no stories met the trending criteria means none passed the current selection and ranking thresholds; it does not mean there
  was no news or no important event during that period.
* **Country only**: stories that belong to that country *on real evidence*, all topics: ≥ 60 % of the story's located articles are in the country,
  ≥ 2 articles are located there, and they are ≥ 25 % of the whole story (`trend_country_*` in `news_config.py`). One stray GDELT hit no longer
  puts a 68-article film story in the UK. **Topic only**: all countries, including stories with no usable location. **Both**: the intersection.
* **Story** = an event cluster; clusters that are the same story split in two (centroid cosine ≥ 0.90 and ≤ 72 h apart) are merged
  (`news_trending.merge_events`). At display time a lower-ranked story ≥ 0.88 similar to one already listed is shown under it as
  "also covered" instead of taking a slot. Exact duplicate articles are counted once.
* **Ranking** (`scripts/lib/news_trending.py`, computed on request, nothing stored): volume, publisher breadth, growth (NULL below
  5 articles across the two windows; a story with no earlier coverage is "new"), recency and statistical evidence, as weighted
  percentile ranks (30/30/20/10/10, same design as `emergence_score`; missing components dropped, weights renormalised).
* **Expected baseline**: how the story's window volume compares with comparable stories (same domain when ≥50 exist) in the same
  geographic scope in earlier, fully observed windows - descriptive, **not** a significance test. A first version reported an FDR q for
  this and gave q = 1 to a 306-article story (candidates and reference come from the same population, so the biggest story is an
  extreme of it by construction); it was rejected.
* **Statistical evidence** comes only from tests with a proper null that the pipeline already ran (emerging event, new development,
  negative-binomial topic surge, all FDR-controlled) and only where they exist (pre-computed country areas). Elsewhere the card says no
  significance claim is made - missing is never shown as "not significant".
* **Taxonomy path** (`lib/news_taxonomy.py`, one rule for the Trending page *and* the dashboard): a level is shown only when **more than half**
  of the story's labelled articles carry the same value there, resolved top-down (domain → topic inside it → category → subtopic). A plurality
  is not enough: a story whose articles are scattered over eight domains is reported as **Mixed topics** (or **Outside the taxonomy** when most
  are `D00`, **Unlabelled** when none is labelled) and is not listed under any single topic; it still appears in a country-only view.
  The path stops at the deepest level that has agreement. Labels are the stored classifier labels, unchanged (`taxonomy_min_agreement`).
  The per-event modal labels stored in `event_cluster` by the pipeline are no longer read by any user-facing view.
* API: `GET /api/news/trending_meta`, `GET /api/news/trending?country=&tax=&window=&limit=`, `GET /api/news/story_articles?ids=&offset=&limit=`.
  Read-only; verified not to change the database file.
* Tests: `scripts/tests/test_news_trending.py` (planted-answer synthetic world), `scripts/tests/test_news_trending_real.py` (every displayed
  number recomputed with SQL on the pilot DB, 3 windows × 5 scope cases), `UI/tests/check_trending_ui.py` (browser).

## Topic filter and source citations (dashboard)

* **Filter**: Domain → Topic → Subtopic dropdowns plus an **All Topics** reset, built from the stored `taxonomy_node` hierarchy
  (in-scope nodes only). Subtopics are grouped under their real parent category. It combines with the area (country / state /
  drawn area / *event* vs *mentioned-in*): every number is recomputed for *area AND topic*.
* **Matching** uses each article's **stored label** only - never headline text. An **event** matches when a **strict majority**
  of its labelled, non-duplicate articles carry the selected node at that level (and its ancestors); the share of its articles that carry
  the node is shown. An event whose articles disagree matches no single topic, only the level they do agree on. Trend rows and node alerts match when the node is the selection or one of its descendants.
* **Subtopic series** are not stored by the pipeline; they are computed on demand, read-only, with the same function
  (`persist=False`), and cached per database version. Their FDR family is the subtopic series of the area.
* **Citations**: every article is shown as *publisher — title ↗ date* linked to the original URL (http/https only; a record without a usable URL
  is shown as "no link"). The advanced dashboard displays UTC; the simple Trending page displays timestamps in IST. Events list up to 3 sources
  (and the full list in the drawer), trend rows load the source articles of that node, and every alert lists the articles it was computed from
  (`Sources:` + "all N").
* There is **no date-range control** in this dashboard (it shows the whole processed period).
* Test: `UI/tests/check_news_filter_ui.py` compares every displayed number and citation with independent SQL on the pilot DB.

## Tests

```bash
.venv/bin/python -m pytest scripts/tests/test_news_*.py -q          # no network, no production data
.venv/bin/python UI/tests/check_news_ui.py --url http://127.0.0.1:8878 --shots /tmp/shots          # headless-browser dashboard test (needs playwright + server)
.venv/bin/python UI/tests/check_news_filter_ui.py --url http://127.0.0.1:8878 --shots /tmp/shots   # topic filter + citations vs independent SQL (advanced page)
.venv/bin/python UI/tests/check_trending_ui.py --url http://127.0.0.1:8878 --shots /tmp/shots      # the Top-trending page
```

Legacy tests (`scripts/tests/test_*.py` without `news`) are standalone scripts; run them as before.

## Known limitations (also shown in the dashboard)

* Event location is machine-extracted (GDELT) for ~35–40 % of articles; ~20 % of events with several located articles
  have members in more than one country. Only **high-confidence** locations (≥3 located members, ≥80 % agreeing) can raise
  alerts. No cheap automatic validator exists (mention-support is circular; a headline place-name check was measured and
  rejected). Human review of `location_inspection_sample.csv` is required.
* Bangladesh has very few event-located articles; its AOI is mostly NULL-with-reason.
* Cluster quality is not human-validated; silver GDELT labels are weak (pairwise recall 0.31).
* A 14-day pilot leaves 7 testable days per series (`baseline_min_days=7`); production should use 28-day baselines.
* New-development detection flags a high share of large, evolving events; needs review before alerts are acted on.
* Ad-hoc AOIs get no multiplicity control across the areas a user tries.
* Publisher registry: 53 % of corpus volume classified (reviewed / ccTLD / pattern), none human-verified yet.
