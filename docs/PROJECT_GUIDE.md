# News Trending & Emerging Events — Project Guide

*A plain-language guide for managers and for new team members. For the engineering reference, see `docs/NEWS_ENGINE.md`.*

---

## 1. What this project does (the 60-second version)

We already collect millions of news articles and have already labelled each one with a **taxonomy** (Domain → Topic → Category → Subtopic) and an **embedding** (a list of 1,024 numbers that captures what the article is about).

This project turns that into an answer to one question:

> **"What is trending right now in the place I care about, or on the topic I care about?"**

It does five things:

1. **Cleans** the articles (reliable dates only, removes exact copies, attaches a location where one exists).
2. **Groups** articles about the same event into **stories** (clustering), so one event is not shown 200 times.
3. **Measures** each story and each topic (volume, publishers, growth, "is this unusual?").
4. **Ranks** stories so the strongest trending ones come first and explains *why* in plain words.
5. **Shows** it in a simple web page (Top 10 trending stories) and an advanced dashboard (map, alerts, charts).

Nothing in the source databases is ever changed. Everything we compute is stored in our own small file (`data/news_*.duckdb`).

---

## 2. The big picture

```
 PostgreSQL (read-only)               OpenSearch (read-only)
 ├ tmc.article_label  (taxonomy)      └ full-articles  (text, date, embedding,
 ├ tmc.taxonomy_node  (names)            GDELT location, publisher domain)
 └ raw.* (titles, urls)
            │                                   │
            └──────────────┬────────────────────┘
                           ▼
   ① INGEST   dates · duplicates · labels · location          (scripts/lib/news_ingest.py)
                           ▼
   ② CLUSTER  articles → events (by meaning, same time)       (scripts/lib/news_cluster.py)
                           ▼
   ③ METRICS  baselines · growth · surprise · alerts          (news_metrics.py, news_events.py)
                           ▼
        our own derived file: data/news_*.duckdb   (never touches the source DBs)
                           ▼
   ④ TRENDING  merge near-duplicate events → rank stories     (news_trending.py, runs on request)
                           ▼
   ⑤ UI   Trending page (/)   +   Advanced dashboard (/news.html)
```

Steps ①–③ are the **pipeline** (run on a schedule). Steps ④–⑤ happen **when someone opens the page** — so the page always reflects the latest stored data.

---

## 3. What data we use (and nothing else)

| Need | Where it comes from | Notes |
|---|---|---|
| Article identity | OpenSearch `_id` = `"{source_table}-{article_id}"` | Verified: 300 of 300 sampled articles matched PostgreSQL exactly |
| Publication date | `published_at` (PostgreSQL label table = OpenSearch, 0 h difference) | **Never** replaced by scrape time. Missing/conflicting dates are excluded and counted |
| Taxonomy labels | `tmc.article_label` + `tmc.taxonomy_node` (version v4.1) | **Reused as-is. We never relabel anything** |
| Meaning of the article | OpenSearch `embedding` (bge-m3, 1,024 numbers) | **Reused as-is. No new model, no retraining** |
| Where the event happened | OpenSearch `gdelt.action_*` (GDELT ActionGeo) | Only ~35–40 % of articles have it (see §10) |
| Which publisher | OpenSearch `source_domain` | Used for "how many different outlets" |
| Title, link | OpenSearch / PostgreSQL | Shown with every citation |
| Country boundaries | Natural Earth (free, public domain), stored locally | Works offline |

**Not used / not available for news:** likes, shares, followers, engagement, virality, hate scores (these only existed for social media; we deliberately do not invent them).

---

## 4. How the data is cleaned (step ①)

1. **Reliable date only.** We use `published_at`. If it is missing, in the future, before year 2000, or flagged as conflicting, the article is *excluded and counted* — never guessed from scrape time.
2. **Exact duplicates.** Articles with the same `content_hash` (identical text, e.g. press releases on many sites) are kept but flagged; they are **counted once**. In the pilot ≈ 8 % were copies.
3. **Labels.** Each article gets its stored taxonomy labels. If an article has no label yet (classifier lag, ≈ 6.6 % in the pilot), it is *still used for clustering* but excluded from topic charts, and the label is attached automatically when it appears.
4. **Location.** GDELT country code (it uses FIPS codes like `UK`, `IN`, `BG`, **not** ISO) is converted to ISO and checked against the point on the map. The place's *precision* is kept: country / state / city.
5. **Missing is never zero.** Whenever there is not enough evidence, the value is `NULL` and shown as "—".

---

## 5. How clustering works (step ②) — "which articles are the same event?"

**Idea:** two articles about the same event have *similar embeddings* and appear *around the same time*.

**Method (online centroid clustering):**

1. Take articles in publication order.
2. For each article, compare its embedding to the **average embedding (centroid)** of every cluster still "open" (updated within the last 72 hours).
3. If the best similarity is **≥ 0.80 (cosine)**, the article joins that cluster and the centroid is updated. Otherwise it starts a new cluster.
4. A cluster counts as an **event** only if it has **≥ 3 articles from ≥ 2 different publishers**. (Most clusters are single articles — 75 % in the pilot — and that is normal.)
5. Cluster IDs are deterministic (hash of the first article), so re-running gives the *same* IDs. We tested this: 100 % identical on 56,635 articles.

**Why this method?** We compared three methods earlier on real data (threshold/centroid, agglomerative, HDBSCAN). They scored similarly, but the centroid method is the only one that can **add new articles incrementally** without re-clustering everything — essential for live updates.

**How we used the existing taxonomy in clustering:** *We did not force clusters to stay inside a taxonomy label.* The classifier's subtopic confidence is low (median 0.36), so a hard partition would split one real event across several labels. Instead the label is **attached afterwards** to each cluster.

### Second step: stories (merging)
Sometimes one real story is split into 2–4 clusters (e.g. "attack", "India's reaction", "flights cancelled"). When the Trending page is built:

- Clusters with **centroid similarity ≥ 0.90** and overlapping in time (≤ 72 h apart) are **merged into one story**. We chose 0.90 by reading real headline pairs: every sampled pair ≥ 0.90 was the same story; 0.85 started chaining unrelated news together.
- A lower-ranked story **≥ 0.88** similar to one already shown is displayed *under* it as "also covered" instead of taking its own slot.

---

## 6. How the existing taxonomy labels are used

- Each article carries Domain / Topic / Category / Subtopic codes from the classifier. We **read** them; we **never change** them.
- A **cluster/story's label** needs **agreement**: a level is shown only if *more than half* of its labelled articles carry that value, resolved top-down (domain → topic inside it → category → subtopic). If they do not agree the story is shown as *Mixed topics* (or *Outside the taxonomy* / *Unlabelled*) and is not listed under any single topic. (An earlier version took the most common label even at ~30 % support, which put an entertainment story under "Terrorism › Narcotics".)
- **Known limit — classifier quality:** when most articles of a story are wrongly labelled the same way (e.g. a Kashmir weather story labelled as terrorism because of the place name), agreement cannot detect it. The upstream gate is still the uncalibrated placeholder (`g0-uncalibrated`) and our scraped articles have no ground truth, so this needs a calibrated gate / better classifier, not a change in this engine.
- The filter "Domain → Topic → Subtopic" uses the *stored* label only — **never headline words**.
- Domain `D00` ("Non-relevant / Other") is excluded from topic charts and alerts.
- **Honest warning:** subtopic labels are noisy (45 % of pilot articles have confidence < 0.3). Domain and topic filters are much more reliable than subtopic filters.

---

## 7. How every metric is calculated

### 7.1 Topic trends inside an area (the "node" series)
For one **area** (e.g. India) and one **topic**, we count articles per day: `V`.

| Metric | Plain meaning | Formula |
|---|---|---|
| **Exposure** `E` | How much other news there was that day in this area (so a busy news day doesn't make every topic look "up") | articles of **all other topics** in the area that day (leave-one-out), smoothed over 7 days |
| **Expected value (baseline)** | What we'd normally expect today | Negative-Binomial distribution fitted on the **previous 28 days** (never includes today), scaled by `E_today / E_average` (limited to ×⅓…×3) and by the topic's own recent trend |
| **Surprise** | How unusual today's count is | `−log10 P(X ≥ V_today)` under that baseline |
| **FDR q-value** | Surprise, corrected for the fact that we test hundreds of topics a day | Benjamini–Hochberg over all topics tested in that area that day |
| **Growth** | Day-over-day change | `log((V+1)/E)` today − yesterday |
| **Acceleration** | Is growth speeding up? | 2nd derivative of the log-rate using a **causal** Savitzky–Golay filter (window 5, order 2; uses only past days) |
| **Burst** | Is the topic in a "burst" state? | Kleinberg 2-state model (≥ 21 days of data) or Poisson CUSUM (short series) |
| **Persistence** | How long does attention last? | AR(1) half-life (`ln 0.5 / ln φ`) with a Ljung–Box check |
| **Publisher diversity** | Are many different outlets covering it? | Hurlbert PIE `= N/(N−1)·(1−Σpᵢ²)`; blank (not 0) if N < 2 |

**Support rules (no guessing):** a number is shown only if the baseline has ≥ 7 days (28 recommended in production), mean ≥ 0.5/day and today's `V ≥ 3`. Days where the area has fewer than 30 articles are treated as **data gaps**, not zero.

**Alert rule:** `q < 0.05` **and** `V ≥ 2 × baseline` **and** support rules met.

### 7.2 Event (story) metrics
| Metric | Formula / rule |
|---|---|
| Articles, publishers | counts of distinct articles / distinct source domains |
| Growth 24 h | `log((articles last 24h + 1) / (articles previous 24h + 1))`; blank unless ≥ 5 articles in 48 h |
| Acceleration 6 h | causal Savitzky–Golay 2nd derivative of log 6-hour counts; blank unless ≥ 8 articles in 48 h |
| Half-life | exponential decay of daily counts after the peak; blank unless ≥ 4 days after peak |
| Syndication share | copies ÷ (articles + copies) |

### 7.3 "Emerging event" test
New events have no history, so we compare them with **similar events in the same area**:
`p = (1 + number of reference events with ≥ k publishers at the same age) / (n_ref + 1)`
Reference = events in the same area (same domain if enough) started in the last 28 days. Needs ≥ 50 reference events; then Benjamini–Hochberg; alert if `q < 0.05` and ≥ 3 publishers and the location is high-confidence.

### 7.4 "New development inside an event"
For a big event (≥ 8 articles, ≥ 3 publishers): are the last 24 h of articles **outside the envelope** of earlier coverage? Rank test of each new article's similarity to the old core vs. how far old articles sit from their own core (Mann–Whitney), plus a 10th-percentile effect floor, coherence ≥ 0.75, and ≥ 3 new articles from ≥ 2 publishers.

### 7.5 The Trending score (what orders the Top 10)
For a window `W` (24 h / 3 days / 7 days) ending at the latest data time:

| Part | Weight | Meaning |
|---|---|---|
| Volume | 30 % | articles in the window |
| Breadth | 30 % | distinct publishers in the window (stops one outlet repeating itself looking "big") |
| Growth | 20 % | this window vs the previous one; blank if < 5 articles total; "new story" if there was nothing before |
| Recency | 10 % | hours since the latest article |
| Evidence | 10 % | number of the pipeline's own FDR-controlled flags (emerging / development / topic surge) — **dropped (not zero) where those tests don't exist** |

Each part is turned into a **percentile rank among the candidate stories**, then the weighted average × 100 = **Trend score**. It is a *relative ranking inside the view*, **not** a probability.

*Expected baseline line* on each card: "more coverage than 99 % of comparable stories (a typical story gets 5 articles)" — a **description**, not a significance test.
> **Important design decision:** an earlier version reported a p-value for "story size vs. other stories". It gave q = 1 to a 306-article story because the story is compared with a group it belongs to. We removed it and take "statistical significance" only from tests with a correct null hypothesis. This is deliberate honesty, not a gap.

---

## 8. The user interface

**Trending page (`/`)** — simple, no map
- Choose a **country**, a **topic** (Domain → Topic → Subtopic), or **both**, and a period.
- Nothing chosen → it asks you to choose (trending needs an area or topic).
- Top 10 cards: rank, title, taxonomy, country/place, related article count, latest activity, **"why it's trending"**, latest source links, **View related articles** (every article with title, publisher, date and link to the original).

**Advanced dashboard (`/news.html`)** — map, area drawing (polygon/box/radius), topic series, alerts, coverage/data-quality panel, methodology, topic filter with citations.

Both pages are **read-only** and run on `127.0.0.1` only.

---

## 9. How to run it

### 9.1 Setup (once)
```bash
cd /home/sherya/anjali/MetricsCritique
source .venv/bin/activate                       # packages are in requirements.txt
export NEWS_ENV_FILE=/path/to/env-with-DB-credentials     # credentials only from environment
.venv/bin/python scripts/21_news_geo_data.py --verify      # boundary files (use --download on a new machine)
```

### 9.2 Pilot (already done: 2026-10-06 → 10-19, India + Bangladesh)
```bash
.venv/bin/python scripts/20_news_run.py plan  --countries IN,BD --from 2026-10-06 --to 2026-10-19   # sizing only
.venv/bin/python scripts/20_news_run.py pilot --from 2026-10-06 --to 2026-10-19                      # ~9 min
.venv/bin/python scripts/23_news_pilot_report.py --validate-geo-live                                 # quality report
```

### 9.3 Run for **all the data we have** (needs your approval — it is blocked until you add the flag)
```bash
.venv/bin/python scripts/20_news_run.py plan --from 2026-08-17 --to <last day>                        # how many articles?
.venv/bin/python scripts/20_news_run.py backfill --from 2026-08-17 --to <last day> \
      --countries IN,BD,... --max-days 7 --i-approve-backfill
```
- Run it repeatedly: it does **7 days per run** and skips finished days, so it can be stopped and resumed safely.
- The dated and labelled data starts **2026-08-17** (earlier data is negligible).
- Safe to re-run: no duplicate rows are created. If a run crashes, the day it was on is simply redone.
- **Know before you do it (not yet changed in code):** the same `--countries` list is used both to *choose which articles to read* and to *choose which areas get baselines/alerts*. Running with no country list reads everything but then stops at the metrics step ("AOI family needs an explicit country list"). To cover **all countries**, a small code change is needed so articles are read from everywhere while alerts are built only for chosen countries (see §11, item 9). The Trending page itself only needs the clusters, so it works for all countries as soon as all articles are clustered.
- Memory/time: the pilot (57 k articles) took ~9 minutes, mostly fetching embeddings from OpenSearch. The full dated corpus is roughly 50× larger, so plan hours, not minutes — hence the chunked `--max-days`. (Estimate; measure with `plan` and a one-week trial first.)

### 9.4 Keep it up to date (live-ish data)
```bash
.venv/bin/python scripts/20_news_run.py incremental --since 2026-10-19T00:00:00Z --db data/news_metrics.duckdb   # first time only
.venv/bin/python scripts/20_news_run.py incremental --db data/news_metrics.duckdb                                  # afterwards
```
What it does each time:
1. Reads only articles whose OpenSearch `ingested_at` is newer than the saved bookmark (with a 6-hour overlap for late arrivals).
2. Attaches labels that arrived late.
3. Puts new articles into existing open clusters (or starts new ones).
4. Recomputes **only the areas affected** (baselines, alerts).
5. Marks an alert **"retracted"** (never deletes it) if late data changes its answer.

**Schedule it** (example, every 30 min — not installed):
```cron
*/30 * * * *  cd /home/sherya/anjali/MetricsCritique && NEWS_ENV_FILE=/path/to/env .venv/bin/python scripts/20_news_run.py incremental --db data/news_metrics.duckdb >> logs/news_incremental.log 2>&1
```
A lock file prevents two runs overlapping (the second exits with code 3). **This is batch processing, not real-time** — "latest" means "as of the last run".

**How the UI updates itself:** the Trending page does **not** keep pre-computed results. Every time you open it or change a filter, the server reads the stored clusters and ranks them on the spot (it caches only until the database file changes). So after an incremental run, a browser refresh shows the new ranking and "data up to …" time. If the pipeline is writing while someone views the page, the server reads a safe snapshot copy instead of failing.

### 9.5 Open the dashboard
```bash
cd /home/sherya/anjali/MetricsCritique && NEWS_UI_DB=data/news_pilot_india_bangladesh.duckdb .venv/bin/python UI/news_server.py 8878
# open http://127.0.0.1:8878/
```
(Use `NEWS_UI_DB=data/news_metrics.duckdb` once the full run exists.)

### 9.6 Tests
```bash
.venv/bin/python -m pytest scripts/tests/test_news_*.py -q          # 231 tests, no network
.venv/bin/python UI/tests/check_trending_ui.py --url http://127.0.0.1:8878 --shots /tmp/shots   # browser test
```

---

## 10. Pilot results (what we measured)

| Item | Result |
|---|---|
| Articles read | 57,119 → 56,638 kept (481 date conflicts excluded) |
| Exact copies | 4,690 (8.3 %) counted once |
| Not yet labelled | 3,762 (6.6 %) — still clustered |
| Articles with an event location | 39.8 % |
| Clusters / events / stories | 25,549 / 2,995 / 2,909 (after merging) |
| Single-article clusters | 74.9 % (normal) |
| Reproducibility | 100 % identical cluster IDs on a second run |
| Precision label check (my proxy vs GDELT's own type) | 93.2 % agreement |
| Events with several located articles that disagree on country | ≈ 20 % ← main data-quality risk |
| Alerts at pilot run | 62 emerging + 265 new-development + 96 topic-trend active |
| Tests | 231 Python + 123 browser checks passing |

---

## 11. Known problems and **my personal to-do / suggestions**

### 11.1 Which GDELT date should we use?
- **Use the article's own `published_at`** (what we do now). It is the real publication time and is identical in PostgreSQL and OpenSearch.
- **Don't use** GDELT's event date (`SQLDATE`/`EventTimeDate`) as the article date: it is the date GDELT *infers* for the event, which can differ from the real publication time (we have not measured how often in our data — worth checking on a sample). Also don't use `DATEADDED` (when GDELT processed it) or our `scraped_at`; both measure processing/collection time, not publication.
- GDELT dates are useful only as a **cross-check**: if `published_at` is more than a few days away from GDELT's date, flag the article as suspicious (old story re-scraped).
- Day boundaries: we bucket by **UTC** (`day_tz` in `news_config.py`). If India-local days matter for reporting, set `day_tz = "Asia/Kolkata"` **before** the full run (changing it later means rebuilding).
- Trusted-source articles: only ~3 % have a usable `published_at` in the label table; `raw.scraped_articles_trusted.published_date` exists but includes impossible values (up to year 2027). **Validate it before trusting it** (reject future dates, old years) and add it as a second date source with a "source" flag.

### 11.2 Location problems (and how to solve them)
| Problem | Evidence | Suggested fix |
|---|---|---|
| Only ~36 % of articles have GDELT `ActionGeo` | OpenSearch counts per day | Add NER-based location (see 11.3) |
| GDELT sometimes geocodes wrongly (e.g. "Delhi University" → Dhaka, "DU") | Found in pilot; ≈ 20 % of multi-located events disagree | Keep our "high-confidence only for alerts" rule; improve with NER agreement check |
| Country-level points sit on the **country centroid** | e.g. India = (20 N, 77 E) | Already handled: centroid points never fill state/polygon areas |
| One article links to several GDELT events | 76 % of linked articles map to > 1 event | Already handled: story location = majority of its articles |
| Mention-only locations | GKG lists places *mentioned* | Already kept separate ("mentioned in" vs "event location") |
| Publisher location | no field exists | Registry file `data/publisher_registry.csv` (53 % of volume classified; **none human-verified yet** — fill the `verified_by` column) |

### 11.3 Articles with **no location** (e.g. `scraped_articles_trusted`) — plan to add NER later
1. **Run NER** (named-entity recognition) on title + first 2–3 paragraphs to get place names. The current News engine does not include a retained NER stage; a future implementation could use spaCy or a multilingual transformer (needs GPU for speed).
2. **Geocode** the place names with a local gazetteer (GeoNames or Natural Earth populated places — free, works offline). Resolve ambiguity with country hints: publisher country, other places in the article, ccTLD.
3. **Store it as a new source:** `geo_source = "ner"` with precision (country/state/city) and the matched text. **Never overwrite GDELT values.**
4. **Combine, don't replace:** if NER and GDELT agree → high confidence. If they disagree (Delhi vs Dhaka) → mark *uncertain* (this fixes the biggest current error). If only NER exists → medium confidence.
5. Code place to change: `news_geo.GeoIndex.article_location()` (one function) + add a column in `article_geo`. All metrics, AOIs and the UI will pick it up automatically.
6. **Test it** the way we tested GDELT: hand-check ~200 articles per precision level and report accuracy before enabling it.
7. Cheap quick win first: **inherit the location from the cluster**. If an unlocated article sits in a cluster whose other articles are located in India, label it "inherited" (lower confidence).

### 11.4 Other suggestions (ordered by value)
1. **Human review of the pilot samples** (`reports/news/cluster_inspection_sample.csv`, `location_inspection_sample.csv`) before approving a full backfill. Silver-label scores are weak; people are the real check.
2. **Use 28-day baselines** (not the 7-day pilot setting) once ≥ 4 weeks of data exist (`baseline_min_days` in `news_config.py`).
3. **Verify the publisher registry** (fill `verified_by`) and extend it past the top 600 domains.
4. **Classifier quality:** the weak point is subtopic confidence (median 0.36). Domain/topic are good enough for filtering; retraining or a confidence filter (e.g. only show subtopics ≥ 0.5) would help. *We deliberately did not change any labels.*
5. **Alert wording review** with analysts: the "new development" detector flags ~30 % of large events — likely right (big stories keep evolving) but needs their judgement.
6. **Schedule + monitor:** add the cron line above and watch `logs/` and the `run_log` table; add an email/Slack hook later (a reference consumer exists in `scripts/notify_poller.py`).
7. **More taxonomy-specific baselines** (sports vs defence have very different "normal" volumes) — the pipeline already compares within the same domain when enough comparison stories exist.
8. **Date-range control** in the UI (the pages currently show the whole processed period).
9. **All-countries run:** change the run script so reading articles (all countries) is separate from choosing which countries get alert baselines.
10. **Scale:** for very large stores, compute story merging in chunks and keep only the last ~14 days of cluster centroids (already the default retention).

---

## 12. FAQ for managers

**Is the original data safe?** Yes. PostgreSQL and OpenSearch are opened read-only, enforced three ways (code guard, read-only session, and a test that proves a write is refused by the server).

**Is it real-time?** No. It is *near-live on a schedule* (e.g. every 30 minutes). The page always shows the state as of the last run.

**Can I trust the ranking?** The numbers on each card are recomputed independently in tests (all match). The ranking is a transparent blend; significance claims come only from proper statistical tests. The main risks are *data* risks: missing/incorrect GDELT locations and noisy subtopic labels.

**Why are some stories listed under strange topics?** We display the classifier's stored label unchanged. (Example: a celebrity attack story labelled "Military & Defence".)

**Why does a country show few stories?** The pilot only ingested articles that name India or Bangladesh, so other countries appear only when those articles report on them.

**What was *not* reused?** Social-media metrics (likes, shares, followers, virality, hate scores) — no data exists for news, so they are removed, not faked.

---

## 13. Glossary
- **Embedding** — 1,024 numbers describing the meaning of an article; similar articles have similar numbers.
- **Cosine similarity** — how alike two embeddings are (1 = identical).
- **Cluster / event** — articles about the same thing at the same time.
- **Story** — one or more near-identical events merged for display.
- **Baseline** — what we'd normally expect; **surprise** — how far today is from it.
- **FDR / q-value** — a correction so that testing hundreds of topics doesn't create false alarms (q < 0.05 ≈ at most 5 % of flagged items are false).
- **AOI** — Area of Interest (country, state, or a drawn shape).
- **Backfill** — processing old data; **incremental** — processing only what is new.
- **NER** — software that finds names of people/places in text.

---

## 14. Where things are

| What | Where |
|---|---|
| Entry point (all run modes) | `scripts/20_news_run.py` |
| Cleaning / location | `scripts/lib/news_ingest.py`, `news_geo.py` |
| Clustering | `scripts/lib/news_cluster.py` |
| Metrics / alerts | `scripts/lib/news_metrics.py`, `news_events.py`, `stats_v2.py` |
| Trending ranking | `scripts/lib/news_trending.py` |
| All tunable numbers | `scripts/lib/news_config.py` (the `PARAMS` block) |
| Web server / API | `UI/news_server.py`, `UI/news_api.py` |
| Pages | `UI/trending.html` (Trending), `UI/news.html` (Advanced) |
| Pilot database | `data/news_pilot_india_bangladesh.duckdb` |
| Quality report | `reports/news/pilot_report.md` |
| Engineering reference | `docs/NEWS_ENGINE.md` |
