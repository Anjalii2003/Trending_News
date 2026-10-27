# News engine - pilot validation report

Database `news_pilot_india_bangladesh.duckdb` - published days 2026-10-06 .. 2026-10-19 - generated 2026-10-20 13:57 UTC. Source PostgreSQL / OpenSearch were only read.

## 1. Data quality and coverage

* documents seen in scope: **57119**; ingested **56,638**; excluded for date conflict: **481**; missing/invalid date: **0** (`scraped_at` is never substituted)
* exact duplicates (same `content_hash`): **4,690** (8.3%) - kept, flagged, not counted
* not yet labelled in PostgreSQL: **3,762** (6.6%) - used for events, excluded from taxonomy series
* event location available (non-duplicate articles): **20,689 of 51,948 (39.8%)** - the rest are `unlocated` and shown as such; geo-inconsistent (FIPS vs polygon disagree): 72; unmapped country code: 24

Corpus-wide event-location coverage per day (OpenSearch counts, whole corpus - NOT only this scope):

| day | n_dated_total | n_action_country | n_action_geo | action_country_% | action_geo_% |
|---|---|---|---|---|---|
| 2026-10-06 00:00:00 | 68822 | 24564 | 24560 | 35.700 | 35.700 |
| 2026-10-07 00:00:00 | 67374 | 23573 | 23568 | 35.000 | 35.000 |
| 2026-10-08 00:00:00 | 70681 | 25771 | 25767 | 36.500 | 36.500 |
| 2026-10-09 00:00:00 | 63818 | 24161 | 24158 | 37.900 | 37.900 |
| 2026-10-10 00:00:00 | 37178 | 12977 | 12975 | 34.900 | 34.900 |
| 2026-10-11 00:00:00 | 34320 | 12558 | 12557 | 36.600 | 36.600 |
| 2026-10-12 00:00:00 | 58210 | 20187 | 20183 | 34.700 | 34.700 |
| 2026-10-13 00:00:00 | 67994 | 24847 | 24844 | 36.500 | 36.500 |
| 2026-10-14 00:00:00 | 61382 | 22179 | 22178 | 36.100 | 36.100 |
| 2026-10-15 00:00:00 | 63393 | 22976 | 22976 | 36.200 | 36.200 |
| 2026-10-16 00:00:00 | 58825 | 21624 | 21624 | 36.800 | 36.800 |
| 2026-10-17 00:00:00 | 35784 | 13172 | 13172 | 36.800 | 36.800 |
| 2026-10-18 00:00:00 | 33107 | 11510 | 11510 | 34.800 | 34.800 |
| 2026-10-19 00:00:00 | 59151 | 20905 | 20903 | 35.300 | 35.300 |

Date provenance of ingested articles:

| confidence | date_src | n |
|---|---|---|
| high | jsonld.datePublished | 48362 |
| medium | opengraph.article:published_time | 8161 |
| medium | html_meta.datePublished | 115 |

Ingest lag (OpenSearch `ingested_at` - `published_at`): median 2.8 h, p90 7.0 h, p99 19.5 h. PostgreSQL-vs-OpenSearch publication-date mismatches >1 h: 0. Documents present in both OpenSearch backing indices: 0.

## 2. Event clustering

* clusters: **25,549**; singletons **19,128 (74.9%)**; events (>= 3 articles from >= 2 publishers): **2,995**

| size_bin | clusters | articles | avg_publishers |
|---|---|---|---|
| 1 | 19128 | 19128.000 | 1.000 |
| 2 | 3297 | 6594.000 | 1.800 |
| 3-5 | 2087 | 7477.000 | 3.100 |
| 6-15 | 741 | 6427.000 | 6.700 |
| 16-50 | 236 | 6115.000 | 14.600 |
| >50 | 60 | 6207.000 | 34.500 |

Largest clusters (a very large cluster may merge related sub-stories - check these first):

| n_articles | n_publishers | n_copies | first_headline |
|---|---|---|---|
| 434 | 203 | 545 | Two pilots among 3 killed in Houthi attacks on Saudi airports |
| 418 | 114 | 85 | CJP, Left groups allege supporters detained ahead of October 10 protest |
| 241 | 87 | 37 | Nana Patekar Dies at 75 After Being Found Unconscious at Goa Residence |
| 200 | 60 | 4 | Tiger Shroff Escapes Unhurt in Mumbai Car Attack |
| 195 | 38 | 2 | 17 dead after stampede at Kolkata Durga Puja pandal; CM Suvendu Adhikari reaches hospital |

**Silver-label check** (GDELT `globaleventid` shared by >=2 articles = weak 'same event' label; 8,847 articles / 2,824 GDELT events): pairwise recall **0.310**, strict precision **0.061** (a lower bound: GDELT splits one real event into several ids). This is NOT comparable with the earlier experiment's numbers (different pair definition and sample). Low strict precision is expected when one real story is reported under many GDELT ids (median distinct GDELT ids per multi-article silver cluster: 1). Silver labels are weak evidence - **human review of `cluster_inspection_sample.csv` is the real validation and has NOT been done by you yet.**

Wrote `cluster_inspection_sample.csv` (42 clusters, stratified by size) with an empty `analyst_verdict` column (SAME_EVENT / MIXED / DIFFERENT_EVENTS).

## 3. Location accuracy (event location, GDELT ActionGeo)

| loc_precision | n |
|---|---|
| unlocated | 31259 |
| city | 16703 |
| country | 3669 |
| adm1 | 317 |

Event clusters by location status (consensus of located members; `ambiguous` = members disagree):

| loc_status | n |
|---|---|
| located | 2063 |
| unlocated | 896 |
| ambiguous | 36 |

Event-location **confidence** (high = >=3 located members and >=80% agree; only high-confidence locations can raise AOI alerts):

| confidence | events |
|---|---|
| high | 954 |
| (not located) | 932 |
| medium | 640 |
| low | 469 |

Of 2,493 clusters with >=2 located articles, **504 (20.2%) have members geocoded to more than one country** - GDELT ActionGeo is noisy. Example found in the pilot: a Delhi University story whose members were geocoded to Dhaka (3 of 4 members). No cheap automatic validator exists: mention-support is circular (99.8% agreement, same geocoder) and a headline place-name check was measured and REJECTED (8.3% of located articles flagged, mostly legitimate cross-border stories).

Wrote `location_inspection_sample.csv` (60 located articles: 30 city, 15 adm1, 15 country) with an empty `analyst_verdict` column.

Cross-check against PostgreSQL `raw.gdelt_events` (read-only):

* 400 of 400 sampled GDELT events found; precision-proxy agreement with `ActionGeo_Type` **93.5%** (n=400); OpenSearch `action_location` == PG `ActionGeo_FullName`: 100.0%
* confusion (rows = PG type, cols = proxy): `{'adm1': {'adm1': 3, 'city': 0, 'country': 0}, 'city': {'adm1': 24, 'city': 305, 'country': 0}, 'country': {'adm1': 0, 'city': 2, 'country': 66}}`

AOI coverage on the last pilot day (scope = everything ingested; unlocated shown, never hidden):

| aoi_id | n_articles | n_located_event | n_mention_only | n_scope_unlocated | n_scope_total | pct_scope_unlocated |
|---|---|---|---|---|---|---|
| country:IN:mentioned | 3242 | 728 | 0 | 2261 | 3584 | 63.100 |
| country:IN:event | 974 | 974 | 2514 | 2261 | 3584 | 63.100 |
| adm1:IND-2447:event | 201 | 201 | 3087 | 2261 | 3584 | 63.100 |
| adm1:IND-2428:event | 195 | 195 | 3091 | 2261 | 3584 | 63.100 |
| country:BD:mentioned | 134 | 27 | 0 | 2261 | 3584 | 63.100 |
| adm1:IND-2446:event | 70 | 70 | 3188 | 2261 | 3584 | 63.100 |
| adm1:IND-3253:event | 51 | 51 | 3203 | 2261 | 3584 | 63.100 |
| adm1:IND-20011:event | 49 | 49 | 3206 | 2261 | 3584 | 63.100 |

## 4. Trending metrics

15 of 47 registered AOIs reach the minimum evidence for node series (>= 100 labelled in-scope articles); the others report NULL with a reason, never zeros. Baseline needs 7 days, so with a 14-day pilot only days 8-14 can be tested (**limitation**: production should use the configured 28-day window with >= 14 baseline days).

| aoi_id | kind | mode | series | supported_days | gap_rows |
|---|---|---|---|---|---|
| country:IN:mentioned | country | mentioned | 135 | 865.000 | 0 |
| country:IN:event | country | event | 112 | 619.000 | 0 |
| country:BD:mentioned | country | mentioned | 63 | 217.000 | 0 |
| adm1:IND-2428:event | adm1 | event | 61 | 255.000 | 0 |
| adm1:IND-2447:event | adm1 | event | 51 | 207.000 | 0 |
| adm1:IND-3263:event | adm1 | event | 28 | 90.000 | 0 |
| adm1:IND-2446:event | adm1 | event | 26 | 83.000 | 26 |
| adm1:IND-3253:event | adm1 | event | 20 | 50.000 | 40 |
| adm1:IND-20011:event | adm1 | event | 19 | 65.000 | 19 |
| country:BD:event | country | event | 15 | 18.000 | 60 |
| adm1:IND-3261:event | adm1 | event | 14 | 10.000 | 70 |
| adm1:IND-2430:event | adm1 | event | 11 | 0.000 | 88 |

**Anti-restatement audit** (Spearman vs V, support-gated rows; thresholds of `scripts/audit_restatement.py`):

| metric | n | spearman | pearson | verdict |
|---|---|---|---|---|
| V_rate | 2479 | 0.148 | 0.240 | clean |
| share_V | 2479 | 0.148 | 0.256 | clean |
| growth | 2479 | -0.036 | -0.069 | clean |
| growth_7d | 2479 | 0.060 | 0.046 | clean |
| momentum | 2479 | -0.077 | -0.111 | clean |
| accel | 2479 | -0.007 | -0.032 | clean |
| accel_smooth | 2479 | -0.016 | -0.040 | clean |
| surprise | 2479 | 0.136 | 0.034 | clean |
| burst_intensity | 2479 | 0.790 | 0.890 | volume (declared) |
| publisher_pie | 2479 | -0.186 | 0.093 | clean |

Verdict: no undeclared volume restatements.

## 5. Alerts (FDR-controlled)

| kind | status | n | min_q | max_q |
|---|---|---|---|---|
| emerging_event | active | 62 | 0.006 | 0.050 |
| new_development | active | 265 | 0.000 | 0.049 |
| node_trend | active | 96 | 0.000 | 0.048 |

Emerging-event tests by support (`support_ok=false` = fewer than 50 reference clusters; result NULL, no alert):

| support_ok | n |
|---|---|
| False | 242 |
| True | 1164 |

New-development tests: 859 (event, AOI, evaluation-time) tests over 212 events, 265 flagged. The statistic was calibrated on synthetic data (false positive <4%, planted sub-story found >95%) **but the real-data flag rate is high because large events genuinely evolve; it needs your review.**

Top active alerts per kind (rationale is the audit trail):


**emerging_event**

| aoi_id | q | severity | rationale |
|---|---|---|---|
| country:BD:event | 0.006 | high | 10 publishers / 10 articles within 19h; exceeded by 0 of 166 reference clusters (aoi_pool); BH q=0.0060: 'Immature, counterproductive': Sheikh Hasina slams Bangladesh PM Rahman for skipping  |
| country:BD:event | 0.006 | high | 8 publishers / 8 articles within 23h; exceeded by 0 of 166 reference clusters (aoi_pool); BH q=0.0060: From climate-driven disasters to Rohingya refugee camps, solar kits help keep the light |
| adm1:IND-3263:event | 0.012 | medium | 26 publishers / 37 articles within 18h; exceeded by 0 of 428 reference clusters (aoi_pool); BH q=0.0117: Tholi Prema Music Director Deva Passes Away At 75 |
| adm1:IND-3257:event | 0.013 | medium | 21 publishers / 50 articles within 8h; exceeded by 0 of 154 reference clusters (aoi_pool); BH q=0.0129: 17 dead after stampede at Kolkata Durga Puja pandal; CM Suvendu Adhikari reaches hospi |
| adm1:IND-3257:event | 0.013 | medium | 14 publishers / 20 articles within 7h; exceeded by 0 of 154 reference clusters (aoi_pool); BH q=0.0129: At least 17 killed after deadly stampede at India religious festival |


**new_development**

| aoi_id | q | severity | rationale |
|---|---|---|---|
| adm1:IND-2428:event | 0.000 | medium | 17 new articles (14 publishers) shifted away from the event's earlier coverage (outside the earlier-coverage envelope; coherence 0.93, rank-test q=0.0000): Rahul Gandhi-led INDIA bloc MPs me |
| country:IN:event | 0.000 | medium | 80 new articles (51 publishers) shifted away from the event's earlier coverage (outside the earlier-coverage envelope; coherence 0.91, rank-test q=0.0000): Actor Imran Khan asked to leave In |
| adm1:IND-2447:event | 0.000 | medium | 80 new articles (51 publishers) shifted away from the event's earlier coverage (outside the earlier-coverage envelope; coherence 0.91, rank-test q=0.0000): Actor Imran Khan asked to leave In |
| adm1:IND-2477:event | 0.000 | medium | 21 new articles (9 publishers) shifted away from the event's earlier coverage (outside the earlier-coverage envelope; coherence 0.92, rank-test q=0.0000): Assam bypoll: Counting of votes beg |
| country:IN:event | 0.000 | medium | 17 new articles (14 publishers) shifted away from the event's earlier coverage (outside the earlier-coverage envelope; coherence 0.93, rank-test q=0.0000): Rahul Gandhi-led INDIA bloc MPs me |


**node_trend**

| aoi_id | q | severity | rationale |
|---|---|---|---|
| country:BD:event | 0.000 | high | category D08.T02.C01: 12 articles vs baseline 1.0 (11.4x), BH q=0.0000 |
| country:BD:event | 0.000 | high | topic D08.T02: 12 articles vs baseline 1.3 (9.3x), BH q=0.0000 |
| adm1:IND-2446:event | 0.000 | high | topic D04.T04: 14 articles vs baseline 1.5 (9.1x), BH q=0.0000 |
| adm1:IND-2447:event | 0.000 | high | topic D02.T01: 17 articles vs baseline 1.9 (8.8x), BH q=0.0000 |
| country:BD:event | 0.000 | high | category D01.T02.C02: 13 articles vs baseline 1.6 (8.0x), BH q=0.0000 |


## 6. Limitations (read before approving a backfill)

* Event location is GDELT ActionGeo for ~40% of articles; the rest are `unlocated` and cannot be in any event-location AOI. Bangladesh has very few event-located articles; its AOI mostly reports NULL with a reason.
* GDELT ActionGeo is itself machine-extracted and may be wrong; accuracy needs the human check of the location sample.
* Mentioned-in AOIs are country-level only and are a different population from event-location AOIs.
* Cluster quality is unvalidated by humans; singletons dominate (most articles are not part of a multi-publisher event).
* A 14-day window leaves only 7 testable days per node series; emerging-event references are limited to <= 14 days.
* Late-arriving data revises node series; revised alerts are marked `retracted`, never deleted.
* Natural Earth boundaries are de facto lines, not an authoritative position on disputed territory; GDELT points that fall in disputed areas may be flagged `geo_inconsistent` and demoted to country precision.
* No publisher-location filter exists yet (see the publisher registry review file).