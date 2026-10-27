#!/usr/bin/env python3
"""News Engine live service: scheduled, resumable, non-overlapping processing for ALL countries.

  news_service.py run            # forever: one cycle every NEWS_INTERVAL_MINUTES (the Docker entry point)
  news_service.py once           # exactly one cycle, then exit (cron / tests / an operator)
  news_service.py health         # exit 0 when the service is alive and not stuck (Docker HEALTHCHECK)
  news_service.py status         # the live summary (from stored records) as JSON

One cycle (same code as the pilot/backfill/incremental runs, see lib/news_pipeline.py):
  1. missing COMPLETE days (checkpointed per day) are processed first, NEWS_CATCHUP_DAYS_PER_CYCLE at a time - this is the
     "resume from the last checkpoint" step; it never touches days already done and never starts a historical backfill:
     the first missing day is the day after the last processed one (or NEWS_START_DATE) and it ends yesterday;
  2. otherwise newly ingested articles since the stored watermark are processed (6 h overlap, idempotent);
  3. clusters, trend metrics and alerts are recomputed for the affected areas;
  4. stories and rankings are materialised from the committed clusters (PostgreSQL tables `story`, `story_event`,
     `trending_ranking`); the dashboard reads the live schema directly (one consistent snapshot per request).
Everything lives in the PostgreSQL schema `news_engine` (no local database). Overlap is impossible: a file lock (this host/volume)
and a PostgreSQL advisory lock (any host) must both be held.
A failed cycle is recorded (run log + error text) and retried on the next tick; checkpoints make the retry resume where
the failure happened. Source systems (PostgreSQL tables, OpenSearch) are only read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import news_config as C  # noqa: E402
from lib import news_geo as G  # noqa: E402
from lib import news_ingest as I  # noqa: E402
from lib import news_live as LV  # noqa: E402
from lib import news_pgstore as PGS  # noqa: E402
from lib import news_stories as NST  # noqa: E402
from lib import news_pipeline as PL  # noqa: E402
from lib import news_relevance as NR  # noqa: E402
from lib import news_sources as NS  # noqa: E402
from lib.news_store import LockBusy, Store, now_utc, run_lock  # noqa: E402

SCOPE = os.environ.get("NEWS_SCOPE_NAME", "all")
HEARTBEAT_FILE = C.STATE_DIR / "service_heartbeat.json"
STOP = threading.Event()


def log(msg: str) -> None:
    print(f"{now_utc():%Y-%m-%dT%H:%M:%SZ} [news-service] {msg}", flush=True)


def env_int(name: str, default: int, lo: int = 0) -> int:
    v = os.environ.get(name)
    try:
        return max(lo, int(v)) if v not in (None, "") else default
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {v!r}")


def run_at_utc() -> Optional[dt.time]:
    """NEWS_RUN_AT_UTC=HH:MM -> run once a day at that time (UTC) instead of every interval."""
    v = os.environ.get("NEWS_RUN_AT_UTC")
    if not v:
        return None
    try:
        return dt.datetime.strptime(v.strip(), "%H:%M").time()
    except ValueError:
        raise SystemExit(f"NEWS_RUN_AT_UTC must be HH:MM (UTC), got {v!r}")


def next_daily(now: dt.datetime, at: dt.time) -> dt.datetime:
    t = now.replace(hour=at.hour, minute=at.minute, second=0, microsecond=0)
    return t if t > now else t + dt.timedelta(days=1)


def interval_seconds() -> int:
    m = os.environ.get("NEWS_INTERVAL_MINUTES")
    try:
        return max(30, int(float(m) * 60)) if m else 30 * 60
    except ValueError:
        raise SystemExit(f"NEWS_INTERVAL_MINUTES must be a number, got {m!r}")


# --------------------------------------------------------------------------
# heartbeat (file for the container health check; PG state for the dashboard)
# --------------------------------------------------------------------------
class Heartbeat(threading.Thread):
    """Beats every 30 s from its own thread/connection, so a long recompute does not look like a dead service."""

    def __init__(self, interval: int):
        super().__init__(daemon=True)
        self.state = {"state": "starting", "pid": os.getpid(), "interval_seconds": interval, "next_run_at": None,
                      "cycle_started_at": None, "last_status": None, "service_started_at": now_utc().isoformat()}
        self._st: Optional[Store] = None
        self._halt = threading.Event()

    def update(self, **kw) -> None:
        self.state.update(kw)
        self._write()

    def _write(self) -> None:
        d = {**self.state, "ts": now_utc().isoformat()}
        tmp = HEARTBEAT_FILE.with_suffix(".tmp")
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(d, default=str))
            os.replace(tmp, HEARTBEAT_FILE)
        except OSError as e:
            log(f"heartbeat file not written: {e}")
        try:
            if self._st is None:
                self._st = Store()
            self._st.set_kv("service_heartbeat", d["ts"])
        except Exception as e:                  # the heartbeat must never take the service down
            log(f"heartbeat to PostgreSQL failed: {type(e).__name__}: {str(e)[:120]}")
            try:
                if self._st:
                    self._st.close()
            finally:
                self._st = None

    def run(self) -> None:
        while not self._halt.wait(30):
            self._write()

    def stop(self) -> None:
        self._halt.set()
        if self._st:
            self._st.close()


def health() -> int:
    """0 = healthy. Unhealthy when the heartbeat is stale, the current cycle has run far too long, or the next run is
    overdue by more than twice the interval (the loop is wedged)."""
    try:
        d = json.loads(HEARTBEAT_FILE.read_text())
        ts = dt.datetime.fromisoformat(d["ts"])
    except Exception as e:
        print(f"unhealthy: no readable heartbeat ({type(e).__name__})")
        return 1
    now = now_utc()
    age = (now - ts).total_seconds()
    if age > 120:
        print(f"unhealthy: heartbeat is {age:.0f}s old")
        return 1
    max_cycle = env_int("NEWS_MAX_CYCLE_MINUTES", 180, 1) * 60
    if d.get("state") == "running" and d.get("cycle_started_at"):
        run_s = (now - dt.datetime.fromisoformat(d["cycle_started_at"])).total_seconds()
        if run_s > max_cycle:
            print(f"unhealthy: cycle running for {run_s / 60:.0f} min (limit {max_cycle // 60})")
            return 1
    if d.get("state") == "idle" and d.get("next_run_at"):
        overdue = (now - dt.datetime.fromisoformat(d["next_run_at"])).total_seconds()
        if overdue > 2 * int(d.get("interval_seconds") or 1800) + 300:
            print(f"unhealthy: next run overdue by {overdue / 60:.0f} min")
            return 1
    print(f"healthy: state={d.get('state')} last_status={d.get('last_status')} heartbeat={age:.0f}s")
    return 0


# --------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------
def start_day(st: Store) -> dt.date:
    """First day the live scope must cover. Never defaults to 'the beginning of the corpus'."""
    v = os.environ.get("NEWS_START_DATE")
    if v:
        d = dt.date.fromisoformat(v)
        if st.get_kv("live_start_day") != d.isoformat():
            st.set_kv("live_start_day", d.isoformat())          # remembered, so a restart without the variable keeps the window
        return d
    kv = st.get_kv("live_start_day")
    if kv:
        return dt.date.fromisoformat(kv)
    last = st.scalar("SELECT max(day) FROM article_fact")
    if last is None:
        raise SystemExit("the live store is empty and NEWS_START_DATE is not set; refusing to guess a start "
                         "(a full historical backfill needs an explicit approval)")
    d = last + dt.timedelta(days=1)
    st.set_kv("live_start_day", d.isoformat())
    return d


def missing_days(st: Store, scope: str, first: dt.date, today: dt.date) -> list[dt.date]:
    """Complete days (first .. yesterday) with no 'done' ingest checkpoint for this scope."""
    out, d = [], first
    while d < today:
        if not st.stage_done("ingest", scope, d):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------
# one cycle
# --------------------------------------------------------------------------
def run_cycle(trigger: str = "scheduled", hb: Optional[Heartbeat] = None, interval: Optional[int] = None) -> dict:
    """One full cycle. Returns {'status', 'run_id', 'catching_up', ...}. Raises LockBusy if another run holds a lock."""
    interval = interval or interval_seconds()
    with run_lock(C.NEWS_LOCK):
        st = Store()
        try:
            if not st.try_lock():
                raise LockBusy("another News Engine service holds the PostgreSQL advisory lock")
            return _cycle(trigger, st, hb, interval)
        finally:
            st.close()


def _cycle(trigger: str, st: Store, hb: Optional[Heartbeat], interval: int) -> dict:
    t0 = now_utc()
    pg = None
    run_id = None
    try:
        today = I.to_day(t0, C.PARAMS["day_tz"])
        first = start_day(st)
        todo = missing_days(st, SCOPE, first, today)
        chunk = todo[:env_int("NEWS_CATCHUP_DAYS_PER_CYCLE", 1, 1)]
        mode = "range" if chunk else "incremental"
        run_id = st.start_run(mode, SCOPE, chunk[0] if chunk else None, chunk[-1] if chunk else None,
                              {"trigger": trigger, "scope": SCOPE, "interval_seconds": interval})
        st.set_kv("scope", SCOPE)
        st.set_kv("interval_seconds", str(interval))
        st.set_kv("catchup_missing_days", str(len(todo)))
        if hb:
            hb.update(state="running", cycle_started_at=t0.isoformat())
        log(f"run {run_id} mode={mode} trigger={trigger} missing_complete_days={len(todo)}"
            + (f" processing {chunk[0]}..{chunk[-1]}" if chunk else ""))
        pg = NS.ReadOnlyPG()
        ctx = I.Ctx(st, pg, NS.ReadOnlyOpenSearch(), G.GeoIndex(), I.Scope(SCOPE, None), run_id)
        if len(todo) > len(chunk):                        # days still waiting: keep the centroids their articles could join
            nxt = todo[len(chunk)]
            ctx.prune_floor = I.day_bounds(nxt, C.PARAMS["day_tz"])[0] - dt.timedelta(hours=C.PARAMS["max_gap_hours"])
        I.load_taxonomy(ctx)
        ensure_reference(ctx)
        if st.get_kv(f"ingest_wm:{SCOPE}") is None:
            # everything ingested from now on is picked up by the incremental stage; set BEFORE the catch-up scans so
            # nothing that arrives while they run can fall between the two
            st.set_kv(f"ingest_wm:{SCOPE}", (t0 - dt.timedelta(hours=1)).isoformat())
        if chunk:
            summary = PL.run_range(ctx, chunk[0], chunk[-1], run_developments=True)
        else:
            summary = PL.run_incremental(ctx, None, run_developments=True)
        if STOP.is_set():
            log("stop requested: finishing the publication of this cycle before exiting")
        ingested = len(ctx.new_keys)
        unique_new = _unique_count(st, ctx.new_keys)
        summary.update(counters=ctx.counters, articles_ingested=ingested, articles_unique_new=unique_new)
        for k, v in ctx.counters.items():
            st.record_quality(run_id, k, v)
        if hb:
            hb.update(state="publishing")
        stories = NST.materialise(st, run_id)                 # stories + rankings, from the clusters just committed
        summary["stories"] = stories
        totals = PGS.run_totals(st)
        st.set_kv("ingest_watermark", st.get_kv(f"ingest_wm:{SCOPE}"))
        st.set_kv("catchup_missing_days", str(max(0, len(todo) - len(chunk))))
        log(f"timings {summary.get('timings')}")
        st.finish_run(run_id, "success", summary, ingested=ingested, unique_new=unique_new, touched=len(ctx.touched), totals=totals)
        remaining = max(0, len(todo) - len(chunk))
        log(f"run {run_id} success: {ingested} new articles ({unique_new} unique), {len(ctx.touched)} clusters touched, "
            f"{len(ctx.recomputed)} areas recomputed, {stories.get('stories')} stories, {remaining} missing days left, "
            f"{(now_utc() - t0).total_seconds():.0f}s")
        return {"status": "success", "run_id": run_id, "catching_up": remaining > 0, "articles_ingested": ingested,
                "articles_unique_new": unique_new, "stories": stories}
    except BaseException as ex:
        msg = f"{type(ex).__name__}: {ex}"[:1500]
        log(f"run {run_id} FAILED: {msg}")
        traceback.print_exc()
        try:
            if run_id:
                st.finish_run(run_id, "failed", {}, error=msg)
        except Exception as e2:
            log(f"could not record the failure: {e2}")
        if isinstance(ex, (KeyboardInterrupt, SystemExit)):
            raise
        return {"status": "failed", "run_id": run_id, "error": msg, "catching_up": False}
    finally:
        if pg:
            pg.close()


def ensure_reference(ctx) -> None:
    """Build the relevance reference once (ground-truth embeddings, read-only). If it cannot be built the cycle still runs;
    stories are then simply not relevance-scored (and, by design, not filtered)."""
    if NR.REFERENCE_FILE.exists():
        return
    try:
        n = NR.build_reference(ctx.pg, ctx.osc, log=log)
        log(f"relevance reference built: {n} ground-truth articles -> {NR.REFERENCE_FILE.name}")
    except Exception as e:
        log(f"relevance reference NOT built ({type(e).__name__}: {str(e)[:150]}); stories will not be relevance-scored")


def _parse_ts(v: str) -> dt.datetime:
    d = dt.datetime.fromisoformat(v.replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def _unique_count(st: Store, keys) -> int:
    import pandas as pd
    if not keys:
        return 0
    k = pd.DataFrame(sorted(keys), columns=["source_table", "article_id"])
    st.con.register("_nk", k)
    try:
        return int(st.scalar("SELECT count(*) FROM article_fact f JOIN _nk USING (source_table, article_id) WHERE NOT f.is_duplicate") or 0)
    finally:
        st.con.unregister("_nk")


# --------------------------------------------------------------------------
# scheduler loop
# --------------------------------------------------------------------------
def serve() -> int:
    interval = interval_seconds()
    hb = Heartbeat(interval)
    hb.start()
    hb.update(state="starting")
    if threading.current_thread() is threading.main_thread():      # (tests run serve() on a worker thread)
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, lambda *_: STOP.set())
    log(f"service started: scope={SCOPE} interval={interval / 60:g} min store=postgresql schema={PGS.schema_name()}")
    failures = 0
    next_run = now_utc()                                  # first cycle immediately (it also resumes any catch-up)
    try:
        while not STOP.is_set():
            wait = (next_run - now_utc()).total_seconds()
            if wait > 0:
                hb.update(state="idle", next_run_at=next_run.isoformat())
                _publish_next(next_run)
                STOP.wait(min(wait, 30))
                continue
            started = now_utc()
            try:
                res = run_cycle("scheduled", hb, interval)
            except LockBusy as e:
                log(f"skipped: {e}")
                res = {"status": "skipped", "catching_up": False}
            if res["status"] == "failed":
                failures += 1
            elif res["status"] == "success":
                failures = 0
            # catching up: continue straight away; otherwise fixed-rate schedule, with a floor after failures
            if res.get("catching_up"):
                next_run = now_utc() + dt.timedelta(seconds=5)
            elif res["status"] == "failed":
                next_run = now_utc() + dt.timedelta(seconds=min(interval, 60 * 2 ** min(failures, 5)))
            elif run_at_utc() is not None:
                next_run = next_daily(now_utc(), run_at_utc())
            else:
                next_run = max(now_utc() + dt.timedelta(seconds=1), started + dt.timedelta(seconds=interval))
            hb.update(state="idle", last_status=res["status"], next_run_at=next_run.isoformat(), cycle_started_at=None)
            _publish_next(next_run)
    finally:
        hb.update(state="stopped")
        hb.stop()
        log("service stopped")
    return 0


def _publish_next(next_run: dt.datetime) -> None:
    """The next scheduled run is a stored record too (read by the dashboard)."""
    try:
        with Store() as st:
            st.set_kv("next_run_at", next_run.isoformat())
    except Exception as e:
        log(f"could not record next_run_at: {type(e).__name__}: {str(e)[:100]}")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "once", "health", "status"])
    a = ap.parse_args()
    if a.cmd == "health":
        return health()
    if a.cmd == "run":
        return serve()
    if a.cmd == "once":
        try:
            res = run_cycle("manual")
        except LockBusy as e:
            log(f"SKIPPED: {e}")
            return 3
        print(json.dumps(res, default=str))
        return 0 if res["status"] == "success" else 1
    if a.cmd == "status":
        with Store(read_only=True) as st:
            print(json.dumps(LV.summary(st), indent=1, default=str))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
