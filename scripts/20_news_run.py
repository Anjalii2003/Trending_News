#!/usr/bin/env python3
"""Stage 20 - News Trending & Emerging Events engine: one entry point for every run mode.

  # 1. pilot: 14 published days, India + Bangladesh, isolated pilot database
  python scripts/20_news_run.py pilot                       # last 14 complete days
  python scripts/20_news_run.py pilot --from 2026-10-06 --to 2026-10-19

  # 2. any explicit range / scope (same code path as the pilot)
  python scripts/20_news_run.py range --countries IN,BD --from 2026-10-01 --to 2026-10-19

  # 3. full historical backfill - REFUSED unless explicitly approved on the command line
  python scripts/20_news_run.py backfill --from 2026-08-17 --to 2026-10-19 --i-approve-backfill

  # 4. ongoing: process what was newly ingested since the last run (call from cron / a timer)
  python scripts/20_news_run.py incremental --since 2026-10-19T00:00:00Z   # first run only
  python scripts/20_news_run.py incremental

  python scripts/20_news_run.py status                      # checkpoints, pending labels, last runs
  python scripts/20_news_run.py plan --from .. --to ..      # read-only: how many docs would be processed

Source systems (PostgreSQL, OpenSearch) are only ever read. Credentials come from the environment
(see scripts/lib/news_config.py). A file lock prevents overlapping runs. Runs are resumable and
idempotent: re-running a range skips days already ingested (use --force-days to redo them).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib import news_config as C  # noqa: E402
from lib import news_geo as G  # noqa: E402
from lib import news_ingest as I  # noqa: E402
from lib import news_pipeline as PL  # noqa: E402
from lib import news_sources as NS  # noqa: E402
from lib.news_store import LockBusy, Store, now_utc, run_lock  # noqa: E402

ALERTS_JSON = C.EDA_ROOT / "UI" / "data" / "news_alerts.json"


def log(msg: str) -> None:
    print(f"[20] {msg}", flush=True)


def parse_day(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def build_scope(a) -> I.Scope:
    if a.mode == "pilot" and not a.countries:
        a.countries = ",".join(C.PILOT_SCOPE["countries"])
    cs = [c.strip().upper() for c in a.countries.split(",")] if a.countries else None
    name = a.scope_name or (C.PILOT_SCOPE["name"] if a.mode == "pilot" else ("_".join(c.lower() for c in cs) if cs else "all"))
    return I.Scope(name, cs)


def resolve_range(a) -> tuple[dt.date, dt.date]:
    if a.to:
        d1 = parse_day(a.to)
    else:  # last COMPLETE day in the configured timezone
        d1 = I.to_day(now_utc(), C.PARAMS["day_tz"]) - dt.timedelta(days=1)
    d0 = parse_day(a.from_) if a.from_ else d1 - dt.timedelta(days=a.days - 1)
    if d0 > d1:
        raise SystemExit("--from must be <= --to")
    return d0, d1


def cmd_status(a) -> int:
    with Store(read_only=True) as st:
        print(f"store: PostgreSQL schema {st.schema}")
        print(st.df("SELECT run_id, mode, scope, date_from, date_to, status, started_at, finished_at FROM run_log "
                    "ORDER BY started_at DESC LIMIT 5").to_string(index=False))
        print(st.df("SELECT stage, scope, status, count(*) days, min(day) first_day, max(day) last_day FROM checkpoint "
                    "GROUP BY stage, scope, status ORDER BY 1,2,3").to_string(index=False))
        print("articles:", st.scalar("SELECT count(*) FROM article_fact"),
              "| pending labels:", st.scalar("SELECT count(*) FROM pending_label"),
              "| excluded:", st.scalar("SELECT count(*) FROM excluded_article"),
              "| clusters:", st.scalar("SELECT count(*) FROM event_cluster"),
              "| events:", st.scalar("SELECT count(*) FROM event_cluster WHERE is_event"),
              "| active alerts:", st.scalar("SELECT count(*) FROM alert WHERE status='active'"))
        print(st.df("SELECT key, value FROM state WHERE key LIKE 'ingest_wm:%'").to_string(index=False))
    return 0


def cmd_plan(a) -> int:
    """Read-only: counts what a run would touch. Writes nothing anywhere."""
    d0, d1 = resolve_range(a)
    scope = build_scope(a)
    geo = G.GeoIndex()
    os_ = NS.ReadOnlyOpenSearch()
    sf = scope.os_filter(geo)
    tot = 0
    for d in PL.daterange(d0, d1):
        s, e = I.day_bounds(d, C.PARAMS["day_tz"])
        flt = [{"range": {"published_at": {"gte": I.iso(s), "lt": I.iso(e)}}}] + ([sf] if sf else [])
        n = os_.count({"bool": {"filter": flt}})
        tot += n
        print(f"  {d}  {n:>8,}")
    print(f"scope={scope.name} {d0}..{d1}: {tot:,} documents (metadata only; vectors fetched for clustering)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["pilot", "range", "backfill", "incremental", "status", "plan"])
    ap.add_argument("--countries", help="ISO alpha-2 list for the AOI scope, e.g. IN,BD")
    ap.add_argument("--scope-name")
    ap.add_argument("--from", dest="from_")
    ap.add_argument("--to")
    ap.add_argument("--days", type=int, default=14, help="pilot window length when --from is omitted")
    ap.add_argument("--since", help="ISO time for the FIRST incremental run")
    ap.add_argument("--force-days", action="store_true", help="re-ingest days already marked done")
    ap.add_argument("--max-days", type=int, help="process at most N not-yet-done days (chunked backfill)")
    ap.add_argument("--no-developments", action="store_true")
    ap.add_argument("--i-approve-backfill", action="store_true",
                    help="REQUIRED for 'backfill'; the user must have reviewed the pilot first")
    a = ap.parse_args()
    a.mode_hint = a.mode

    if a.mode == "status":
        return cmd_status(a)
    if a.mode == "plan":
        return cmd_plan(a)
    if a.mode == "backfill" and not a.i_approve_backfill:
        log("REFUSED: a historical backfill needs --i-approve-backfill (review the pilot report first).")
        return 2

    scope = build_scope(a)
    d0 = d1 = None
    if a.mode in ("pilot", "range", "backfill"):
        d0, d1 = resolve_range(a)
        if a.mode == "backfill" and not a.from_:
            log("backfill requires an explicit --from")
            return 2
    try:
        with run_lock():
            st = Store()
            geo = G.GeoIndex()
            pg = NS.ReadOnlyPG()
            try:
                run_id = st.start_run(a.mode, scope.name, d0, d1, {"countries": scope.countries, **C.PARAMS})
                ctx = I.Ctx(st, pg, NS.ReadOnlyOpenSearch(), geo, scope, run_id)
                I.load_taxonomy(ctx)
                log(f"run {run_id} mode={a.mode} scope={scope.name} range={d0}..{d1} schema={st.schema}")
                try:
                    if a.mode == "incremental":
                        since = dt.datetime.fromisoformat(a.since.replace("Z", "+00:00")) if a.since else None
                        summary = PL.run_incremental(ctx, since, run_developments=not a.no_developments)
                    else:
                        summary = PL.run_range(ctx, d0, d1, force_days=a.force_days, max_days=a.max_days,
                                               run_developments=not a.no_developments)
                    summary["counters"] = ctx.counters
                    for k, v in ctx.counters.items():
                        st.record_quality(run_id, k, v)
                    n = PL.export_alerts(st, ALERTS_JSON)
                    st.finish_run(run_id, "success", summary)
                    log(f"success; {n} active alerts exported to {ALERTS_JSON}")
                    print(json.dumps(summary, indent=1, default=str))
                except BaseException as ex:
                    st.finish_run(run_id, "failed", {"counters": ctx.counters}, error=f"{type(ex).__name__}: {ex}"[:1000])
                    raise
            finally:
                pg.close()
                st.close()
    except LockBusy as e:
        log(f"SKIPPED: {e}")
        return 3
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
