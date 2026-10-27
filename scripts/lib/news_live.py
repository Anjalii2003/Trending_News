"""Live processing summary: every value is read from stored records of the `news_engine` PostgreSQL schema (run log, state and
the derived tables). Nothing is hardcoded and nothing is estimated.

Fields (see `summary`): totals of articles processed / unique, clusters / qualifying events, the latest run's article count,
last successful update, latest publication date in the processed data, processing status (running / successful / failed),
pending labels, excluded articles, recent errors and the next scheduled run.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Optional


STALE_RUNNING_SECONDS = 15 * 60       # a 'running' run whose service stopped beating this long ago is reported as failed
STALE_HEARTBEAT_SECONDS = 180          # scheduler considered alive if it beat within this


def _iso(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dt.datetime):
        if v.tzinfo is None:
            v = v.replace(tzinfo=dt.timezone.utc)
        return v.astimezone(dt.timezone.utc).isoformat()
    return str(v)


def _parse(v) -> Optional[dt.datetime]:
    if not v:
        return None
    if isinstance(v, dt.datetime):
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)
    try:
        d = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _run_view(r: dict, now: dt.datetime, alive: Optional[bool]) -> dict:
    status = r.get("status")
    shown = {"running": "running", "success": "successful"}.get(status, "failed")
    err = r.get("error")
    if status == "running" and alive is False:
        hb = _parse(r.get("heartbeat_at"))
        if hb is None or (now - hb).total_seconds() > STALE_RUNNING_SECONDS:
            shown, err = "failed", err or "the scheduler stopped reporting while this run was in progress (interrupted)"
    if status == "abandoned":
        err = err or "interrupted before it finished (process stopped); the next run resumed from the last checkpoint"
    st, fin = _parse(r.get("started_at")), _parse(r.get("finished_at"))
    return {"run_id": r.get("run_id"), "mode": r.get("mode"), "trigger": r.get("trigger"), "status": shown, "raw_status": status,
            "started_at": _iso(st), "finished_at": _iso(fin),
            "duration_seconds": None if not (st and fin) else round((fin - st).total_seconds(), 1),
            "articles_ingested": r.get("articles_ingested"), "articles_unique_new": r.get("articles_unique_new"),
            "clusters_touched": r.get("clusters_touched"), "date_from": _iso(r.get("date_from")), "date_to": _iso(r.get("date_to")),
            "error": err}


def _rows(st, sql: str, params=None) -> list[dict]:
    cur = st.con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def summary(st, now: Optional[dt.datetime] = None) -> dict:
    """The live summary from a (read-only) Store."""
    now = now or dt.datetime.now(dt.timezone.utc)
    state = st.all_state()
    hb = _parse(state.get("service_heartbeat"))
    alive = None if hb is None else (now - hb).total_seconds() <= STALE_HEARTBEAT_SECONDS
    runs = _rows(st, "SELECT * FROM run_log ORDER BY started_at DESC LIMIT 1")
    last_ok = _rows(st, "SELECT * FROM run_log WHERE status = 'success' ORDER BY finished_at DESC LIMIT 1")
    a = _rows(st, "SELECT count(*) AS n, count(*) FILTER (WHERE NOT is_duplicate) AS uniq, max(published_at) AS latest, "
                  "count(*) FILTER (WHERE NOT has_label AND NOT is_duplicate) AS unlabelled FROM article_fact")[0]
    c = _rows(st, "SELECT count(*) AS n, count(*) FILTER (WHERE is_event) AS events FROM event_cluster")[0]
    al = st.scalar("SELECT count(*) FROM alert WHERE status = 'active'")
    fails = _rows(st, "SELECT run_id, started_at, finished_at, status, error FROM run_log WHERE status IN ('failed','abandoned') "
                      "ORDER BY started_at DESC LIMIT 5")
    n_fail24 = st.scalar("SELECT count(*) FROM run_log WHERE status IN ('failed','abandoned') AND started_at > now() - interval '24 hours'")
    tot = (last_ok[0].get("totals") if last_ok else None) or {}
    if isinstance(tot, str):
        tot = json.loads(tot)
    # live counters beat the last run's recorded ones for anything the store can answer now
    pending = st.scalar("SELECT count(*) FROM pending_label")
    ex = {r[0]: int(r[1]) for r in st.con.execute("SELECT reason, count(*) FROM excluded_article GROUP BY 1").fetchall()}
    latest = _run_view(runs[0], now, alive) if runs else None
    consecutive = 0
    for r in _rows(st, "SELECT status FROM run_log ORDER BY started_at DESC LIMIT 20"):
        if r["status"] in ("failed", "abandoned"):
            consecutive += 1
        elif r["status"] == "success":
            break
    nxt = _parse(state.get("next_run_at"))
    running = bool(latest and latest["status"] == "running")
    catching = (None if state.get("catchup_missing_days") is None else int(state["catchup_missing_days"]))
    # what the scheduler is doing NOW; a stored next_run_at that is in the past while a run is in progress is not "next"
    phase = "running" if running else ("catching_up" if (catching or 0) > 0 and nxt and nxt <= now + dt.timedelta(minutes=1) else "idle")
    return {
        "source": "postgresql", "generated_at": _iso(now),
        "status": latest["status"] if latest else "never_run",
        "latest_run": latest,
        "last_successful_update": _iso(_parse(last_ok[0]["finished_at"])) if last_ok else None,
        "last_successful_run": _run_view(last_ok[0], now, alive) if last_ok else None,
        "latest_publication_date": _iso(a["latest"]),
        "articles": {"processed_total": int(a["n"]), "unique_total": int(a["uniq"]),
                     "duplicates_total": int(a["n"]) - int(a["uniq"]),
                     # a run in progress has no final count yet: report the last COMPLETED run's (labelled as such in the UI)
                     "processed_in_latest_run": (latest["articles_ingested"] if latest and latest["status"] != "running"
                                                 else (last_ok[0].get("articles_ingested") if last_ok else None))},
        "clusters": {"total": int(c["n"]), "qualifying_events": int(c["events"])},
        "pending_labels": int(pending or 0), "unlabelled_unique_articles": int(a["unlabelled"]),
        "excluded": {"total": int(sum(ex.values())), "by_reason": ex},
        "active_alerts": int(al or 0),
        "errors": {"recent": [{"run_id": f["run_id"], "at": _iso(f["started_at"]), "status": f["status"], "message": f["error"]} for f in fails],
                   "failed_runs_24h": int(n_fail24 or 0), "consecutive_failures": consecutive},
        "schedule": {"interval_minutes": None if not state.get("interval_seconds") else round(float(state["interval_seconds"]) / 60, 2),
                     "next_run_at": None if running else _iso(nxt), "phase": phase, "scheduler_alive": alive, "last_heartbeat": _iso(hb),
                     "catching_up_days": catching, "scope": state.get("scope")},
        "state": {k: v for k, v in state.items() if k.startswith(("initial_sync", "seeded", "migrated"))},
    }
