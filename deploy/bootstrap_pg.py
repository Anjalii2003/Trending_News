#!/usr/bin/env python3
"""One-time provisioning of the News Engine's PostgreSQL footprint (schema + two least-privilege roles).

  python deploy/bootstrap_pg.py                 # DRY RUN: shows what it would do, changes nothing
  python deploy/bootstrap_pg.py --apply         # creates schema news_engine + roles, writes deploy/news.env + dashboard.env
  python deploy/bootstrap_pg.py --verify        # read-only audit of the privileges the roles actually have

Needs ADMIN credentials, taken from the environment file named by NEWS_ENV_FILE (SOURCE_DATABASE_URL) - they are used for this
provisioning only and are never written anywhere. The engine itself afterwards runs as `news_engine_app`, which can only
SELECT two source tables and write its own schema; the dashboard runs as `news_engine_reader` (SELECT on the schema).
Opensearch credentials are copied into deploy/news.env (mode 600, git-ignored) because the ingest container needs them.
The SQL it executes is deploy/sql/bootstrap_admin.sql with generated passwords.
"""
from __future__ import annotations

import argparse
import os
import secrets
import stat
import sys
from pathlib import Path

import psycopg2
from psycopg2 import sql

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from lib import news_config as C  # noqa: E402

DEPLOY = ROOT / "deploy"
SCHEMA, APP, READER = "news_engine", "news_engine_app", "news_engine_reader"
SOURCE_TABLES = [("tmc", "article_label"), ("tmc", "taxonomy_node")]


def admin_conn():
    s = C.pg_settings()                    # same resolution as the engine: NEWS_PG_* | SOURCE_DATABASE_URL (via NEWS_ENV_FILE)
    return psycopg2.connect(host=s["host"], port=s["port"], user=s["user"], password=s["password"], dbname=s["dbname"],
                            connect_timeout=30, application_name="news_engine_bootstrap"), s


def role_exists(cur, name) -> bool:
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", [name])
    return cur.fetchone() is not None


def plan(cur, dbname) -> list[str]:
    out = []
    cur.execute("SELECT 1 FROM pg_namespace WHERE nspname = %s", [SCHEMA])
    out.append(("schema exists already - kept" if cur.fetchone() else f"CREATE SCHEMA {SCHEMA}"))
    for r in (APP, READER):
        out.append(f"role {r} exists already - kept (password unchanged)" if role_exists(cur, r) else f"CREATE ROLE {r} LOGIN (no superuser/createdb/createrole)")
    out += [f"ALTER SCHEMA {SCHEMA} OWNER TO {APP}", f"GRANT CONNECT ON DATABASE {dbname} TO {APP}, {READER}",
            f"GRANT USAGE ON SCHEMA tmc TO {APP}; GRANT SELECT ON tmc.article_label, tmc.taxonomy_node TO {APP}",
            f"GRANT SELECT (source_table, source_id, gt_domain_id) ON tmc.a0_ground_truth TO {APP}  (column-level: ids + domain only)",
            f"GRANT USAGE ON SCHEMA {SCHEMA} TO {READER}; default SELECT on {SCHEMA} tables to {READER}"]
    return out


def apply(conn, dbname, app_pw, reader_pw):
    made = {}
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA)))
        for role, pw in ((APP, app_pw), (READER, reader_pw)):
            if role_exists(cur, role):
                made[role] = False
                continue
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION CONNECTION LIMIT 12")
                        .format(sql.Identifier(role), sql.Literal(pw)))
            made[role] = True
        I = sql.Identifier
        stmts = [
            sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(I(SCHEMA), I(APP)),
            sql.SQL("ALTER ROLE {} SET search_path = {}").format(I(APP), I(SCHEMA)),
            sql.SQL("ALTER ROLE {} SET search_path = {}").format(I(READER), I(SCHEMA)),
            sql.SQL("GRANT CONNECT ON DATABASE {} TO {}, {}").format(I(dbname), I(APP), I(READER)),
            sql.SQL("GRANT USAGE ON SCHEMA tmc TO {}").format(I(APP)),
            *[sql.SQL("GRANT SELECT ON {}.{} TO {}").format(I(s), I(t), I(APP)) for s, t in SOURCE_TABLES],
            # relevance reference: the classifier's ground-truth ids + domain ONLY (no title/content columns)
            sql.SQL("GRANT SELECT (source_table, source_id, gt_domain_id) ON tmc.a0_ground_truth TO {}").format(I(APP)),
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(I(SCHEMA), I(READER)),
            sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(I(SCHEMA), I(READER)),
            sql.SQL("ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} GRANT SELECT ON TABLES TO {}").format(I(APP), I(SCHEMA), I(READER)),
        ]
        for st in stmts:
            cur.execute(st)
    conn.commit()
    return made


def verify(conn) -> list[str]:
    """Read-only audit. Returns problems (empty = the roles have exactly the intended privileges)."""
    problems, cur = [], conn.cursor()
    for r in (APP, READER):
        if not role_exists(cur, r):
            problems.append(f"role {r} does not exist")
            continue
        cur.execute("SELECT rolsuper, rolcreatedb, rolcreaterole, rolreplication FROM pg_roles WHERE rolname = %s", [r])
        if any(cur.fetchone()):
            problems.append(f"{r} has an elevated attribute")
        # every table/view/matview/sequence outside the derived schema: which privileges does the role have?
        cur.execute("""SELECT n.nspname, c.relname, c.relkind,
                              has_table_privilege(%s, c.oid, 'INSERT') ins, has_table_privilege(%s, c.oid, 'UPDATE') upd,
                              has_table_privilege(%s, c.oid, 'DELETE') del, has_table_privilege(%s, c.oid, 'TRUNCATE') trn,
                              has_table_privilege(%s, c.oid, 'SELECT') sel
                       FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                       WHERE c.relkind IN ('r','p','v','m','f') AND n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')""",
                    [r] * 5)
        for ns, rel, kind, ins, upd, dele, trn, sel in cur.fetchall():
            if ns == SCHEMA:
                if r == READER and (ins or upd or dele or trn):
                    problems.append(f"{r} can write {ns}.{rel}")
                continue
            if ins or upd or dele or trn:
                problems.append(f"{r} can WRITE {ns}.{rel}")
            if sel and (ns, rel) not in SOURCE_TABLES:
                problems.append(f"{r} can read {ns}.{rel} (not needed)")
        cur.execute("SELECT nspname FROM pg_namespace WHERE has_schema_privilege(%s, oid, 'CREATE') AND nspname <> %s AND nspname NOT LIKE 'pg_%%'", [r, SCHEMA])
        for (ns,) in cur.fetchall():
            problems.append(f"{r} can CREATE objects in schema {ns}")
    conn.rollback()
    return problems


def write_env(path: Path, items: dict, header: str) -> None:
    path.write_text(header + "\n".join(f"{k}={v}" for k, v in items.items()) + "\n")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    conn, s = admin_conn()
    try:
        if a.verify:
            probs = verify(conn)
            print("privilege audit:", "OK - exactly the intended privileges" if not probs else "")
            for p in probs:
                print("  PROBLEM:", p)
            return 1 if probs else 0
        with conn.cursor() as cur:
            steps = plan(cur, s["dbname"])
        conn.rollback()
        print(f"target: database {s['dbname']} on {s['host']}:{s['port']} (admin user {s['user']}; credentials not stored)")
        for x in steps:
            print("  -", x)
        if not a.apply:
            print("\nDRY RUN - nothing changed. Re-run with --apply to execute.")
            return 0
        app_pw, reader_pw = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        made = apply(conn, s["dbname"], app_pw, reader_pw)
        print("\napplied. roles created:", {k: v for k, v in made.items()})
        probs = verify(conn)
        for p in probs:
            print("  PROBLEM:", p)
        if probs:
            return 1
        if not all(made.values()):
            print("note: a role already existed, so its password was not changed and deploy/*.env were NOT rewritten for it.")
        if all(made.values()):
            os_ = C.os_settings()
            common = {"NEWS_PG_HOST": s["host"], "NEWS_PG_PORT": s["port"], "NEWS_PG_DB": s["dbname"]}
            write_env(DEPLOY / "news.env", {**common, "NEWS_PG_USER": APP, "NEWS_PG_PASSWORD": app_pw,
                                             "NEWS_OS_URL": os_["url"], "NEWS_OS_USER": os_["user"], "NEWS_OS_PASSWORD": os_["password"],
                                             "NEWS_OS_VERIFY_CERTS": str(os_["verify"]).lower()},
                      "# Ingest service: PostgreSQL as news_engine_app (SELECT 2 source tables + own schema) and OpenSearch (read-only client).\n"
                      "# Generated by deploy/bootstrap_pg.py. Mode 600. NEVER commit.\n")
            write_env(DEPLOY / "dashboard.env", {"NEWS_DERIVED_PG_HOST": s["host"], "NEWS_DERIVED_PG_PORT": s["port"], "NEWS_DERIVED_PG_DB": s["dbname"],
                                                  "NEWS_DERIVED_PG_USER": READER, "NEWS_DERIVED_PG_PASSWORD": reader_pw},
                      "# Dashboard: PostgreSQL as news_engine_reader (SELECT on schema news_engine only). Generated by deploy/bootstrap_pg.py. Mode 600.\n")
            print("wrote deploy/news.env and deploy/dashboard.env (mode 600)")
        print("privilege audit: OK - exactly the intended privileges")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
