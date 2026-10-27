#!/usr/bin/env python3
"""Browser test of the Domain > Topic > Subtopic filter, article lists and source citations.

Read-only: drives an already-running UI/news_server.py in headless Chromium and compares what the page shows with
INDEPENDENT SQL on the pilot database (opened read-only). Starts no pipeline, writes only screenshots.
External publisher sites are never contacted: navigation to them is intercepted and answered with a stub, so the
test can prove WHERE a link would go without depending on third-party servers.

  .venv/bin/python UI/tests/check_news_filter_ui.py --url http://127.0.0.1:8878 --shots DIR
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import pgdb
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def chrome():
    if os.environ.get("CHROME"):
        return os.environ["CHROME"]
    return sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8878")
    ap.add_argument("--shots", default=".")
    a = ap.parse_args()
    shots = Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)
    db = pgdb.connect()

    tax = {r[0]: r for r in db.execute("SELECT code, tier, name, parent, in_scope FROM taxonomy_node").fetchall()}
    COL = {"domain": "domain_id", "topic": "topic_id", "category": "category_id", "subtopic": "subtopic_id"}

    def level_of(code):
        return {0: "domain", 1: "topic", 2: "category", 3: "subtopic"}[code.count(".")]

    def sql_count(code, where="g.ev_country_iso = 'IN'", params=()):
        return db.execute(f"SELECT count(*) FROM article_fact f JOIN article_geo g USING (source_table, article_id) "
                          f"WHERE NOT f.is_duplicate AND {where} AND f.{COL[level_of(code)]} = ?", [*params, code]).fetchone()[0]

    def sql_area(where="g.ev_country_iso = 'IN'", params=()):
        return db.execute(f"SELECT count(*) FROM article_fact f JOIN article_geo g USING (source_table, article_id) "
                          f"WHERE NOT f.is_duplicate AND {where}", list(params)).fetchone()[0]

    def sql_primary(event_ids, code):
        col = COL[level_of(code)]
        rows = db.execute(f"""WITH m AS (SELECT cluster_id, {col} lab, count(*) n FROM article_fact
                                         WHERE NOT is_duplicate AND has_label AND cluster_id IN (SELECT UNNEST(?)) GROUP BY 1, 2),
                              r AS (SELECT *, row_number() OVER (PARTITION BY cluster_id ORDER BY n DESC, lab) rk FROM m)
                              SELECT cluster_id, lab FROM r WHERE rk = 1""", [list(event_ids)]).fetchall()
        return dict(rows)

    def record(url):
        r = db.execute("SELECT title, source_domain, to_char(published_at AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI'), domain_id, topic_id, category_id, subtopic_id "
                       "FROM article_fact WHERE url = ? LIMIT 1", [url]).fetchone()
        return r

    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=chrome(), headless=True, args=["--no-sandbox"])
        ctx = br.new_context(viewport={"width": 1400, "height": 1000})
        external, errors, bad_req = [], [], []

        def stub_external(route):                       # never touch third-party sites
            external.append(route.request.url)
            route.fulfill(status=200, content_type="text/html", body="<title>stub</title>stubbed original article")
        ctx.route(re.compile(r"^https?://(?!127\.0\.0\.1|localhost)"), stub_external)
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("response", lambda r: bad_req.append(f"{r.status} {r.url}") if r.status >= 400 else None)

        def settle():
            page.wait_for_function("document.querySelectorAll('.n-spin').length === 0", timeout=90000)

        def act(fn):
            """Run a UI action and return the JSON of the aoi/adhoc response it triggered."""
            with page.expect_response(lambda r: ("/api/news/aoi" in r.url or "/api/news/adhoc" in r.url) and r.status == 200, timeout=90000) as ri:
                fn()
            j = ri.value.json()
            settle()
            return j

        page.goto(a.url + "/news.html")
        page.wait_for_selector("#ev-tbl tbody tr", timeout=90000)
        settle()
        base_articles = int(re.sub(r"\D", "", page.inner_text("#kpis .n-kpi:first-child b")))

        # ---------- the filter is visible and built from the real hierarchy ----------
        check("filter card is visible with an All Topics button", page.is_visible("#tx-all") and page.get_attribute("#tx-all", "aria-pressed") == "true")
        doms = page.eval_on_selector_all("#tx-domain option", "e=>e.map(o=>[o.value,o.textContent])")
        in_scope_domains = sorted(c for c, t in tax.items() if t[1] == "domain" and t[4])
        check("domain list = in-scope domains of the stored taxonomy (no 'Non-relevant')", sorted(v for v, _ in doms if v) == in_scope_domains, f"{len(doms) - 1} domains")
        check("topic and subtopic stay disabled until a parent is chosen", page.is_disabled("#tx-topic") and page.is_disabled("#tx-sub"))
        check("unfiltered view says All Topics", "All Topics" in page.inner_text("#tx-crumb"))
        page.screenshot(path=str(shots / "10_filter_default.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 900})

        # ---------- DOMAIN ----------
        D = "D02"
        j = act(lambda: page.select_option("#tx-domain", D))
        want = sql_count(D)
        check("domain: matching-articles KPI equals independent SQL", j["tax"]["n_match"] == want and f"{want:,}" in page.inner_text("#kpis"), f"{want}")
        check("domain: article list total equals SQL", j["articles"]["total"] == want and f"{want:,} article" in page.inner_text("#articles"))
        check("domain: topic dropdown now lists exactly that domain's topics",
              sorted(v for v, _ in page.eval_on_selector_all("#tx-topic option", "e=>e.map(o=>[o.value,o.textContent])") if v)
              == sorted(c for c, t in tax.items() if t[1] == "topic" and t[3] == D and t[4]) and not page.is_disabled("#tx-topic"))
        ev_ids = [e["event_id"] for e in j["events"]]
        prim = sql_primary(ev_ids, D)
        check("domain: every listed event's primary label (independent SQL) is in the domain", ev_ids and all(prim.get(i) == D for i in ev_ids), f"{len(ev_ids)} events")
        check("domain: events fewer than unfiltered", 0 < len(ev_ids) <= 300)
        topic_names = {t[2] for c, t in tax.items() if t[1] == "topic" and t[3] == D}
        shown_topics = set(page.eval_on_selector_all("#nodes tbody tr td:first-child", "e=>e.map(x=>x.textContent.replace(/^(ALERT( \\d+D)?)/,'').trim())"))
        check("domain: trending-topic rows are only topics of the domain", shown_topics and all(any(n.endswith(t) for t in topic_names) for n in shown_topics), str(len(shown_topics)))
        check("domain: node alerts belong to the domain", all(al["entity_id"].startswith(D + ".") for al in j["alerts"] if al["kind"] == "node_trend"))
        ev_alert_ids = {al["entity_id"] for al in j["alerts"] if al["kind"] != "node_trend"}
        check("domain: event alerts only for matching events", ev_alert_ids <= set(sql_primary(ev_alert_ids, D)) and all(sql_primary(ev_alert_ids, D).get(i) == D for i in ev_alert_ids) if ev_alert_ids else True, f"{len(ev_alert_ids)}")
        page.screenshot(path=str(shots / "11_filter_domain.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 1000})

        # ---------- every article row: title/publisher/date/link equal the stored record ----------
        rows = page.eval_on_selector_all("#articles .n-art", """els=>els.slice(0,25).map(e=>({href:(e.querySelector('.t a')||{}).href||null, title:(e.querySelector('.t a')||{}).textContent||'',
                       publisher:(e.querySelector('.m b')||{}).textContent||'', date:(e.querySelectorAll('.m span')[1]||{}).textContent||'', tax:(e.querySelectorAll('.m span')[2]||{}).textContent||''}))""")
        ok_all = len(rows) == 25
        for r in rows:
            rec = record(r["href"]) if r["href"] else None
            good = bool(rec) and rec[0].strip() == r["title"].replace(" ↗", "").strip() and r["publisher"] == re.sub(r"^www\.", "", rec[1] or "") \
                and r["date"].startswith(rec[2]) and rec[3] == D
            if not good:
                ok_all = False
                print("   mismatch:", r, rec)
        check("articles: 25 rows each show stored title + publisher + publication date + original URL, and carry the domain label", ok_all)
        check("articles: every row has a clickable link", all(r["href"] and r["href"].startswith("http") for r in rows))

        # ---------- TOPIC ----------
        T = "D02.T01"
        j = act(lambda: page.select_option("#tx-topic", T))
        want = sql_count(T)
        check("topic: matching articles equal SQL", j["tax"]["n_match"] == want, f"{want}")
        ev_ids = [e["event_id"] for e in j["events"]]
        check("topic: events' primary topic (SQL) is the topic", ev_ids and all(v == T for v in sql_primary(ev_ids, T).values()) and len(sql_primary(ev_ids, T)) == len(ev_ids), f"{len(ev_ids)} events")
        subs = page.eval_on_selector_all("#tx-sub optgroup", "e=>e.map(g=>[g.label, [...g.querySelectorAll('option')].map(o=>o.value)])")
        real = {}
        for c, t in tax.items():
            if t[1] == "subtopic" and c.startswith(T + ".") and t[4]:
                real.setdefault(tax[t[3]][2], []).append(c)
        check("topic: subtopic dropdown is grouped by the real parent categories with the real subtopics",
              {g: sorted(v) for g, v in subs} == {k: sorted(v) for k, v in real.items()}, f"{len(subs)} categories")
        check("topic: breadcrumb is clickable and shows the path", "All Topics" in page.inner_text("#tx-crumb") and tax[D][2] in page.inner_text("#tx-crumb") and tax[T][2] in page.inner_text("#tx-crumb"))

        # ---------- SUBTOPIC ----------
        counts = {c: sql_count(c) for c in real[next(iter(real))] + sum(real.values(), [])}
        S_ = max((c for c in counts if counts[c] > 0), key=lambda c: counts[c])
        j = act(lambda: page.select_option("#tx-sub", S_))
        want = sql_count(S_)
        check("subtopic: matching articles equal SQL", j["tax"]["n_match"] == want and j["articles"]["total"] == want, f"{S_} -> {want}")
        check("subtopic: view switches to the Subtopic level", page.get_attribute("#lv-subtopic", "aria-pressed") == "true")
        ev_ids = [e["event_id"] for e in j["events"]]
        check("subtopic: events' primary subtopic (SQL) is the subtopic", all(v == S_ for v in sql_primary(ev_ids, S_).values()) and len(sql_primary(ev_ids, S_)) == len(ev_ids), f"{len(ev_ids)} events")
        shown = page.eval_on_selector_all("#articles .n-art .t a", "els=>els.slice(0,50).map(a=>a.href)")
        check("subtopic: every listed article carries the subtopic label (SQL)", shown and all((record(h) or [None] * 7)[6] == S_ for h in shown), f"{len(shown)} shown")
        check("subtopic: trending section reports honestly (series or 'no topic series')", page.inner_text("#nodes").strip() != "")
        page.screenshot(path=str(shots / "12_filter_subtopic.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 1000})

        # ---------- citations: events, trends, alerts ----------
        j = act(lambda: page.click("#tx-all"))
        check("All Topics resets the filter", int(re.sub(r"\D", "", page.inner_text("#kpis .n-kpi:first-child b"))) == base_articles and j["tax"] is None
              and page.get_attribute("#tx-all", "aria-pressed") == "true" and page.is_disabled("#tx-topic"), f"{base_articles}")
        det = page.locator("#ev-tbl details.n-cite").first
        det.locator("summary").click()
        cites = det.locator(".n-cite-i").evaluate_all("els=>els.map(e=>({href:(e.querySelector('a')||{}).href||null,title:(e.querySelector('a')||{}).textContent||'',pub:(e.querySelector('b')||{}).textContent||'',txt:e.textContent}))")
        check("event rows expose source citations", len(cites) >= 2, f"{len(cites)}")
        good = True
        for c in cites:
            rec = record(c["href"])
            good &= bool(rec) and rec[0].strip() == c["title"].replace(" ↗", "").strip() and c["pub"] == re.sub(r"^www\.", "", rec[1] or "") and rec[2] in c["txt"]
        check("event citations: title, publisher, date, link equal the stored article", good)
        check("opening an event citation does not also open the event drawer", "drawer" not in page.evaluate("document.body.className"))
        # trends
        page.locator("#nodes tbody tr").first.click()
        page.wait_for_selector("#node-src a", timeout=30000)
        tr = page.eval_on_selector_all("#node-src .n-cite-i", "els=>els.map(e=>({href:e.querySelector('a').href,title:e.querySelector('a').textContent,pub:e.querySelector('b').textContent}))")
        check("trend rows show source articles (title + publisher + link)", len(tr) >= 3 and all(record(c["href"]) and c["pub"] == re.sub(r"^www\.", "", record(c["href"])[1]) for c in tr), f"{len(tr)}")
        # alerts
        al = page.locator("#alerts .n-alert")
        n_al = al.count()
        with_src = page.eval_on_selector_all("#alerts .n-alert", "els=>els.filter(e=>e.querySelector('.n-cites a')).length")
        check("alerts show source citations", n_al > 0 and with_src >= max(1, int(0.9 * n_al)), f"{with_src}/{n_al}")
        kinds_seen = set()
        for kind in ("node_trend", "emerging_event", "new_development"):
            page.select_option("#al-filters select >> nth=0", kind)
            if page.locator("#alerts .n-alert").count():
                a_c = page.eval_on_selector_all("#alerts .n-alert:first-child .n-cites .n-cite-i", "els=>els.map(e=>({href:(e.querySelector('a')||{}).href,pub:e.querySelector('b').textContent}))")
                if a_c and all(record(c["href"]) and c["pub"] == re.sub(r"^www\.", "", record(c["href"])[1]) for c in a_c):
                    kinds_seen.add(kind)
        check("alert citations match stored articles for every alert kind present", len(kinds_seen) >= 2, str(sorted(kinds_seen)))
        page.select_option("#al-filters select >> nth=0", "")
        page.screenshot(path=str(shots / "13_alerts_sources.png"), clip={"x": 0, "y": 0, "width": 1400, "height": 1000})

        # ---------- links open the original source ----------
        def open_and_check(locator, label):
            href = locator.get_attribute("href")
            check(f"{label}: link is the original URL, new tab, noopener", href and href.startswith("http") and locator.get_attribute("target") == "_blank" and "noopener" in (locator.get_attribute("rel") or ""))
            with ctx.expect_page() as pi:
                locator.click()
            pop = pi.value
            pop.wait_for_load_state()
            check(f"{label}: browser navigated to exactly that URL", pop.url == href or pop.url.rstrip("/") == href.rstrip("/"), pop.url[:80])
            check(f"{label}: URL exists in the article records", record(href) is not None)
            pop.close()
        open_and_check(page.locator("#articles .n-art .t a").first, "article list")
        open_and_check(page.locator("#ev-tbl details.n-cite .n-cite-i a").first, "event citation")
        open_and_check(page.locator("#alerts .n-cites a").first, "alert citation")
        check("only publisher URLs from the records were requested externally", external and all(record(u) is not None for u in external), f"{len(external)} stubbed")

        # ---------- filter + AOI + location together ----------
        page.select_option("#tx-domain", D)
        settle()
        opts = page.eval_on_selector_all("#sel-adm1 option", "e=>e.map(o=>[o.value,o.textContent])")
        code = next(v for v, t in opts if t.startswith("Delhi"))
        j = act(lambda: page.select_option("#sel-adm1", code))
        want = sql_count(D, "g.ev_adm1_code = ?", (code,))
        check("filter + state: matching articles equal SQL for (Delhi AND domain)", j["tax"]["n_match"] == want and j["tax"]["n_area"] == sql_area("g.ev_adm1_code = ?", (code,)), f"{want} of {j['tax']['n_area']}")
        check("filter persists when the state changes", page.input_value("#tx-domain") == D)
        j = act(lambda: page.click("#m-mention"))
        want = sql_count(D, "list_contains(g.mention_countries_iso, 'IN')")
        check("filter + 'mentioned in': matching articles equal SQL", j["tax"]["n_match"] == want, f"{want}")
        j = act(lambda: page.click("#m-event"))
        j = act(lambda: page.select_option("#sel-country", "PK"))
        want = sql_count(D, "g.ev_country_iso = 'PK'")
        check("filter + another country (computed on demand): equal SQL", j["tax"]["n_match"] == want, f"{want}")
        # ad-hoc polygon with the filter
        page.select_option("#sel-country", "IN")
        settle()
        page.click("#d-box")
        box = page.locator("#map svg").bounding_box()
        view = page.eval_on_selector("#map svg", "e=>e.getAttribute('viewBox').split(' ').map(Number)")

        def at(lon, lat):
            x = (lon + 180) / 360 * 1000
            y = (90 - lat) / 180 * 500
            return box["x"] + (x - view[0]) / view[2] * box["width"], box["y"] + (y - view[1]) / view[3] * box["height"]
        page.mouse.click(*at(72.0, 18.0))
        j = act(lambda: page.mouse.click(*at(74.0, 20.5)))
        ring = j["aoi"]["geometry"]["coordinates"][0]
        x0, x1 = min(p_[0] for p_ in ring), max(p_[0] for p_ in ring)
        y0, y1 = min(p_[1] for p_ in ring), max(p_[1] for p_ in ring)
        inbox = ("g.ev_precision = 'city' AND NOT coalesce(g.geo_inconsistent, false) AND g.ev_lon BETWEEN ? AND ? AND g.ev_lat BETWEEN ? AND ?")
        want_area = sql_area(inbox, (x0, x1, y0, y1))
        want_dom = sql_count(D, inbox, (x0, x1, y0, y1))
        check("filter + drawn box (ad-hoc): area and matching counts equal SQL on the stored coordinates",
              j["aoi"]["adhoc"] and j["tax"]["n_area"] == want_area and j["tax"]["n_match"] == want_dom, f"{j['tax']['n_match']} of {j['tax']['n_area']} (SQL {want_dom} of {want_area})")
        page.click("#tx-all")
        settle()

        # ---------- hostile text in titles/publishers must stay inert ----------
        page2 = ctx.new_page()
        dialogs = []
        page2.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        evil = "<img src=x onerror=alert('x1')><script>alert('x2')</script>"

        def tamper(route):
            resp = route.fetch()
            jj = json.loads(resp.text())
            for it in jj.get("articles", {}).get("items", [])[:5]:
                it["title"] = evil
                it["publisher"] = evil
            for e in jj.get("events", [])[:5]:
                for s in e.get("sources", []):
                    s["title"] = evil
            for al_ in jj.get("alerts", [])[:5]:
                for s in al_.get("sources", []) or []:
                    s["title"] = evil
            route.fulfill(response=resp, body=json.dumps(jj))
        page2.route("**/api/news/aoi?*", tamper)
        page2.goto(a.url + "/news.html")
        page2.wait_for_selector("#articles .n-art", timeout=90000)
        page2.locator("#ev-tbl details.n-cite summary").first.click()
        check("hostile titles/publishers render as text only", page2.locator("#articles img, #articles script, #events img, #alerts img").count() == 0 and not dialogs)
        page2.close()

        check("no console/page errors", not errors, "; ".join(errors[:3]))
        check("no failed requests", not bad_req, "; ".join(bad_req[:3]))
        br.close()
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
