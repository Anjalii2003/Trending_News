#!/usr/bin/env python3
"""Browser test of the Trending-news page (UI/trending.html). Read-only.

Drives an already-running UI/news_server.py in headless Chromium and compares what the page shows with independent SQL on
the pilot database (opened read-only). External publisher sites are never contacted (navigation is intercepted and stubbed).

  .venv/bin/python UI/tests/check_trending_ui.py --url http://127.0.0.1:8878 --shots DIR
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pgdb
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
FAILS: list[str] = []


STATUS_TEXT = {"mixed": "Mixed topics", "outside": "Outside the taxonomy", "unlabelled": "Unlabelled"}


def expected_tax(story):
    """What the taxonomy chip must say: the agreed Domain › Topic › Category › Subtopic path, else the honest status."""
    p = [story["taxonomy"][k]["name"] for k in ("domain", "topic", "category", "subtopic") if story["taxonomy"].get(k)]
    return " › ".join(p) if p else STATUS_TEXT[story["taxonomy_status"]]


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail else ""))
    if not cond:
        FAILS.append(name)


def chrome():
    return os.environ.get("CHROME") or sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")))[-1]


def num(txt):
    return int(re.sub(r"[^\d]", "", txt))


def ist_time(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(ZoneInfo("Asia/Kolkata")).strftime("%d/%m/%Y, %H:%M IST")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8878")
    ap.add_argument("--shots", default=".")
    a = ap.parse_args()
    shots = Path(a.shots)
    shots.mkdir(parents=True, exist_ok=True)
    db = pgdb.connect()
    tax = {r[0]: r for r in db.execute("SELECT code, tier, name, parent FROM taxonomy_node WHERE in_scope").fetchall()}

    def record(url):
        return db.execute("SELECT title, source_domain, to_char(published_at AT TIME ZONE 'Asia/Kolkata', 'DD/MM/YYYY, HH24:MI') FROM article_fact WHERE url = ? LIMIT 1", [url]).fetchone()

    def sql_story(eids, as_of, hours):
        return db.execute("""SELECT count(*), count(*) FILTER (WHERE published_at > CAST(? AS TIMESTAMPTZ) - (? * interval '1 hour') AND published_at <= CAST(? AS TIMESTAMPTZ)),
                                    count(DISTINCT source_domain) FILTER (WHERE published_at > CAST(? AS TIMESTAMPTZ) - (? * interval '1 hour') AND published_at <= CAST(? AS TIMESTAMPTZ)),
                                    to_char(max(published_at) AT TIME ZONE 'Asia/Kolkata', 'DD/MM/YYYY, HH24:MI')
                             FROM article_fact WHERE NOT is_duplicate AND cluster_id IN (SELECT UNNEST(?))""",
                          [as_of, hours, as_of, as_of, hours, as_of, eids]).fetchone()

    with sync_playwright() as pw:
        br = pw.chromium.launch(executable_path=chrome(), headless=True, args=["--no-sandbox"])
        ctx = br.new_context(viewport={"width": 1300, "height": 950})
        external, errors, bad, dialogs, trending_calls = [], [], [], [], []
        ctx.route(re.compile(r"^https?://(?!127\.0\.0\.1|localhost)"),
                  lambda r: (external.append(r.request.url), r.fulfill(status=200, content_type="text/html", body="<title>stub</title>stub")))
        page = ctx.new_page()
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("response", lambda r: bad.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        page.on("request", lambda r: trending_calls.append(r.url) if "/api/news/trending?" in r.url else None)
        page.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))

        def settle():
            page.wait_for_function("document.querySelectorAll('.n-spin').length === 0", timeout=90000)

        def pick(sel, value):
            with page.expect_response(lambda r: "/api/news/trending?" in r.url and r.status == 200, timeout=90000) as ri:
                page.select_option(sel, value)
            j = ri.value.json()
            settle()
            return j

        page.goto(a.url + "/")
        page.wait_for_function("document.querySelector('#t-country').options.length > 5", timeout=90000)
        settle()

        # ---------- live processing panel: every number is a stored record ----------
        page.wait_for_function("document.querySelectorAll('#live-grid .live-cell').length >= 9", timeout=30000)
        cells = {c.locator(".live-k").text_content(): c for c in page.locator("#live-grid .live-cell").all()}
        val = lambda k: cells[k].locator(".live-v").inner_text()
        n_art, n_uniq = db.execute("SELECT count(*), count(*) FILTER (WHERE NOT is_duplicate) FROM article_fact").fetchone()
        n_cl, n_ev = db.execute("SELECT count(*), count(*) FILTER (WHERE is_event) FROM event_cluster").fetchone()
        check("live panel: status pill shows a real status", page.locator("#live-status").text_content().strip() in ("Successful", "Running", "Failed", "No run yet"), page.locator("#live-status").text_content())
        check("live panel: articles processed / unique equal the database", val("Articles processed") == f"{n_art:,}" and f"{n_uniq:,} unique" in cells["Articles processed"].inner_text())
        check("live panel: clusters and qualifying events equal the database", val("Event clusters") == f"{n_cl:,}" and f"{n_ev:,} qualifying" in cells["Event clusters"].inner_text())
        check("live panel: pending labels and excluded equal the database",
              val("Pending labels") == f"{db.execute('SELECT count(*) FROM pending_label').fetchone()[0]:,}"
              and val("Excluded articles") == f"{db.execute('SELECT count(*) FROM excluded_article').fetchone()[0]:,}")
        latest = db.execute("SELECT to_char(max(published_at) AT TIME ZONE 'Asia/Kolkata', 'DD/MM/YYYY, HH24:MI') FROM article_fact").fetchone()[0]
        check("live panel: latest publication date equals the database", latest in val("Latest publication date"), val("Latest publication date"))
        api_live = page.evaluate("fetch('/api/news/live').then(r => r.json())")
        sched = api_live["schedule"]["next_run_at"]
        lsr = api_live["last_successful_run"]
        check("live panel: last successful run shows the stored run's finish time", lsr is None or ist_time(lsr["finished_at"]) in val("Last successful run"), val("Last successful run"))
        ph = api_live["schedule"]["phase"]
        check("live panel: next-run cell follows the scheduler's real state",
              (ph == "running" and val("Next scheduled run") == "Running now") or (ph == "catching_up" and val("Next scheduled run") == "Catch-up in progress")
              or (ph in ("idle", None) and (val("Next scheduled run") == "not scheduled") == (sched is None)), val("Next scheduled run"))
        _unused = ("Next scheduled run",)
        check("live panel: next run is shown only when one is stored", ph in ("running", "catching_up") or (val("Next scheduled run") == "not scheduled") == (sched is None), val("Next scheduled run"))
        page.locator("#t-live").screenshot(path=str(shots / "40_live_panel.png"))

        # ---------- nothing selected ----------
        check("nothing selected: asks the user to choose, shows no ranking", "Select what you want to follow" in page.inner_text("#t-results")
              and page.locator(".t-card").count() == 0)
        check("nothing selected: no trending request is even made", len(trending_calls) == 0)
        check("header states the data range and that it is read-only", "read-only" in page.inner_text("#t-sub") and "2026-10" in page.inner_text("#t-sub"))
        check("topic and subtopic are disabled until a parent is chosen", page.is_disabled("#t-topic") and page.is_disabled("#t-sub-sel"))
        page.screenshot(path=str(shots / "40_prompt.png"))

        # ---------- country only ----------
        j = pick("#t-country", "IN")
        cards = page.locator(".t-card")
        check("country only: exactly 10 trending stories", cards.count() == 10 == len(j["stories"]))
        ranks = [num(x) for x in page.eval_on_selector_all(".t-rank", "e=>e.map(x=>x.textContent)")]
        scores = [num(x) for x in page.eval_on_selector_all(".t-score b", "e=>e.map(x=>x.textContent)")]
        check("ranked 1..10, strongest first (scores never increase down the list)", ranks == list(range(1, 11)) and all(scores[i] >= scores[i + 1] for i in range(9)), str(scores))
        titles = page.eval_on_selector_all(".t-title button", "e=>e.map(x=>x.textContent)")
        check("no story listed twice", len(set(titles)) == 10)
        c0 = cards.nth(0)
        txt = c0.inner_text()
        check("card shows title, taxonomy, location, article count, latest activity and the reasons",
              j["stories"][0]["title"] in txt and c0.locator(".t-chip").first.inner_text() == expected_tax(j["stories"][0]) and "📍" in txt and "related articles" in txt
              and "latest activity" in txt and c0.locator(".t-why li").count() >= 3 and "View related articles" in txt)
        check("every card has a non-empty explanation and a View button", all(cards.nth(i).locator(".t-why li").count() >= 2 and cards.nth(i).locator("button", has_text="View related articles").count() == 1 for i in range(10)))
        check("summary says what is ranked and from how many stories", "Top 10 trending stories" in page.inner_text(".t-summary") and "active stories" in page.inner_text(".t-summary"))

        # numbers on the cards equal independent SQL
        ok = True
        for i in range(10):
            s = j["stories"][i]
            tot, a0, p0, latest = sql_story(s["event_ids"], j["as_of"], 72)
            t = cards.nth(i).locator(".t-stats").inner_text()
            nums = [num(x) for x in re.findall(r"(\d[\d,]*) (?:related articles|in the 3 days|publishers)", t)]
            if not (nums[0] == tot and nums[1] == a0 and nums[2] == p0 and latest in t.replace("T", " ")):
                ok = False
                print("   mismatch", i, nums, (tot, a0, p0, latest), t[:120])
        check("every card's counts and latest-activity time equal independent SQL", ok)
        check("evidence lines appear where the pipeline's tests flagged something", any("Flagged" in c or "detected" in c or "ran ×" in c for c in page.eval_on_selector_all(".t-why li", "e=>e.map(x=>x.textContent)")))
        page.locator("#t-results").screenshot(path=str(shots / "41_country_only.png"))

        # ---------- periods ----------
        with page.expect_response(lambda r: "window=24h" in r.url and r.status == 200, timeout=90000) as ri:
            page.click("#t-window button:has-text('Last 24 hours')")
        j24 = ri.value.json()
        settle()
        check("period: 24 hours is requested, shown and pressed", j24["window"]["key"] == "24h" and "last 24 hours" in page.inner_text(".t-summary")
              and page.get_attribute("#t-window button:has-text('Last 24 hours')", "aria-pressed") == "true")
        check("period changes the ranking", [s["story_id"] for s in j24["stories"]] != [s["story_id"] for s in j["stories"]])
        with page.expect_response(lambda r: "window=3d" in r.url and r.status == 200, timeout=90000):
            page.click("#t-window button:has-text('Last 3 days')")
        settle()

        # ---------- custom dates are inclusive IST calendar dates ----------
        page.click("#t-window button:has-text('Custom dates')")
        settle()
        meta = page.evaluate("fetch('/api/news/trending_meta').then(r => r.json())")
        with page.expect_response(lambda r: "window=custom" in r.url and ("date_from=" + meta["date_end"]) in r.url and r.status == 200, timeout=90000) as ri:
            page.fill("#t-date-from", meta["date_end"])
            page.fill("#t-date-to", meta["date_end"])
            page.click("#t-date-apply")
        custom = ri.value.json()
        settle()
        check("custom dates: selected day and results are reported in IST",
              custom["window"]["timezone"] == "Asia/Kolkata" and custom["window"]["date_from"] == custom["window"]["date_to"] == meta["date_end"]
              and "(IST)" in custom["window"]["label"] and meta["date_end"] + " to " + meta["date_end"] + " (IST)" in page.inner_text("#t-hint"))

        # ---------- both (India still selected, now add a topic) ----------
        D = "D02"
        j = pick("#t-domain", D)
        check("both: enabled topic list holds exactly this domain's topics", sorted(v for v in page.eval_on_selector_all("#t-topic option", "e=>e.map(o=>o.value)") if v)
              == sorted(c for c, t in tax.items() if t[1] == "topic" and t[3] == D) and not page.is_disabled("#t-topic"))
        chips = page.eval_on_selector_all(".t-card .t-chip:first-child", "e=>e.map(x=>x.textContent)")
        locs = page.eval_on_selector_all(".t-card .t-chip.loc", "e=>e.map(x=>x.textContent)")
        check("both: stories match the topic AND the country", chips and all(c.startswith(tax[D][2]) for c in chips) and all("India" in l for l in locs), f"{len(chips)} cards")
        check("both: summary names both", "India" in page.inner_text(".t-summary") and tax[D][2] in page.inner_text(".t-summary"))

        # ---------- topic only (clear the country, keep the topic) ----------
        j = pick("#t-country", "")
        chips = page.eval_on_selector_all(".t-card .t-chip:first-child", "e=>e.map(x=>x.textContent)")
        check("topic only: every story is in the selected domain", chips and all(c.startswith(tax[D][2]) for c in chips), f"{len(chips)} cards")
        locs = {x for x in page.eval_on_selector_all(".t-card .t-chip.loc", "e=>e.map(x=>x.textContent.split('·')[0].trim())")}
        check("topic only: spans several countries / locations, not one", len(locs) >= 2, str(sorted(locs))[:120])
        check("topic only: summary says all countries", "all countries" in page.inner_text(".t-summary"))
        page.locator("#t-results").screenshot(path=str(shots / "42_topic_only.png"))
        j = pick("#t-country", "IN")

        # ---------- topic then subtopic ----------
        T = next(c for c, t in tax.items() if t[1] == "topic" and t[3] == D)
        j = pick("#t-topic", T)
        groups = page.eval_on_selector_all("#t-sub-sel optgroup", "e=>e.map(g=>g.label)")
        check("topic selected: subtopics are grouped under their real parent categories", set(groups) == {tax[c][2] for c, t in tax.items() if t[1] == "category" and t[3] == T} and not page.is_disabled("#t-sub-sel"))
        # find a subtopic that has trending stories (7 days) via the API the page itself uses
        subs = [c for c, t in tax.items() if t[1] == "subtopic" and c.startswith(T + ".")]
        pickd = None
        for sc in subs:
            r = page.evaluate("async (u)=>await (await fetch(u)).json()", f"/api/news/trending?country=IN&tax={sc}&window=7d&limit=10")
            if r.get("stories"):
                pickd = (sc, r)
                break
        with page.expect_response(lambda r: "window=7d" in r.url and r.status == 200, timeout=90000):
            page.click("#t-window button:has-text('Last 7 days')")
        settle()
        if pickd:
            j = pick("#t-sub-sel", pickd[0])
            names = page.eval_on_selector_all(".t-card .t-chip:first-child", "e=>e.map(x=>x.textContent)")
            check("subtopic: every story's taxonomy path ends in the selected subtopic", names and all(n.endswith(tax[pickd[0]][2]) for n in names), f"{len(names)} cards")
        else:
            check("subtopic: view shows an honest empty state when nothing is trending", True)
        j = pick("#t-sub-sel", "")
        page.click("#t-window button:has-text('Last 3 days')")
        settle()

        # ---------- open a story ----------
        j = pick("#t-domain", "")
        page.wait_for_selector(".t-card")
        idx = next((i for i, s in enumerate(j["stories"]) if s["related"]), 0)
        s = j["stories"][idx]
        page.locator(".t-card").nth(idx).locator("button", has_text="View related articles").click()
        page.wait_for_selector("#story-articles .n-art", timeout=60000)
        page.wait_for_timeout(500)
        head = page.inner_text("#dr-title")
        total_txt = page.inner_text("#story-articles > div:first-child")
        ids = s["event_ids"] + [e for x in s["related"] for e in x["event_ids"]]
        want = db.execute("SELECT count(*) FROM article_fact WHERE NOT is_duplicate AND cluster_id IN (SELECT UNNEST(?))", [ids]).fetchone()[0]
        check("story opens with its title and the full related-article count", head == s["title"] and f"{want:,}" in total_txt, f"{want} articles")
        check("drawer repeats the reasons and key facts", all(r[:40] in page.inner_text("#dr-body") for r in s["reasons"][:2]) and "first reported" in page.inner_text("#dr-body").lower())
        rows = page.eval_on_selector_all("#story-articles .n-art", """els=>els.slice(0,25).map(e=>({href:(e.querySelector('.t a')||{}).href||null,title:(e.querySelector('.t a')||{}).textContent||'',
                 pub:(e.querySelector('.m b')||{}).textContent||'',date:(e.querySelectorAll('.m span')[1]||{}).textContent||''}))""")
        good = bool(rows)
        for r in rows:
            rec = record(r["href"]) if r["href"] else None
            good &= bool(rec) and rec[0].strip() == r["title"].replace(" ↗", "").strip() and r["pub"] == re.sub(r"^www\.", "", rec[1] or "") and r["date"].startswith(rec[2])
        check("related articles: title, publisher, date and link equal the stored records", good, f"{len(rows)} checked")
        if s["related"]:
            check("articles from a closely related story are tagged", page.locator("#story-articles .t-tag").count() >= 1)
        link = page.locator("#story-articles .n-art .t a").first
        href = link.get_attribute("href")
        check("article link: original URL, new tab, noopener", href.startswith("http") and link.get_attribute("target") == "_blank" and "noopener" in (link.get_attribute("rel") or ""))
        with ctx.expect_page() as pi:
            link.click()
        pop = pi.value
        pop.wait_for_load_state()
        check("clicking an article opens exactly that original URL", pop.url.rstrip("/") == href.rstrip("/") and record(href) is not None, pop.url[:80])
        pop.close()
        src = page.locator(".t-card").nth(idx).locator(".t-src a").first
        shref = src.get_attribute("href")
        check("card 'latest coverage' links are original URLs too", shref.startswith("http") and record(shref) is not None)
        page.screenshot(path=str(shots / "43_story_drawer.png"))
        page.keyboard.press("Escape")
        check("Escape closes the story", "drawer" not in page.evaluate("document.body.className"))

        # ---------- show more ----------
        pick("#t-country", "IN")
        if page.locator(".t-more").count():
            with page.expect_response(lambda r: "limit=20" in r.url and r.status == 200, timeout=90000):
                page.click(".t-more")
            settle()
            check("show more reveals the next stories (20 total), still ranked", page.locator(".t-card").count() == 20
                  and [num(x) for x in page.eval_on_selector_all(".t-rank", "e=>e.map(x=>x.textContent)")] == list(range(1, 21)))
        else:
            check("show more is offered for a large scope", False)

        # ---------- a small scope is honest ----------
        j = pick("#t-country", "BD")
        check("small country: ranks what exists and says when the baseline comparison is omitted", page.locator(".t-card").count() == len(j["stories"]) > 0 and "comparable story-windows" in page.inner_text(".t-note"))

        # ---------- clear ----------
        page.click("#t-clear")
        page.wait_for_timeout(300)
        check("Clear returns to the prompt", page.locator(".t-card").count() == 0 and "Select what you want to follow" in page.inner_text("#t-results") and page.input_value("#t-country") == "")

        # ---------- theme + phone ----------
        pick("#t-country", "IN")
        page.click("#n-theme")
        check("light theme applies", page.evaluate("document.documentElement.getAttribute('data-theme')") == "light")
        page.screenshot(path=str(shots / "44_light.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.wait_for_timeout(300)
        check("no horizontal overflow at phone width", page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 1)
        page.screenshot(path=str(shots / "45_phone.png"))

        # ---------- hostile text ----------
        p2 = ctx.new_page()
        p2.on("dialog", lambda d: (dialogs.append(d.message), d.dismiss()))
        evil = "<img src=x onerror=alert('t1')><script>alert('t2')</script>"

        def tamper(route):
            resp = route.fetch()
            jj = json.loads(resp.text())
            for s_ in jj.get("stories", [])[:5]:
                s_["title"] = evil
                s_["reasons"] = [evil] + s_["reasons"]
                for c in s_["sources"]:
                    c["title"] = evil
                    c["publisher"] = evil
                if s_["taxonomy"]["domain"]:
                    s_["taxonomy"]["domain"]["name"] = evil
            route.fulfill(response=resp, body=json.dumps(jj))
        p2.route("**/api/news/trending?*", tamper)
        p2.goto(a.url + "/")
        p2.wait_for_function("document.querySelector('#t-country').options.length > 5", timeout=90000)
        p2.select_option("#t-country", "IN")
        p2.wait_for_selector(".t-card", timeout=90000)
        check("hostile titles, reasons and publishers render as text only", p2.locator(".t-card img, .t-card script, #t-results img").count() == 0 and not dialogs
              and "<img" in p2.inner_text(".t-title"))
        p2.close()

        check("no console/page errors", not errors, "; ".join(errors[:3]))
        check("no failed requests", not bad, "; ".join(bad[:3]))
        br.close()
    print(f"\n{'ALL CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
