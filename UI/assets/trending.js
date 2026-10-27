/* Trending news page. Vanilla JS. All corpus text is inserted with textContent (never innerHTML); links are limited to http(s). */
(function () {
  'use strict';
  var S = { meta: null, methodology: null, country: '', tax: '', win: '3d', dateFrom: '', dateTo: '', limit: 10, req: 0, res: null, by: {}, kids: {}, cnames: {} };

  // ---------- helpers ----------
  function $(id) { return document.getElementById(id); }
  function add(e, k) {
    if (k == null || k === false) return;
    if (Array.isArray(k)) { k.forEach(function (x) { add(e, x); }); return; }
    e.appendChild(typeof k === 'object' ? k : document.createTextNode(String(k)));
  }
  function h(tag, attrs) {
    var e = document.createElement(tag), a, i;
    if (attrs) for (a in attrs) {
      if (attrs[a] == null || attrs[a] === false) continue;
      if (a === 'class') e.className = attrs[a];
      else if (a.slice(0, 2) === 'on') e.addEventListener(a.slice(2), attrs[a]);
      else if (a === 'href') { if (/^https?:\/\//i.test(attrs[a])) { e.setAttribute('href', attrs[a]); e.setAttribute('target', '_blank'); e.setAttribute('rel', 'noopener noreferrer'); } }
      else e.setAttribute(a, attrs[a] === true ? '' : attrs[a]);
    }
    for (i = 2; i < arguments.length; i++) add(e, arguments[i]);
    return e;
  }
  function clear(e) { while (e.firstChild) e.removeChild(e.firstChild); return e; }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function when(s) {
    return s ? new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', year: 'numeric', month: '2-digit', day: '2-digit',
      hour: '2-digit', minute: '2-digit', hourCycle: 'h23' }).format(new Date(s)) + ' IST' : '';
  }
  function api(path) {
    return fetch(path).then(function (r) { return r.json().then(function (j) { if (!r.ok) throw new Error(j && j.error ? j.error : 'HTTP ' + r.status); return j; }); });
  }
  function badge(t, cls, title) { return h('span', { class: 'n-badge ' + (cls || ''), title: title || null }, t); }
  function errBox(host, m) { clear(host).appendChild(h('div', { class: 'n-err', role: 'alert' }, 'Could not load: ' + m)); }

  function initTheme() {
    var t = 'dark';
    try { t = localStorage.getItem('news-theme') || 'dark'; } catch (e) { /* storage may be unavailable */ }
    document.documentElement.setAttribute('data-theme', t);
    $('n-theme').addEventListener('click', function () {
      t = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', t);
      try { localStorage.setItem('news-theme', t); } catch (e) { /* ignore */ }
    });
  }


  // ---------- live processing summary (every value comes from /api/news/live = stored records) ----------
  function fmtN(v) { return isNum(v) ? v.toLocaleString() : '—'; }
  function ago(iso) {
    if (!iso) return '';
    var s = (Date.now() - Date.parse(iso)) / 1000;
    if (!isFinite(s)) return '';
    var a = Math.abs(s), t = a < 90 ? Math.round(a) + ' s' : a < 5400 ? Math.round(a / 60) + ' min' : a < 129600 ? (a / 3600).toFixed(1) + ' h' : Math.round(a / 86400) + ' d';
    return s >= 0 ? t + ' ago' : 'in ' + t;
  }
  function liveCell(label, value, sub, title) {
    return h('div', { class: 'live-cell', title: title || null }, [h('span', { class: 'live-k' }, label), h('b', { class: 'live-v' }, value), sub ? h('small', { class: 'live-s' }, sub) : null]);
  }
  function renderLive(j) {
    var pill = $('live-status'), grid = clear($('live-grid')), note = $('live-note');
    var st = j.status || 'never_run', sch = j.schedule || {}, lr = j.latest_run, A = j.articles || {}, C = j.clusters || {}, E = j.errors || {}, X = j.excluded || {};
    pill.className = 'live-pill ' + st;
    pill.textContent = st === 'successful' ? 'Successful' : st === 'running' ? 'Running' : st === 'failed' ? 'Failed' : 'No run yet';
    $('live-asof').textContent = 'updated ' + when(j.generated_at) + (j.source ? ' · numbers: ' + j.source : '') + (j.last_successful_update ? ' · stories as of the last successful run ' + when(j.last_successful_update) : '');
    var reasons = Object.keys(X.by_reason || {}).map(function (k) { return k + ': ' + X.by_reason[k]; }).join('\n');
    var lastErr = (E.recent && E.recent[0]) ? E.recent[0] : null;
    var ls = j.last_successful_run;
    grid.appendChild(liveCell('Last successful run', j.last_successful_update ? when(j.last_successful_update) : '—',
      ls ? ago(j.last_successful_update) + ' · ' + (ls.mode || '') + (isNum(ls.articles_ingested) ? ' · ' + fmtN(ls.articles_ingested) + ' articles' : '') + (ls.duration_seconds != null ? ' · ' + Math.round(ls.duration_seconds / 60 * 10) / 10 + ' min' : '') : 'none yet',
      ls ? 'Run ' + ls.run_id : null));
    grid.appendChild(liveCell('Latest publication date', j.latest_publication_date ? when(j.latest_publication_date) : '—', ago(j.latest_publication_date), 'Newest published_at among the processed articles'));
    grid.appendChild(liveCell('Articles processed', fmtN(A.processed_total), fmtN(A.unique_total) + ' unique · ' + fmtN(A.duplicates_total) + ' duplicates'));
    grid.appendChild(liveCell('Latest run', isNum(A.processed_in_latest_run) ? fmtN(A.processed_in_latest_run) + ' articles' : '—', lr ? (lr.mode || '') + (lr.duration_seconds != null ? ' · ' + Math.round(lr.duration_seconds) + ' s' : '') + (lr.status === 'running' ? ' · in progress' : '') : 'no run recorded'));
    grid.appendChild(liveCell('Event clusters', fmtN(C.total), fmtN(C.qualifying_events) + ' qualifying events (≥3 articles, ≥2 publishers)'));
    grid.appendChild(liveCell('Pending labels', fmtN(j.pending_labels), fmtN(j.unlabelled_unique_articles) + ' unique articles unlabelled', 'Articles waiting for the classifier; they are attached when the label appears'));
    grid.appendChild(liveCell('Excluded articles', fmtN(X.total), 'no usable publication date', reasons || null));
    grid.appendChild(liveCell('Errors (24 h)', fmtN(E.failed_runs_24h), lastErr ? (lastErr.message || lastErr.status || '').slice(0, 70) : 'none recorded', lastErr ? (lastErr.message || '') : null));
    var every = sch.interval_minutes ? ' · then every ' + (sch.interval_minutes % 60 === 0 && sch.interval_minutes >= 60 ? sch.interval_minutes / 60 + ' h' : sch.interval_minutes + ' min') : '';
    if (sch.phase === 'running') grid.appendChild(liveCell('Next scheduled run', 'Running now', lr && lr.started_at ? 'started ' + ago(lr.started_at) + every : every.slice(3)));
    else if (sch.phase === 'catching_up') grid.appendChild(liveCell('Next scheduled run', 'Catch-up in progress', sch.catching_up_days + ' day(s) left' + every));
    else grid.appendChild(liveCell('Next scheduled run', sch.next_run_at ? when(sch.next_run_at) : 'not scheduled',
      sch.next_run_at ? ago(sch.next_run_at) + (sch.interval_minutes ? ' · every ' + (sch.interval_minutes % 60 === 0 && sch.interval_minutes >= 60 ? sch.interval_minutes / 60 + ' h' : sch.interval_minutes + ' min') : '') : ''));
    var msgs = [];
    if (j.note) msgs.push(j.note);
    if (sch.scheduler_alive === false) msgs.push('The scheduler has not reported for a while; data may be stale.');
    if (sch.catching_up_days > 0) msgs.push('Catching up: ' + sch.catching_up_days + ' complete day(s) still to process before live updates begin.');
    if (st === 'failed' && lr && lr.error) msgs.push('Latest run failed: ' + lr.error);
    note.hidden = !msgs.length; note.textContent = msgs.join(' ');
  }
  function loadLive() {
    api('/api/news/live').then(renderLive).catch(function (e) {
      $('live-status').className = 'live-pill failed'; $('live-status').textContent = 'Unavailable';
      var n = $('live-note'); n.hidden = false; n.textContent = 'Live summary unavailable: ' + e.message;
    });
  }
  function startLive() {
    loadLive();
    setInterval(function () { if (!document.hidden) loadLive(); }, 30000);
  }

  // ---------- citations ----------
  function citeNode(c) {
    return h('span', { class: 'n-cite-i' }, [h('b', null, c.publisher || 'unknown publisher'), ' — ',
      c.url ? [h('a', { href: c.url, title: 'Open the original article: ' + c.url }, c.title || '(no title)'), ' ↗'] : [c.title || '(no title)', ' ', badge('no link', 'b-low')],
      h('span', { class: 'n-muted' }, '  · ' + when(c.published_at))]);
  }

  // ---------- taxonomy (Domain > Topic > Subtopic) ----------
  function buildTax() {
    S.meta.taxonomy.forEach(function (t) { S.by[t.code] = t; (S.kids[t.parent || ''] = S.kids[t.parent || ''] || []).push(t); });
    Object.keys(S.kids).forEach(function (k) { S.kids[k].sort(function (a, b) { return a.code < b.code ? -1 : 1; }); });
  }
  function parts(code) { if (!code) return []; var p = code.split('.'), c = '', out = []; p.forEach(function (x, i) { c = i ? c + '.' + x : x; out.push(c); }); return out; }
  function opt(v, label, sel) { return h('option', { value: v, selected: sel }, label); }
  function fillCountry() {
    var s = clear($('t-country')); s.appendChild(opt('', 'Any country', !S.country));
    S.meta.countries.forEach(function (c) { s.appendChild(opt(c.iso2, c.name + ' (' + c.n_stories.toLocaleString() + ')', c.iso2 === S.country)); });
  }
  function fillTax() {
    var p = parts(S.tax), dom = p[0] || '', top = p[1] || '', sub = p[3] || '';
    var d = clear($('t-domain')); d.appendChild(opt('', 'All topics', !dom));
    (S.kids[''] || []).forEach(function (x) { d.appendChild(opt(x.code, x.name, x.code === dom)); });
    var t = clear($('t-topic')); t.appendChild(opt('', dom ? 'All topics in this domain' : '—', !top)); t.disabled = !dom;
    (S.kids[dom] || []).forEach(function (x) { t.appendChild(opt(x.code, x.name, x.code === top)); });
    var u = clear($('t-sub-sel')); u.appendChild(opt('', top ? 'All subtopics in this topic' : '—', !sub)); u.disabled = !top;
    (S.kids[top] || []).forEach(function (cat) {
      var g = h('optgroup', { label: cat.name }); (S.kids[cat.code] || []).forEach(function (x) { g.appendChild(opt(x.code, x.name, x.code === sub)); }); u.appendChild(g);
    });
  }
  function fillWindow() {
    var seg = clear($('t-window'));
    S.meta.windows.forEach(function (w) { seg.appendChild(h('button', { type: 'button', 'aria-pressed': String(w.key === S.win), onclick: function () { S.win = w.key; S.limit = S.meta.top_n; fillWindow(); load(); } }, w.label.replace('last ', 'Last '))); });
    seg.appendChild(h('button', { type: 'button', 'aria-pressed': String(S.win === 'custom'), onclick: function () {
      S.win = 'custom'; S.limit = S.meta.top_n; fillWindow(); load();
    } }, 'Custom dates'));
    $('t-custom').hidden = S.win !== 'custom';
  }
  function initCustomDates() {
    var from = $('t-date-from'), to = $('t-date-to'), end = S.meta.date_end, start = S.meta.date_start;
    if (!start || !end) { from.disabled = true; to.disabled = true; $('t-date-apply').disabled = true; return; }
    from.min = to.min = start; from.max = to.max = end;
    to.value = end;
    var d = new Date(end + 'T00:00:00Z');
    d.setUTCDate(d.getUTCDate() - 6);
    var recent = d.toISOString().slice(0, 10);
    from.value = recent < start ? start : recent;
    S.dateFrom = from.value; S.dateTo = to.value;
  }
  function applyCustomDates() {
    var from = $('t-date-from').value, to = $('t-date-to').value, err = $('t-date-error');
    err.textContent = '';
    if (!from || !to) { err.textContent = 'Choose both a start and end date.'; return; }
    if (from > to) { err.textContent = 'The start date must be on or before the end date.'; return; }
    if (from < S.meta.date_start || to > S.meta.date_end) {
      err.textContent = 'Choose dates within the available data: ' + S.meta.date_start + ' to ' + S.meta.date_end + ' IST.';
      return;
    }
    S.dateFrom = from; S.dateTo = to; S.win = 'custom'; S.limit = S.meta.top_n;
    fillWindow(); load();
  }
  function changed() { S.limit = S.meta.top_n; fillTax(); load(); }

  // ---------- results ----------
  function prompt(msg) {
    clear($('t-results')).appendChild(h('div', { class: 'n-card t-empty' }, [h('b', null, 'Select what you want to follow'), msg || 'Trending depends on the area or topic you care about. Choose a country, a topic, or both above.']));
  }
  function scopeText() {
    var c = S.country ? (S.cnames[S.country] || S.country) : null, t = S.tax && S.by[S.tax] ? S.by[S.tax].name : null;
    return c && t ? t + ' in ' + c : (c ? c + ' — all topics' : t + ' — all countries');
  }
  function load() {
    var my = ++S.req, host = $('t-results');
    if (!S.country && !S.tax) { S.res = null; prompt(); $('t-hint').textContent = 'Choose a country, a topic, or both. With a country only you get its trending stories across every topic; with a topic only, across every country.'; return; }
    var selectedWindow = S.win === 'custom' ? S.dateFrom + ' to ' + S.dateTo + ' (IST)' : S.meta.windows.filter(function (w) { return w.key === S.win; })[0].label;
    $('t-hint').textContent = scopeText() + ' · ' + selectedWindow;
    clear(host).appendChild(h('div', { class: 'n-card n-spin' }, 'Ranking stories…'));
    var q = '/api/news/trending?window=' + encodeURIComponent(S.win) + '&limit=' + S.limit + (S.country ? '&country=' + encodeURIComponent(S.country) : '') + (S.tax ? '&tax=' + encodeURIComponent(S.tax) : '');
    if (S.win === 'custom') q += '&date_from=' + encodeURIComponent(S.dateFrom) + '&date_to=' + encodeURIComponent(S.dateTo);
    api(q).then(function (r) { if (my !== S.req) return; S.res = r; render(r); }).catch(function (e) { if (my !== S.req) return; errBox(clear(host), e.message); });
  }
  function render(r) {
    var host = clear($('t-results'));
    if (r.needs_selection) { prompt(r.message); return; }
    host.appendChild(h('div', { class: 't-summary' }, [
      h('h2', null, (r.stories.length ? 'Top ' + r.stories.length + ' trending stories' : 'No trending stories') + ' · ' + scopeText()),
      h('span', { class: 'n-muted' }, r.window.label + (r.window.key === 'custom' ? '' : ' · data up to ' + when(r.as_of)) +
        ' · ranked from ' + r.n_candidates.toLocaleString() + ' active stories (of ' + r.n_scope_stories.toLocaleString() + ' in ' + (S.country ? (S.cnames[S.country] || S.country) : 'all countries') + ')')]));
    (r.notes || []).forEach(function (n) { host.appendChild(h('div', { class: 't-note' }, n)); });
    if (!r.stories.length) { host.appendChild(h('div', { class: 'n-card t-empty' }, [
      h('b', null, 'No stories met the trending criteria for this selection and period.'),
      (r.notes && r.notes[0]) || 'No story had enough recent coverage from multiple publishers to be listed.',
      ' This does not mean there was no news or no important event. Try a longer period or a broader selection.'
    ])); return; }
    var list = h('div', { class: 't-list' }); host.appendChild(list);
    r.stories.forEach(function (s) { list.appendChild(card(s, r)); });
    if (r.stories.length >= S.limit && S.limit < S.meta.max_n && r.n_candidates > r.stories.length)
      host.appendChild(h('button', { class: 't-more', onclick: function () { S.limit = Math.min(S.meta.max_n, S.limit + 10); load(); } }, 'Show more stories'));
  }
  function relCount(s) {   // articles of closely related stories folded under this one (they are in the drawer too)
    var n = (s.related || []).reduce(function (a, x) { return a + x.n_articles; }, 0);
    return n ? ' (+' + n.toLocaleString() + ' in closely related coverage)' : '';
  }
  var TAX_STATUS = { mixed: ['Mixed topics', 'The articles in this story are labelled with different topics and no single one has majority support, so no category is shown.'],
                     outside: ['Outside the taxonomy', 'Most of this story\u2019s articles are labelled Non-relevant / Other.'],
                     unlabelled: ['Unlabelled', 'None of this story\u2019s articles carries a taxonomy label.'] };
  // the path a MAJORITY of the story's labelled articles agree on (Domain › Topic › Category › Subtopic), or an honest status
  function taxChip(s) {
    var t = s.taxonomy, path = [t.domain, t.topic, t.category, t.subtopic].filter(Boolean).map(function (x) { return x.name; });
    if (path.length) return h('span', { class: 't-chip tax', title: 'Domain › Topic › Category › Subtopic that most of this story\u2019s labelled articles agree on' }, path.join(' › '));
    var st = TAX_STATUS[s.taxonomy_status] || TAX_STATUS.unlabelled;
    return h('span', { class: 't-chip tax none', title: st[1] }, st[0]);
  }
  function locChip(s) {
    var l = s.location, name = l.country ? (S.cnames[l.country] || l.country) : null;
    if (!name) return h('span', { class: 't-chip loc', title: 'No usable event location for this story' }, '📍 Location unknown');
    return h('span', { class: 't-chip loc', title: 'Where the event happened (GDELT). ' + (l.uncertain ? 'Uncertain: few located articles, or they disagree.' : '') }, '📍 ' + name + (l.place ? ' · ' + l.place : '') + (l.uncertain ? ' (uncertain)' : ''));
  }
  function card(s, r) {
    var w = s.in_window, wl = r.window.label.replace('last ', '');
    return h('article', { class: 't-card', 'aria-label': 'Trending story ' + s.rank }, [
      h('div', { class: 't-rank', 'aria-hidden': 'true' }, String(s.rank)),
      h('div', null, [
        h('h3', { class: 't-title' }, h('button', { type: 'button', onclick: function () { openStory(s, r); } }, s.title || '(no title)')),
        h('div', { class: 't-chips' }, [taxChip(s), locChip(s)]),
        h('div', { class: 't-stats' }, [h('b', null, s.n_articles.toLocaleString()), ' related articles' + relCount(s) + ' · ', h('b', null, w.articles.toLocaleString()), ' in the ' + wl + ' · ', h('b', null, s.n_publishers.toLocaleString()), ' publishers · latest activity ',
          h('b', null, when(s.latest_activity))]),
        h('ul', { class: 't-why', 'aria-label': 'Why it is trending' }, s.reasons.map(function (t, i) { return h('li', { class: i >= 2 && /pipeline|Flagged|detected|ran ×/.test(t) ? 'ev' : null }, t); })),
        (s.sources && s.sources.length) ? h('div', { class: 't-src' }, [h('span', { class: 'n-muted' }, 'Latest coverage: '), s.sources.map(citeNode)]) : null,
        (s.related && s.related.length) ? h('div', { class: 't-related' }, 'Also covered (closely related): ' + s.related.slice(0, 2).map(function (x) { return x.title + ' (' + x.articles_in_window + ' in the ' + wl + ')'; }).join(' · ')) : null,
        h('div', { class: 't-actions' }, h('button', { type: 'button', onclick: function () { openStory(s, r); } }, 'View related articles'))]),
      h('div', { class: 't-score', title: 'Relative ranking inside this view (weighted percentile of volume, publisher breadth, growth, recency and statistical evidence). Not a probability.' }, [h('b', null, String(Math.round(s.score))), h('span', null, 'Trend score'), h('div', { class: 't-bar' }, h('i', { style: 'width:' + Math.max(2, Math.min(100, s.score)) + '%' }))])]);
  }

  // ---------- story drawer ----------
  function artRow(c, relIds) {
    var meta = [h('span', null, h('b', null, c.publisher || 'unknown publisher')), h('span', null, when(c.published_at))];
    if (c.loc_status === 'located' && (c.place || c.country)) meta.push(h('span', { title: 'event location (GDELT)' }, '📍 ' + (c.place || S.cnames[c.country] || c.country)));
    if (c.tax_subtopic || c.tax_topic) meta.push(h('span', null, '▸ ' + (c.tax_subtopic || c.tax_topic)));
    if (relIds && c.event_id && relIds[c.event_id]) meta.push(h('span', { class: 't-tag', title: 'from a closely related story listed under this one' }, 'related story'));
    return h('div', { class: 'n-art' }, [h('div', { class: 't' }, c.url ? [h('a', { href: c.url, title: 'Open the original article: ' + c.url }, c.title || '(no title)'), ' ↗'] : [c.title || '(no title)', ' ', badge('no link', 'b-low')]), h('div', { class: 'm' }, meta)]);
  }
  function openStory(s, r) {
    var body = clear($('dr-body')), ids = s.event_ids.slice(), rel = {};
    (s.related || []).forEach(function (x) { x.event_ids.forEach(function (e) { ids.push(e); rel[e] = true; }); });
    ids = ids.slice(0, 100);
    $('dr-title').textContent = s.title || 'Story';
    document.body.classList.add('drawer'); $('n-drawer').setAttribute('aria-hidden', 'false');
    var wl = r.window.label.replace('last ', '');
    body.appendChild(h('div', { class: 't-chips', style: 'margin:8px 0' }, [taxChip(s), locChip(s)]));
    body.appendChild(h('div', { class: 't-stats-grid' }, [kv('Related articles', s.n_articles.toLocaleString() + relCount(s).replace(' in closely related coverage', ' related')), kv('Publishers', s.n_publishers.toLocaleString()), kv('In the ' + wl, s.in_window.articles.toLocaleString()),
      kv('First reported', when(s.first_seen)), kv('Latest activity', when(s.latest_activity)), kv('Trend score', String(Math.round(s.score)) + ' / 100')]));
    body.appendChild(h('div', { class: 'n-2', style: 'margin:6px 0 2px' }, h('b', null, 'Why it is trending')));
    body.appendChild(h('ul', { class: 't-why' }, s.reasons.map(function (t) { return h('li', null, t); })));
    if (s.related && s.related.length) body.appendChild(h('div', { class: 't-note' }, 'Includes closely related coverage: ' + s.related.map(function (x) { return x.title + ' (' + x.n_articles + ' articles)'; }).join(' · ')));
    var box = body.appendChild(h('div', { id: 'story-articles' }, h('div', { class: 'n-spin' }, 'loading articles…')));
    function fetchPage(off) { return api('/api/news/story_articles?ids=' + ids.join(',') + '&offset=' + off + '&limit=50'); }
    fetchPage(0).then(function (p) {
      clear(box);
      box.appendChild(h('div', { class: 'n-2', style: 'margin:10px 0 4px' }, [h('b', null, 'Related articles'), ' — ' + p.total.toLocaleString() + ' (newest first)' + (p.exact_copies ? ' · ' + p.exact_copies.toLocaleString() + ' exact copies not listed' : '') + '. Links open the original publisher.']));
      var list = h('div', null, p.items.map(function (c) { return artRow(c, rel); })), shown = p.items.length, btn = null;
      box.appendChild(list);
      function sync() { if (shown >= p.total) { if (btn) btn.remove(); return; } btn = btn || box.appendChild(h('button', { style: 'margin-top:8px', onclick: function () { btn.disabled = true; fetchPage(shown).then(function (q) { q.items.forEach(function (c) { list.appendChild(artRow(c, rel)); }); shown += q.items.length; btn.disabled = false; sync(); }).catch(function (e) { btn.disabled = false; btn.textContent = 'Retry (' + e.message + ')'; }); } }, '')); btn.textContent = 'Show more (' + (p.total - shown).toLocaleString() + ' remaining)'; }
      sync();
    }).catch(function (e) { errBox(clear(box), e.message); });
  }
  function kv(l, v) { return h('div', { class: 'n-kpi' }, [h('span', null, l), h('b', { style: 'font-size:16px' }, v)]); }
  function closeStory() { document.body.classList.remove('drawer'); $('n-drawer').setAttribute('aria-hidden', 'true'); }

  // ---------- method ----------
  function renderHow() {
    var m = S.meta || {}, b = clear($('t-how-body')), P = S.methodology || m.methodology || {};
    function section(title, intro, headers, rows) {
      var body = h('div', null);
      if (intro) body.appendChild(h('p', { class: 'n-2' }, intro));
      if (headers && rows) body.appendChild(h('div', { class: 'n-tbl-wrap' }, h('table', { class: 'n-tbl' }, [
        h('thead', null, h('tr', null, headers.map(function (x) { return h('th', null, x); }))),
        h('tbody', null, rows.map(function (row) { return h('tr', null, row.map(function (x) { return h('td', null, x == null ? '—' : String(x)); })); }))
      ])));
      b.appendChild(h('details', { class: 'n-det t-method' }, [h('summary', null, title), body]));
    }
    function val(key) {
      var v = P[key];
      return Array.isArray(v) ? v.join(', ') : v && typeof v === 'object' ? JSON.stringify(v) : String(v);
    }
    b.appendChild(h('p', { class: 'n-lede' }, 'A displayed story is a group of semantically similar event clusters. The page first applies eligibility and selected filters, then ranks the survivors. The score is relative to the other stories in this exact view; it is not a probability or a judgment of importance.'));
    b.appendChild(h('ul', { class: 't-why' }, [
      h('li', null, 'Volume is the number of unique, non-duplicate articles in the selected period. Breadth is the number of distinct publisher domains in that period.'),
      h('li', null, 'Preset periods are rolling 24 hours, 3 days, or 7 days ending at the latest article time in the loaded data. Custom dates include both selected IST calendar dates; growth compares with the equally long preceding interval.'),
      h('li', null, 'A story needs at least ' + val('trend_min_window_articles') + ' articles from ' + val('trend_min_window_publishers') + ' publishers in the selected period. Country/topic/relevance filters are applied before ranking.'),
      h('li', null, 'A missing growth or evidence component is omitted, not treated as zero; the remaining score weights are renormalized. If no story passes the selected gates, the empty state means “none qualified in this data/view,” not “there was no news.”'),
      h('li', null, 'Statistical flags are separate pipeline tests. They contribute a percentile-ranked evidence count when available; a story does not need a flag to be eligible or rank.')]));

    section('1. Story building and eligibility',
      'Article vectors are grouped online by cosine similarity. A separate merge combines event clusters into displayed stories. Exact duplicates are excluded from unique article counts.',
      ['Step', 'Rule / threshold', 'Meaning'], [
        ['Article → event cluster', 'Cosine ≥ ' + val('sim_threshold') + '; cluster gap ≤ ' + val('max_gap_hours') + ' hours', 'Join the most similar eligible centroid; otherwise start a new cluster.'],
        ['Qualifying event', '≥ ' + val('event_min_articles') + ' articles and ≥ ' + val('event_min_publishers') + ' publishers', 'Smaller clusters are not candidates for story merging.'],
        ['Event clusters → displayed story', 'Cosine ≥ ' + val('trend_merge_cos') + '; time gap ≤ ' + val('trend_merge_gap_hours') + ' hours', 'Connected components form stories; merging is transitive.'],
        ['Related coverage fold', 'Cosine ≥ ' + val('trend_fold_cos') + '; same time-gap rule', 'A related lower-ranked story may appear under a listed story; not transitive.'],
        ['Story in selected window', '≥ ' + val('trend_min_window_articles') + ' articles and ≥ ' + val('trend_min_window_publishers') + ' publishers', 'Hard candidate minimum, after combining member-event articles.'],
        ['Country membership', '≥ ' + (100 * P.trend_country_min_share) + '% of located articles; ≥ ' + val('trend_country_min_located') + ' located there; ≥ ' + (100 * P.trend_country_min_coverage) + '% of all story articles', 'All three location checks must pass.'],
        ['Taxonomy membership', 'Strict majority: > ' + (100 * P.taxonomy_min_agreement) + '%', 'Classifier labels are reused; a tie or plurality is not enough.'],
        ['Relevance gate', 'Mean cosine of 5 nearest examples ≥ ' + val('relevance_min_similarity') + '; selected-domain agreement ≥ ' + (100 * P.relevance_min_agreement) + '% of 10 nearest', 'Applied only when a relevance score exists; unscored stories pass.']
      ]);

    section('2. Final Trend score — calculation you can reproduce',
      'For each eligible story, calculate a percentile for each available component across the eligible stories in the current filtered view. Percentile rank is (average rank − 0.5) / number of scored stories; ties use average rank. A single scored story gets percentile 0.5.',
      ['Component', 'Value ranked (higher is better)', 'Weight'], [
        ['Volume', 'Current-window unique article count', (100 * P.trend_weights.volume) + '%'],
        ['Publisher breadth', 'Distinct current-window publisher domains', (100 * P.trend_weights.breadth) + '%'],
        ['Growth', 'ln((current articles + 1) / (previous articles + 1)); only when both intervals are complete and combined count ≥ ' + val('trend_growth_min_total'), (100 * P.trend_weights.growth) + '%'],
        ['Recency', 'Negative hours since the latest article (newer ranks higher)', (100 * P.trend_weights.recency) + '%'],
        ['Statistical evidence', 'Count of available flagged signal entries for the story, percentile-ranked', (100 * P.trend_weights.evidence) + '%']
      ]);
    b.appendChild(h('div', { class: 't-note' }, 'Final score = 100 × sum(weight × component percentile) / sum(weights for available components). The weights are ' +
      Object.keys(P.trend_weights || {}).map(function (k) { return k + ' ' + (100 * P.trend_weights[k]) + '%'; }).join(', ') +
      '. Unavailable components are omitted from both sums. Stories sort by score, then current-window article count, then story ID. The list shows up to ' +
      val('trend_top_n') + '; “show more” is capped at ' + val('trend_max_n') + '.'));

    section('3. Statistical evidence — tests and alert thresholds',
      'Benjamini–Hochberg (BH) adjusts p-values for multiple tests. For sorted p-values p(i) among m tests, q(i) = min over j ≥ i of (m × p(j) / j), capped at 1 and made monotonic. q < ' +
        val('fdr_alpha') + ' is a configured false-discovery-rate threshold, not the probability that a particular story is true or important.',
      ['Signal', 'Test / evidence formula', 'Required support and alert gate'], [
        ['Emerging event', 'Compare distinct publishers k at the event’s age to recent reference clusters: p = (1 + references with ≥ k publishers) / (n_reference + 1). Use same taxonomy domain if it has enough references; otherwise use the AOI pool.', 'Event is ≤ ' + val('emerging_window_hours') + ' hours old; ≥ ' + val('event_min_articles') + ' articles; not D00; high-confidence location (≥ ' + val('loc_high_min_located') + ' located and ≥ ' + (100 * P.loc_high_min_share) + '% agree); reference pool ≥ ' + val('emerging_min_ref') + '; q < ' + val('fdr_alpha') + '; ≥ ' + val('emerging_min_publishers') + ' publishers. High severity: q < ' + val('severity_q_high') + ' and ≥ ' + (2 * P.emerging_min_publishers) + ' publishers.'],
        ['New development', 'One-sided Mann–Whitney U: compare recent article cosine-to-old-centroid values with old articles’ leave-one-out cosine values. Also require median recent similarity below the old-similarity ' + val('dev_envelope_pct') + 'th percentile; coherence = mean cosine of recent vectors to their own centroid.', 'Established non-D00 event ≥ ' + val('dev_min_event_articles') + ' articles / ' + val('dev_min_event_publishers') + ' publishers; high-confidence location; ≥80% of event articles need usable embeddings; recent window ' + val('dev_recent_hours') + 'h; ≥ ' + val('dev_min_new_articles') + ' new articles / ' + val('dev_min_new_publishers') + ' publishers; ≥ ' + (P.dev_min_event_articles - P.dev_min_new_articles) + ' old articles; coherence ≥ ' + val('dev_min_coherence') + '; q < ' + val('fdr_alpha') + '. Tests capped at ' + val('dev_max_events_per_run') + ' events/run.'],
        ['Topic surge', 'Negative-binomial upper-tail P(X ≥ V | fitted baseline); report surprise = −log10(p). BH uses mid-p = P(X > V) + 0.5P(X = V).', 'AOI ≥ ' + val('aoi_min_total_articles') + ' labelled articles; node ≥ ' + val('node_min_total_v') + ' total; day exposure ≥ ' + val('min_exposure_day') + '; ≥ ' + val('baseline_min_days') + ' baseline days with mean ≥ ' + val('baseline_min_mean') + '; V ≥ ' + val('support_min_v') + '; alert requires q < ' + val('fdr_alpha') + ' and V ≥ ' + val('alert_min_ratio') + ' × baseline. High severity: q < ' + val('severity_q_high') + ' and V ≥ ' + val('severity_min_v_high') + '.']
      ]);
    b.appendChild(h('div', { class: 't-note' }, 'No applicable or insufficient-reference test is “unavailable,” not a zero or a failed significance test. Emerging/development tests require high-confidence event location; the topic-surge signal is a country/AOI × topic/category × day test. Statistical significance is not causation or editorial importance.'));

    section('4. All thresholds used by this Trending page',
      'Only settings that affect this page’s story building, filters, ranking, or the three evidence signals are listed here. Values are served from the same configuration used by the application; unrelated dashboard/pipeline metrics are intentionally omitted.',
      ['What it controls', 'Configured threshold'], [
        ['Article clustering', 'Cosine ≥ ' + val('sim_threshold') + '; maximum cluster gap ' + val('max_gap_hours') + 'h; centroid retained ' + val('centroid_retention_days') + ' days.'],
        ['Qualifying event', '≥ ' + val('event_min_articles') + ' unique articles and ≥ ' + val('event_min_publishers') + ' publishers.'],
        ['Event location consensus', 'Assign location when ≥ ' + (100 * P.location_consensus_min_share) + '% of located articles agree; at least ' + val('location_min_located') + ' located article(s). High confidence for evidence: ≥ ' + val('loc_high_min_located') + ' located and ≥ ' + (100 * P.loc_high_min_share) + '% agreement.'],
        ['Merge / related coverage', 'Merge cosine ≥ ' + val('trend_merge_cos') + '; time gap ≤ ' + val('trend_merge_gap_hours') + 'h; related-story fold cosine ≥ ' + val('trend_fold_cos') + '.'],
        ['Window eligibility / growth', 'At least ' + val('trend_min_window_articles') + ' articles and ' + val('trend_min_window_publishers') + ' publishers; growth needs ≥ ' + val('trend_growth_min_total') + ' articles across current and previous periods; data completeness tolerance ' + val('trend_window_tolerance_hours') + 'h.'],
        ['Typical-coverage comparison', 'At least ' + val('trend_min_ref') + ' comparable prior story-windows; otherwise that comparison is omitted (it is not a significance test).'],
        ['Country / taxonomy / relevance', 'Country: ≥ ' + (100 * P.trend_country_min_share) + '% of located articles, ≥ ' + val('trend_country_min_located') + ' located articles, and ≥ ' + (100 * P.trend_country_min_coverage) + '% of all story articles. Taxonomy: > ' + (100 * P.taxonomy_min_agreement) + '% agreement. Relevance: similarity ≥ ' + val('relevance_min_similarity') + '; domain agreement ≥ ' + (100 * P.relevance_min_agreement) + '%.'],
        ['Ranking weights / output', Object.keys(P.trend_weights).map(function (k) { return k + ' ' + (100 * P.trend_weights[k]) + '%'; }).join(', ') + '; show ' + val('trend_top_n') + ', maximum ' + val('trend_max_n') + '.'],
        ['Multiple testing', val('fdr_method').toUpperCase() + ' FDR; q threshold < ' + val('fdr_alpha') + '; node-test FDR uses ' + val('fdr_pvalue_kind') + ' p-values.'],
        ['Emerging event', 'Within ' + val('emerging_window_hours') + 'h; reference window ' + val('emerging_ref_days') + ' days; minimum ' + val('emerging_min_ref') + ' references and ' + val('emerging_min_publishers') + ' publishers. High severity: q < ' + val('severity_q_high') + ' and ≥ ' + (2 * P.emerging_min_publishers) + ' publishers.'],
        ['New development', 'Established event ≥ ' + val('dev_min_event_articles') + ' articles / ' + val('dev_min_event_publishers') + ' publishers; recent window ' + val('dev_recent_hours') + 'h; ≥ ' + val('dev_min_new_articles') + ' new articles / ' + val('dev_min_new_publishers') + ' publishers; coherence ≥ ' + val('dev_min_coherence') + '; old-core envelope percentile ' + val('dev_envelope_pct') + '; maximum ' + val('dev_max_events_per_run') + ' tests/run.'],
        ['Topic-surge evidence', 'AOI ≥ ' + val('aoi_min_total_articles') + ' labelled articles; node ≥ ' + val('node_min_total_v') + '; day exposure ≥ ' + val('min_exposure_day') + '; baseline ' + val('baseline_window') + ' days, minimum ' + val('baseline_min_days') + ' days and mean ' + val('baseline_min_mean') + '; count support V ≥ ' + val('support_min_v') + '; alert effect ≥ ' + val('alert_min_ratio') + '× baseline. High severity: q < ' + val('severity_q_high') + ' and V ≥ ' + val('severity_min_v_high') + '.']
      ]);

    b.appendChild(h('div', { class: 't-note' }, 'Limits: location comes from machine-extracted GDELT event location; labels are upstream classifier outputs. Embedding similarity and statistical tests are signals, not human verification. Check the Live processing panel and “Data up to” timestamp before relying on freshness.'));
  }

  // ---------- boot ----------
  function boot() {
    initTheme();
    startLive();
    api('/api/news/trending_methodology').then(function (x) {
      S.methodology = x.methodology;
      renderHow();
    }).catch(function (e) {
      clear($('t-how-body')).appendChild(h('div', { class: 'n-err', role: 'alert' }, 'Could not load Trending methodology: ' + e.message));
    });
    api('/api/news/trending_meta').then(function (m) {
      S.meta = m; m.countries.forEach(function (c) { S.cnames[c.iso2] = c.name; });
      buildTax(); fillCountry(); fillTax(); initCustomDates(); fillWindow(); renderHow();
      $('t-sub').textContent = 'Data up to ' + when(m.as_of) + ' (' + (m.date_start || '').slice(0, 10) + ' → ' + (m.date_end || '').slice(0, 10) + ' IST) · ' + m.n_stories.toLocaleString() + ' stories from ' + m.n_events.toLocaleString() + ' event clusters · read-only';
      prompt();
    }).catch(function (e) { $('t-sub').textContent = 'could not load'; errBox($('t-results'), e.message); });
    $('t-country').addEventListener('change', function (e) { S.country = e.target.value; changed(); });
    $('t-domain').addEventListener('change', function (e) { S.tax = e.target.value; changed(); });
    $('t-topic').addEventListener('change', function (e) { S.tax = e.target.value || $('t-domain').value; changed(); });
    $('t-sub-sel').addEventListener('change', function (e) { S.tax = e.target.value || $('t-topic').value; changed(); });
    $('t-date-apply').addEventListener('click', applyCustomDates);
    $('t-clear').addEventListener('click', function () { S.country = ''; S.tax = ''; fillCountry(); changed(); });
    $('dr-close').addEventListener('click', closeStory); $('n-backdrop').addEventListener('click', closeStory);
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeStory(); });
  }
  boot();
})();
