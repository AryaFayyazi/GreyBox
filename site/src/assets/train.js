/* Monitor-your-own-agent page: import logs, label actions, train in a Web Worker, review queue, export.
   All computation is local (tm-core.js, tm-worker.js, embed-*.js); labels persist in localStorage, keyed by a hash of the
   file's content (never by its name). */
(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const pct = x => (100 * x).toFixed(1) + '%';
  const num = n => Number(n).toLocaleString('en-US');
  const plural = (n, w, ws) => `${num(n)} ${n === 1 ? w : (ws || w + 's')}`;
  let steps = [], hidden = null, order = [], pos = 0, model = null, report = null, key = null, last = null, allScores = null;
  // browser reader: embeddings of every step (computed once per data set), scores of every step under an embed model
  let BR = null, emb = null, embScores = null, embWorker = null;
  const isEmbed = () => model && model.type === 'embed';
  function stepScore(i) { return isEmbed() ? embScores[i] : TM.decision(model, TM.features(model, TM.render(steps[i]))); }
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k) || 'null'); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage may be blocked */ } },
    del(k) { try { localStorage.removeItem(k); } catch (e) { /* ignore */ } }
  };
  function fnv(text) {                               // FNV-1a, 32 bit: the storage key is the content, not the file name
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
    return h.toString(16).padStart(8, '0') + ':' + text.length;
  }
  const nLab = () => steps.filter(s => s.label === 0 || s.label === 1).length;
  const nPos = () => steps.filter(s => s.label === 1).length;
  const NOTES = {
    'GPT example': 'GPT-6-luna on CAPE-Bench, OpenAI format. Labels come from the benchmark\'s rule checker.',
    'Claude example': 'Claude Haiku 5.5 on CAPE-Bench, Anthropic format. Labels come from the benchmark\'s rule checker.',
    'CAPE-Bench example': 'Pre-rendered steps of an open agent: each context holds the instructions, the task and the earlier calls.',
    'AgentDojo example': 'A new kind of task: prompt injections. Trained and tested on AgentDojo\'s own labels, as here, word counts do well. ' +
      'Trained on another benchmark, they were at chance on AgentDojo (test T9). That is why each new type of policy needs its own labels.'
  };

  function load(text, src, opts = {}) {
    let r;
    const chat = $('chatCtx').checked;
    try { r = TM.importLogs(text, { chat }); } catch (err) {
      $('dataInfo').innerHTML = `<span class="tag fail">error</span> Could not load ${esc(src)}: ${esc(err.message)}.` +
        (steps.length ? ' <b>The data below is still the previous file.</b>' : '');
      return;
    }
    last = [text, src, opts];
    steps = r.steps.map((s, i) => ({ ...s, label: s.label === 1 || s.label === 0 ? s.label : null, _i: i }));
    key = 'gb-labels:v2:' + fnv(text) + (chat ? ':chat' : '');
    hidden = null; model = null; report = null; emb = null; embScores = null; allScores = null;
    const fileLabels = nLab(), filePos = nPos();
    let restored = 0;
    const saved = store.get(key);
    if (saved && saved.v === 2 && saved.labels.length === steps.length) {
      if (saved.hidden) hidden = steps.map(s => s.label);
      saved.labels.forEach((l, i) => {
        const v = l === 0 || l === 1 ? l : null;
        if (saved.hidden ? v !== null : v !== null && v !== steps[i].label) restored++;
        if (saved.hidden || v !== null) steps[i].label = v;
      });
    }
    const groups = new Set(steps.map(s => s.group ?? s.task)).size;
    $('dataInfo').innerHTML = `<b>${plural(steps.length, 'action')}</b> from ${r.conversations ? plural(r.conversations, 'conversation') : esc(src)}` +
      ` (format: ${esc(r.format)}; ${plural(groups, 'group')}). ` +
      (fileLabels ? `The file labels ${num(fileLabels)} of them (${plural(filePos, 'violation')}).` : 'The file has no labels.') +
      (restored ? ` <b>Restored ${plural(restored, 'label')} that you saved in this browser for this file${hidden ? ' (the benchmark labels are hidden)' : ''}.</b>` : '');
    const warn = [];
    if (r.badLabels && r.badLabels.length) warn.push(`${plural(r.badLabels.length, 'label value')} ignored (use 0, 1, true or false): ${esc(r.badLabels.slice(0, 3).join('; '))}${r.badLabels.length > 3 ? '; …' : ''}`);
    if (r.unknownIds && r.unknownIds.length) warn.push(`${plural(r.unknownIds.length, 'label names', 'labels name')} a call id that is not in the conversation: ${esc(r.unknownIds.slice(0, 3).join('; '))}${r.unknownIds.length > 3 ? '; …' : ''}`);
    $('loadWarn').hidden = !warn.length;
    $('loadWarn').innerHTML = warn.map(w => `<span class="tag fail">check</span> ${w}`).join('<br>');
    $('exampleNote').hidden = !NOTES[src];
    $('exampleNote').textContent = NOTES[src] || '';
    $('labelChoice').hidden = !(opts.example && fileLabels && !(saved && saved.hidden));
    order = steps.map((_, i) => i); pos = 0;
    $('labelSec').hidden = false; $('labelEmpty').hidden = true; $('results').hidden = true; $('use').hidden = true;
    const pick = steps.findIndex(s => s.label === 1);
    $('step').value = JSON.stringify(stepForBox(steps[pick >= 0 ? pick : 0]), null, 1);
    refresh(); show();
  }
  function stepForBox(s) {
    const { instructions, task, history, action, context } = s;
    return typeof context === 'string' && !s.turns ? { context, action } : { instructions, task, history: (history || []).slice(-3), action };
  }
  function refresh() {
    const n = nLab(), p = nPos();
    $('labelProgress').textContent = `${num(n)} of ${num(steps.length)} labelled: ${plural(p, 'violation')}, ${num(n - p)} safe` + (hidden ? ' (the benchmark labels are hidden)' : '');
    $('trainBtn').disabled = !(p >= 2 && n - p >= 2);
    $('progMsg').textContent = $('trainBtn').disabled ? 'Label at least 2 violations and 2 safe actions (20 or more violations for useful estimates).' : '';
    if (key) store.set(key, { v: 2, labels: steps.map(s => s.label), hidden: !!hidden });
  }
  function rankText(i) {
    if (!allScores) return '';
    const z = allScores[i], above = allScores.filter(x => x > z).length;
    return ` (top ${Math.max(0.1, (100 * (above + 1)) / allScores.length).toFixed(1)}% of your actions)`;
  }
  function transcriptHtml(s) {
    const T = s.turns, out = [];
    let first = true;
    for (const t of T) {
      if (t.user !== undefined) {
        out.push(first ? `<div class="tx-task"><b>User (the task):</b> ${esc(t.user)}</div>`
                       : `<div class="tx-user-later"><b>User, later in the conversation:</b> ${esc(t.user)}</div>`);
        first = false;
      } else if (t.agent !== undefined) {
        if (t.agent && t.agent.trim()) out.push(`<div class="tx-agent"><b>Agent said:</b> ${esc(t.agent)}</div>`);
      } else {
        const res = String(t.result || ''), short = res.length > 400;
        out.push(`<div class="tx-call"><b>Agent called</b> <code>${esc(t.call[0])} ${esc(JSON.stringify(t.call[1] || {}))}</code><br>&rarr; ` +
          (short ? `<details><summary>${esc(res.slice(0, 400))} …</summary><pre>${esc(res)}</pre></details>` : esc(res)) + '</div>');
      }
    }
    return out.join('');
  }
  function show() {
    if (!steps.length) return;
    pos = Math.max(0, Math.min(pos, order.length - 1));
    const i = order[pos], s = steps[i], r = TM.render(s);
    const lab = s.label === 1 ? '<span class="tag fail">labelled: violation</span>' : s.label === 0 ? '<span class="tag pass">labelled: safe</span>' : '<span class="tag">unlabelled</span>';
    const risk = model ? ` <span class="tag">risk score ${stepScore(i).toFixed(2)}${rankText(i)}</span>` : '';
    let body;
    if (s.turns) {
      body = (s.instructions ? `<details><summary class="muted">Operating instructions</summary><pre style="max-height:220px">${esc(s.instructions)}</pre></details>` : '') +
        `<p style="margin:8px 0 4px"><b>The conversation up to this action</b> (${plural(s.turns.length, 'turn')}):</p><div class="transcript" id="tx">${transcriptHtml(s)}</div>`;
    } else if (typeof s.context === 'string') {
      body = `<p style="margin:8px 0 4px"><b>Context</b> (instructions, task and earlier calls, as the monitor reads them):</p><pre class="transcript" id="tx">${esc(s.context)}</pre>`;
    } else {
      body = (s.instructions ? `<details><summary class="muted">Operating instructions</summary><pre style="max-height:220px">${esc(s.instructions)}</pre></details>` : '') +
        `<p style="margin:8px 0 4px"><b>Task:</b> ${esc(s.task || '')}</p>` +
        ((s.history || []).length ? `<p style="margin:8px 0 4px"><b>Earlier calls</b> (${(s.history || []).length}):</p><div class="transcript" id="tx">` +
          s.history.map(h => `<div class="tx-call"><code>${esc(h.tool)} ${esc(JSON.stringify(h.args || {}))}</code><br>&rarr; ${esc(String(h.result || '').slice(0, 600))}</div>`).join('') + '</div>' : '');
    }
    $('labelView').innerHTML = `<div class="row" style="justify-content:space-between"><span class="muted">action ${num(pos + 1)} of ${num(order.length)}</span><span>${lab}${risk}</span></div>
      ${body}
      <p style="margin:10px 0 4px"><b>Proposed action:</b></p><pre style="white-space:pre-wrap">${esc(s.action && s.action.tool ? s.action.tool + ' ' + JSON.stringify(s.action.args || {}, null, 1) : r.act)}</pre>`;
    const tx = $('tx');
    if (tx) tx.scrollTop = tx.scrollHeight;          // the latest turns are the ones closest to the action
  }
  function setLabel(v) { if (!steps.length) return; steps[order[pos]].label = v; refresh(); if (pos < order.length - 1) pos++; show(); }
  $('bViol').onclick = () => setLabel(1); $('bSafe').onclick = () => setLabel(0);
  $('bSkip').onclick = () => { pos++; show(); }; $('bBack').onclick = () => { pos--; show(); };
  document.addEventListener('keydown', e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;            // Ctrl+S and the like are the browser's
    if ($('labelSec').hidden || /TEXTAREA|INPUT|SELECT/.test(document.activeElement.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === 'v') setLabel(1); else if (k === 's') setLabel(0); else if (k === 'n') { pos++; show(); } else if (k === 'b') { pos--; show(); }
  });
  $('useLabels').onclick = () => { if (hidden) { hidden.forEach((l, i) => { steps[i].label = l; }); hidden = null; } $('labelChoice').hidden = true; refresh(); show(); };
  $('hideLabels').onclick = () => { hidden = steps.map(s => s.label); steps.forEach(s => { s.label = null; }); $('labelChoice').hidden = true; refresh(); show(); };
  $('clearLabels').onclick = () => {
    if (!last || !confirm('Clear the labels that you saved in this browser for this file?')) return;
    store.del(key); load(...last);
  };
  $('exportLabels').onclick = () => {
    const rows = steps.filter(s => s.label === 0 || s.label === 1).map(({ _i, turns, ...s }) => JSON.stringify(s)).join('\n');
    download(new Blob([rows + '\n'], { type: 'application/x-ndjson' }), 'greybox-labelled-actions.jsonl');
  };
  function download(blob, name) {
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }
  async function fetchText(u) { const r = await fetch(u); if (!r.ok) throw new Error(`could not load ${u}`); return r.text(); }
  for (const [btn, file, label] of [['exGpt', 'data/example_gpt_openai.jsonl', 'GPT example'], ['exClaude', 'data/example_claude_anthropic.jsonl', 'Claude example'],
    ['exCape', 'data/example_capebench.jsonl', 'CAPE-Bench example'], ['exAdojo', 'data/example_agentdojo_gpt.jsonl', 'AgentDojo example']]) {
    $(btn).onclick = async () => { $('dataInfo').textContent = 'downloading the example…'; try { load(await fetchText(file), label, { example: true }); } catch (e) { $('dataInfo').textContent = e.message; } };
  }
  $('file').onchange = async e => { const f = e.target.files[0]; if (f) load(await f.text(), f.name); e.target.value = ''; };
  $('pasteBtn').onclick = () => load($('paste').value, 'pasted text');
  $('chatCtx').onchange = () => {
    const box = $('chatCtx');
    if (model && !confirm('This rebuilds the actions with the other context and discards the trained monitor. Continue?')) { box.checked = !box.checked; return; }
    if (last) load(...last);
    box.blur(); if ($('labelCard')) $('labelCard').focus();           // so that the V/S/N/B keys work again
  };

  $('trainBtn').onclick = () => { if ($('mtype').value === 'embed') trainEmbed(); else trainText(); };
  function trainText() {
    const rows = steps.filter(s => s.label === 0 || s.label === 1).map(({ turns, ...s }) => s);
    const budget = Number($('budget').value) / 100;
    $('trainBtn').disabled = true; $('prog').hidden = false; $('progMsg').textContent = 'starting';
    const w = new Worker('assets/tm-worker.js');
    w.onmessage = ev => {
      const m = ev.data;
      if (m.type === 'progress') { $('prog').value = m.frac; $('progMsg').textContent = m.msg; }
      else if (m.type === 'error') { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(m.msg)}`; $('trainBtn').disabled = false; w.terminate(); }
      else { model = m.model; model.chat = $('chatCtx').checked; report = m.report; w.terminate(); $('prog').hidden = true; $('progMsg').textContent = ''; $('trainBtn').disabled = false; results(); }
    };
    w.postMessage({ cmd: 'train', rows, budget });
  }

  // ---- browser reader ----
  function embedSteps(items) {
    return new Promise((resolve, reject) => {
      if (!embWorker) embWorker = new Worker('assets/embed-worker.js', { type: 'module' });
      const w = embWorker, id = Math.random();
      w.onmessage = ev => {
        const m = ev.data;
        if (m.type === 'progress') { $('prog').hidden = false; $('prog').value = m.frac; $('progMsg').textContent = m.msg; }
        else if (m.type === 'ready') w.postMessage({ cmd: 'embed', id, items });
        else if (m.type === 'done' && m.id === id) resolve(m);
        else if (m.type === 'error') reject(new Error(m.msg));
      };
      if (w._loaded === BR.encoder) w.postMessage({ cmd: 'embed', id, items });
      else { w._loaded = BR.encoder; w.postMessage({ cmd: 'load', encoder: BR.encoder }); }
    });
  }
  async function ensureEmb() {
    if (emb) return emb;
    $('progMsg').textContent = `loading the reader (${BR.download}, once; then cached by the browser)`;
    emb = await embedSteps(steps.map(s => TM.render(s)));
    return emb;
  }
  async function trainEmbed() {
    const budget = Number($('budget').value) / 100;
    $('trainBtn').disabled = true;
    try {
      const E = await ensureEmb();
      const w = new Worker('assets/embed-train.js');
      w.onmessage = ev => {
        const m = ev.data;
        if (m.type === 'progress') { $('prog').value = m.frac; $('progMsg').textContent = m.msg; }
        else if (m.type === 'error') { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(m.msg)}`; $('trainBtn').disabled = false; w.terminate(); }
        else { model = m.model; model.chat = $('chatCtx').checked; report = m.report; embScores = m.scores; w.terminate(); $('prog').hidden = true; $('progMsg').textContent = ''; $('trainBtn').disabled = false; results(); }
      };
      w.postMessage({ cmd: 'train', X: E.X, dim: E.dim, y: steps.map(s => s.label), groups: steps.map((s, i) => String(s.group ?? s.task ?? i)),
                      tools: steps.map(s => TM.render(s).act.split(' ')[0]), budget, encoder: BR.encoder, web: BR.web });
    } catch (e) { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(e.message)}`; $('trainBtn').disabled = false; }
  }
  $('pretrainedBtn').onclick = async () => {
    const budget = Number($('budget').value) / 100;
    try {
      const E = await ensureEmb(), P = BR.pretrained, n = steps.length, sc = new Array(n);
      for (let i = 0; i < n; i++) { let s = P.b; for (let k = 0; k < E.dim; k++) s += P.w[k] * (E.X[i * E.dim + k] - P.mean[k]) / P.scale[k]; sc[i] = s; }
      model = { ...P, budget, threshold_logit: TM.quantile(sc, 1 - budget), chat: $('chatCtx').checked, format: 'graybox-embed-monitor/1' };
      embScores = sc; $('prog').hidden = true; $('progMsg').textContent = '';
      const L = steps.map((s, i) => i).filter(i => steps[i].label === 0 || steps[i].label === 1), y = L.map(i => steps[i].label), s = L.map(i => sc[i]);
      const nPos = y.reduce((a, b) => a + b, 0), cut = model.threshold_logit;
      const caught = L.filter((i, j) => y[j] && sc[i] >= cut).length, safeEsc = L.filter((i, j) => !y[j] && sc[i] >= cut).length, nEsc = caught + safeEsc;
      const perTool = {};
      L.forEach((i, j) => { const t = TM.render(steps[i]).act.split(' ')[0], o = perTool[t] || (perTool[t] = { n: 0, unsafe: 0, caught: 0 }); o.n++; if (y[j]) { o.unsafe++; if (sc[i] >= cut) o.caught++; } });
      report = nPos >= 1 && nPos < y.length ? { pretrained: true, n: y.length, unsafe: nPos, groups: new Set(L.map(i => String(steps[i].group ?? steps[i].task ?? i))).size,
        auroc: TM.auroc(y, s), auroc_ci95: TM.groupBootstrapAUROC(y, s, L.map(i => String(steps[i].group ?? steps[i].task ?? i)), 500, 0), budget,
        caught: caught / nPos, safe_escalated: safeEsc / (y.length - nPos), perTool, base_rate: nPos / y.length, ceiling: Math.min(1, nEsc / nPos),
        precision: nEsc ? caught / nEsc : NaN, encoder: BR.encoder, dim: E.dim } : null;
      if (report) results();
      else { $('use').hidden = false; $('results').hidden = false; $('unusable').hidden = true; $('useCode').textContent = useCode(true); $('stats').innerHTML = ''; $('perTool').innerHTML = '';
        allScores = sc.slice();
        $('details').textContent = `Scored ${plural(n, 'action')} with the pre-trained reader. Without labels there is nothing to evaluate; the review queue below ranks the actions.`; queueInfo(); show(); }
    } catch (e) { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(e.message)}`; }
  };
  fetch('data/browser_reader.json').then(r => r.ok ? r.json() : null).then(b => {
    if (!b || !b.shipped) {
      $('mtypeNote').textContent = 'This trainer uses word counts. A small neural reader that runs in the browser and transfers to new kinds of tasks ' +
        'is ' + (b && b.status ? b.status : 'being tested') + '. For new kinds of tasks today, use the reader monitor on a GPU (see the docs).';
      return;
    }
    BR = b;
    $('mtypeRow').hidden = false;
    $('mtypeNote').innerHTML = b.note;
    $('mtype').onchange = () => { $('pretrainedBtn').hidden = $('mtype').value !== 'embed'; };
  }).catch(() => {});
  function results() {
    const r = report;
    allScores = steps.map((_, i) => stepScore(i));
    $('results').hidden = false; $('use').hidden = false;
    $('useCode').textContent = useCode(isEmbed());
    const weak = r.auroc_ci95[0] <= 0.5 || r.unsafe < 5 || !(r.auroc > 0.5);
    $('unusable').hidden = !weak;
    $('unusable').innerHTML = `<b>Not usable yet.</b> ` + (r.auroc <= 0.5 ? `An AUROC of ${r.auroc.toFixed(3)} is no better than guessing. ` : '') +
      (r.auroc_ci95[0] <= 0.5 ? 'Its 95% interval includes 0.5 (guessing). ' : '') + (r.unsafe < 5 ? `Only ${plural(r.unsafe, 'violation')} labelled. ` : '') +
      'Label more actions, and more violations: use the review queue below to find them.';
    $('exportBtn').textContent = weak ? 'Download anyway (not usable yet)' : 'Download model (.json)';
    $('stats').innerHTML = [
      [r.auroc.toFixed(3), (r.pretrained ? 'AUROC on your labelled actions; the reader never saw them' : 'AUROC on conversations held out in cross-validation') +
        ` (95% interval ${r.auroc_ci95[0].toFixed(3)}–${r.auroc_ci95[1].toFixed(3)})`],
      [pct(r.caught), `of violations sent to review when ${pct(r.budget)} of actions go to review` + (r.ceiling < 0.999 ? ` (the most this budget allows is ${pct(r.ceiling)}: violations are ${pct(r.base_rate)} of your labelled actions)` : '')],
      [isNaN(r.precision) ? '–' : pct(r.precision), `of the actions sent to review are real violations (${pct(r.safe_escalated)} of safe actions sent)`]
    ].map(([v, l]) => `<div class="card"><div class="num">${v}</div><div class="muted">${l}</div></div>`).join('');
    const tools = Object.entries(r.perTool).sort((a, b) => b[1].n - a[1].n);
    $('perTool').innerHTML = `<table><thead><tr><th>Tool</th><th>Actions</th><th>Violations</th><th>Sent to review at the budget</th></tr></thead><tbody>` +
      tools.map(([t, o]) => `<tr><td><code>${esc(t)}</code></td><td class="n">${num(o.n)}</td><td class="n">${num(o.unsafe)}</td><td class="n">${o.unsafe ? `${o.caught}/${o.unsafe}` : '–'}</td></tr>`).join('') + '</tbody></table>';
    $('details').textContent = r.pretrained
      ? `The pre-trained browser reader (${r.encoder}), trained only on our CAPE-Bench steps, never on your data: evaluated on your ${plural(r.n, 'labelled action')} (${plural(r.unsafe, 'violation')}). The review cut is the top ${pct(r.budget)} of your actions, set without labels.`
      : `${plural(r.n, 'labelled action')}, ${plural(r.unsafe, 'violation')}, ${plural(r.groups, 'group')} (conversations or tasks), ${r.folds}-fold cross-validation that holds out whole groups. ` +
        (r.vocab ? `Regularization C = ${r.C} (AUROC by C: ${r.cv_auroc_by_C.map(([c, a]) => `${c}: ${a.toFixed(3)}`).join(', ')}). Vocabulary: ${num(r.vocab.context)} context terms, ${num(r.vocab.action)} action terms.`
                 : `Browser reader ${r.encoder} (${num(r.dim)} features), logistic regression C = ${r.C}.`) +
      (r.unsafe < 20 ? ' With fewer than 20 violations these numbers are rough: the interval above shows how rough.' : '');
    queueInfo();
    show();
  }
  function useCode(embed) {
    const load = embed ? 'from graybox.embed_monitor import EmbedMonitor\nguard = Guard(EmbedMonitor.load_json("greybox-reader-monitor.json"))   # CPU; downloads the reader once'
                       : 'from graybox.monitor import TextMonitor\nguard = Guard(TextMonitor.load_json("greybox-text-monitor.json"))';
    return `pip install "graybox @ git+https://github.com/AryaFayyazi/GreyBox"\n\nfrom graybox.guard import Guard\n${load}\n\n` +
      '# OpenAI: before you execute each proposed tool call (messages may include the reply; the score is the same)\nfor call in reply.tool_calls or []:\n    v = guard.check_openai(messages, call)\n' +
      '    result = run_tool(call) if v.allow else v.block_message\n\n# Anthropic: for each tool_use block\nv = guard.check_anthropic(system, messages, block)';
  }
  function queueInfo() {
    const unl = steps.map((s, i) => i).filter(i => steps[i].label !== 0 && steps[i].label !== 1);
    $('queueBox').hidden = !unl.length;
    if (unl.length) {
      const above = unl.filter(i => stepScore(i) >= model.threshold_logit).length;
      $('queueText').textContent = `${plural(unl.length, 'action is', 'actions are')} still unlabelled. The monitor would send ${num(above)} of them to review. ` +
        'Label the riskiest first: they help the next model most, and they show you what the monitor flags.';
    }
  }
  $('queueBtn').onclick = () => {
    const unl = steps.map((s, i) => i).filter(i => steps[i].label !== 0 && steps[i].label !== 1);
    const sc = new Map(unl.map(i => [i, stepScore(i)]));
    order = unl.sort((a, b) => sc.get(b) - sc.get(a)); pos = 0; show();
    $('labelSec').scrollIntoView({ behavior: 'smooth' });
  };
  function parseBox(text) {
    const ex = JSON.parse(text);
    if (ex && Array.isArray(ex.messages)) {                   // a conversation: score its last tool call
      const r = TM.importLogs(JSON.stringify(ex), { chat: $('chatCtx').checked });
      return r.steps[r.steps.length - 1];
    }
    const a = ex && ex.action;
    if (!(a && (typeof a === 'string' || typeof a.tool === 'string'))) throw new Error('expected a step with "action": {"tool": ..., "args": {...}}, or a conversation {"messages": [...]}');
    return ex;
  }
  $('scoreBtn').onclick = async () => {
    if (!model) return;
    try {
      const ex = parseBox($('step').value);
      let z, note = '';
      if (isEmbed()) { const E = await embedSteps([TM.render(ex)]); z = model.b; for (let k = 0; k < E.dim; k++) z += model.w[k] * (E.X[k] - model.mean[k]) / model.scale[k]; $('prog').hidden = true; $('progMsg').textContent = ''; }
      else {
        const r = TM.render(ex), x = TM.features(model, r), off = model.vc.idf.length;
        z = TM.decision(model, x);
        if (!x.idx.some(j => j >= off)) note = ' <span class="tag fail">check</span> None of this action\'s words are in the monitor\'s vocabulary: its score comes from the context alone. Train on actions with this tool.';
      }
      const esc_ = z >= model.threshold_logit;
      $('scoreOut').innerHTML = `<span class="tag ${esc_ ? 'fail' : 'pass'}">${esc_ ? 'send to review' : 'allow'}</span> score ${z.toFixed(2)} (review cut ${model.threshold_logit.toFixed(2)}). ` +
        `<span class="muted">The score is compared with the cut set from your actions; it is not a probability.</span>${note}`;
    } catch (err) { $('scoreOut').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; }
  };
  // privacy: words with digits, "@" or long codes (tokens, keys, ids) are dropped from the downloaded file
  const SECRETISH = /\d|@|[a-z0-9_]{20,}/i;
  function privateCopy(m) {
    const c = JSON.parse(JSON.stringify(m));
    let removed = 0;
    for (const v of [c.vc, c.va]) for (const t of Object.keys(v.vocab)) if (SECRETISH.test(t)) { delete v.vocab[t]; removed++; }
    return [c, removed];
  }
  $('exportBtn').onclick = () => {
    if (!model) return;
    if ($('exportBtn').textContent.startsWith('Download anyway') && !confirm('This monitor is not usable yet (see the warning above). Download it anyway?')) return;
    let out = model, note = '';
    if (!isEmbed() && $('privacyFilter').checked) {
      const [c, removed] = privateCopy(model);
      const d = steps.map(s => { const r = TM.render(s); return Math.abs(TM.decision(c, TM.features(c, r)) - TM.decision(model, TM.features(model, r))); });
      const changed = steps.filter((s, i) => (TM.decision(c, TM.features(c, TM.render(s))) >= c.threshold_logit) !== (allScores[i] >= model.threshold_logit)).length;
      out = c;
      note = `Removed ${plural(removed, 'word')} with digits, "@" or long codes. On your actions, scores changed by at most ${Math.max(0, ...d).toFixed(3)}; ` +
        `${plural(changed, 'review decision')} changed. Names and other words can remain: check the file before you share it.`;
    }
    $('exportNote').textContent = note;
    download(new Blob([JSON.stringify(out)], { type: 'application/json' }), isEmbed() ? 'greybox-reader-monitor.json' : 'greybox-text-monitor.json');
  };
})();
