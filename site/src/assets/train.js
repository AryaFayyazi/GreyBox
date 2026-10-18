/* Monitor-your-own-agent page: import logs, label actions, train in a Web Worker, review queue, export.
   All computation is local (tm-core.js, tm-worker.js); labels persist in localStorage when available. */
(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const pct = x => (100 * x).toFixed(x < 0.995 ? 1 : 2) + '%';
  let steps = [], hidden = null, order = [], pos = 0, model = null, report = null, key = 'gb-labels', last = null;
  // browser reader: embeddings of every step (computed once per data set), scores of every step under an embed model
  let BR = null, emb = null, embScores = null, embWorker = null;
  const isEmbed = () => model && model.type === 'embed';
  function stepScore(i) { return isEmbed() ? embScores[i] : TM.decision(model, TM.features(model, TM.render(steps[i]))); }
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k) || 'null'); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage may be blocked */ } }
  };
  const nLab = () => steps.filter(s => s.label === 0 || s.label === 1).length;
  const nPos = () => steps.filter(s => s.label === 1).length;

  function load(text, src, opts = {}) {
    let r;
    last = [text, src, opts];
    const chat = $('chatCtx').checked;
    try { r = TM.importLogs(text, { chat }); } catch (err) { $('dataInfo').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; return; }
    steps = r.steps.map((s, i) => ({ ...s, label: s.label === true ? 1 : s.label === false ? 0 : s.label ?? null, _i: i }));
    key = 'gb-labels:' + src + ':' + steps.length + (chat ? ':chat' : '');
    hidden = null; model = null; report = null; emb = null; embScores = null;
    const saved = store.get(key);
    if (saved && saved.length === steps.length && !opts.example) saved.forEach((l, i) => { if (l === 0 || l === 1) steps[i].label = l; });
    const groups = new Set(steps.map(s => s.group ?? s.task)).size;
    const fileLabels = nLab();
    $('dataInfo').innerHTML = `<b>${steps.length.toLocaleString()}</b> actions from ${r.conversations ? r.conversations.toLocaleString() + ' conversations' : esc(src)}` +
      ` (format: ${esc(r.format)}; ${groups.toLocaleString()} groups). ${fileLabels ? `<b>${fileLabels}</b> come with labels (${nPos()} violations).` : 'None are labelled yet.'}`;
    $('labelChoice').hidden = !(opts.example && fileLabels);
    order = steps.map((_, i) => i); pos = 0;
    $('labelSec').hidden = false; $('results').hidden = true; $('use').hidden = true;
    refresh(); show();
  }
  function refresh() {
    const n = nLab(), p = nPos();
    $('labelProgress').textContent = `${n} of ${steps.length} labelled: ${p} violations, ${n - p} safe` + (hidden ? ' (the benchmark labels are hidden)' : '');
    $('trainBtn').disabled = !(p >= 2 && n - p >= 2);
    $('progMsg').textContent = $('trainBtn').disabled ? 'Label at least 2 violations and 2 safe actions (20+ violations for useful estimates).' : '';
    store.set(key, steps.map(s => s.label));
  }
  function show() {
    if (!steps.length) return;
    pos = Math.max(0, Math.min(pos, order.length - 1));
    const s = steps[order[pos]], r = TM.render(s);
    const hist = (s.history || []).slice(-2).map(h => `<div class="muted" style="font:12px/1.4 var(--mono)">${esc(h.tool)} ${esc(JSON.stringify(h.args || {})).slice(0, 160)}<br>&rarr; ${esc(String(h.result || '')).slice(0, 220)}</div>`).join('');
    const lab = s.label === 1 ? '<span class="tag fail">labelled: violation</span>' : s.label === 0 ? '<span class="tag pass">labelled: safe</span>' : '<span class="tag">unlabelled</span>';
    const risk = model ? ` <span class="tag">risk score ${stepScore(order[pos]).toFixed(2)}</span>` : '';
    $('labelView').innerHTML = `<div class="row" style="justify-content:space-between"><span class="muted">action ${pos + 1} of ${order.length}</span><span>${lab}${risk}</span></div>
      <p style="margin:8px 0 4px"><b>Task:</b> ${esc(s.task || '').slice(0, 400)}</p>
      ${s.instructions ? `<details><summary class="muted">Operating instructions</summary><pre style="max-height:200px">${esc(s.instructions)}</pre></details>` : ''}
      ${hist ? `<p style="margin:8px 0 4px"><b>Earlier calls</b> (last ${Math.min(2, s.history.length)} of ${s.history.length}):</p>${hist}` : ''}
      <p style="margin:10px 0 4px"><b>Proposed action:</b></p><pre style="white-space:pre-wrap">${esc(s.action && s.action.tool ? s.action.tool + ' ' + JSON.stringify(s.action.args || {}, null, 1) : r.act)}</pre>`;
  }
  function setLabel(v) { if (!steps.length) return; steps[order[pos]].label = v; refresh(); if (pos < order.length - 1) pos++; show(); }
  $('bViol').onclick = () => setLabel(1); $('bSafe').onclick = () => setLabel(0);
  $('bSkip').onclick = () => { pos++; show(); }; $('bBack').onclick = () => { pos--; show(); };
  document.addEventListener('keydown', e => {
    if ($('labelSec').hidden || /TEXTAREA|INPUT|SELECT/.test(document.activeElement.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === 'v') setLabel(1); else if (k === 's') setLabel(0); else if (k === 'n') { pos++; show(); } else if (k === 'b') { pos--; show(); }
  });
  $('useLabels').onclick = () => { if (hidden) { hidden.forEach((l, i) => { steps[i].label = l; }); hidden = null; } $('labelChoice').hidden = true; refresh(); show(); };
  $('hideLabels').onclick = () => { hidden = steps.map(s => s.label); steps.forEach(s => { s.label = null; }); $('labelChoice').hidden = true; refresh(); show(); };
  $('exportLabels').onclick = () => {
    const rows = steps.filter(s => s.label === 0 || s.label === 1).map(({ _i, ...s }) => JSON.stringify(s)).join('\n');
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
  $('file').onchange = async e => { const f = e.target.files[0]; if (f) load(await f.text(), f.name); };
  $('pasteBtn').onclick = () => load($('paste').value, 'pasted text');
  $('chatCtx').onchange = () => { if (last) load(...last); };     // rebuild the steps with the other context

  $('trainBtn').onclick = () => { if ($('mtype').value === 'embed') trainEmbed(); else trainText(); };
  function trainText() {
    const rows = steps.filter(s => s.label === 0 || s.label === 1);
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
      else { $('use').hidden = false; $('results').hidden = false; $('useCode').textContent = useCode(true); $('stats').innerHTML = ''; $('perTool').innerHTML = '';
        $('details').textContent = `Scored ${n.toLocaleString()} actions with the pre-trained reader. Without labels there is nothing to evaluate; the review queue below ranks the actions.`; queueInfo(); show(); }
    } catch (e) { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(e.message)}`; }
  };
  fetch('data/browser_reader.json').then(r => r.ok ? r.json() : null).then(b => {
    BR = b;
    if (!b) return;
    $('mtypeRow').hidden = false;
    $('mtypeNote').innerHTML = b.note;
    $('mtype').onchange = () => { $('pretrainedBtn').hidden = $('mtype').value !== 'embed'; };
  }).catch(() => {});
  function results() {
    const r = report;
    $('results').hidden = false; $('use').hidden = false;
    $('useCode').textContent = useCode(isEmbed());
    $('stats').innerHTML = [
      [r.auroc.toFixed(3), (r.pretrained ? 'AUROC on your labelled actions; the reader never saw them' : 'AUROC on conversations held out in cross-validation') +
        ` (95% CI ${r.auroc_ci95[0].toFixed(3)}–${r.auroc_ci95[1].toFixed(3)})`],
      [pct(r.caught), `of violations escalated when ${pct(r.budget)} of actions go to review` + (r.ceiling < 0.999 ? ` (the most this budget allows is ${pct(r.ceiling)}: violations are ${pct(r.base_rate)} of your labelled actions)` : '')],
      [pct(r.precision), `of escalated actions are real violations (${pct(r.safe_escalated)} of safe actions escalated)`]
    ].map(([v, l]) => `<div class="card"><div class="num">${v}</div><div class="muted">${l}</div></div>`).join('');
    const tools = Object.entries(r.perTool).sort((a, b) => b[1].n - a[1].n);
    $('perTool').innerHTML = `<table><thead><tr><th>Tool</th><th>Actions</th><th>Violations</th><th>Escalated at the budget</th></tr></thead><tbody>` +
      tools.map(([t, o]) => `<tr><td><code>${esc(t)}</code></td><td class="n">${o.n}</td><td class="n">${o.unsafe}</td><td class="n">${o.unsafe ? `${o.caught}/${o.unsafe}` : '–'}</td></tr>`).join('') + '</tbody></table>';
    $('details').textContent = r.pretrained
      ? `The pre-trained browser reader (${r.encoder}), trained only on our CAPE-Bench steps, never on your data: evaluated on your ${r.n.toLocaleString()} labelled actions (${r.unsafe} violations). The cut is the top ${pct(r.budget)} of your actions, set without labels.`
      : `${r.n.toLocaleString()} labelled actions, ${r.unsafe} violations, ${r.groups} groups, ${r.folds}-fold grouped cross-validation. ` +
        (r.vocab ? `C = ${r.C} (AUROC by C: ${r.cv_auroc_by_C.map(([c, a]) => `${c}: ${a.toFixed(3)}`).join(', ')}). Vocabulary: ${r.vocab.context.toLocaleString()} context terms, ${r.vocab.action.toLocaleString()} action terms.`
                 : `Browser reader ${r.encoder} (${r.dim} features), logistic regression C = ${r.C}.`) +
      (r.unsafe < 20 ? ' Fewer than 20 violations: treat these numbers as rough.' : '');
    queueInfo();
    show();
  }
  function useCode(embed) {
    const load = embed ? 'from graybox.embed_monitor import EmbedMonitor\nguard = Guard(EmbedMonitor.load_json("greybox-reader-monitor.json"))   # CPU; downloads the reader once'
                       : 'from graybox.monitor import TextMonitor\nguard = Guard(TextMonitor.load_json("greybox-text-monitor.json"))';
    return `pip install "graybox @ git+https://github.com/AryaFayyazi/GreyBox"\n\nfrom graybox.guard import Guard\n${load}\n\n` +
      '# OpenAI: before you execute each proposed tool call\nfor call in reply.tool_calls or []:\n    v = guard.check_openai(messages, call)\n' +
      '    result = run_tool(call) if v.allow else v.block_message\n\n# Anthropic: for each tool_use block\nv = guard.check_anthropic(system, messages, block)';
  }
  function queueInfo() {
    const unl = steps.map((s, i) => i).filter(i => steps[i].label !== 0 && steps[i].label !== 1);
    $('queueBox').hidden = !unl.length;
    if (unl.length) {
      const above = unl.filter(i => stepScore(i) >= model.threshold_logit).length;
      $('queueText').textContent = `${unl.length.toLocaleString()} actions are still unlabelled. The monitor would send ${above.toLocaleString()} of them to review. ` +
        'Label the riskiest first: they help the next model most, and they show you what the monitor flags.';
    }
  }
  $('queueBtn').onclick = () => {
    const unl = order.length ? steps.map((s, i) => i).filter(i => steps[i].label !== 0 && steps[i].label !== 1) : [];
    const sc = new Map(unl.map(i => [i, stepScore(i)]));
    order = unl.sort((a, b) => sc.get(b) - sc.get(a)); pos = 0; show();
    $('labelSec').scrollIntoView({ behavior: 'smooth' });
  };
  $('scoreBtn').onclick = async () => {
    if (!model) return;
    try {
      const ex = JSON.parse($('step').value);
      let z;
      if (isEmbed()) { const E = await embedSteps([TM.render(ex)]); z = model.b; for (let k = 0; k < E.dim; k++) z += model.w[k] * (E.X[k] - model.mean[k]) / model.scale[k]; $('prog').hidden = true; $('progMsg').textContent = ''; }
      else z = TM.decision(model, TM.features(model, TM.render(ex)));
      const esc_ = z >= model.threshold_logit;
      $('scoreOut').innerHTML = `<span class="tag ${esc_ ? 'fail' : 'pass'}">${esc_ ? 'send to review' : 'allow'}</span> score ${z.toFixed(2)} (threshold ${model.threshold_logit.toFixed(2)}). ` +
        `<span class="muted">Act on the threshold, not on the probability: probabilities do not transfer between agents.</span>`;
    } catch (err) { $('scoreOut').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; }
  };
  $('exportBtn').onclick = () => { if (model) download(new Blob([JSON.stringify(model)], { type: 'application/json' }), isEmbed() ? 'greybox-reader-monitor.json' : 'greybox-text-monitor.json'); };
})();
