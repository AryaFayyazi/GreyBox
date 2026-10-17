/* Monitor-your-own-agent page: import logs, label actions, train in a Web Worker, review queue, export.
   All computation is local (tm-core.js, tm-worker.js); labels persist in localStorage when available. */
(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const pct = x => (100 * x).toFixed(x < 0.995 ? 1 : 2) + '%';
  let steps = [], hidden = null, order = [], pos = 0, model = null, report = null, key = 'gb-labels';
  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k) || 'null'); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage may be blocked */ } }
  };
  const nLab = () => steps.filter(s => s.label === 0 || s.label === 1).length;
  const nPos = () => steps.filter(s => s.label === 1).length;

  function load(text, src, opts = {}) {
    let r;
    try { r = TM.importLogs(text); } catch (err) { $('dataInfo').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; return; }
    steps = r.steps.map((s, i) => ({ ...s, label: s.label === true ? 1 : s.label === false ? 0 : s.label ?? null, _i: i }));
    key = 'gb-labels:' + src + ':' + steps.length;
    hidden = null; model = null; report = null;
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
    const risk = model ? ` <span class="tag">risk score ${TM.decision(model, TM.features(model, r)).toFixed(2)}</span>` : '';
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
    ['exCape', 'data/example_capebench.jsonl', 'CAPE-Bench example']]) {
    $(btn).onclick = async () => { $('dataInfo').textContent = 'downloading the example…'; try { load(await fetchText(file), label, { example: true }); } catch (e) { $('dataInfo').textContent = e.message; } };
  }
  $('file').onchange = async e => { const f = e.target.files[0]; if (f) load(await f.text(), f.name); };
  $('pasteBtn').onclick = () => load($('paste').value, 'pasted text');

  $('trainBtn').onclick = () => {
    const rows = steps.filter(s => s.label === 0 || s.label === 1);
    const budget = Number($('budget').value) / 100;
    $('trainBtn').disabled = true; $('prog').hidden = false; $('progMsg').textContent = 'starting';
    const w = new Worker('assets/tm-worker.js');
    w.onmessage = ev => {
      const m = ev.data;
      if (m.type === 'progress') { $('prog').value = m.frac; $('progMsg').textContent = m.msg; }
      else if (m.type === 'error') { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(m.msg)}`; $('trainBtn').disabled = false; w.terminate(); }
      else { model = m.model; report = m.report; w.terminate(); $('prog').hidden = true; $('progMsg').textContent = ''; $('trainBtn').disabled = false; results(); }
    };
    w.postMessage({ cmd: 'train', rows, budget });
  };
  function results() {
    const r = report;
    $('results').hidden = false; $('use').hidden = false;
    $('stats').innerHTML = [
      [r.auroc.toFixed(3), `AUROC on conversations held out in cross-validation (95% CI ${r.auroc_ci95[0].toFixed(3)}–${r.auroc_ci95[1].toFixed(3)})`],
      [pct(r.caught), `of violations escalated when ${pct(r.budget)} of actions go to review` + (r.ceiling < 0.999 ? ` (the most this budget allows is ${pct(r.ceiling)}: violations are ${pct(r.base_rate)} of your labelled actions)` : '')],
      [pct(r.precision), `of escalated actions are real violations (${pct(r.safe_escalated)} of safe actions escalated)`]
    ].map(([v, l]) => `<div class="card"><div class="num">${v}</div><div class="muted">${l}</div></div>`).join('');
    const tools = Object.entries(r.perTool).sort((a, b) => b[1].n - a[1].n);
    $('perTool').innerHTML = `<table><thead><tr><th>Tool</th><th>Actions</th><th>Violations</th><th>Escalated at the budget</th></tr></thead><tbody>` +
      tools.map(([t, o]) => `<tr><td><code>${esc(t)}</code></td><td class="n">${o.n}</td><td class="n">${o.unsafe}</td><td class="n">${o.unsafe ? `${o.caught}/${o.unsafe}` : '–'}</td></tr>`).join('') + '</tbody></table>';
    $('details').textContent = `${r.n.toLocaleString()} labelled actions, ${r.unsafe} violations, ${r.groups} groups, ${r.folds}-fold grouped cross-validation. ` +
      `C = ${r.C} (AUROC by C: ${r.cv_auroc_by_C.map(([c, a]) => `${c}: ${a.toFixed(3)}`).join(', ')}). Vocabulary: ${r.vocab.context.toLocaleString()} context terms, ${r.vocab.action.toLocaleString()} action terms.` +
      (r.unsafe < 20 ? ' Fewer than 20 violations: treat these numbers as rough.' : '');
    const unl = steps.filter(s => s.label !== 0 && s.label !== 1);
    $('queueBox').hidden = !unl.length;
    if (unl.length) {
      const sc = unl.map(s => TM.decision(model, TM.features(model, TM.render(s))));
      const above = sc.filter(z => z >= model.threshold_logit).length;
      $('queueText').textContent = `${unl.length.toLocaleString()} actions are still unlabelled. The monitor would send ${above.toLocaleString()} of them to review. ` +
        'Label the riskiest first: they help the next model most, and they show you what the monitor flags.';
    }
    show();
  }
  $('queueBtn').onclick = () => {
    const unl = order.length ? steps.map((s, i) => i).filter(i => steps[i].label !== 0 && steps[i].label !== 1) : [];
    const sc = new Map(unl.map(i => [i, TM.decision(model, TM.features(model, TM.render(steps[i])))]));
    order = unl.sort((a, b) => sc.get(b) - sc.get(a)); pos = 0; show();
    $('labelSec').scrollIntoView({ behavior: 'smooth' });
  };
  $('scoreBtn').onclick = () => {
    if (!model) return;
    try {
      const ex = JSON.parse($('step').value), z = TM.decision(model, TM.features(model, TM.render(ex))), esc_ = z >= model.threshold_logit;
      $('scoreOut').innerHTML = `<span class="tag ${esc_ ? 'fail' : 'pass'}">${esc_ ? 'send to review' : 'allow'}</span> score ${z.toFixed(2)} (threshold ${model.threshold_logit.toFixed(2)}). ` +
        `<span class="muted">Act on the threshold, not on the probability: probabilities do not transfer between agents.</span>`;
    } catch (err) { $('scoreOut').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; }
  };
  $('exportBtn').onclick = () => { if (model) download(new Blob([JSON.stringify(model)], { type: 'application/json' }), 'greybox-text-monitor.json'); };
})();
