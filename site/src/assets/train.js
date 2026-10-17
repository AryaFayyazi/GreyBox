/* Train-your-own-monitor page. All computation is local: tm-core.js in a Web Worker (tm-worker.js). */
(() => {
  const $ = id => document.getElementById(id);
  const esc = s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;');
  const pct = x => (100 * x).toFixed(x < 0.995 ? 1 : 2) + '%';
  let rows = null, model = null, report = null;

  function parse(text) {
    text = text.trim();
    if (!text) throw new Error('no data');
    let out;
    if (text[0] === '[') out = JSON.parse(text);
    else out = text.split(/\r?\n/).filter(l => l.trim()).map((l, i) => { try { return JSON.parse(l); } catch (e) { throw new Error(`line ${i + 1}: not valid JSON`); } });
    out.forEach((r, i) => {
      const pre = typeof r.context === 'string' && typeof r.action === 'string';
      const st = r.action && typeof r.action === 'object' && typeof r.action.tool === 'string';
      if (!pre && !st) throw new Error(`row ${i + 1}: needs either "context" + "action" strings, or "instructions", "task", "history" and "action": {"tool", "args"}`);
      if (!('label' in r)) throw new Error(`row ${i + 1}: missing "label" (1 = violation, 0 = safe)`);
    });
    return out;
  }
  function loaded(r, src) {
    rows = r; model = null; report = null;
    const pos = r.filter(x => x.label === 1 || x.label === true).length;
    const groups = new Set(r.map(x => String(x.group ?? x.task ?? (x.context || '').slice(0, 400)))).size;
    $('dataInfo').innerHTML = `<b>${r.length.toLocaleString()}</b> steps loaded from ${esc(src)}: <b>${pos}</b> violations, ` +
      `${(r.length - pos).toLocaleString()} safe, ${groups.toLocaleString()} task groups.` +
      (pos < 20 ? ` <span class="tag fail">fewer than 20 violations: results will be unreliable</span>` : '') +
      (groups < 10 ? ` <span class="tag fail">fewer than 10 task groups: cross-validation cannot estimate new-task performance</span>` : '');
    $('trainBtn').disabled = false; $('results').hidden = true; $('use').hidden = true;
  }
  $('file').onchange = async e => {
    const f = e.target.files[0]; if (!f) return;
    try { loaded(parse(await f.text()), f.name); } catch (err) { $('dataInfo').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; }
  };
  $('pasteBtn').onclick = () => { try { loaded(parse($('paste').value), 'pasted text'); } catch (err) { $('dataInfo').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; } };
  $('exampleBtn').onclick = async () => {
    $('dataInfo').textContent = 'downloading the example (about 1 MB compressed)…';
    try { const t = await (await fetch('data/example_capebench.jsonl')).text(); loaded(parse(t), 'the CAPE-Bench example'); }
    catch (err) { $('dataInfo').innerHTML = `<span class="tag fail">error</span> could not load the example: ${esc(err.message)}`; }
  };

  $('trainBtn').onclick = () => {
    if (!rows) return;
    const budget = Number($('budget').value) / 100;
    $('trainBtn').disabled = true; $('prog').hidden = false; $('progMsg').textContent = 'starting';
    const w = new Worker('assets/tm-worker.js');
    w.onmessage = ev => {
      const m = ev.data;
      if (m.type === 'progress') { $('prog').value = m.frac; $('progMsg').textContent = m.msg; }
      else if (m.type === 'error') { $('progMsg').innerHTML = `<span class="tag fail">error</span> ${esc(m.msg)}`; $('trainBtn').disabled = false; w.terminate(); }
      else { model = m.model; report = m.report; w.terminate(); $('prog').hidden = true; $('progMsg').textContent = ''; $('trainBtn').disabled = false; show(); }
    };
    w.postMessage({ cmd: 'train', rows, budget });
  };

  function show() {
    const r = report;
    $('results').hidden = false; $('use').hidden = false;
    $('stats').innerHTML = [
      [r.auroc.toFixed(3), `AUROC on tasks held out in cross-validation (95% CI ${r.auroc_ci95[0].toFixed(3)}–${r.auroc_ci95[1].toFixed(3)}, bootstrap over tasks)`],
      [pct(r.caught), `of violations escalated when ${pct(r.budget)} of all steps go to review` +
        (r.ceiling < 0.999 ? ` (the most this budget allows is ${pct(r.ceiling)}: violations are ${pct(r.base_rate)} of your steps)` : '')],
      [pct(r.precision), `of escalated steps are real violations (${pct(r.safe_escalated)} of safe steps escalated)`]
    ].map(([v, l]) => `<div class="card"><div class="num">${v}</div><div class="muted">${l}</div></div>`).join('');
    const tools = Object.entries(r.perTool).sort((a, b) => b[1].n - a[1].n);
    $('perTool').innerHTML = `<table><thead><tr><th>Tool</th><th>Steps</th><th>Violations</th><th>Escalated at the budget</th></tr></thead><tbody>` +
      tools.map(([t, o]) => `<tr><td><code>${esc(t)}</code></td><td class="n">${o.n}</td><td class="n">${o.unsafe}</td><td class="n">${o.unsafe ? `${o.caught}/${o.unsafe}` : '–'}</td></tr>`).join('') + '</tbody></table>';
    $('details').textContent = `${r.n.toLocaleString()} steps, ${r.unsafe} violations, ${r.groups} task groups, ${r.folds}-fold grouped cross-validation. ` +
      `Regularisation chosen by cross-validation: C = ${r.C} (AUROC by C: ${r.cv_auroc_by_C.map(([c, a]) => `${c}: ${a.toFixed(3)}`).join(', ')}). ` +
      `Vocabulary: ${r.vocab.context.toLocaleString()} context terms, ${r.vocab.action.toLocaleString()} action terms.`;
  }

  $('scoreBtn').onclick = () => {
    if (!model) return;
    try {
      const ex = JSON.parse($('step').value);
      const r = TM.render(ex), z = TM.decision(model, TM.features(model, r)), p = TM.sigmoid(z);
      const esc_ = z >= model.threshold_logit;
      $('scoreOut').innerHTML = `<span class="tag ${esc_ ? 'fail' : 'pass'}">${esc_ ? 'escalate for review' : 'allow'}</span> ` +
        `score ${z.toFixed(2)} (cut ${model.threshold_logit.toFixed(2)}); model probability ${p.toFixed(4)}. ` +
        `<span class="muted">Act on the cut, not the probability: probabilities do not transfer between agents.</span>`;
    } catch (err) { $('scoreOut').innerHTML = `<span class="tag fail">error</span> ${esc(err.message)}`; }
  };
  $('exportBtn').onclick = () => {
    if (!model) return;
    const blob = new Blob([JSON.stringify(model)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'greybox-text-monitor.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };
})();
