/* Trains the text monitor off the main thread. Protocol:
   in:  {cmd: 'train', rows: [...], grid: [0.1, 1, 10], folds: 5, budget: 0.05}
   out: {type: 'progress', msg, frac} ... then {type: 'done', model, report} or {type: 'error', msg} */
importScripts('tm-core.js');

self.onmessage = ev => {
  try { train(ev.data); } catch (e) { self.postMessage({ type: 'error', msg: String(e && e.message || e) }); }
};

function train({ rows, grid = [0.1, 1, 10], folds = 5, budget = 0.05, privacy = false, canonical = false }) {
  const drop = privacy ? TM.privacyDrop : null;   // applied in every fold and in the final fit: the report is about the saved model
  const P = (msg, frac) => self.postMessage({ type: 'progress', msg, frac });
  const R = [], y = [], groups = [], tools = [];
  rows.forEach((ex, i) => {
    if (ex.label !== 0 && ex.label !== 1 && ex.label !== true && ex.label !== false) throw new Error(`row ${i + 1}: "label" must be 0 or 1`);
    const r0 = TM.render(ex), r = canonical ? TM.canonical(r0) : r0;   // canonical mode: decisive fields only
    R.push(privacy ? { ctx: TM.maskPII(r.ctx), act: TM.maskPII(r.act) } : r); y.push(ex.label ? 1 : 0);   // emails, URLs masked
    groups.push(String(ex.group ?? ex.task ?? r.ctx.slice(0, 400)));
    tools.push(r.act.split(' ')[0]);
  });
  const nPos = y.reduce((a, b) => a + b, 0), nGroups = new Set(groups).size;
  if (nPos < 2 || nPos > y.length - 2) throw new Error('need at least 2 unsafe and 2 safe examples');
  const k = Math.min(folds, nGroups);
  if (k < 2) throw new Error('need at least 2 groups (tasks) for cross-validation');
  if (new Set(groups.filter((g, i) => y[i])).size < 2 || new Set(groups.filter((g, i) => !y[i])).size < 2)
    throw new Error('cross-validation needs violations in at least 2 groups (tasks) and safe actions in at least 2');
  // repeated grouped CV: 3 fold assignments (graybox.stats.group_folds); one assignment can move the estimate a lot
  const REPS = 3, reps = [];
  const steps = REPS * k * (grid.length + 1) + 2; let done = 0;
  for (let rep = 0; rep < REPS; rep++) {
    const fold = TM.groupFolds(groups, k, rep, y), oof = Object.fromEntries(grid.map(C => [C, new Float64Array(y.length)]));
    for (let f = 0; f < k; f++) {
      const tr = [], te = [];
      fold.forEach((v, i) => (v === f ? te : tr).push(i));
      P(`repeat ${rep + 1}/${REPS}, fold ${f + 1}/${k}: building vocabulary`, done / steps);
      const vc = TM.fitVectorizer(tr.map(i => R[i].ctx), 3, 200000, drop), va = TM.fitVectorizer(tr.map(i => R[i].act), 3, 200000, drop);
      const base = { vc, va }, Xtr = tr.map(i => TM.features(base, R[i])), Xte = te.map(i => TM.features(base, R[i]));
      done++;
      for (const C of grid) {
        P(`repeat ${rep + 1}/${REPS}, fold ${f + 1}/${k}: fitting C = ${C}`, done / steps);
        const lr = TM.fitLR({ rows: Xtr, dim: vc.idf.length + va.idf.length }, tr.map(i => y[i]), C);
        te.forEach((i, j) => { oof[C][i] = TM.decision(lr, Xte[j]); });
        done++;
      }
    }
    reps.push(oof);
  }
  const per = Object.fromEntries(grid.map(C => [C, reps.map(o => TM.auroc(y, Array.from(o[C])))]));
  const mean = a => a.reduce((x, z) => x + z, 0) / a.length;
  const cvAuc = Object.fromEntries(grid.map(C => [C, mean(per[C])]));
  const best = grid.reduce((a, b) => (cvAuc[b] > cvAuc[a] ? b : a));
  const oof = reps[0];                               // the first assignment's scores: interval, histogram and per-tool table
  P(`final fit on all ${y.length} examples (C = ${best})`, done / steps);
  const model = TM.fit(R, y, best, { drop });
  model.privacy = privacy;
  if (privacy) model.mask_pii = true;                // the rows were masked: scoring masks the same way
  if (canonical) model.canonical = true;             // the rows were canonical: scoring reads the same form
  done++;
  P('evaluating', done / steps);
  const s = Array.from(oof[best]), cut = TM.quantile(s, 1 - budget);
  const caught = s.filter((v, i) => y[i] && v >= cut).length, safeEsc = s.filter((v, i) => !y[i] && v >= cut).length;
  const ci = TM.groupBootstrapAUROC(y, s, groups, 500, 0);
  const perTool = {};
  tools.forEach((t, i) => { const o = perTool[t] || (perTool[t] = { n: 0, unsafe: 0, caught: 0 }); o.n++; if (y[i]) o.unsafe++; });
  reps.forEach(o => { const z = Array.from(o[best]), c = TM.quantile(z, 1 - budget);           // caught: mean over the 3 repeats, as the headline
    tools.forEach((t, i) => { if (y[i] && z[i] >= c) perTool[t].caught += 1 / reps.length; }); });
  // the report's catch rate uses the top `budget` of the out-of-fold scores (a cross-validated estimate). The saved cut is
  // replaced on the page (train.js results()) by the top `budget` of the final model's scores on ALL loaded actions.
  model.threshold_logit = cut;
  model.cut_source = 'out-of-fold scores of the training actions (replaced on the page)';
  model.budget = budget;
  model.format = 'graybox-text-monitor/1';
  model.trained_on = { n: y.length, unsafe: nPos, groups: nGroups };
  const nEsc = caught + safeEsc;
  const atBudget = reps.map(o => { const z = Array.from(o[best]), c = TM.quantile(z, 1 - budget);
    const tp = z.filter((v, i) => y[i] && v >= c).length, fp = z.filter((v, i) => !y[i] && v >= c).length;
    return { caught: tp / nPos, safe: fp / (y.length - nPos), precision: tp + fp ? tp / (tp + fp) : NaN }; });   // mean over the 3 repeats
  const report = { n: y.length, unsafe: nPos, groups: nGroups, folds: k, repeats: REPS, cv_auroc_by_C: grid.map(C => [C, cvAuc[C]]), C: best, auroc: cvAuc[best],
                   auroc_range: [Math.min(...per[best]), Math.max(...per[best])], auroc_ci95: ci,
                   auroc_within_tool: mean(reps.map(o => TM.withinAUROC(y, Array.from(o[best]), tools))),
                   budget, caught: mean(atBudget.map(r => r.caught)), safe_escalated: mean(atBudget.map(r => r.safe)), perTool,
                   base_rate: nPos / y.length, ceiling: Math.min(1, nEsc / nPos), precision: mean(atBudget.map(r => r.precision)),
                   vocab: { context: model.vc.idf.length, action: model.va.idf.length } };
  self.postMessage({ type: 'done', model, report });
}
