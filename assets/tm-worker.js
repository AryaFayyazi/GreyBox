/* Trains the text monitor off the main thread. Protocol:
   in:  {cmd: 'train', rows: [...], grid: [0.1, 1, 10], folds: 5, budget: 0.05}
   out: {type: 'progress', msg, frac} ... then {type: 'done', model, report} or {type: 'error', msg} */
importScripts('tm-core.js');

self.onmessage = ev => {
  try { train(ev.data); } catch (e) { self.postMessage({ type: 'error', msg: String(e && e.message || e) }); }
};

function train({ rows, grid = [0.1, 1, 10], folds = 5, budget = 0.05 }) {
  const P = (msg, frac) => self.postMessage({ type: 'progress', msg, frac });
  const R = [], y = [], groups = [], tools = [];
  rows.forEach((ex, i) => {
    if (ex.label !== 0 && ex.label !== 1 && ex.label !== true && ex.label !== false) throw new Error(`row ${i + 1}: "label" must be 0 or 1`);
    const r = TM.render(ex);
    R.push(r); y.push(ex.label ? 1 : 0);
    groups.push(String(ex.group ?? ex.task ?? r.ctx.slice(0, 400)));
    tools.push(r.act.split(' ')[0]);
  });
  const nPos = y.reduce((a, b) => a + b, 0), nGroups = new Set(groups).size;
  if (nPos < 2 || nPos > y.length - 2) throw new Error('need at least 2 unsafe and 2 safe examples');
  const k = Math.min(folds, nGroups);
  if (k < 2) throw new Error('need at least 2 groups (tasks) for cross-validation');
  const fold = TM.groupFolds(groups, k), oof = Object.fromEntries(grid.map(C => [C, new Float64Array(y.length)]));
  const steps = k * (grid.length + 1) + 2; let done = 0;
  for (let f = 0; f < k; f++) {
    const tr = [], te = [];
    fold.forEach((v, i) => (v === f ? te : tr).push(i));
    P(`fold ${f + 1}/${k}: building vocabulary`, done / steps);
    const vc = TM.fitVectorizer(tr.map(i => R[i].ctx)), va = TM.fitVectorizer(tr.map(i => R[i].act));
    const base = { vc, va }, Xtr = tr.map(i => TM.features(base, R[i])), Xte = te.map(i => TM.features(base, R[i]));
    done++;
    for (const C of grid) {
      P(`fold ${f + 1}/${k}: fitting C = ${C}`, done / steps);
      const lr = TM.fitLR({ rows: Xtr, dim: vc.idf.length + va.idf.length }, tr.map(i => y[i]), C);
      te.forEach((i, j) => { oof[C][i] = TM.decision(lr, Xte[j]); });
      done++;
    }
  }
  const cvAuc = Object.fromEntries(grid.map(C => [C, TM.auroc(y, Array.from(oof[C]))]));
  const best = grid.reduce((a, b) => (cvAuc[b] > cvAuc[a] ? b : a));
  P(`final fit on all ${y.length} examples (C = ${best})`, done / steps);
  const model = TM.fit(R, y, best);
  done++;
  P('evaluating', done / steps);
  const s = Array.from(oof[best]), cut = TM.quantile(s, 1 - budget);
  const caught = s.filter((v, i) => y[i] && v >= cut).length, safeEsc = s.filter((v, i) => !y[i] && v >= cut).length;
  const ci = TM.groupBootstrapAUROC(y, s, groups, 500, 0);
  const perTool = {};
  tools.forEach((t, i) => { const o = perTool[t] || (perTool[t] = { n: 0, unsafe: 0, caught: 0 }); o.n++; if (y[i]) { o.unsafe++; if (s[i] >= cut) o.caught++; } });
  // the frozen gate cut for deployment: (1 - budget) quantile of the model's own scores on all training rows (label-free)
  // the escalation cut: the top `budget` of the OUT-OF-FOLD scores (in-sample scores are overfit: a cut set on them
  // escalates far fewer than `budget` of new actions). The same scores give the report's catch rate.
  model.threshold_logit = cut;
  model.cut_source = 'out-of-fold scores of the training actions';
  model.budget = budget;
  model.format = 'graybox-text-monitor/1';
  model.trained_on = { n: y.length, unsafe: nPos, groups: nGroups };
  const nEsc = caught + safeEsc;
  const report = { n: y.length, unsafe: nPos, groups: nGroups, folds: k, cv_auroc_by_C: grid.map(C => [C, cvAuc[C]]), C: best, auroc: cvAuc[best], auroc_ci95: ci,
                   budget, caught: caught / nPos, safe_escalated: safeEsc / (y.length - nPos), perTool,
                   base_rate: nPos / y.length, ceiling: Math.min(1, nEsc / nPos), precision: nEsc ? caught / nEsc : NaN,
                   vocab: { context: model.vc.idf.length, action: model.va.idf.length } };
  self.postMessage({ type: 'done', model, report });
}
