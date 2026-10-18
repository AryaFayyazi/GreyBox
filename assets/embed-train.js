/* Trains the browser-reader monitor on embeddings (from embed-worker.js), off the main thread. Same recipe as
   graybox.embed.fit_probe: standardize -> logistic regression, C = 0.05 (no grid: the tested setting).
   in:  {cmd: 'train', X: Float32Array (n * dim), dim, y: [0/1 or null], groups, tools, budget, encoder, web}
        rows with y null are scored (review queue) but not used for training
   out: {type: 'progress', msg, frac} ... then {type: 'done', model, report, scores} or {type: 'error', msg} */
importScripts('tm-core.js');
const C = 0.05;

self.onmessage = ev => {
  try { train(ev.data); } catch (e) { self.postMessage({ type: 'error', msg: String(e && e.message || e) }); }
};

function scaler(X, dim, idx) {
  const mean = new Float64Array(dim), sq = new Float64Array(dim);
  for (const i of idx) for (let k = 0; k < dim; k++) { const v = X[i * dim + k]; mean[k] += v; sq[k] += v * v; }
  const n = idx.length, scale = new Float64Array(dim);
  for (let k = 0; k < dim; k++) { mean[k] /= n; const v = sq[k] / n - mean[k] * mean[k]; scale[k] = v > 1e-24 ? Math.sqrt(v) : 1; }
  return { mean, scale };
}
const allIdx = dim => Array.from({ length: dim }, (_, k) => k);
function rows(X, dim, idx, sc) {
  const ii = allIdx(dim);
  return idx.map(i => ({ idx: ii, val: Array.from({ length: dim }, (_, k) => (X[i * dim + k] - sc.mean[k]) / sc.scale[k]) }));
}
function scoreAll(X, dim, n, m) {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) { let s = m.b; for (let k = 0; k < dim; k++) s += m.w[k] * (X[i * dim + k] - m.mean[k]) / m.scale[k]; out[i] = s; }
  return out;
}

function train({ X, dim, y: yAll, groups: gAll, tools: tAll, budget = 0.05, folds = 5, encoder, web }) {
  const P = (msg, frac) => self.postMessage({ type: 'progress', msg, frac });
  const n = yAll.length, L = [];
  for (let i = 0; i < n; i++) if (yAll[i] === 0 || yAll[i] === 1) L.push(i);
  const y = L.map(i => yAll[i]), groups = L.map(i => gAll[i]), tools = L.map(i => tAll[i]);
  const nPos = y.reduce((a, b) => a + b, 0), nGroups = new Set(groups).size;
  if (nPos < 2 || nPos > y.length - 2) throw new Error('need at least 2 unsafe and 2 safe examples');
  const k = Math.min(folds, nGroups);
  if (k < 2) throw new Error('need at least 2 groups (tasks) for cross-validation');
  if (new Set(groups.filter((g, i) => y[i])).size < 2 || new Set(groups.filter((g, i) => !y[i])).size < 2)
    throw new Error('cross-validation needs violations in at least 2 groups (tasks) and safe actions in at least 2');
  const REPS = 3, reps = [];                          // the same repeated, stratified group folds as the text trainer
  for (let rep = 0; rep < REPS; rep++) {
    const fold = TM.groupFolds(groups, k, rep, y), o = new Float64Array(y.length);
    for (let f = 0; f < k; f++) {
      P(`repeat ${rep + 1}/${REPS}, fold ${f + 1}/${k}: fitting`, (rep * k + f) / (REPS * k + 1));
      const tr = [], te = [];
      fold.forEach((v, j) => (v === f ? te : tr).push(j));
      const sc = scaler(X, dim, tr.map(j => L[j]));
      const lr = TM.fitLR({ rows: rows(X, dim, tr.map(j => L[j]), sc), dim }, tr.map(j => y[j]), C);
      const s = scoreAll(X, dim, n, { ...sc, w: lr.w, b: lr.b });
      te.forEach(j => { o[j] = s[L[j]]; });
    }
    reps.push(o);
  }
  const oof = reps[0], aucs = reps.map(o => TM.auroc(y, Array.from(o)));
  P(`final fit on all ${y.length} labelled actions`, (REPS * k) / (REPS * k + 1));
  const sc = scaler(X, dim, L);
  const lr = TM.fitLR({ rows: rows(X, dim, L, sc), dim }, y, C);
  const model = { type: 'embed', format: 'graybox-embed-monitor/1', encoder, web, mean: Array.from(sc.mean), scale: Array.from(sc.scale),
                  w: Array.from(lr.w), b: lr.b, budget, trained_on: { n: y.length, unsafe: nPos, groups: nGroups } };
  const all = scoreAll(X, dim, n, model);
  const s = Array.from(oof), cut = TM.quantile(s, 1 - budget);
  model.threshold_logit = cut;                       // the top `budget` of the out-of-fold scores (in-sample scores are overfit)
  model.cut_source = 'out-of-fold scores of the training actions';
  const caught = s.filter((v, i) => y[i] && v >= cut).length, safeEsc = s.filter((v, i) => !y[i] && v >= cut).length, nEsc = caught + safeEsc;
  const perTool = {};
  tools.forEach((t, i) => { const o = perTool[t] || (perTool[t] = { n: 0, unsafe: 0, caught: 0 }); o.n++; if (y[i]) o.unsafe++; });
  reps.forEach(o => { const z = Array.from(o), c = TM.quantile(z, 1 - budget);                 // caught: mean over the 3 repeats
    tools.forEach((t, i) => { if (y[i] && z[i] >= c) perTool[t].caught += 1 / reps.length; }); });
  const meanAuc = aucs.reduce((x, z) => x + z, 0) / aucs.length;
  const report = { n: y.length, unsafe: nPos, groups: nGroups, folds: k, repeats: REPS, C, cv_auroc_by_C: [[C, meanAuc]], auroc: meanAuc,
                   auroc_range: [Math.min(...aucs), Math.max(...aucs)],
                   auroc_ci95: TM.groupBootstrapAUROC(y, s, groups, 500, 0), budget, perTool, base_rate: nPos / y.length,
                   ceiling: Math.min(1, nEsc / nPos), encoder, dim };
  const at = reps.map(o => { const z = Array.from(o), c = TM.quantile(z, 1 - budget);       // rates: mean over the 3 repeats
    const tp = z.filter((v, i) => y[i] && v >= c).length, fp = z.filter((v, i) => !y[i] && v >= c).length;
    return [tp / nPos, fp / (y.length - nPos), tp + fp ? tp / (tp + fp) : NaN]; });
  [report.caught, report.safe_escalated, report.precision] = [0, 1, 2].map(j => at.reduce((x, r) => x + r[j], 0) / at.length);
  self.postMessage({ type: 'done', model, report, scores: Array.from(all) });
}
