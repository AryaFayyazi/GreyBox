/* Grey Box text monitor core: the same model as graybox.monitor.TextMonitor (Python SDK), in the browser.
   - render(): the SDK's ctx_text / act_text (Python json.dumps separators ", " and ": ", ensure_ascii=False)
   - TF-IDF as scikit-learn TfidfVectorizer(ngram_range=(1,2), min_df=3, max_features, sublinear_tf=True):
     lowercase, token pattern \b\w\w+\b, smooth idf ln((1+n)/(1+df)) + 1, l2 row norm; one vectorizer for the
     context and one for the action, concatenated
   - logistic regression as scikit-learn LogisticRegression(C, lbfgs): minimise C * sum(logloss) + 0.5 * ||w||^2,
     intercept unpenalised
   Runs in a Web Worker (importScripts) and in the page. No data leaves the browser. */
(function (root) {
  'use strict';

  // ---- Python-compatible json.dumps (default separators, ensure_ascii=False) ----
  function pyStr(s) {
    let o = '"';
    for (const ch of s) {
      const c = ch.codePointAt(0);
      if (ch === '"') o += '\\"';
      else if (ch === '\\') o += '\\\\';
      else if (ch === '\n') o += '\\n';
      else if (ch === '\r') o += '\\r';
      else if (ch === '\t') o += '\\t';
      else if (ch === '\b') o += '\\b';
      else if (ch === '\f') o += '\\f';
      else if (c < 0x20) o += '\\u' + c.toString(16).padStart(4, '0');
      else o += ch;
    }
    return o + '"';
  }
  function pyNum(x) {
    if (Number.isInteger(x) && !Object.is(x, -0)) return String(x);
    if (!isFinite(x)) return isNaN(x) ? 'NaN' : (x > 0 ? 'Infinity' : '-Infinity');
    return String(x);   // Python prints floats like 1.0 as "1.0"; JSON numbers that were floats with .0 become ints here
  }
  function pyDumps(v) {
    if (v === null || v === undefined) return 'null';
    if (v === true) return 'true';
    if (v === false) return 'false';
    if (typeof v === 'number') return pyNum(v);
    if (typeof v === 'string') return pyStr(v);
    if (Array.isArray(v)) return '[' + v.map(pyDumps).join(', ') + ']';
    return '{' + Object.keys(v).map(k => pyStr(k) + ': ' + pyDumps(v[k])).join(', ') + '}';
  }
  const cp = s => Array.from(s);   // Python slices by code point
  function ctxText(instructions, task, history) {
    const lines = ['[Operating instructions]\n' + (instructions || ''), '[User task]\n' + (task || '')];
    for (const s of history || []) {
      lines.push('[Agent called] ' + s.tool + ' ' + pyDumps(s.args || {}));
      const r = cp(String(s.result || ''));
      lines.push('[Tool result] ' + r.slice(0, 2000).join('') + (r.length > 2000 ? ' ...' : ''));
    }
    const out = lines.join('\n\n'), o = cp(out);
    return o.length < 30000 ? out : o.slice(0, 6000).join('') + '\n...\n' + o.slice(-24000).join('');
  }
  function actText(tool, args) {
    return tool === '__final__' ? 'final answer to the user (no tool call)' : tool + ' ' + pyDumps(args || {});
  }
  // One example: {context, action, label} pre-rendered, or {instructions, task, history, action: {tool, args}, label}
  function render(ex) {
    if (typeof ex.context === 'string' && typeof ex.action === 'string') return { ctx: ex.context, act: ex.action };
    const a = ex.action || {};
    return { ctx: ctxText(ex.instructions, ex.task, ex.history), act: actText(a.tool, a.args) };
  }

  // ---- TF-IDF ----
  const TOK = /[\p{L}\p{N}_]{2,}/gu;   // scikit-learn's \b\w\w+\b (Python \w: letters, numbers, underscore): maximal runs of 2+
  function terms(doc) {
    const t = doc.toLowerCase().match(TOK) || [];
    const out = t.slice();
    for (let i = 0; i + 1 < t.length; i++) out.push(t[i] + ' ' + t[i + 1]);
    return out;
  }
  function fitVectorizer(docs, minDf = 3, maxFeatures = 200000) {
    const df = new Map(), tf = new Map();
    for (const d of docs) {
      const seen = new Set();
      for (const w of terms(d)) {
        tf.set(w, (tf.get(w) || 0) + 1);
        if (!seen.has(w)) { seen.add(w); df.set(w, (df.get(w) || 0) + 1); }
      }
    }
    let keep = [...df.keys()].filter(w => df.get(w) >= minDf);
    if (keep.length > maxFeatures) keep = keep.sort((a, b) => tf.get(b) - tf.get(a) || (a < b ? -1 : 1)).slice(0, maxFeatures);
    keep.sort();
    const n = docs.length, vocab = {}, idf = new Float64Array(keep.length);
    keep.forEach((w, i) => { vocab[w] = i; idf[i] = Math.log((1 + n) / (1 + df.get(w))) + 1; });
    return { vocab, idf: Array.from(idf) };
  }
  function transform(v, doc) {
    const c = new Map();
    for (const w of terms(doc)) { const j = v.vocab[w]; if (j !== undefined) c.set(j, (c.get(j) || 0) + 1); }
    const idx = [...c.keys()].sort((a, b) => a - b), val = idx.map(j => (1 + Math.log(c.get(j))) * v.idf[j]);
    const nrm = Math.sqrt(val.reduce((s, x) => s + x * x, 0)) || 1;
    return { idx, val: val.map(x => x / nrm) };
  }
  function features(model, r) {
    const a = transform(model.vc, r.ctx), b = transform(model.va, r.act), off = model.vc.idf.length;
    return { idx: a.idx.concat(b.idx.map(j => j + off)), val: a.val.concat(b.val) };
  }

  // ---- logistic regression, L-BFGS ----
  function fitLR(X, y, C, maxIter = 500, tol = 1e-4) {
    const d = X.dim, n = X.rows.length, m = 10;
    let w = new Float64Array(d + 1);   // last entry: intercept
    const z = new Float64Array(n);
    function fg(w, g) {
      g.fill(0);
      let f = 0;
      for (let i = 0; i < n; i++) {
        const r = X.rows[i]; let s = w[d];
        for (let k = 0; k < r.idx.length; k++) s += w[r.idx[k]] * r.val[k];
        z[i] = s;
        const yy = y[i] ? 1 : -1, m_ = yy * s;
        f += m_ > 0 ? Math.log1p(Math.exp(-m_)) : -m_ + Math.log1p(Math.exp(m_));
        const coef = -yy / (1 + Math.exp(m_)) * C;
        for (let k = 0; k < r.idx.length; k++) g[r.idx[k]] += coef * r.val[k];
        g[d] += coef;
      }
      f *= C;
      for (let j = 0; j < d; j++) { f += 0.5 * w[j] * w[j]; g[j] += w[j]; }
      return f;
    }
    let g = new Float64Array(d + 1), f = fg(w, g);
    const S = [], Y = [], R = [];
    for (let it = 0; it < maxIter; it++) {
      let gmax = 0; for (let j = 0; j <= d; j++) gmax = Math.max(gmax, Math.abs(g[j]));
      if (gmax <= tol) break;
      // two-loop recursion
      const q = Float64Array.from(g), al = [];
      for (let k = S.length - 1; k >= 0; k--) { let a = 0; for (let j = 0; j <= d; j++) a += S[k][j] * q[j]; a *= R[k]; al[k] = a; for (let j = 0; j <= d; j++) q[j] -= a * Y[k][j]; }
      if (S.length) { const k = S.length - 1; let sy = 0, yy = 0; for (let j = 0; j <= d; j++) { sy += S[k][j] * Y[k][j]; yy += Y[k][j] * Y[k][j]; } const h = sy / yy; for (let j = 0; j <= d; j++) q[j] *= h; }
      else { const h = 1 / Math.max(1, gmax); for (let j = 0; j <= d; j++) q[j] *= h; }
      for (let k = 0; k < S.length; k++) { let b = 0; for (let j = 0; j <= d; j++) b += Y[k][j] * q[j]; b *= R[k]; for (let j = 0; j <= d; j++) q[j] += S[k][j] * (al[k] - b); }
      let gd = 0; for (let j = 0; j <= d; j++) gd -= g[j] * q[j];
      if (gd >= 0) { S.length = Y.length = R.length = 0; continue; }   // not a descent direction: reset memory
      let step = 1, wn = new Float64Array(d + 1), gn = new Float64Array(d + 1), fn;
      for (let ls = 0; ls < 40; ls++) {
        for (let j = 0; j <= d; j++) wn[j] = w[j] - step * q[j];
        fn = fg(wn, gn);
        if (fn <= f + 1e-4 * step * gd) break;
        step *= 0.5;
      }
      const s = new Float64Array(d + 1), yv = new Float64Array(d + 1); let sy = 0;
      for (let j = 0; j <= d; j++) { s[j] = wn[j] - w[j]; yv[j] = gn[j] - g[j]; sy += s[j] * yv[j]; }
      if (sy > 1e-10) { S.push(s); Y.push(yv); R.push(1 / sy); if (S.length > m) { S.shift(); Y.shift(); R.shift(); } }
      const df_ = f - fn; w = wn; g = gn; f = fn;
      if (Math.abs(df_) <= 1e-12 * Math.max(1, Math.abs(f))) break;
    }
    return { w: Array.from(w.subarray(0, d)), b: w[d] };
  }
  const sigmoid = t => t >= 0 ? 1 / (1 + Math.exp(-t)) : Math.exp(t) / (1 + Math.exp(t));
  function decision(model, x) { let s = model.b; for (let k = 0; k < x.idx.length; k++) s += model.w[x.idx[k]] * x.val[k]; return s; }

  function fit(rendered, labels, C, opts = {}) {
    const vc = fitVectorizer(rendered.map(r => r.ctx), 3, opts.maxFeatures || 200000);
    const va = fitVectorizer(rendered.map(r => r.act), 3, opts.maxFeatures || 200000);
    const model = { vc, va, C };
    const rows = rendered.map(r => features(model, r));
    const lr = fitLR({ rows, dim: vc.idf.length + va.idf.length }, labels, C);
    model.w = lr.w; model.b = lr.b;
    return model;
  }
  const score = (model, r) => sigmoid(decision(model, features(model, r)));

  // ---- evaluation ----
  function auroc(y, s) {
    const idx = s.map((v, i) => i).sort((a, b) => s[a] - s[b]);
    let rank = 0, sumPos = 0, nPos = 0;
    for (let i = 0; i < idx.length;) {
      let j = i; while (j + 1 < idx.length && s[idx[j + 1]] === s[idx[i]]) j++;
      const avg = (i + j) / 2 + 1;
      for (let k = i; k <= j; k++) if (y[idx[k]]) { sumPos += avg; nPos++; }
      i = j + 1;
    }
    const nNeg = y.length - nPos;
    return nPos && nNeg ? (sumPos - nPos * (nPos + 1) / 2) / (nPos * nNeg) : NaN;
  }
  function quantile(a, q) { const s = Float64Array.from(a).sort(); const p = (s.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p); return s[lo] + (s[hi] - s[lo]) * (p - lo); }
  // GroupKFold like scikit-learn: groups sorted by size (desc), each assigned to the currently smallest fold
  function groupFolds(groups, k) {
    const size = new Map(); groups.forEach(g => size.set(g, (size.get(g) || 0) + 1));
    const order = [...size.keys()].sort((a, b) => size.get(b) - size.get(a) || (a < b ? -1 : 1));
    const load = new Array(k).fill(0), fold = new Map();
    for (const g of order) { let f = 0; for (let i = 1; i < k; i++) if (load[i] < load[f]) f = i; fold.set(g, f); load[f] += size.get(g); }
    return groups.map(g => fold.get(g));
  }
  function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function groupBootstrapAUROC(y, s, groups, B = 500, seed = 0) {
    const ug = [...new Set(groups)], by = new Map(ug.map(g => [g, []]));
    groups.forEach((g, i) => by.get(g).push(i));
    const rnd = mulberry32(seed + 1), out = [];
    for (let b = 0; b < B; b++) {
      const yy = [], ss = [];
      for (let k = 0; k < ug.length; k++) for (const i of by.get(ug[Math.floor(rnd() * ug.length)])) { yy.push(y[i]); ss.push(s[i]); }
      const a = auroc(yy, ss); if (!isNaN(a)) out.push(a);
    }
    return [quantile(out, 0.025), quantile(out, 0.975)];
  }

  root.TM = { pyDumps, ctxText, actText, render, terms, fitVectorizer, transform, features, fitLR, fit, score, decision, sigmoid,
              auroc, quantile, groupFolds, groupBootstrapAUROC };
})(typeof self !== 'undefined' ? self : this);
