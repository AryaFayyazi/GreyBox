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
  // A number parsed from Python-rendered JSON keeps Python's text: an int stays exact, a float prints as Python's repr
  // (50.0 stays "50.0", 1e-05 stays "1e-05"). Used where a rendered action is parsed and dumped again (canonical mode).
  class PyNum { constructor(t) { this.t = t; } }
  class PyDict { constructor(e) { this.e = e; } }   // [key, value] pairs in source order (JS objects move integer-like keys first)
  function pyFloat(x) {
    if (!isFinite(x)) return isNaN(x) ? 'NaN' : (x > 0 ? 'Infinity' : '-Infinity');
    if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
    const [m, e] = x.toExponential().split('e'), exp = Number(e), d = m.replace('-', '').replace('.', ''), sg = x < 0 ? '-' : '';
    if (exp >= 16 || exp < -4) return sg + d[0] + (d.length > 1 ? '.' + d.slice(1) : '') + 'e' + (exp < 0 ? '-' : '+') + String(Math.abs(exp)).padStart(2, '0');
    if (exp < 0) return sg + '0.' + '0'.repeat(-exp - 1) + d;
    return sg + (d.length <= exp + 1 ? d + '0'.repeat(exp + 1 - d.length) + '.0' : d.slice(0, exp + 1) + '.' + d.slice(exp + 1));
  }
  function parsePyJSON(src) {                  // JSON.parse, with numbers as PyNum; throws on invalid input
    let i = 0;
    const ws = () => { while (i < src.length && ' \t\n\r'.includes(src[i])) i++; };
    const val = () => {
      ws();
      const c = src[i];
      if (c === '{') {
        i++; const o = new Map(); ws();
        const done = () => new PyDict([...o.entries()]);   // duplicate keys: the last value, at the first key's place (as Python)
        if (src[i] === '}') { i++; return done(); }
        for (;;) { ws(); if (src[i] !== '"') throw new SyntaxError('key'); const k = str(); ws(); if (src[i++] !== ':') throw new SyntaxError(':');
          o.set(k, val()); ws(); if (src[i] === ',') { i++; continue; } if (src[i++] === '}') return done(); throw new SyntaxError('}'); }
      }
      if (c === '[') {
        i++; const a = []; ws();
        if (src[i] === ']') { i++; return a; }
        for (;;) { a.push(val()); ws(); if (src[i] === ',') { i++; continue; } if (src[i++] === ']') return a; throw new SyntaxError(']'); }
      }
      if (c === '"') return str();
      for (const [w, v] of [['true', true], ['false', false], ['null', null], ['NaN', new PyNum('NaN')], ['Infinity', new PyNum('Infinity')], ['-Infinity', new PyNum('-Infinity')]])
        if (src.startsWith(w, i)) { i += w.length; return v; }
      const m = /^-?(?:0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?/.exec(src.slice(i));
      if (!m) throw new SyntaxError('value');
      i += m[0].length;
      return new PyNum(m[1] || m[2] ? pyFloat(Number(m[0])) : BigInt(m[0]).toString());
    };
    const str = () => {
      const m = /^"(?:[^"\\\u0000-\u001f]|\\["\\/bfnrt]|\\u[0-9a-fA-F]{4})*"/.exec(src.slice(i));
      if (!m) throw new SyntaxError('string');
      i += m[0].length; return JSON.parse(m[0]);
    };
    const v = val(); ws();
    if (i !== src.length) throw new SyntaxError('trailing');
    return v;
  }
  function pyDumps(v) {
    if (v instanceof PyNum) return v.t;
    if (v instanceof PyDict) return '{' + v.e.map(([k, x]) => pyStr(k) + ': ' + pyDumps(x)).join(', ') + '}';
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
    const a = ex.action || {};
    if (typeof ex.context === 'string') return { ctx: ex.context, act: typeof ex.action === 'string' ? ex.action : actText(a.tool, a.args) };
    return { ctx: ctxText(ex.instructions, ex.task, ex.history), act: actText(a.tool, a.args) };
  }
  // Conversation-aware context (graybox.chat.chat_ctx): every user turn and agent message, in order.
  // turns: {user} | {agent} | {call: [tool, args], result}
  function chatCtx(instructions, turns) {
    const lines = ['[Operating instructions]\n' + (instructions || '')];
    let first = true;
    for (const t of turns) {
      if (t.user !== undefined) { lines.push(first ? '[User task]\n' + t.user : '[User] ' + t.user); first = false; }
      else if (t.agent !== undefined) { if (t.agent && t.agent.trim()) lines.push('[Agent said] ' + t.agent); }
      else {
        const r = cp(String(t.result || ''));
        lines.push('[Agent called] ' + t.call[0] + ' ' + pyDumps(t.call[1] || {}));
        lines.push('[Tool result] ' + r.slice(0, 2000).join('') + (r.length > 2000 ? ' ...' : ''));
      }
    }
    const out = lines.join('\n\n'), o = cp(out);
    return o.length < 30000 ? out : o.slice(0, 6000).join('') + '\n...\n' + o.slice(-24000).join('');
  }

  // ---- TF-IDF ----
  const TOK = /[\p{L}\p{N}_]{2,}/gu;   // scikit-learn's \b\w\w+\b (Python \w: letters, numbers, underscore): maximal runs of 2+
  function terms(doc) {
    const t = doc.toLowerCase().match(TOK) || [];
    const out = t.slice();
    for (let i = 0; i + 1 < t.length; i++) out.push(t[i] + ' ' + t[i + 1]);
    return out;
  }
  // privacy mode: words that identify people or secrets are left out of the vocabulary before training
  const SECRETISH = /\d|@|[a-z0-9_]{20,}/i;
  function randomish(w) {                      // a long token with few vowels: keys, ids, hashes ("kzxqvbnmwqprtj")
    return w.split(/[_\s]+/).some(t => t.length >= 10 && (t.match(/[aeiouy]/gi) || []).length / t.length < 0.2);
  }
  function privacyDrop(w, df, n) { return SECRETISH.test(w) || randomish(w) || df < Math.max(5, 0.01 * n); }
  function fitVectorizer(docs, minDf = 3, maxFeatures = 200000, drop = null) {
    const df = new Map(), tf = new Map();
    for (const d of docs) {
      const seen = new Set();
      for (const w of terms(d)) {
        tf.set(w, (tf.get(w) || 0) + 1);
        if (!seen.has(w)) { seen.add(w); df.set(w, (df.get(w) || 0) + 1); }
      }
    }
    let keep = [...df.keys()].filter(w => df.get(w) >= minDf && !(drop && drop(w, df.get(w), docs.length)));
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
  // privacy mode: the personal part of emails and the path of URLs masked before tokenizing; domains and hosts stay
  // (as graybox.monitor.mask_pii)
  function maskPII(text) {
    return String(text).replace(/[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)/g, ' xemailx $1 ')
      .replace(/(?:https?:\/\/|www\.)([A-Za-z0-9.-]+)[^\s"'<>]*/gi, ' xurlx $1 ');
  }
  // canonical mode: the decisive fields of each call only (as graybox.monitor.canonical_args). A field named like free
  // text becomes "[free text]"; in other strings quoted prose (3+ words) and '#' comments are removed.
  const FREE_TEXT_FIELDS = new Set(['body', 'subject', 'message', 'msg', 'text', 'content', 'contents', 'description', 'desc', 'note',
    'notes', 'comment', 'comments', 'reason', 'justification', 'summary', 'title', 'explanation', 'details', 'memo', 'caption',
    'remarks', 'rationale', 'purpose']);
  function canonicalArgs(v) {
    if (Array.isArray(v)) return v.map(canonicalArgs);
    if (v instanceof PyDict) return new PyDict(v.e.map(([k, x]) => [k, FREE_TEXT_FIELDS.has(k.toLowerCase()) && typeof x === 'string' ? '[free text]' : canonicalArgs(x)]));
    if (v && typeof v === 'object' && !(v instanceof PyNum)) {
      const o = {};
      for (const k of Object.keys(v)) o[k] = FREE_TEXT_FIELDS.has(String(k).toLowerCase()) && typeof v[k] === 'string' ? '[free text]' : canonicalArgs(v[k]);
      return o;
    }
    if (typeof v === 'string') {
      const prose = m => (m.slice(1, -1).split(/\s+/).filter(Boolean).length >= 3 ? '' : m);
      return v.replace(/'[^']*'|"[^"]*"/g, prose).replace(/\s#.*$/gm, '').replace(/\s+/g, ' ').trim();
    }
    return v;
  }
  function canonicalActText(act) {
    const i = act.indexOf(' '), tool = i < 0 ? act : act.slice(0, i);
    let a; try { a = parsePyJSON(i < 0 ? '' : act.slice(i + 1)); } catch (e) { return act; }
    return a instanceof PyDict ? actText(tool, canonicalArgs(a)) : act;
  }
  const canonicalCtxText = ctx => ctx.replace(/^\[Agent called\] (.+)$/gm, (m, c) => '[Agent called] ' + canonicalActText(c));
  const canonical = r => ({ ctx: canonicalCtxText(r.ctx), act: canonicalActText(r.act) });
  function features(model, r) {
    if (model.canonical) r = canonical(r);
    if (model.mask_pii) r = { ctx: maskPII(r.ctx), act: maskPII(r.act) };
    const a = transform(model.vc, r.ctx), b = transform(model.va, r.act), off = model.vc.idf.length;
    return { idx: a.idx.concat(b.idx.map(j => j + off)), val: a.val.concat(b.val) };
  }

  // ---- logistic regression, L-BFGS ----
  function fitLR(X, y, C, maxIter = 3000, tol = 1e-6) {    // tight, as graybox.monitor (scikit-learn tol=1e-6): the two trainers give the same model
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
    if (opts.canonical) rendered = rendered.map(canonical);
    if (opts.mask) rendered = rendered.map(r => ({ ctx: maskPII(r.ctx), act: maskPII(r.act) }));
    const vc = fitVectorizer(rendered.map(r => r.ctx), 3, opts.maxFeatures || 200000, opts.drop || null);
    const va = fitVectorizer(rendered.map(r => r.act), 3, opts.maxFeatures || 200000, opts.drop || null);
    const model = { vc, va, C };
    const rows = rendered.map(r => features(model, r));
    const lr = fitLR({ rows, dim: vc.idf.length + va.idf.length }, labels, C);
    model.w = lr.w; model.b = lr.b;
    if (opts.mask) model.mask_pii = true;            // scoring masks emails and URLs the same way
    if (opts.canonical) model.canonical = true;      // scoring reads the canonical action the same way
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
  // pair-weighted AUROC within strata (tools) that hold both classes, as graybox.stats.within_auroc: 0.5 = the tool alone
  function withinAUROC(y, s, strata) {
    const by = new Map(); strata.forEach((k, i) => { if (!by.has(k)) by.set(k, []); by.get(k).push(i); });
    let num = 0, den = 0;
    for (const ix of by.values()) {
      const yy = ix.map(i => y[i]), np = yy.reduce((a, b) => a + b, 0), nn = yy.length - np;
      if (!np || !nn) continue;
      num += np * nn * auroc(yy, ix.map(i => s[i])); den += np * nn;
    }
    return den ? num / den : NaN;
  }
  function quantile(a, q) { const s = Float64Array.from(a).sort(); const p = (s.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p); return s[lo] + (s[hi] - s[lo]) * (p - lo); }
  // GroupKFold like scikit-learn: groups sorted by size (desc), each assigned to the currently smallest fold
  // FNV-1a over UTF-8 bytes, as graybox.stats._fnv1a
  function fnv1a(text) { let h = 0x811c9dc5; for (const b of new TextEncoder().encode(text)) h = Math.imul(h ^ b, 0x01000193) >>> 0; return h; }
  // as graybox.stats.group_folds: groups with a violation first, then the largest, same-size groups in an order fixed by a hash of (rep, group),
  // each to the currently smallest fold. Identical folds in Python and here; rep gives another assignment (repeated CV)
  function groupFolds(groups, k, rep = 0, y = null) {
    const size = new Map(); groups.forEach(g => size.set(g, (size.get(g) || 0) + 1));
    const hv = new Map([...size.keys()].map(g => [g, fnv1a(`${rep}:${g}`)]));
    const pos = new Set(y ? groups.filter((g, i) => y[i]) : []);       // groups with a violation first: every training fold has some
    const order = [...size.keys()].sort((a, b) => (pos.has(b) - pos.has(a)) || size.get(b) - size.get(a) || hv.get(a) - hv.get(b) || (a < b ? -1 : a > b ? 1 : 0));
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

  // ---- import agent logs: OpenAI Chat Completions or Anthropic Messages conversations -> one step per tool call ----
  // A step holds what a monitor sees before the call runs: system prompt, first user message, earlier calls with
  // their results, and the call. Same rule as graybox.guard.steps_from_openai / steps_from_anthropic (Python).
  function textOf(c) {
    if (c == null) return '';
    if (typeof c === 'string') return c;
    if (Array.isArray(c)) return c.map(b => (typeof b === 'string' ? b : b && (b.text ?? (b.type === 'tool_result' ? textOf(b.content) : '')) || '')).filter(Boolean).join('\n');
    return typeof c.text === 'string' ? c.text : '';
  }
  function parseArgs(a) {
    if (a == null) return {};
    if (typeof a === 'object') return a;
    try { const v = JSON.parse(a); return v && typeof v === 'object' && !Array.isArray(v) ? v : { _raw: a }; } catch (e) { return { _raw: a }; }
  }
  // label values: 0/1, true/false, or the strings "0", "1", "true", "false"; anything else is reported, never guessed
  function parseLabel(v) {
    if (v === 1 || v === true || v === '1' || v === 'true') return 1;
    if (v === 0 || v === false || v === '0' || v === 'false') return 0;
    return undefined;
  }
  function conversationSteps(conv, gid, chat = false, info = null) {
    // conv: {messages: [...], system?} or a bare array of messages
    const msgs = Array.isArray(conv) ? conv : (conv.messages || []);
    let system = Array.isArray(conv) ? '' : textOf(conv.system);
    const sysMsg = msgs.find(m => m.role === 'system' || m.role === 'developer');
    if (!system && sysMsg) system = textOf(sysMsg.content);
    const firstUser = msgs.find(m => m.role === 'user' && textOf(m.content).trim() && !(Array.isArray(m.content) && m.content.every(b => b.type === 'tool_result')));
    const task = firstUser ? textOf(firstUser.content) : '';
    const results = {};
    for (const m of msgs) {
      if (m.role === 'tool' && m.tool_call_id) results[m.tool_call_id] = textOf(m.content);
      if (m.role === 'user' && Array.isArray(m.content)) for (const b of m.content) if (b && b.type === 'tool_result') results[b.tool_use_id] = textOf(b.content);
    }
    const labels = (!Array.isArray(conv) && conv.labels) || {};          // optional {call_id: 0/1}
    const seen = new Set();
    const steps = [], hist = [], turns = [];
    const isResults = c => Array.isArray(c) && c.every(b => b && b.type === 'tool_result');
    for (const m of msgs) {
      if (m.role === 'user') {
        if (!isResults(m.content) && textOf(m.content).trim())
          turns.push({ user: Array.isArray(m.content) ? m.content.filter(b => !(b && b.type === 'tool_result')).map(textOf).join('\n') : textOf(m.content) });
        continue;
      }
      if (m.role !== 'assistant') continue;
      const said = typeof m.content === 'string' ? m.content : (m.content || []).filter(b => b && typeof b === 'object' && b.type === 'text').map(b => b.text ?? '').join('\n');
      let calls = [];
      for (const tc of m.tool_calls || []) calls.push({ id: tc.id, tool: (tc.function || {}).name, args: parseArgs((tc.function || {}).arguments) });
      if (Array.isArray(m.content)) for (const b of m.content) if (b && b.type === 'tool_use') calls.push({ id: b.id, tool: b.name, args: parseArgs(b.input) });
      calls = calls.filter(c => c.tool);
      // a step sees the earlier calls of its own reply with empty results (none has run yet); the reply's text joins later steps
      calls.forEach((c, k) => {
        const sib = calls.slice(0, k).map(x => ({ tool: x.tool, args: x.args, result: '' }));
        let lab = null;
        if (c.id != null && labels[c.id] !== undefined) {
          seen.add(String(c.id));
          lab = parseLabel(labels[c.id]);
          if (lab === undefined) { lab = null; if (info) info.badLabels.push(`${gid}: ${c.id} = ${JSON.stringify(labels[c.id])}`); }
        }
        const sibTurns = sib.map(h => ({ call: [h.tool, h.args], result: '' }));
        const st = { instructions: system, task, history: hist.map(h => ({ ...h })).concat(sib), action: { tool: c.tool, args: c.args }, group: task ? 'task: ' + task : gid, conversation: gid, label: lab,
                     call_id: c.id ?? null, turns: turns.concat(sibTurns) };     // the whole conversation before the call, for people to read
        if (chat) st.context = chatCtx(system, turns.concat(sibTurns));
        steps.push(st);
      });
      turns.push({ agent: said });
      for (const c of calls) {
        hist.push({ tool: c.tool, args: c.args, result: results[c.id] ?? '' });
        turns.push({ call: [c.tool, c.args], result: results[c.id] ?? '' });
      }
    }
    if (info) for (const k of Object.keys(labels)) if (!seen.has(String(k))) info.unknownIds.push(`${gid}: ${k}`);
    return steps;
  }
  function importLogs(text, opts = {}) {
    text = text.trim();
    if (!text) throw new Error('the file is empty');
    let convs, whole;
    try { whole = JSON.parse(text); } catch (e) { whole = undefined; }      // one JSON document, or JSON lines
    if (whole !== undefined) {
      if (Array.isArray(whole) && whole.length && (whole[0].role !== undefined)) convs = [whole];   // one conversation as a message list
      else convs = Array.isArray(whole) ? whole : [whole];
    } else {
      convs = [];
      text.split(/\r?\n/).forEach((l, i) => {                                  // i + 1 = the physical line number
        if (!l.trim()) return;
        try { convs.push(JSON.parse(l)); } catch (e) { throw new Error(`line ${i + 1}: not valid JSON`); }
      });
    }
    const info = { badLabels: [], unknownIds: [] };
    // each line is a labelled step or a conversation, as in Python (graybox.train_reader.load_steps); a file may mix them
    const isStep = c => c && !Array.isArray(c) && (c.action !== undefined || c.context !== undefined);
    const steps = [], count = { openai: 0, anthropic: 0, none: 0, steps: 0 };
    convs.forEach((c, i) => {
      if (isStep(c)) {
        count.steps++;
        if (c.label === undefined || c.label === null) { steps.push({ ...c, label: null }); return; }
        const l = parseLabel(c.label);
        if (l === undefined) info.badLabels.push(`line ${i + 1} = ${JSON.stringify(c.label)}`);
        steps.push({ ...c, label: l === undefined ? null : l });
        return;
      }
      const msgs = Array.isArray(c) ? c : (c.messages || []);
      const oa = msgs.some(m => m && m.tool_calls), an = msgs.some(m => m && Array.isArray(m.content) && m.content.some(b => b && b.type === 'tool_use'));
      count[oa ? 'openai' : an ? 'anthropic' : 'none']++;
      steps.push(...conversationSteps(c, String(c.id ?? c.conversation_id ?? `conversation-${i + 1}`), !!opts.chat, info));
    });
    if (!steps.length) throw new Error('no tool calls found: expected labelled steps, OpenAI messages with "tool_calls" or Anthropic messages with "tool_use" blocks');
    const kinds = [['steps', 'labelled step'], ['openai', 'OpenAI conversation'], ['anthropic', 'Anthropic conversation']].filter(([k]) => count[k]);
    const fmt = kinds.length > 1 ? 'mixed: ' + kinds.map(([k, w]) => `${count[k]} ${w}${count[k] === 1 ? '' : 's'}`).join(', ')
      : kinds.length ? kinds[0][0] : 'unknown';
    info.noToolCalls = count.none;                     // conversations without any tool call (they give no steps)
    return { steps, format: fmt, conversations: count.steps === convs.length ? null : convs.length - count.steps, ...info };
  }

  root.TM = { pyDumps, ctxText, actText, render, terms, fitVectorizer, privacyDrop, maskPII, canonicalArgs, canonicalActText, canonicalCtxText, canonical, parsePyJSON, pyFloat, transform, features, fitLR, fit, score, decision, sigmoid,
              auroc, withinAUROC, quantile, groupFolds, groupBootstrapAUROC, importLogs, conversationSteps, chatCtx, parseLabel };
})(typeof self !== 'undefined' ? self : this);
