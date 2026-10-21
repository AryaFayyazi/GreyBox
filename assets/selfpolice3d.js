// Two rotatable 3D views of real hidden states for the Self-policing page (data: data/selfpolice3d.json, built by
// web/build_selfpolice3d.py from stored states; every number shown is computed there). Drag to rotate, scroll to zoom,
// hover a point to read it. Palette: the validated categorical slots in fixed order.
(() => {
  const SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"];
  const SAFE = "#2a78d6", VIOL = "#eb6834", INK = "#1b2126";

  function dotTexture(ring) {
    const c = document.createElement("canvas"); c.width = c.height = 64; const x = c.getContext("2d");
    x.beginPath(); x.arc(32, 32, ring ? 26 : 22, 0, 2 * Math.PI);
    if (ring) { x.lineWidth = 7; x.strokeStyle = "#fff"; x.stroke(); } else { x.fillStyle = "#fff"; x.fill(); }
    const t = new THREE.CanvasTexture(c); return t;
  }

  // robust display scaling: each axis divided by its 98th percentile of |value|; points beyond 1.25 are drawn at the edge
  function robust(P) {
    const q = [0, 1, 2].map((k) => { const a = P.map((p) => Math.abs(p[k])).sort((x, y) => x - y); return a[Math.floor(0.98 * (a.length - 1))] || 1; });
    const m = Math.max(...q);
    return P.map((p) => p.map((v, k) => Math.max(-1.25, Math.min(1.25, v / m))));
  }

  // one view: points P [[x,y,z]], colours C ["#..."], optional ring mask, labels for the tooltip
  function view(canvas, tip, P, C, rings, label, axes, shared) {
    const W = () => canvas.clientWidth, H = () => canvas.clientHeight;
    const r = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    r.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    const st = shared || { dist: 3.6, az: 0.7, el: 0.35, views: [] };
    let dist = st.dist, az = st.az, el = st.el;
    const group = new THREE.Group(); scene.add(group);
    const pos = new Float32Array(P.length * 3), col = new Float32Array(P.length * 3);
    P.forEach((p, i) => { pos.set(p, 3 * i); const c = new THREE.Color(C[i]); col.set([c.r, c.g, c.b], 3 * i); });
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    group.add(new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.055, vertexColors: true, map: dotTexture(false), transparent: true, alphaTest: 0.4, depthWrite: false, opacity: 0.9 })));
    if (rings) {
      const idx = rings.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
      const rp = new Float32Array(idx.length * 3); idx.forEach((i, k) => rp.set(P[i], 3 * k));
      const rg = new THREE.BufferGeometry(); rg.setAttribute("position", new THREE.BufferAttribute(rp, 3));
      group.add(new THREE.Points(rg, new THREE.PointsMaterial({ size: 0.095, color: INK, map: dotTexture(true), transparent: true, alphaTest: 0.3, depthWrite: false, opacity: 0.7 })));
    }
    // axes with labels
    const axMat = new THREE.LineBasicMaterial({ color: 0x8a949e });
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((d, k) => {
      const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...d.map((v) => -1.1 * v)), new THREE.Vector3(...d.map((v) => 1.1 * v))]);
      group.add(new THREE.Line(g, axMat));
      const c = document.createElement("canvas"); c.width = 512; c.height = 64; const x = c.getContext("2d");
      x.font = "28px Archivo, sans-serif"; x.fillStyle = "#5d6873"; x.fillText(axes[k], 4, 40);
      const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false })); s.scale.set(1.0, 0.125, 1);
      s.position.set(...d.map((v) => 1.25 * v)); group.add(s);
    });
    const ray = new THREE.Raycaster(); ray.params.Points.threshold = 0.03;
    const draw = () => {
      const w = W(), h = H(); if (canvas.width !== Math.floor(w * r.getPixelRatio()) || canvas.height !== Math.floor(h * r.getPixelRatio())) { r.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
      cam.position.set(dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el), dist * Math.cos(el) * Math.cos(az)); cam.lookAt(0, 0, 0);
      r.render(scene, cam);
    };
    let drag = null;
    const sync = () => { st.dist = dist; st.az = az; st.el = el; st.views.forEach((v) => v !== self && v.set(dist, az, el)); };
    canvas.addEventListener("pointerdown", (e) => { drag = [e.clientX, e.clientY, az, el]; tip.hidden = true; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointerup", () => { drag = null; });
    canvas.addEventListener("pointerleave", () => { tip.hidden = true; });
    canvas.addEventListener("pointermove", (e) => {
      if (drag) { az = drag[2] - (e.clientX - drag[0]) * 0.01; el = Math.max(-1.4, Math.min(1.4, drag[3] + (e.clientY - drag[1]) * 0.01)); draw(); sync(); return; }
      const b = canvas.getBoundingClientRect(); const m = new THREE.Vector2(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
      ray.setFromCamera(m, cam); const hit = ray.intersectObject(group.children[0])[0];
      if (hit) { tip.textContent = label(hit.index); tip.style.left = `${e.clientX - b.left + 12}px`; tip.style.top = `${e.clientY - b.top + 12}px`; tip.hidden = false; } else tip.hidden = true;
    });
    canvas.addEventListener("wheel", (e) => { e.preventDefault(); dist = Math.max(1.6, Math.min(8, dist * (e.deltaY > 0 ? 1.08 : 0.93))); draw(); sync(); }, { passive: false });
    window.addEventListener("resize", draw); draw();
    const self = { set(d, a, e2) { dist = d; az = a; el = e2; draw(); }, draw };
    st.views.push(self);
    return self;
  }

  const legend = (el, items) => { el.innerHTML = items.map(([c, t, ring]) => `<span style="margin-right:14px;white-space:nowrap"><span style="display:inline-block;width:11px;height:11px;border-radius:50%;${ring ? `border:2px solid ${INK};` : `background:${c};`}vertical-align:-1px"></span> ${t}</span>`).join(""); };

  fetch("data/selfpolice3d.json").then((r) => r.json()).then((D) => {
    // A. inside the agent
    const A = D.agent, sA = A.stats;
    const xm = A.x.reduce((a, b) => a + b, 0) / A.x.length;
    const PA = robust(A.x.map((x, i) => [x - xm, A.y[i], A.z[i]]));
    view(document.getElementById("cvA"), document.getElementById("tipA"), PA, A.viol.map((v) => (v ? VIOL : SAFE)), A.flag,
      (i) => `${A.template[i]} · checker: ${A.viol[i] ? "violation" : "allowed"} · the agent's own audit: ${A.flag[i] ? "flagged" : "not flagged"}`,
      ["violation readout →", "direction 2", "direction 3"]);
    legend(document.getElementById("legA"), [[VIOL, "violation (checker)"], [SAFE, "allowed"], [INK, "flagged by the agent's own audit", true]]);
    document.getElementById("statA").innerHTML =
      `Gemma-4, ${sA.n.toLocaleString()} decision steps (${sA.violations} violations), its state at layer ${sA.layer}. The horizontal axis is the ` +
      `direction in the agent's own state that best separates violations (a linear readout, cross-fitted by scenario). How well it separates ` +
      `violations from allowed calls of the same kind (AUROC; 1 = perfect, 0.5 = chance): ` +
      `<b>${sA.readout_auroc_within}</b>. The agent's own spoken audit reaches <b>${sA.selfaudit_auroc_within}</b> and flags ` +
      `<b>${Math.round(sA.selfaudit_flag_violation * 100)}%</b> of the violations when all four answers must say "no" (97% when most must), ` +
      `${(sA.selfaudit_flag_safe * 100).toFixed(1)}% of allowed calls. ` +
      `Directions 2 and 3 hold ${Math.round(sA.variance_yz * 100)}% of the remaining variance. So the information that the call is a violation is ` +
      `inside the agent, and most of it reaches its answer when asked. Display: scaled by the 98th percentile of each axis.`;
    // B. what answers reveal
    const G = D.geometry, sG = G.stats, kinds = [...new Set(G.kind)].sort();
    const colB = G.kind.map((k) => SLOTS[kinds.indexOf(k) % SLOTS.length]);
    const lab = (i) => `${G.kind[i]}: ${G.text[i]}`;
    const shared = { dist: 3.6, az: 0.7, el: 0.35, views: [] };                 // the two panels turn and zoom together
    view(document.getElementById("cvB1"), document.getElementById("tipB1"), robust(G.hidden), colB, null, lab, ["", "", ""], shared);
    view(document.getElementById("cvB2"), document.getElementById("tipB2"), robust(G.answers), colB, null, lab, ["", "", ""], shared);
    legend(document.getElementById("legB"), kinds.map((k, i) => [SLOTS[i % SLOTS.length], k]));
    document.getElementById("statB").innerHTML =
      `${sG.model}, ${sG.shown} of ${sG.contexts} confirmation contexts. Left: classical scaling of its hidden states (layer ${sG.layer}; the three ` +
      `axes hold ${Math.round(sG.mds_variance_hidden * 100)}% of the variance; classical scaling places points so that their distances match). Right: the same contexts placed only from the model's yes/no answers ` +
      `to 88 questions (Fisher–Rao distances; ${Math.round(sG.mds_variance_answers * 100)}%), turned onto the left by a rotation and one scale, ` +
      `with no other fitting. Over all ${sG.contexts.toLocaleString()} contexts, the two distance matrices agree at Spearman <b>${sG.spearman_all_pairs}</b> ` +
      `(preregistered, G1). The geometry an outsider can read is the geometry models share; it is not unique to this model (G6). ` +
      `Display: each view is scaled by its 98th percentile; the few points beyond are drawn at the edge.`;
  });
})();
