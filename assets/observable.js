// The live observable state of a proposed call (no hidden states: a closed API never returns them; paper, Proposition 1).
// The agent answers a fixed battery of 8 yes/no questions about its own proposed call, each in a side branch; each
// answer moves the point. Coordinates: theta_j = 2 arcsin sqrt(p_j), p_j = (yes + 3/8)/(answers + 3/4) (paper, Theorem 1a, d),
// projected on the fixed 3D basis of the reference map (data/observable_ref.json, web/build_observable_ref.py).
(() => {
  const $ = (id) => document.getElementById(id);
  if (!$("cvO")) return;
  fetch("data/observable_ref.json").then((r) => r.json()).then((R) => {
    const G = window.GB3D, S = window.GBSelf;
    const scale = G.robustScale(R.points);
    const v = G.view($("cvO"), $("tipO"), G.robust(R.points, scale), R.viol.map((x) => (x ? G.VIOL : G.SAFE)), null,
      (i) => `${R.template[i]} · checker: ${R.viol[i] ? "violation" : "allowed"} (Gemma-4 reference)`, ["toward violation →", "axis 2", "axis 3"]);
    const live = v.marker(G.INK);
    G.legend($("legO"), [[G.VIOL, "reference: violation"], [G.SAFE, "reference: allowed"], [G.INK, "your call (live), sphere = ±1 standard error"]]);
    const st = R.stats;
    $("obsStat").innerHTML = `Reference: ${st.calls} recorded calls of ${st.model} with the checker's labels, ${st.answers_per_question} answers ` +
      `per question. The three axes hold ${Math.round(st.variance_3d * 100)}% of the variance of the 8 coordinates; along the first axis, ` +
      `violations and allowed calls separate with AUROC ${st.auroc_axis1} (the single "does it comply?" question alone: ${st.auroc_q0}). ` +
      `Your model is placed in the same map; that one model's answer geometry is shared with others is our finding G6, so treat ` +
      `the position as indicative for a model other than the reference.`;
    const project = (k, m) => {
      const th = k.map((kk, j) => 2 * Math.asin(Math.sqrt((kk + 3 / 8) / (m[j] + 3 / 4))));
      const x = R.basis.map((w) => w.reduce((acc, wj, j) => acc + wj * (th[j] - R.mean[j]), 0));
      const se = Math.sqrt(R.basis.map((w) => w.reduce((acc, wj, j) => acc + (wj * wj) / (m[j] + 0.5), 0)).reduce((a, b) => a + b, 0) / 3);
      return [x.map((c) => Math.max(-1.25, Math.min(1.25, c / scale))), se / scale];
    };
    let busy = false;
    $("run").addEventListener("click", async () => {
      if (!$("obs").checked || busy) return;
      const name = $("tool").value.trim(); let args = $("args").value.trim() || "{}";
      try { args = JSON.stringify(JSON.parse(args)); } catch (e) { return; }
      if (!name) return;
      busy = true; live.hide();
      const n = Math.max(1, Math.min(8, +$("n").value || 3)), key = $("key").value.trim(), model = $("model").value.trim();
      const k = Array(8).fill(0), m = Array(8).fill(0), turns = S.toTurns($("conv").value);
      let done = 0;
      const prog = () => { $("obsProg").textContent = `${done} of ${8 * n} answers in` + (m.every((x) => x > 0) ? "" : " (the point appears when every question has an answer)"); };
      prog();
      const one = async (j) => {
        const q = R.frame.replace("{name}", name).replace("{args}", args).replace("{q}", R.questions[j]);
        const get = () => ($("prov").value === "anthropic" ? S.askAnthropic(key, model, $("policy").value, turns, q, 1, 2048)
          : S.askOpenAI($("base").value.trim(), key, model, $("policy").value, turns, q, 1, 2048));
        for (let r = 0; r < n; r++) {
          try {
            const [a] = await get(); const vv = S.parseAnswer(a);
            if (vv !== null) { m[j] += 1; k[j] += vv ? 0 : 1; }
          } catch (e) { $("obsProg").textContent = `A call failed: ${e.message}`; }
          done += 1; prog();
          if (m.every((x) => x > 0)) { const [p, r_] = project(k, m); live.set(p, r_); }
        }
      };
      await Promise.all([0, 1, 2, 3, 4, 5, 6, 7].map(one));
      $("obsRead").innerHTML = R.labels.map((l, j) => `${l}: <b>${m[j] ? k[j] + "/" + m[j] : "–"}</b> yes`).join(" · ");
      busy = false;
    });
  }).catch(() => { $("obsStat").textContent = "The map's reference data is being built; it will appear here soon."; $("obs").disabled = true; });
})();
