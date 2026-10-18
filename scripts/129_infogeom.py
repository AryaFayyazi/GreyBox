#!/usr/bin/env python
"""G-series: the observable latent geometry of language models, training-free (docs/theory_infogeom.md;
docs/prereg_G_information_geometry.md). No model is trained or fine-tuned; every quantity is a distance, a rank
correlation, or a linear-algebra solve.

  stage=open      G1: 8 open chat models. Answer geometry (Fisher-Rao, Theorem 1) vs the hidden-state geometry (cosine,
                  invariant under the networks' orthogonal gauge), Spearman over all context pairs. Layer chosen on DEV
                  contexts only; endpoints on the CONFIRM split.            (CPU)
  stage=claude    G2: Claude Haiku 5.5's own answer geometry (24 questions, 8 samples, Theorem 5 estimator) on 192
                  confirm contexts: split-half reliability, and distinctness from surface-text geometry. No other model
                  is used to estimate it.                                    (CPU)
  stage=collect   G3: Qwen3-0.6B in float32 on model-generated text: centered log-probabilities projected on their
                  rank-d span, and the true normalized final states for held-out validation contexts.   (GPU)
  stage=solve     G3: solve b_i' S b_i = d for S (Theorem 6), recover the Gram matrix of held-out states; phase
                  transition in the number of contexts N.                                         (GPU)
python scripts/129_infogeom.py stage=...  [split=dev for a dry run of G1 on development contexts only]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
L2 = Path("/data/models/graybox_ws/outputs/l2")
OUT = Path("/data/models/graybox_ws/outputs/infogeom")
RES = ROOT / "results/t1"
MODELS = ["gemma3_4b", "gptoss20b", "llama31_8b", "mistral7b", "olmo3_7b", "phi4mini", "qwen25_7b", "qwen3_8b"]
D = OmegaConf.create({"stage": "open", "split": "confirm", "n_boot": 500, "n_samples": 8,
                      "exact_model": "Qwen/Qwen3-0.6B", "seed": 0, "n_seq": 9000, "seq_len": 128, "n_val": 2000, "basis_ctx": 3000})


# ---------------------------------------------------------------- geometry helpers
def theta(p):
    return 2 * np.arcsin(np.sqrt(np.clip(p, 0, 1)))


def theta_sampled(k, n):
    """Anscombe-corrected angle from k yes out of n (Theorem 5)."""
    return 2 * np.arcsin(np.sqrt((k + 3 / 8) / (n + 3 / 4)))


def sq_dist(X):
    G = X @ X.T
    nrm = np.diag(G)
    return np.maximum(nrm[:, None] + nrm[None, :] - 2 * G, 0)


def cos_dist(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return 1 - Xn @ Xn.T


def upper(M):
    return M[np.triu_indices(len(M), 1)]


def rho(A, B):
    from scipy.stats import rankdata
    a, b = rankdata(upper(A)), rankdata(upper(B))
    return float(np.corrcoef(a, b)[0, 1])


def p_yes(m):
    z = np.load(L2 / m / "psi.npz")
    # P(yes | yes or no); the project's rule (scripts/13, 16, 18) gives 1/2 where both masses vanish (2 cells of 38,400)
    return np.clip((z["p_yes"] + 1e-6) / (z["p_yes"] + z["p_no"] + 2e-6), 0, 1), list(z["items"])


def hidden(m):
    h = np.load(L2 / m / "hidden.npz")
    return h["h"].astype(np.float64), [int(x) for x in h["layers"]]


def contexts():
    return pd.read_parquet(L2 / "contexts.parquet")


def select_layer(m, dev_idx):
    """The layer whose cosine geometry best matches the exact answer geometry on 1,000 DEV contexts (seed 0)."""
    p, _ = p_yes(m)
    H, layers = hidden(m)
    Dq = np.sqrt(sq_dist(theta(p[dev_idx])))
    r = [rho(Dq, cos_dist(H[dev_idx, k])) for k in range(H.shape[1])]
    k = int(np.argmax(r))
    return k, layers[k], [round(x, 4) for x in r]


def boot(fn, n, n_boot, seed):
    """Bootstrap over contexts: resample n contexts with replacement; pairs of distinct draws only."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        s = np.unique(rng.integers(0, n, n))
        out.append(fn(s))
    return np.array(out)


def ci(a):
    return [round(float(np.mean(a)), 4), round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


# ---------------------------------------------------------------- G1: open models
def stage_open(cfg):
    C = contexts()
    dev = np.where(C.split.values == "dev")[0]
    rng = np.random.default_rng(0)
    dev_sel = rng.choice(dev, 1000, replace=False)
    if cfg.split == "dev":                                      # dry run: evaluate on other dev contexts only
        test = rng.choice(np.setdiff1d(dev, dev_sel), 800, replace=False)
    else:
        test = np.where(C.split.values == "confirm")[0]
    emb = np.load(L2 / "minilm.npy").astype(np.float64)
    res = {"split": cfg.split, "n_test": int(len(test)), "models": {}}
    for mi, m in enumerate(MODELS):
        k, layer, dev_r = select_layer(m, dev_sel)
        p, items = p_yes(m)
        H, _ = hidden(m)
        P, Ht, E = p[test], H[test, k], emb[test]
        Dh = cos_dist(Ht)
        Dq = np.sqrt(sq_dist(theta(P)))
        Dtext = cos_dist(E)
        Draw = np.sqrt(sq_dist(P))
        r2 = np.random.default_rng(129 + mi)
        n = cfg.n_samples
        ks = r2.binomial(n, P)
        Dn = np.sqrt(np.maximum(sq_dist(theta_sampled(ks, n)) - 2 * P.shape[1] / (n + 0.5), 0))
        subsets = {M: [r2.choice(P.shape[1], M, replace=False) for _ in range(5)] for M in (4, 8, 16, 32, 64)}
        conv = {M: float(np.mean([rho(np.sqrt(sq_dist(theta(P[:, s]))), Dh) for s in subsets[M]])) for M in subsets}
        conv[P.shape[1]] = rho(Dq, Dh)
        r = {"layer": layer, "dev_layer_rho": dev_r, "rho_answers": round(rho(Dq, Dh), 4), "rho_text": round(rho(Dtext, Dh), 4),
             "rho_raw_p": round(rho(Draw, Dh), 4), f"rho_sampled_n{n}": round(rho(Dn, Dh), 4),
             "convergence": {str(M): round(v, 4) for M, v in sorted(conv.items())}}
        sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731
        b1 = boot(lambda s: rho(sub(Dq, s), sub(Dh, s)) - rho(sub(Dtext, s), sub(Dh, s)), len(test), cfg.n_boot, 1291 + mi)
        b2 = boot(lambda s: rho(sub(Dn, s), sub(Dh, s)) / rho(sub(Dq, s), sub(Dh, s)), len(test), cfg.n_boot, 1301 + mi)
        b3 = boot(lambda s: rho(sub(Dq, s), sub(Dh, s)) - float(np.mean([rho(np.sqrt(sq_dist(theta(P[np.ix_(s, q)]))), sub(Dh, s))
                                                                           for q in subsets[8]])), len(test), cfg.n_boot, 1311 + mi)
        r["F1_answers_minus_text"], r["F2_sampled_over_exact"], r["F3_full_minus_8"] = ci(b1), ci(b2), ci(b3)
        res["models"][m] = r
        print(m, json.dumps({k_: r[k_] for k_ in ("layer", "rho_answers", "rho_text", f"rho_sampled_n{n}", "F1_answers_minus_text")}), flush=True)
    ms = res["models"].values()
    res["F1_pass"] = all(x["F1_answers_minus_text"][1] > 0 for x in ms)
    res["F2_pass"] = sum(x["F2_sampled_over_exact"][1] > 0.95 for x in ms) >= 7
    res["F3_pass"] = sum(x["F3_full_minus_8"][1] > 0 for x in ms) >= 7
    name = "g1_open_geometry.json" if cfg.split == "confirm" else "g1_open_geometry_DRYRUN_dev.json"
    (RES / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: res[k_] for k_ in ("F1_pass", "F2_pass", "F3_pass")}))


# ---------------------------------------------------------------- G2: Claude (its own answers only)
def claude_counts():
    """Claude's sampled answers: counts (yes, no, other) of 8 samples per context and question."""
    d = json.loads(Path("/data/models/graybox_ws/outputs/l3/claude_haiku55_run.json").read_text())
    return d, np.array([r["counts"] for r in d["results"]], int)


def angle_dist(k, n):
    """Debiased Fisher-Rao distances (Theorem 5) from k yes of n valid answers per (context, question)."""
    th = theta_sampled(k, np.maximum(n, 1))
    var = 1 / (np.maximum(n, 1) + 0.5)
    D2 = sq_dist(th) - (var.sum(1)[:, None] + var.sum(1)[None, :])
    D = np.sqrt(np.maximum(D2, 0))
    np.fill_diagonal(D, 0)
    return D


def stage_claude(cfg):
    C = contexts()
    d, cnt = claude_counts()
    ids = [r["id"] for r in d["results"]]
    pos = C.reset_index().set_index("id").loc[ids, "index"].values
    items = d["items"]
    k, n = cnt[:, :, 0], cnt[:, :, 0] + cnt[:, :, 1]
    Dc = angle_dist(k, n)
    # split-half: the 8 samples of each (context, question) are exchangeable; rebuild them from the counts, permute
    # them (seed 2290) and give the first 4 to half A, the last 4 to half B
    rng = np.random.default_rng(2290)
    lab = np.zeros(cnt.shape[:2] + (8,), int)                          # 0 yes, 1 no, 2 other
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            s_ = np.repeat([0, 1, 2], cnt[i, j])
            lab[i, j] = rng.permutation(np.pad(s_, (0, 8 - len(s_)), constant_values=2))
    A_, B_ = lab[:, :, :4], lab[:, :, 4:]
    kA, nA = (A_ == 0).sum(2), (A_ < 2).sum(2)
    kB, nB = (B_ == 0).sum(2), (B_ < 2).sum(2)
    DA, DB = angle_dist(kA, nA), angle_dist(kB, nB)
    emb = np.load(L2 / "minilm.npy").astype(np.float64)[pos]
    Dtext = cos_dist(emb)
    sub = lambda A, s: A[np.ix_(s, s)]  # noqa: E731
    bR = boot(lambda s: rho(sub(DA, s), sub(DB, s)), len(ids), cfg.n_boot, 2291)
    bT = boot(lambda s: rho(sub(DA, s), sub(DB, s)) - 0.5 * (rho(sub(DA, s), sub(Dtext, s)) + rho(sub(DB, s), sub(Dtext, s))), len(ids), cfg.n_boot, 2292)
    res = {"n_contexts": len(ids), "questions": len(items), "samples": 8,
           "C1_split_half_rho": ci(bR), "C2_split_half_minus_text": ci(bT),
           "rho_claude_full_vs_text": round(rho(Dc, Dtext), 4)}
    res["C1_pass"] = res["C1_split_half_rho"][1] > 0.7
    res["C2_pass"] = res["C2_split_half_minus_text"][1] > 0
    # reported only, not an estimator: Claude's answer geometry vs open models' hidden geometry on the same contexts
    C_ = contexts()
    dev = np.where(C_.split.values == "dev")[0]
    dev_sel = np.random.default_rng(0).choice(dev, 1000, replace=False)
    rep = {}
    for m in MODELS:
        kk, _, _ = select_layer(m, dev_sel)
        H, _ = hidden(m)
        rep[m] = round(rho(Dc, cos_dist(H[pos, kk])), 4)
    res["reported_only_claude_answers_vs_open_hidden"] = rep
    # MDS coordinates of Claude's observable geometry (Theorem 5: exact up to a rigid motion), for the figure
    J = np.eye(len(ids)) - 1 / len(ids)
    Bm = -0.5 * J @ (Dc ** 2) @ J
    ev, V = np.linalg.eigh(Bm)
    o = np.argsort(ev)[::-1]
    ev, V = ev[o], V[:, o]
    res["mds_eigenvalues_top10"] = [round(float(x), 4) for x in ev[:10]]
    res["mds_negative_mass_share"] = round(float(-ev[ev < 0].sum() / np.abs(ev).sum()), 4)
    np.savez(OUT / "claude_mds.npz", X=V[:, :2] * np.sqrt(np.maximum(ev[:2], 0)), ids=np.array(ids),
             harmful=C_.harmful.values[pos], source=C_.source.values[pos])
    (RES / "g2_claude_geometry.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: res[k_] for k_ in ("C1_split_half_rho", "C1_pass", "C2_split_half_minus_text", "C2_pass")}, indent=1))


# ---------------------------------------------------------------- G3: exact recovery from log-probabilities
def load_exact(cfg):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = snapshot_download(cfg.exact_model, cache_dir="/data/models", local_files_only=True)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32, device_map="cuda").eval()
    return tok, model


def stage_collect(cfg):
    """Model-generated sequences (sampled from the model itself, temperature 1, seed cfg.seed); every position is a
    context. Stores b = Q_d' c (centered log-probabilities projected on their rank-d span; Q_d from the first basis_ctx
    contexts) for all contexts, and, for n_val held-out validation contexts only, the true normalized final state
    u = x / rms(x), x the input of the final RMSNorm (its gain gamma is folded into the output map)."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    tok, model = load_exact(cfg)
    out = OUT / f"seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    seqs = []
    for i in range(0, cfg.n_seq, 256):
        inp = torch.full((min(256, cfg.n_seq - i), 1), bos, device="cuda")
        with torch.no_grad():
            g = model.generate(inp, attention_mask=torch.ones_like(inp), do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                               max_new_tokens=cfg.seq_len, min_new_tokens=cfg.seq_len, pad_token_id=tok.eos_token_id)
        seqs.append(g[:, 1:1 + cfg.seq_len].cpu())
    S = torch.cat(seqs)
    print("generated", tuple(S.shape), flush=True)
    dmod, eps = model.config.hidden_size, model.model.norm.variance_epsilon
    cap = {}
    hook = model.model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0]))

    def run(batch):
        with torch.no_grad():
            lo = model(input_ids=batch.cuda()).logits.float()
            lp = torch.log_softmax(lo, -1)
            c = (lp - lp.mean(-1, keepdim=True)).reshape(-1, lp.shape[-1])
            x = cap["x"].float()
            u = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)).reshape(-1, dmod)
        return c, u
    nb = -(-cfg.basis_ctx // cfg.seq_len)
    Cb = torch.cat([run(S[i:i + 1])[0] for i in range(nb)])[: cfg.basis_ctx]
    _, sv, Vh = torch.linalg.svd(Cb, full_matrices=False)                 # Cb = U diag(sv) Vh;  Vh: (basis_ctx, V)
    Q = Vh[:dmod].T.contiguous()                                          # (V, d): the span of the centered log-probs
    np.save(out / "singular_values.npy", sv.cpu().numpy())
    del Cb, Vh
    n_total = S.shape[0] * cfg.seq_len
    val = np.zeros(n_total, bool)
    val[np.random.default_rng(cfg.seed + 1).choice(n_total, cfg.n_val, replace=False)] = True
    B = np.zeros((n_total, dmod), np.float32)
    Uv, resid = [], []
    for i in range(0, S.shape[0], 16):
        c, u = run(S[i:i + 16])
        bq = c @ Q
        lo, hi = i * cfg.seq_len, i * cfg.seq_len + c.shape[0]
        B[lo:hi] = bq.cpu().numpy()
        if i % 1024 == 0:                                                 # the rank-d model of c holds: relative residual
            resid.append(float(((c - bq @ Q.T).norm() / c.norm()).cpu()))
        m = val[lo:hi]
        if m.any():
            Uv.append(u[torch.tensor(m, device="cuda")].cpu().numpy())
    hook.remove()
    np.save(out / "B.npy", B)
    np.save(out / "U_val.npy", np.concatenate(Uv))
    np.save(out / "val_mask.npy", val)
    info = {"seed": cfg.seed, "contexts": int(n_total), "d": dmod, "vocab": int(Q.shape[0]), "validation": int(val.sum()),
            "rank_residual_max": max(resid), "sv_d_over_sv_d+1": float(sv[dmod - 1] / sv[dmod]) if len(sv) > dmod else None}
    (out / "collect_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


def solve_S(Bt, target, max_iter=5000, tol=1e-12):
    """Least squares for symmetric S in b_i' S b_i = target (all i), by conjugate gradient on the normal equations.
    The operator S -> sum_i (b_i' S b_i) b_i b_i' is symmetric positive semidefinite on symmetric matrices. CG from S = 0
    returns the minimum-norm solution when the system is underdetermined (N < d(d+1)/2)."""
    import torch
    d = Bt.shape[1]

    def A(Sm):
        q = ((Bt @ Sm) * Bt).sum(1)
        return Bt.T @ (Bt * q[:, None])
    rhs = (Bt.T * target) @ Bt
    S = torch.zeros(d, d, dtype=Bt.dtype, device=Bt.device)
    r = rhs.clone(); pdir = r.clone(); rs = (r * r).sum(); r0 = torch.sqrt(rs)
    it = 0
    for it in range(1, max_iter + 1):
        Ap = A(pdir)
        alpha = rs / (pdir * Ap).sum()
        S = S + alpha * pdir
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if torch.sqrt(rs_new) < tol * r0:
            break
        pdir = r + (rs_new / rs) * pdir
        rs = rs_new
    return (S + S.T) / 2, it, float(torch.sqrt(rs_new) / r0)


def cos_upper(G):
    n = np.sqrt(np.clip(np.diag(G), 1e-300, None))
    return upper(G / n[:, None] / n[None, :])


def stage_solve(cfg):
    """Theorem 6 on the held-out validation contexts. B is whitened on the training contexts first (any invertible map
    of B is allowed: U = K B for an unknown K), which conditions the CG. Recovered Gram B_v S B_v' vs the true U_v U_v'.
    Baselines without the sphere constraint: cosines of the raw centered log-probabilities, and of the whitened B."""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    out = OUT / f"seed{cfg.seed}"
    val = np.load(out / "val_mask.npy")
    Uv = np.load(out / "U_val.npy").astype(np.float64)
    B = np.load(out / "B.npy", mmap_mode="r")
    d = B.shape[1]
    need = d * (d + 1) // 2
    tr_idx = np.where(~val)[0]
    Bv = np.asarray(B[val], np.float64)
    true_cos = cos_upper(Uv @ Uv.T)
    res = {"seed": cfg.seed, "d": d, "unknowns": need, "n_train_available": int(len(tr_idx)), "n_validation": int(val.sum()),
           "norm_check_true_u": float(np.median(np.abs((Uv * Uv).sum(1) - d)) / d), "by_N": {}}
    res["baseline_raw_centered_logprob_pearson"] = round(float(np.corrcoef(cos_upper(Bv @ Bv.T), true_cos)[0, 1]), 6)
    rng = np.random.default_rng(cfg.seed + 3)
    fracs = [f for f in (0.25, 0.5, 0.75, 0.9, 1.1, 1.5, 2.0) if int(f * need) <= len(tr_idx)]
    for frac in fracs:
        N = int(frac * need)
        sel = np.sort(rng.choice(tr_idx, N, replace=False))
        Bt = torch.tensor(np.asarray(B[sel]), dtype=torch.float64, device="cuda")
        M = Bt.T @ Bt / N
        ev, V = torch.linalg.eigh(M)
        W = V @ torch.diag(ev.rsqrt()) @ V.T                              # whitening map (symmetric)
        Bt = Bt @ W
        Bvw = torch.tensor(Bv, device="cuda") @ W
        if frac == fracs[0]:
            res["baseline_whitened_no_constraint_pearson"] = round(float(np.corrcoef(cos_upper((Bvw @ Bvw.T).cpu().numpy()), true_cos)[0, 1]), 6)
        Sm, iters, rel = solve_S(Bt, float(d))
        G = (Bvw @ Sm @ Bvw.T).cpu().numpy()
        rec = cos_upper(G)
        res["by_N"][f"{frac}x"] = {"N": N, "cg_iterations": iters, "cg_rel_residual": rel,
                                   "pearson_cos": round(float(np.corrcoef(rec, true_cos)[0, 1]), 8),
                                   "median_abs_err_cos": float(np.median(np.abs(rec - true_cos))),
                                   "max_abs_err_cos": float(np.max(np.abs(rec - true_cos))),
                                   "norm_constraint_val_median_rel": float(np.median(np.abs(np.diag(G) - d)) / d)}
        print(frac, json.dumps(res["by_N"][f"{frac}x"]), flush=True)
        del Bt
        torch.cuda.empty_cache()
    full = res["by_N"][f"{fracs[-1]}x"]
    below = {k: v for k, v in res["by_N"].items() if float(k[:-1]) < 1}
    res["E1_pearson_cos_at_largest_N"] = full["pearson_cos"]
    res["E1_pass"] = full["pearson_cos"] > 0.99
    best_below = min(v["median_abs_err_cos"] for v in below.values())
    res["E2_err_below_best_vs_largest"] = [best_below, full["median_abs_err_cos"]]
    res["E2_pass"] = full["median_abs_err_cos"] < 0.1 * best_below
    base = max(res["baseline_raw_centered_logprob_pearson"], res["baseline_whitened_no_constraint_pearson"])
    res["E3_pass"] = full["pearson_cos"] > base
    name = "g3_exact_recovery.json" if cfg.seed == 0 else f"g3_exact_recovery_PILOT_seed{cfg.seed}.json"
    (RES / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k != "by_N"}, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    {"open": stage_open, "claude": stage_claude, "collect": stage_collect, "solve": stage_solve}[cfg.stage](cfg)


if __name__ == "__main__":
    main()
