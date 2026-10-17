#!/usr/bin/env python
"""L0 of the ground-truth ladder: a program compiled into transformer weights (tracr) as the black box.

Program (causal shuffle-Dyck-2): at every position, output 1 iff the prefix so far is balanced in both bracket types
and never went negative. Its hidden state is known exactly: the running balances of "()" and "{}" (stored in the
residual stream as balance / length) and the "ever negative" flag. Myhill-Nerode: two prefixes that never went
negative are behaviourally equivalent iff their balance vectors are equal, so the minimal state is (b_(), b_{}).

Black box: f(sequence) = the model's output at the last position. Battery: every bracket string of length 1..L.
We test:
  (1) exact recovery of the minimal state from Ψ (the battery answers), by a linear map;
  (2) linear recovery of the residual stream at the prefix's last position: which coordinates are behaviourally
      visible, and which are not;
  (3) the gauge theorem: rotate the residual stream by a random orthogonal G. Every output is unchanged, so Ψ and
      everything decoded from it are unchanged, while a white-box readout that uses the known layout breaks.

Run with the tracr venv:  /data/models/graybox_ws/venv_tracr/bin/python scripts/06_tracr_l0.py
"""
import itertools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tracr.compiler import compiling
from tracr.compiler.lib import make_frac_prevs
from tracr.rasp import rasp

RES = Path("/home/arya/graybox/results/tracr")
MAXLEN, PREFIX_MAX, SUFFIX_MAX, N_BATTERY = 14, 5, 8, 256  # prefix+suffix <= 13 < MAXLEN
VOCAB = ["(", ")", "{", "}"]
POOL_EXH = 5
MLP_EXACT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # tracr numeric-MLP discretisation (100 is inexact at MAXLEN 14)


def pair_balance(open_t, close_t, name):
    o = rasp.numerical(make_frac_prevs(rasp.numerical(rasp.tokens == open_t).named(f"is_{name}_open"))).named(f"opens_{name}")
    c = rasp.numerical(make_frac_prevs(rasp.numerical(rasp.tokens == close_t).named(f"is_{name}_close"))).named(f"closes_{name}")
    return rasp.numerical(rasp.LinearSequenceMap(o, c, 1, -1)).named(f"balance_{name}")


def program():
    b1, b2 = pair_balance("(", ")", "round"), pair_balance("{", "}", "curly")
    neg = rasp.numerical(rasp.Map(lambda x: x, (b1 < 0) | (b2 < 0))).named("neg_now")
    prev = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ).named("prev")
    ever_neg = rasp.numerical(rasp.Aggregate(prev, neg, default=0)).named("ever_neg")
    zero = ((b1 == 0) & (b2 == 0)).named("all_zero")
    ok = (~(ever_neg > 0)).named("never_neg")
    return (zero & ok).named("balanced_so_far")


def reference(seq):
    """Pure-Python ground truth: per-position output and the minimal state after the full sequence."""
    b1 = b2 = 0
    neg = False
    for ch in seq:
        b1 += (ch == "(") - (ch == ")")
        b2 += (ch == "{") - (ch == "}")
        neg |= b1 < 0 or b2 < 0
    return int(b1 == 0 and b2 == 0 and not neg), (b1, b2, int(neg))


def rotate_params(params, G):
    """Apply residual-stream change of basis x -> x G (G orthogonal): writes get @G, reads get G^T@."""
    out = {}
    for mod, p in params.items():
        p = dict(p)
        if mod in ("token_embed", "pos_embed"):
            p["embeddings"] = p["embeddings"] @ G
        elif any(mod.endswith(s) for s in ("query", "key", "value")) or mod.endswith("mlp/linear_1"):
            p["w"] = G.T @ p["w"]
        elif mod.endswith("attn/linear") or mod.endswith("mlp/linear_2"):
            p["w"] = p["w"] @ G
            if "b" in p:
                p["b"] = p["b"] @ G
        out[mod] = p
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    model = compiling.compile_rasp_to_model(program(), vocab=set(VOCAB), max_seq_len=MAXLEN, causal=True,
                                            compiler_bos="BOS", mlp_exactness=MLP_EXACT)
    print("residual dim", len(model.residual_labels), "param modules", list(model.params.keys()), flush=True)

    labs = model.residual_labels
    out_idx = [labs.index("balanced_so_far_1:False"), labs.index("balanced_so_far_1:True")]
    U = np.zeros((len(labs), 2))
    U[out_idx, [0, 1]] = 1.0                       # tracr's unembedding: projection onto the output directions
    I = np.eye(len(labs))

    def run(params, seq, basis=I):
        """Black-box call. basis = G for the rotated model, whose unembedding reads G^T-rotated residuals."""
        tokens = model.input_encoder.encode(["BOS"] + list(seq))
        out = model.forward(params, jnp.array([tokens]))
        h = np.asarray(out.transformer_output.residuals[-1][0])
        return (h @ basis.T @ U).argmax(-1), h        # (per-position output in {0,1}, final residual stream)

    # 1) the compiled model reproduces the program exactly on every sequence up to MAXLEN-1
    allseq = [s for n in range(1, MAXLEN) for s in itertools.product(VOCAB, repeat=n)] if MAXLEN <= 8 else \
        [tuple(np.random.default_rng(i).choice(VOCAB, size=np.random.default_rng(i).integers(1, MAXLEN))) for i in range(3000)]
    errs = 0
    for s in allseq[:3000]:
        dec, _ = run(model.params, s)
        errs += int(dec[-1]) != reference(s)[0]
        if len(s) <= 4:
            errs += int(dec[-1]) != int(model.apply(["BOS"] + list(s)).decoded[-1])   # our unembedding == tracr's
    print(f"compiled model vs program: {errs} mismatches on {min(3000, len(allseq))} sequences", flush=True)

    # 2) prefixes, battery, hidden states (batched by sequence length under jit)
    prefixes = [s for n in range(1, PREFIX_MAX + 1) for s in itertools.product(VOCAB, repeat=n)]
    # fixed generic battery: N_BATTERY random bracket strings, lengths uniform in 1..SUFFIX_MAX (seeded, state-independent)
    brng = np.random.default_rng(123)
    battery = [tuple(brng.choice(VOCAB, size=int(brng.integers(1, SUFFIX_MAX + 1)))) for _ in range(N_BATTERY)]
    G = np.linalg.qr(np.random.default_rng(0).normal(size=(len(model.residual_labels),) * 2))[0]
    rparams = rotate_params(model.params, jnp.array(G))
    fwd = jax.jit(lambda prm, toks: model.forward(prm, toks).transformer_output.residuals[-1])

    def batch_last(params, seqs, basis):
        """Final residual at the last position and the black-box output there, for many sequences."""
        out_h, out_y = [None] * len(seqs), [None] * len(seqs)
        by_len = {}
        for i, sq in enumerate(seqs):
            by_len.setdefault(len(sq), []).append(i)
        for n, idx in by_len.items():
            for c in range(0, len(idx), 4096):
                chunk = idx[c:c + 4096]
                toks = jnp.array([model.input_encoder.encode(["BOS"] + list(seqs[i])) for i in chunk])
                h = np.asarray(fwd(params, toks))[:, -1]
                y = (h @ basis.T @ U).argmax(-1)
                for j, i in enumerate(chunk):
                    out_h[i], out_y[i] = h[j], int(y[j])
        return np.array(out_h), np.array(out_y)

    H, _ = batch_last(model.params, prefixes, I)
    Hrot, _ = batch_last(rparams, prefixes, G)
    queries = [p + q for p in prefixes for q in battery]
    _, y = batch_last(model.params, queries, I)
    _, yr = batch_last(rparams, queries, G)
    psi, psi_rot = y.reshape(len(prefixes), len(battery)), yr.reshape(len(prefixes), len(battery))
    state = np.array([reference(p)[1] for p in prefixes])
    ref_y = np.array([reference(qq)[0] for qq in queries]).reshape(psi.shape)
    res = {"battery_answers_vs_program_mismatches": int((psi != ref_y).sum()), "n_prefixes": len(prefixes), "n_battery": len(battery), "residual_dim": int(H.shape[1]),
           "compiled_vs_program_mismatches": errs,
           "gauge_outputs_identical": bool((psi == psi_rot).all()),
           "max_abs_hidden_change_under_rotation": float(np.abs(H - Hrot).max())}
    # Myhill-Nerode check: distinct Ψ classes among never-negative prefixes == distinct balance vectors
    nn = state[:, 2] == 0
    res["n_distinct_minimal_states"] = int(len({tuple(x) for x in state[nn, :2]}))
    res["n_distinct_psi_classes"] = int(len({tuple(x) for x in psi[nn]}))
    res["psi_class_determines_state"] = bool(all(len({tuple(state[nn][i, :2]) for i in np.flatnonzero((psi[nn] == v).all(1))}) == 1
                                                 for v in np.unique(psi[nn], axis=0)))
    # linear recovery with an intercept (held-out prefixes: random 30%)
    rng = np.random.default_rng(1)
    te = rng.random(len(prefixes)) < 0.3

    def ridge_r2(X, Y, alpha=1e-3):
        X1 = np.hstack([X, np.ones((len(X), 1))])
        W = np.linalg.solve(X1[~te].T @ X1[~te] + alpha * np.eye(X1.shape[1]), X1[~te].T @ Y[~te])
        P = X1[te] @ W
        ss = ((Y[te] - Y[te].mean(0)) ** 2).sum(0)
        return np.where(ss > 1e-12, 1 - ((Y[te] - P) ** 2).sum(0) / np.maximum(ss, 1e-12), np.nan)

    lengths = np.array([len(p) for p in prefixes], float)
    Y_state = np.column_stack([state[:, 0], state[:, 1], state[:, 2]]).astype(float)
    res["R2_minimal_state_from_psi"] = dict(zip(["balance_round", "balance_curly", "ever_negative"],
                                                np.round(ridge_r2(psi.astype(float), Y_state), 4).tolist()))
    # never-negative prefixes: the balance vector is the Myhill-Nerode state; recover it on held-out prefixes
    te_all = te.copy()
    te = te_all[nn]
    Xn, Yn = psi[nn].astype(float), Y_state[nn][:, :2]
    res["R2_balance_from_psi_never_negative_linear"] = np.round(ridge_r2(Xn, Yn), 4).tolist()
    tr_classes = {}
    for x, yv in zip(map(tuple, Xn[~te]), map(tuple, Yn[~te])):
        tr_classes.setdefault(x, set()).add(yv)
    look = [tr_classes.get(tuple(x)) for x in Xn[te]]
    res["exact_state_by_psi_lookup_heldout"] = round(float(np.mean([l is not None and len(l) == 1 and tuple(y) in l
                                                                   for l, y in zip(look, Yn[te])])), 4)
    # negative prefixes: all behaviourally equivalent (output 0 forever), so their balance is inert
    res["negative_prefixes_single_psi_class"] = bool(len({tuple(x) for x in psi[~nn]}) == 1)
    te = te_all
    # residual stream, by labelled component (behaviourally relevant vs not)
    r2 = ridge_r2(np.hstack([psi, np.eye(PREFIX_MAX + 1)[lengths.astype(int)]]).astype(float), H)
    groups = {}
    for lab, v in zip(labs, r2):
        key = lab.split(":")[0]
        groups.setdefault(key, []).append(v)
    res["R2_residual_by_component_from_psi_and_length"] = {k: round(float(np.nanmean(v)), 4) for k, v in groups.items()}
    # white-box readout with the known layout, before and after the gauge rotation
    idx = [i for i, l in enumerate(labs) if l.startswith("balance_round")]
    wb = H[:, idx].sum(1) * (lengths + 1)
    wb_rot = Hrot[:, idx].sum(1) * (lengths + 1)
    res["whitebox_known_layout_corr_with_true_balance"] = round(float(np.corrcoef(wb, state[:, 0])[0, 1]), 4)
    res["whitebox_known_layout_corr_after_rotation"] = round(float(np.corrcoef(wb_rot, state[:, 0])[0, 1]), 4)
    res["greybox_R2_after_rotation"] = dict(zip(["balance_round", "balance_curly", "ever_negative"],
                                                np.round(ridge_r2(psi_rot.astype(float), Y_state), 4).tolist()))
    # 4) designed battery, black-box only: greedily pick tests from a candidate pool that maximise the number of
    #    distinct answer patterns on TRAINING prefixes (no internals used), then evaluate on held-out prefixes.
    crng = np.random.default_rng(321)
    # candidate pool: exhaustive up to length POOL_EXH (contains Myhill-Nerode witnesses for balances <= POOL_EXH)
    # plus random longer strings; still generic: no knowledge of the program is used
    pool = sorted({s for n in range(1, POOL_EXH + 1) for s in itertools.product(VOCAB, repeat=n)} |
                  {tuple(crng.choice(VOCAB, size=int(crng.integers(1, SUFFIX_MAX + 1)))) for _ in range(2000)})
    _, yp = batch_last(model.params, [p + q for p in prefixes for q in pool], I)
    A = yp.reshape(len(prefixes), len(pool))
    # selection uses ONLY black-box answers (no state labels), so it may see every prefix without leakage;
    # only the state decoder below is fit on train prefixes and scored on held-out ones
    trn = np.arange(len(prefixes))
    res["pool_separates_all_states"] = bool(all(
        len({tuple(state[nn][i, :2]) for i in np.flatnonzero((A[nn] == v).all(1))}) == 1 for v in np.unique(A[nn], axis=0)))
    res["pool_n_classes_never_negative"] = int(len({tuple(x) for x in A[nn]}))
    chosen = []
    cid = np.zeros(len(prefixes), np.int64)                 # current behavioural class id of every prefix
    for _ in range(64):
        M = np.sort(cid[trn, None] * 2 + A[trn], axis=0)    # (n_train, n_pool): refined class keys per candidate
        n_cls = 1 + (np.diff(M, axis=0) != 0).sum(0)
        n_cls[chosen] = -1
        best = int(n_cls.argmax())
        if n_cls[best] <= len(np.unique(cid[trn])):
            break
        chosen.append(best)
        _, cid = np.unique(cid * 2 + A[:, best], return_inverse=True)
    D = A[:, chosen]
    res["designed_pool_size"] = len(pool)
    res["designed_battery_size"] = len(chosen)
    res["designed_battery_n_classes_train"] = int(len(set(map(tuple, D[trn]))))
    Dn = D[nn].astype(float)
    te_n = te[nn]
    cls = {}
    for x, yv in zip(map(tuple, Dn[~te_n]), map(tuple, Y_state[nn][~te_n][:, :2])):
        cls.setdefault(x, set()).add(yv)
    look = [cls.get(tuple(x)) for x in Dn[te_n]]
    res["designed_exact_state_heldout"] = round(float(np.mean([l is not None and len(l) == 1 and tuple(y) in l
                                                              for l, y in zip(look, Y_state[nn][te_n][:, :2])])), 4)
    res["designed_psi_class_determines_state"] = bool(all(
        len({tuple(Y_state[nn][i, :2]) for i in np.flatnonzero((D[nn] == v).all(1))}) == 1 for v in np.unique(D[nn], axis=0)))
    res["designed_n_classes_never_negative"] = int(len({tuple(x) for x in D[nn]}))
    te_save = te
    te = te_n
    res["designed_R2_balance_linear"] = np.round(ridge_r2(Dn, Y_state[nn][:, :2]), 4).tolist()
    te = te_save
    res["designed_battery"] = ["".join(pool[j]) for j in chosen]
    (RES / "l0_dyck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
