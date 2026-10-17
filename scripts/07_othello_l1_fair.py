#!/usr/bin/env python
"""L1 fair comparison on the cached Othello states (scripts/05): identical decoders for every observation channel.

Decoders: linear and a one-hidden-layer MLP (512 units), trained on train games, scored on held-out games.
Metrics: occupied-square accuracy (mine vs theirs, the part that needs flip tracking), all-square accuracy,
exact-board rate.
Hidden-vector recovery: cross-validated R² of the top-64 residual principal components (layers 4 and 6) from
θ(Ψ) and from input-only features, with ridge and with the MLP.

python scripts/07_othello_l1_fair.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("/data/models/graybox_ws/outputs/othello")
RES = Path("/home/arya/graybox/results/othello")
torch.set_num_threads(32)


def standardize(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    return (torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32))


def make(kind, din, dout):
    if kind == "linear":
        return nn.Linear(din, dout)
    return nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, dout))


def fit(model, Xtr, lossfn, epochs, wd, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(Xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            loss = lossfn(model(Xtr[b]), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


def decode_board(kind, Xtr, Ytr, Xte, Yte, epochs=40, wd=1e-2):
    Xtr, Xte = standardize(Xtr, Xte)
    Ytr_t, Yte_t = torch.tensor(Ytr), torch.tensor(Yte)
    m = fit(make(kind, Xtr.shape[1], 192), Xtr,
            lambda out, b: F.cross_entropy(out.view(-1, 64, 3).permute(0, 2, 1), Ytr_t[b]), epochs, wd)
    with torch.no_grad():
        pred = m(Xte).view(-1, 64, 3).argmax(-1)
    ok = (pred == Yte_t)
    occ = Yte_t > 0
    return {"occupied_acc": round(float(ok[occ].float().mean()), 4), "square_acc": round(float(ok.float().mean()), 4),
            "board_exact": round(float(ok.all(1).float().mean()), 4)}


def recover_hidden(kind, Xtr, Htr, Xte, Hte, epochs=40, wd=1e-2):
    Xtr, Xte = standardize(Xtr, Xte)
    Htr_t, Hte_t = torch.tensor(Htr, dtype=torch.float32), torch.tensor(Hte, dtype=torch.float32)
    m = fit(make(kind, Xtr.shape[1], Htr.shape[1]), Xtr, lambda out, b: F.mse_loss(out, Htr_t[b]), epochs, wd)
    with torch.no_grad():
        P = m(Xte)
    r2 = 1 - ((Hte_t - P) ** 2).sum(0) / ((Hte_t - Hte_t.mean(0)) ** 2).sum(0)
    return r2.numpy()


def main():
    Z = dict(np.load(OUT / "l1_g3000_p10_s0.npz"))
    games = np.unique(Z["game"])
    rng = np.random.default_rng(1)
    test_games = set(rng.choice(games, int(len(games) * 0.2), replace=False).tolist())
    te = np.array([g in test_games for g in Z["game"]])
    tr = ~te
    Y = Z["board"]
    fr = lambda p: 2 * np.sqrt(np.clip(p, 0, 1))  # noqa: E731
    feats = {"greybox P0 (1 query)": fr(Z["P0"]),
             "greybox P1 (61 queries)": np.hstack([fr(Z["P0"]), fr(Z["P1"].reshape(len(Y), -1))]),
             "input-only (who placed each square)": Z["hist"],
             "white-box L4": Z["resid4"].astype(np.float32), "white-box L6": Z["resid6"].astype(np.float32),
             "white-box L8 (final)": Z["resid8"].astype(np.float32)}
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "board": {}, "hidden_recovery": {}}
    for kind in ("linear", "mlp"):
        for k, X in feats.items():
            t0 = time.time()
            r = decode_board(kind, X[tr], Y[tr], X[te], Y[te])
            res["board"][f"{kind} | {k}"] = r
            print(f"board  {kind:6s} {k:38s} {r} ({time.time()-t0:.0f}s)", flush=True)
    # hidden-vector recovery: top-64 PCs of the residual stream
    for layer in (4, 6, 8):
        H = Z[f"resid{layer}"].astype(np.float32)
        mu = H[tr].mean(0)
        U, S, Vt = np.linalg.svd(H[tr] - mu, full_matrices=False)
        V = Vt[:64].T
        w = S[:64] ** 2 / (S ** 2).sum()
        Ptr, Pte = (H[tr] - mu) @ V, (H[te] - mu) @ V
        for kind in ("linear", "mlp"):
            for k in ("greybox P1 (61 queries)", "input-only (who placed each square)"):
                r2 = recover_hidden(kind, feats[k][tr], Ptr, feats[k][te], Pte)
                key = f"L{layer} | {kind} | {k}"
                res["hidden_recovery"][key] = {"var_in_top64": round(float(w.sum()), 4),
                                               "R2_weighted": round(float((r2 * w).sum() / w.sum()), 4),
                                               "R2_pc1to5": np.round(r2[:5], 3).tolist()}
                print("hidden", key, res["hidden_recovery"][key], flush=True)
    (RES / "l1_fair.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    sys.exit(main())
