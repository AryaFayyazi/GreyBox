#!/usr/bin/env python
"""L1 of the ground-truth ladder: recover Othello-GPT's board (its known latent world model) from black-box queries.

Black-box observables at a game prefix x (moves so far):
  P0   next-move distribution            (1 query; the logprob tier)
  P1   next-move distribution after appending each square token s, for all 60 s (a FIXED, state-independent battery)
  P1_n the same estimated from n sampled moves per query (the text-only tier)
Features use the categorical Fisher-Rao embedding 2*sqrt(p).
Ground truth: the true board (mine / theirs / empty relative to the side to move; Nanda 2023).
White box: linear probes on the residual stream (Li et al. 2023; Nanda 2023), layer by layer.
Input-only baseline: move-history features (which squares were played, and by whom).
All decoders are the same linear multinomial model, trained on train games and scored on held-out games.

python scripts/05_othello_l1.py n_games=3000 per_game=10
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.othello import SQ, TOK, OthelloGPT, board_to_relative, boards_along, random_game  # noqa: E402

OUT = Path("/data/models/graybox_ws/outputs/othello")
RES = Path("/home/arya/graybox/results/othello")
D = OmegaConf.create({"n_games": 3000, "per_game": 10, "test_frac": 0.2, "seed": 0, "epochs": 60, "wd": 1e-3,
                      "threads": 32, "stage": "all"})


def collect(cfg, m):
    rng = np.random.default_rng(cfg.seed)
    rows = {k: [] for k in ("game", "t", "board", "P0", "P1", "hist", "legal")}
    resid = {l: [] for l in range(9)}
    t0 = time.time()
    for gi in range(cfg.n_games):
        g = random_game(rng)[:59]
        if len(g) < 12:
            continue
        bs = boards_along(g)
        ts = np.sort(rng.choice(np.arange(len(g) - 1), size=min(cfg.per_game, len(g) - 1), replace=False))
        ids = torch.tensor([[TOK[s] for s in g]])
        with torch.no_grad():
            lg, rs, kv = m.forward_cache(ids)
            pos = np.repeat(ts, 60)
            tok = np.tile(np.arange(1, 61), len(ts))
            ext = torch.softmax(m.extend(kv, pos, tok)[:, 1:], -1).view(len(ts), 60, 60).numpy()
            p0 = torch.softmax(lg[0, ts, 1:], -1).numpy()
        for j, t in enumerate(ts):
            b, player, legal = bs[t]
            rows["game"].append(gi)
            rows["t"].append(int(t))
            rows["board"].append(board_to_relative(b, player))
            rows["P0"].append(p0[j].astype(np.float32))
            rows["P1"].append(ext[j].astype(np.float32))
            h = np.zeros(128, np.float32)          # history: square placed by the side to move / by the opponent
            for k, mv in enumerate(g[: t + 1]):
                colour = bs[k][0][mv]              # the stone just placed has the mover's colour
                h[mv + (0 if colour == player else 64)] = 1.0
            rows["hist"].append(h)
            rows["legal"].append(np.isin(np.arange(64), legal).astype(np.int8))
            for l in range(9):
                resid[l].append(rs[l][0, t].numpy().astype(np.float16))
        if gi % 500 == 0:
            print(f"{gi}/{cfg.n_games} games, {len(rows['t'])} states, {time.time()-t0:.0f}s", flush=True)
    out = {k: np.array(v) for k, v in rows.items()}
    out.update({f"resid{l}": np.array(v) for l, v in resid.items()})
    return out


def linear_decode(Xtr, Ytr, Xte, Yte, epochs, wd, seed=0):
    """Multinomial linear decoder for 64 squares x 3 classes; standardized features; Adam; returns test metrics."""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    Ytr, Yte = torch.tensor(Ytr), torch.tensor(Yte)
    W = torch.nn.Linear(Xtr.shape[1], 64 * 3)
    opt = torch.optim.AdamW(W.parameters(), lr=3e-3, weight_decay=wd)
    n = len(Xtr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 512):
            b = perm[i:i + 512]
            loss = F.cross_entropy(W(Xtr[b]).view(-1, 64, 3).permute(0, 2, 1), Ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = W(Xte).view(-1, 64, 3).argmax(-1)
    acc_sq = (pred == Yte).float()
    return {"square_acc": round(float(acc_sq.mean()), 4), "board_exact": round(float(acc_sq.all(1).float().mean()), 4),
            "acc_by_class": {c: round(float(acc_sq[Yte == i].mean()), 4) for i, c in enumerate(["empty", "mine", "theirs"])}}


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    torch.set_num_threads(cfg.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    cache = OUT / f"l1_g{cfg.n_games}_p{cfg.per_game}_s{cfg.seed}.npz"
    if cache.exists():
        Z = dict(np.load(cache))
    else:
        Z = collect(cfg, OthelloGPT.load())
        np.savez_compressed(cache, **Z)
    games = np.unique(Z["game"])
    rng = np.random.default_rng(cfg.seed + 1)
    test_games = set(rng.choice(games, int(len(games) * cfg.test_frac), replace=False).tolist())
    te = np.array([g in test_games for g in Z["game"]])
    tr = ~te
    Y = Z["board"]
    fr = lambda p: 2 * np.sqrt(np.clip(p, 0, 1))  # noqa: E731  categorical Fisher-Rao embedding
    feats = {"P0 (1 query)": fr(Z["P0"]), "P1 (61 queries)": np.hstack([fr(Z["P0"]), fr(Z["P1"].reshape(len(Y), -1))]),
             "input-only: who placed each square": Z["hist"], "input-only: move number": np.eye(60)[Z["t"]]}
    srng = np.random.default_rng(7)
    for n in (1, 8, 64):
        smp = lambda p: srng.multinomial(n, p / p.sum(-1, keepdims=True)) / n  # noqa: E731
        P0s = np.stack([smp(p) for p in Z["P0"].astype(np.float64)])
        P1s = np.stack([np.stack([smp(q) for q in p]) for p in Z["P1"].astype(np.float64)])
        feats[f"P1 sampled n={n}"] = np.hstack([fr(P0s), fr(P1s.reshape(len(Y), -1))])
    for l in range(9):
        feats[f"white-box resid L{l}"] = Z[f"resid{l}"].astype(np.float32)
    res = {"n_states": int(len(Y)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
           "majority_square_acc": round(float((Y[te] == np.array([np.bincount(Y[tr][:, s], minlength=3).argmax()
                                                                      for s in range(64)])).mean()), 4),
           "decode": {}}
    for k, X in feats.items():
        t0 = time.time()
        res["decode"][k] = linear_decode(X[tr], Y[tr], X[te], Y[te], cfg.epochs, cfg.wd)
        print(f"{k:32s} {res['decode'][k]}  ({time.time()-t0:.0f}s)", flush=True)
    (RES / "l1_decode.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
