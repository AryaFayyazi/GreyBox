#!/usr/bin/env python
"""T1 in a controlled setting: for a label defined by the model's own future behaviour, no probe beats rollouts.

Othello-GPT, fresh games (seed 77). At each sampled position x:
  Y      the model plays H=3 moves of self-play from x (sampling at T=1 from its own next-move distribution);
         Y = 1 if any of those moves is a corner (squares 0, 7, 56, 63). One draw per position (the realized label).
  R_K    fraction of K further independent rollouts (same rule) with a corner, K in {1, 2, 4, 8, 16, 64}
  WB     logistic probe on the true layer-6 hidden state at x (trained on train games)
  PSI    logistic on θ of the 61-query battery at x (trained on train games)
  WB+PSI both
Units are all sampled positions (selection independent of the future), so Proposition 1 applies.
Metrics on held-out games: AUROC and Brier score for each forecaster; bootstrap over games.

python scripts/23_t1_othello.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graybox.othello import SQ, TOK, OthelloGPT, random_game  # noqa: E402

RES = Path("/home/arya/graybox/results/othello")
D = OmegaConf.create({"n_games": 1500, "per_game": 4, "H": 3, "Kmax": 64, "seed": 77, "threads": 48, "n_boot": 1000})
CORNERS = {TOK[s] for s in (0, 7, 56, 63)}


@torch.no_grad()
def rollouts(m, kv, t, p0, n, H, gen):
    """n independent H-move self-play continuations from prefix position t (first move from p0, the model's own
    next-move distribution); returns an (n,) indicator that some move in the continuation is a corner."""
    pos = np.full(n, t)
    seq = torch.multinomial(p0.expand(n, -1), 1, replacement=True, generator=gen) + 1
    hit = np.isin(seq[:, 0].numpy(), list(CORNERS))
    for _ in range(1, H):
        p = torch.softmax(m.extend_multi(kv, pos, seq)[:, 1:], -1)
        nxt = torch.multinomial(p, 1, generator=gen) + 1
        hit |= np.isin(nxt[:, 0].numpy(), list(CORNERS))
        seq = torch.cat([seq, nxt], 1)
    return hit.astype(float)


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    torch.set_num_threads(cfg.threads)
    m = OthelloGPT.load()
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    rows = {k: [] for k in ("game", "y", "R", "h", "psi")}
    for gi in range(cfg.n_games):
        g = random_game(rng)[:59]
        if len(g) < 20:
            continue
        cand = np.arange(5, min(len(g), 59 - cfg.H) - 1)
        ts = np.sort(rng.choice(cand, size=min(cfg.per_game, len(cand)), replace=False))
        with torch.no_grad():
            lg, rs, kv = m.forward_cache(torch.tensor([[TOK[s] for s in g]]))
            p1 = torch.softmax(m.extend(kv, np.repeat(ts, 60), np.tile(np.arange(1, 61), len(ts)))[:, 1:], -1).numpy()
        for j, t in enumerate(ts):
            p0 = torch.softmax(lg[0, t, 1:], -1)[None]
            draws = rollouts(m, kv, int(t), p0, 1 + cfg.Kmax, cfg.H, gen)   # first draw = the realized label
            rows["game"].append(gi)
            rows["y"].append(draws[0])
            rows["R"].append(draws[1:])
            rows["h"].append(rs[6][0, t].numpy())
            rows["psi"].append(np.concatenate([2 * np.sqrt(p0[0].numpy()), 2 * np.sqrt(p1[j * 60:(j + 1) * 60].ravel())]))
        if gi % 300 == 0:
            print(f"{gi}/{cfg.n_games} games, {len(rows['y'])} positions", flush=True)
    Z = {k: np.array(v) for k, v in rows.items()}
    y = Z["y"].astype(int)
    games = np.unique(Z["game"])
    te = np.isin(Z["game"], np.random.default_rng(1).choice(games, len(games) // 3, replace=False))
    tr = ~te
    P = {}
    for K in (1, 2, 4, 8, 16, 64):
        P[f"R_K{K}"] = (Z["R"][:, :K].sum(1) + 0.5) / (K + 1)
    if cfg.H == 1:   # exact behavioural forecast from ONE logprob query: P(next move is a corner)
        P["exact_1query"] = Z["psi"][:, [TOK[s_] - 1 for s_ in (0, 7, 56, 63)]].__pow__(2).sum(1) / 4.0
    lr = lambda C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=4000))  # noqa: E731
    for name, X, C in (("WB_L6", Z["h"], 0.05), ("PSI", Z["psi"], 0.05), ("WB+PSI", np.hstack([Z["h"], Z["psi"]]), 0.05)):
        P[name] = lr(C).fit(X[tr], y[tr]).predict_proba(X)[:, 1]
    gte = Z["game"][te]
    ug = np.unique(gte)
    mem = {g_: np.flatnonzero(gte == g_) for g_ in ug}
    brng = np.random.default_rng(3)
    boots = [np.concatenate([mem[g_] for g_ in brng.choice(ug, len(ug))]) for _ in range(cfg.n_boot)]
    res = {"n_positions": int(len(y)), "n_test": int(te.sum()), "base_rate": round(float(y.mean()), 4), "forecasters": {}}
    yt = y[te]
    for k, p in P.items():
        pt = p[te]
        a = roc_auc_score(yt, pt)
        bs = [roc_auc_score(yt[b], pt[b]) for b in boots]
        d = [roc_auc_score(yt[b], P["R_K64"][te][b]) - roc_auc_score(yt[b], pt[b]) for b in boots]
        res["forecasters"][k] = {"auroc": round(float(a), 4), "ci": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                                 "brier": round(float(np.mean((pt - yt) ** 2)), 5),
                                 "R64_minus_this": [round(float(np.mean(d)), 4), round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]}
        print(k, res["forecasters"][k], flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / ("t1_othello.json" if cfg.H == 3 else f"t1_othello_H{cfg.H}.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
