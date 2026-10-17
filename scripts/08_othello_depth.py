#!/usr/bin/env python
"""L1 depth study: how much of Othello-GPT's latent state do black-box batteries recover as they look deeper?

Batteries (all black-box: only next-move distributions after appended moves are observed):
  B0         next-move distribution (1 query)
  B1         + the 60 single-move suffixes (61 queries; fixed)
  B2         + 256 fixed random two-move suffixes (317 queries; fixed)
  ON-d       on-policy: K sampled continuations of length 1..d from the model itself; per depth j, the mean of
             2*sqrt(p_next) over the K samples (fixed 60-dim coordinates per depth; K*d queries)
Ground truth: board (mine/theirs/empty) and the residual stream (L6). Metrics on held-out games:
  board occupied-accuracy / exact (MLP decoder); hidden recovery R² (top-64 PCs, variance-weighted);
  probe-subspace R²: R² of the residual's projection onto the span of the white-box board probe (the directions that
  carry the board), which variance-weighted R² can hide.

python scripts/08_othello_depth.py n_games=1200 per_game=8 K=16
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graybox.othello import TOK, OthelloGPT, board_to_relative, boards_along, random_game  # noqa: E402

fair = importlib.import_module("07_othello_l1_fair")
OUT = Path("/data/models/graybox_ws/outputs/othello")
RES = Path("/home/arya/graybox/results/othello")
D = OmegaConf.create({"n_games": 1200, "per_game": 8, "K": 16, "max_depth": 4, "n_pairs": 256, "seed": 10,
                      "threads": 48})


def fr(p):
    return 2 * np.sqrt(np.clip(p, 0, 1))


def collect(cfg, m):
    rng = np.random.default_rng(cfg.seed)
    pairs = torch.tensor(np.random.default_rng(99).integers(1, 61, size=(cfg.n_pairs, 2)))
    rows = {k: [] for k in ("game", "board", "B0", "B1", "B2", "ON", "resid6")}
    t0 = time.time()
    for gi in range(cfg.n_games):
        g = random_game(rng)[:59]
        if len(g) < 12:
            continue
        bs = boards_along(g)
        cand = np.arange(min(len(g) - 1, 59 - cfg.max_depth))      # t + max_depth <= 58 (context 59)
        ts = np.sort(rng.choice(cand, size=min(cfg.per_game, len(cand)), replace=False))
        with torch.no_grad():
            lg, rs, kv = m.forward_cache(torch.tensor([[TOK[s] for s in g]]))
            p0 = torch.softmax(lg[0, ts, 1:], -1)
            b1 = torch.softmax(m.extend(kv, np.repeat(ts, 60), np.tile(np.arange(1, 61), len(ts)))[:, 1:], -1)
            b2 = torch.softmax(m.extend_multi(kv, np.repeat(ts, cfg.n_pairs), pairs.repeat(len(ts), 1))[:, 1:], -1)
            # on-policy rollouts: K continuations per state, depth 1..max_depth, sampling from the model itself
            pos = np.repeat(ts, cfg.K)
            seq = torch.multinomial(p0.repeat_interleave(cfg.K, 0), 1) + 1          # first sampled move (token)
            on = []
            for d in range(1, cfg.max_depth + 1):
                pd = torch.softmax(m.extend_multi(kv, pos, seq)[:, 1:], -1)        # next-move dist after d moves
                on.append(fr(pd.numpy()).reshape(len(ts), cfg.K, 60).mean(1))
                if d < cfg.max_depth:
                    seq = torch.cat([seq, torch.multinomial(pd, 1) + 1], 1)
        for j, t in enumerate(ts):
            b, player, _ = bs[t]
            rows["game"].append(gi)
            rows["board"].append(board_to_relative(b, player))
            rows["B0"].append(p0[j].numpy())
            rows["B1"].append(b1[j * 60:(j + 1) * 60].numpy().ravel())
            rows["B2"].append(b2[j * cfg.n_pairs:(j + 1) * cfg.n_pairs].numpy().ravel())
            rows["ON"].append(np.concatenate([o[j] for o in on]))
            rows["resid6"].append(rs[6][0, t].numpy())
        if gi % 200 == 0:
            print(f"{gi}/{cfg.n_games} games {len(rows['game'])} states {time.time()-t0:.0f}s", flush=True)
    return {k: np.array(v, dtype=np.float32 if k not in ("game", "board") else None) for k, v in rows.items()}


def probe_subspace(Htr, Ytr):
    """Orthonormal basis of the span of a linear white-box board probe fitted on train states."""
    Xtr, _ = fair.standardize(Htr, Htr[:2])
    W = torch.nn.Linear(Htr.shape[1], 192)
    fair.fit(W, Xtr, lambda out, b: F.cross_entropy(out.view(-1, 64, 3).permute(0, 2, 1), torch.tensor(Ytr)[b]), 30, 1e-2)
    Q, _ = np.linalg.qr(W.weight.detach().numpy().T / (Htr.std(0)[:, None] + 1e-6))
    return Q                                                      # (512, 192)


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    torch.set_num_threads(cfg.threads)
    cache = OUT / f"depth_g{cfg.n_games}_p{cfg.per_game}_K{cfg.K}_s{cfg.seed}.npz"
    if cache.exists():
        Z = dict(np.load(cache))
    else:
        Z = collect(cfg, OthelloGPT.load())
        np.savez_compressed(cache, **Z)
    games = np.unique(Z["game"])
    te = np.isin(Z["game"], np.random.default_rng(1).choice(games, int(len(games) * 0.2), replace=False))
    tr = ~te
    Y, H = Z["board"], Z["resid6"]
    on = Z["ON"]
    feats = {"B0 (1 query)": fr(Z["B0"]),
             "B1 (61 queries)": np.hstack([fr(Z["B0"]), fr(Z["B1"])]),
             "B2 (317 queries)": np.hstack([fr(Z["B0"]), fr(Z["B1"]), fr(Z["B2"])])}
    for d in range(1, cfg.max_depth + 1):
        feats[f"ON-{d} (K={cfg.K}, {cfg.K * d} queries)"] = np.hstack([fr(Z["B0"]), on[:, :60 * d]])
    feats["B1 + ON-4"] = np.hstack([feats["B1 (61 queries)"], on])
    feats["white-box L6 (reference)"] = H
    mu = H[tr].mean(0)
    _, S, Vt = np.linalg.svd(H[tr] - mu, full_matrices=False)
    V, w = Vt[:64].T, S[:64] ** 2 / (S ** 2).sum()
    Q = probe_subspace(H[tr], Y[tr])
    Ptr, Pte = (H[tr] - mu) @ V, (H[te] - mu) @ V
    Btr, Bte = (H[tr] - mu) @ Q, (H[te] - mu) @ Q
    res = {"n_train": int(tr.sum()), "n_test": int(te.sum()), "var_in_top64": round(float(w.sum()), 4), "rows": {}}
    for k, X in feats.items():
        t0 = time.time()
        r = fair.decode_board("mlp", X[tr], Y[tr], X[te], Y[te])
        if not k.startswith("white-box"):
            r2 = fair.recover_hidden("mlp", X[tr], Ptr, X[te], Pte)
            r2b = fair.recover_hidden("mlp", X[tr], Btr, X[te], Bte)
            vb = Bte.var(0)
            r.update({"hidden_R2_top64_weighted": round(float((r2 * w).sum() / w.sum()), 4),
                      "probe_subspace_R2": round(float((r2b * vb).sum() / vb.sum()), 4)})
        res["rows"][k] = r
        print(f"{k:34s} {r} ({time.time()-t0:.0f}s)", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "l1_depth.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
