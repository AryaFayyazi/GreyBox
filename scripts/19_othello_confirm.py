#!/usr/bin/env python
"""Preregistered confirmation of the L1 claim (docs/prereg_L1_othello.md) on FRESH Othello games.

stage=freeze   train the input-only comparators on the development cache (same split and MLP recipe as the grey-box
               decoders in models/othello/demo.pt) and save them; nothing here touches fresh games
stage=confirm  generate fresh games (seed 2026, never used before), query the black box (B1, 61 queries), apply ONLY
               frozen decoders, and test E1-E3 with a game-cluster bootstrap

python scripts/19_othello_confirm.py stage=freeze
python scripts/19_othello_confirm.py stage=confirm
"""
import importlib
import json
import sys
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
MOD = Path("/data/models/graybox_ws/models/othello")
RES = Path("/home/arya/graybox/results/othello")
D = OmegaConf.create({"stage": "confirm", "n_games": 1000, "per_game": 10, "seed": 2026, "n_boot": 2000})


def mlp(n_in, n_out):
    return torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(512, n_out))


def freeze():
    ck = torch.load(MOD / "demo.pt", weights_only=False)
    Z = dict(np.load(OUT / "l1_g3000_p10_s0.npz"))
    games = np.unique(Z["game"])
    te = np.isin(Z["game"], np.random.default_rng(1).choice(games, int(len(games) * 0.2), replace=False))
    tr = ~te
    Xh = Z["hist"].astype(np.float32)
    hmu, comps = ck["hmu"], ck["comps"]
    Ptr = (Z[f"resid{ck['layer']}"].astype(np.float32)[tr] - hmu) @ comps.T
    mu, sd = Xh[tr].mean(0), Xh[tr].std(0) + 1e-6
    Xs = torch.tensor((Xh[tr] - mu) / sd)
    torch.manual_seed(0)
    in2h = fair.fit(fair.make("mlp", Xh.shape[1], ck["npc"]), Xs, lambda o, b: F.mse_loss(o, torch.tensor(Ptr)[b]), 40, 1e-2)
    in2b = fair.fit(fair.make("mlp", Xh.shape[1], 192), Xs,
                    lambda o, b: F.cross_entropy(o.view(-1, 64, 3).permute(0, 2, 1), torch.tensor(Z["board"][tr])[b]), 40, 1e-2)
    torch.save({"mu": mu, "sd": sd, "in2h": in2h.state_dict(), "in2b": in2b.state_dict()}, MOD / "input_only.pt")
    print("frozen input-only comparators saved")


def confirm(cfg):
    ck = torch.load(MOD / "demo.pt", weights_only=False)
    io = torch.load(MOD / "input_only.pt", weights_only=False)
    psi2h, psi2b, in2h, in2b = mlp(ck["n_in"], ck["npc"]), mlp(ck["n_in"], 192), mlp(128, ck["npc"]), mlp(128, 192)
    psi2h.load_state_dict(ck["psi2h"]); psi2b.load_state_dict(ck["psi2b"]); in2h.load_state_dict(io["in2h"]); in2b.load_state_dict(io["in2b"])
    wb = torch.nn.Linear(512, 192); wb.load_state_dict(ck["wb"])
    for m in (psi2h, psi2b, in2h, in2b, wb):
        m.eval()
    model = OthelloGPT.load()
    rng = np.random.default_rng(cfg.seed)
    rows = {k: [] for k in ("game", "X", "hist", "board", "h")}
    for gi in range(cfg.n_games):
        g = random_game(rng)[:59]
        if len(g) < 12:
            continue
        bs = boards_along(g)
        ts = np.sort(rng.choice(np.arange(len(g) - 1), size=min(cfg.per_game, len(g) - 1), replace=False))
        with torch.no_grad():
            lg, rs, kv = model.forward_cache(torch.tensor([[TOK[s] for s in g]]))
            p0 = torch.softmax(lg[0, ts, 1:], -1).numpy()
            p1 = torch.softmax(model.extend(kv, np.repeat(ts, 60), np.tile(np.arange(1, 61), len(ts)))[:, 1:], -1).numpy()
        for j, t in enumerate(ts):
            b, player, _ = bs[t]
            h = np.zeros(128, np.float32)
            for k, mv in enumerate(g[: t + 1]):
                h[mv + (0 if bs[k][0][mv] == player else 64)] = 1.0
            rows["game"].append(gi)
            rows["X"].append(np.concatenate([2 * np.sqrt(p0[j]), 2 * np.sqrt(p1[j * 60:(j + 1) * 60].ravel())]))
            rows["hist"].append(h)
            rows["board"].append(board_to_relative(b, player))
            rows["h"].append(rs[ck["layer"]][0, t].numpy())
    Z = {k: np.array(v) for k, v in rows.items()}
    P = (Z["h"] - ck["hmu"]) @ ck["comps"].T
    w = None
    with torch.no_grad():
        Ph_g = psi2h(torch.tensor(((Z["X"] - ck["xmu"]) / ck["xsd"]), dtype=torch.float32)).numpy()
        Ph_i = in2h(torch.tensor(((Z["hist"] - io["mu"]) / io["sd"]), dtype=torch.float32)).numpy()
        Bg = psi2b(torch.tensor(((Z["X"] - ck["xmu"]) / ck["xsd"]), dtype=torch.float32)).view(-1, 64, 3).argmax(-1).numpy()
        Bi = in2b(torch.tensor(((Z["hist"] - io["mu"]) / io["sd"]), dtype=torch.float32)).view(-1, 64, 3).argmax(-1).numpy()
        Bw = wb(torch.tensor(((Z["h"] - ck["hmu"]) / ck["hsd"]), dtype=torch.float32)).view(-1, 64, 3).argmax(-1).numpy()
    # weights: the development frame's variance shares (fixed before confirmation)
    w = np.asarray(ck["calib_true_pc"]).var(0)
    w = w / w.sum()

    def r2w(Pred, idx):
        Y = P[idx]
        r2 = 1 - ((Y - Pred[idx]) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
        return float((r2 * w).sum())

    def occ_acc(B, idx):
        Y = Z["board"][idx]
        return float((B[idx] == Y)[Y > 0].mean())

    allidx = np.arange(len(P))
    res = {"n_states": len(P), "n_games": int(len(np.unique(Z["game"]))),
           "E1_greybox_hidden_R2": round(r2w(Ph_g, allidx), 4), "input_only_hidden_R2": round(r2w(Ph_i, allidx), 4),
           "greybox_occupied_acc": round(occ_acc(Bg, allidx), 4), "input_only_occupied_acc": round(occ_acc(Bi, allidx), 4),
           "whitebox_occupied_acc": round(occ_acc(Bw, allidx), 4)}
    games = np.unique(Z["game"])
    members = {g_: np.flatnonzero(Z["game"] == g_) for g_ in games}
    brng = np.random.default_rng(7)
    d2, da, e1 = [], [], []
    for _ in range(cfg.n_boot):
        idx = np.concatenate([members[g_] for g_ in brng.choice(games, len(games))])
        e1.append(r2w(Ph_g, idx))
        d2.append(r2w(Ph_g, idx) - r2w(Ph_i, idx))
        da.append(occ_acc(Bg, idx) - occ_acc(Bi, idx))
    res["E1_CI95"] = [round(float(np.percentile(e1, 2.5)), 4), round(float(np.percentile(e1, 97.5)), 4)]
    res["E1_pass"] = bool(np.percentile(e1, 5) > 0.90)
    res["E2_delta_R2"] = [round(float(np.mean(d2)), 4), round(float(np.percentile(d2, 2.5)), 4), round(float(np.percentile(d2, 97.5)), 4)]
    res["E2_p_one_sided"] = round(float(np.mean(np.array(d2) <= 0)), 5)
    res["E3_delta_occupied_acc"] = [round(float(np.mean(da)), 4), round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)]
    res["E3_p_one_sided"] = round(float(np.mean(np.array(da) <= 0)), 5)
    p_sorted = sorted([("E2", res["E2_p_one_sided"]), ("E3", res["E3_p_one_sided"])], key=lambda x: x[1])
    holm = {}
    for i, (k, p) in enumerate(p_sorted):
        holm[k] = min(1.0, max(p * (2 - i), holm.get(p_sorted[i - 1][0], 0) if i else 0))
    res["holm"] = holm
    res["E2_pass"], res["E3_pass"] = holm["E2"] < 0.05, holm["E3"] < 0.05
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "l1_confirm.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    torch.set_num_threads(32)
    freeze() if cfg.stage == "freeze" else confirm(cfg)


if __name__ == "__main__":
    main()
