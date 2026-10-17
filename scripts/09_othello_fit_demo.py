#!/usr/bin/env python
"""Fit and freeze the decoders the Othello demo serves (from the cached L1 states of scripts/05).

Saved to /data/models/graybox_ws/models/othello/demo.pt:
  - feature standardization for θ(Ψ) (61-query battery, 2*sqrt(p) features)
  - psi->hidden MLP (predicts the top-64 PCs of the TRUE layer-6 residual) and the PCA frame (mean, components)
  - psi->board MLP (mine / theirs / empty per square)
  - white-box board probe on the true layer-6 residual (used for the gauge demo)
  - held-out metrics, so the viewer can print exactly how good each decoder is
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
fair = importlib.import_module("07_othello_l1_fair")
OUT = Path("/data/models/graybox_ws/outputs/othello")
MOD = Path("/data/models/graybox_ws/models/othello")
LAYER, NPC = 6, 64


def main():
    torch.set_num_threads(32)
    Z = dict(np.load(OUT / "l1_g3000_p10_s0.npz"))
    games = np.unique(Z["game"])
    te = np.isin(Z["game"], np.random.default_rng(1).choice(games, int(len(games) * 0.2), replace=False))
    tr = ~te
    X = np.hstack([2 * np.sqrt(Z["P0"]), 2 * np.sqrt(Z["P1"].reshape(len(Z["P0"]), -1))]).astype(np.float32)
    H = Z[f"resid{LAYER}"].astype(np.float32)
    Y = Z["board"]
    xmu, xsd = X[tr].mean(0), X[tr].std(0) + 1e-6
    hmu = H[tr].mean(0)
    _, S, Vt = np.linalg.svd(H[tr] - hmu, full_matrices=False)
    comps = Vt[:NPC]
    w = S[:NPC] ** 2 / (S ** 2).sum()
    Ptr, Pte = (H[tr] - hmu) @ comps.T, (H[te] - hmu) @ comps.T
    Xs_tr, Xs_te = torch.tensor((X[tr] - xmu) / xsd), torch.tensor((X[te] - xmu) / xsd)
    torch.manual_seed(0)
    psi2h = fair.fit(fair.make("mlp", X.shape[1], NPC), Xs_tr,
                     lambda o, b: F.mse_loss(o, torch.tensor(Ptr)[b]), 40, 1e-2)
    psi2b = fair.fit(fair.make("mlp", X.shape[1], 192), Xs_tr,
                     lambda o, b: F.cross_entropy(o.view(-1, 64, 3).permute(0, 2, 1), torch.tensor(Y[tr])[b]), 40, 1e-2)
    Hs_tr = torch.tensor((H[tr] - hmu) / (H[tr].std(0) + 1e-6))
    hsd = H[tr].std(0) + 1e-6
    wb = fair.fit(torch.nn.Linear(H.shape[1], 192), Hs_tr,
                  lambda o, b: F.cross_entropy(o.view(-1, 64, 3).permute(0, 2, 1), torch.tensor(Y[tr])[b]), 30, 1e-2)
    with torch.no_grad():
        Ph = psi2h(Xs_te).numpy()
        pb = psi2b(Xs_te).view(-1, 64, 3).argmax(-1).numpy()
        pw = wb(torch.tensor((H[te] - hmu) / hsd)).view(-1, 64, 3).argmax(-1).numpy()
    r2 = 1 - ((Pte - Ph) ** 2).sum(0) / ((Pte - Pte.mean(0)) ** 2).sum(0)
    occ = Y[te] > 0
    metrics = {"heldout_states": int(te.sum()),
               "hidden_R2_top64_weighted": round(float((r2 * w).sum() / w.sum()), 4),
               "hidden_R2_pc1to3": np.round(r2[:3], 4).tolist(),
               "var_in_top64": round(float(w.sum()), 4),
               "greybox_board_occupied_acc": round(float((pb == Y[te])[occ].mean()), 4),
               "greybox_board_square_acc": round(float((pb == Y[te]).mean()), 4),
               "whitebox_board_occupied_acc": round(float((pw == Y[te])[occ].mean()), 4)}
    MOD.mkdir(parents=True, exist_ok=True)
    torch.save({"xmu": xmu, "xsd": xsd, "hmu": hmu, "comps": comps, "hsd": hsd, "layer": LAYER,
                "psi2h": psi2h.state_dict(), "psi2b": psi2b.state_dict(), "wb": wb.state_dict(),
                "n_in": X.shape[1], "npc": NPC, "metrics": metrics,
                "calib_true_pc": Ptr[:4000].astype(np.float32)}, MOD / "demo.pt")
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
