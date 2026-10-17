#!/usr/bin/env python
"""Generality test (L1b): black-box recovery of the latent state of TRAINED transformers on random hidden-state worlds.

For each world seed:
  world   a random probabilistic automaton: n states (6-12), alphabet A=4; per state a transition for each symbol and
          a Dirichlet next-symbol distribution. Sequences are sampled from it (the true state is known at every step).
  model   a small causal transformer (2 layers, d=64) trained on 20k sequences to predict the next symbol.
  black box  next-symbol distributions only. Battery: no suffix + every suffix up to length 2 (1 + 4 + 16 = 21
          queries); each answer is a 4-way distribution (Fisher-Rao features 2*sqrt(p)).
Measured on held-out sequences:
  state decoding accuracy (multinomial logistic) from the grey box vs from the true hidden state (white box) vs from
  input-only features (counts of symbols, last two symbols); hidden-state R² (top-16 PCs of the last layer) from the
  grey box vs input-only. Also: is the Bayes state (the automaton state) identifiable at all from the battery? (the
  number of distinct true next-symbol distribution vectors per state, computed from the automaton itself).

python scripts/21_random_automata.py n_worlds=16
"""
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

RES = Path("/home/arya/graybox/results/automata")
D = OmegaConf.create({"n_worlds": 16, "A": 4, "T": 32, "n_train": 20000, "n_test": 2000, "epochs": 3, "threads": 48})


class World:
    def __init__(self, seed, A):
        r = np.random.default_rng(seed)
        self.n = int(r.integers(6, 13))
        self.A = A
        # each symbol permutes the states: the chain is doubly stochastic (all states visited evenly), and tracking
        # the state means composing permutations, which shallow input summaries cannot do
        self.delta = np.stack([r.permutation(self.n) for _ in range(A)], 1)
        self.emit = r.dirichlet(np.full(A, 0.5), size=self.n)          # next-symbol distribution per state

    def sample(self, rng, N, T):
        X = np.zeros((N, T), np.int64)
        S = np.zeros((N, T), np.int64)                                 # state BEFORE emitting X[:, t]
        s = np.zeros(N, np.int64)
        for t in range(T):
            S[:, t] = s
            u = rng.random(N)[:, None]
            X[:, t] = (u > np.cumsum(self.emit[s], 1)).sum(1).clip(0, self.A - 1)
            s = self.delta[s, X[:, t]]
        return X, S


class TinyLM(nn.Module):
    def __init__(self, A, T, d=64, L=2, h=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(A + 1, d), nn.Parameter(torch.randn(T + 4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, L)
        self.out = nn.Linear(d, A)
        self.A = A

    def forward(self, x, return_h=False):
        T = x.shape[1]
        mask = torch.triu(torch.full((T, T), float("-inf")), 1)
        h = self.tf(self.emb(x) + self.pos[:T], mask=mask, is_causal=True)
        return (self.out(h), h) if return_h else self.out(h)


def train_lm(world, cfg, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    X, _ = world.sample(rng, cfg.n_train, cfg.T)
    bos = np.full((cfg.n_train, 1), cfg.A)                             # BOS token = A
    inp = torch.tensor(np.hstack([bos, X[:, :-1]]))
    tgt = torch.tensor(X)
    m = TinyLM(cfg.A, cfg.T)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.01)
    for _ in range(cfg.epochs):
        perm = torch.randperm(cfg.n_train)
        for i in range(0, cfg.n_train, 256):
            b = perm[i:i + 256]
            loss = F.cross_entropy(m(inp[b]).reshape(-1, cfg.A), tgt[b].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return m.eval()


def main():
    cfg = OmegaConf.merge(D, OmegaConf.from_cli())
    torch.set_num_threads(cfg.threads)
    suffixes = [()] + [s for n in (1, 2) for s in itertools.product(range(cfg.A), repeat=n)]
    out = []
    for w in range(cfg.n_worlds):
        t0 = time.time()
        world = World(1000 + w, cfg.A)
        m = train_lm(world, cfg, w)
        rng = np.random.default_rng(5000 + w)
        X, S = world.sample(rng, cfg.n_test, cfg.T)
        # one position per test sequence (t in [4, T-4)) so suffixes fit; prefix = BOS + X[:t]
        ts = rng.integers(4, cfg.T - 4, size=cfg.n_test)
        feats, hid, states, inp = [], [], [], []
        with torch.no_grad():
            for i in range(0, cfg.n_test, 200):
                rows = range(i, min(i + 200, cfg.n_test))
                batch_f, batch_h = [], []
                for j in rows:
                    pre = [cfg.A] + X[j, : ts[j]].tolist()
                    lens = [len(pre) + len(s) for s in suffixes]
                    padded = torch.full((len(suffixes), max(lens)), 0)
                    for k, s in enumerate(suffixes):
                        padded[k, : lens[k]] = torch.tensor(pre + list(s))
                    logits, h = m(padded, return_h=True)
                    P = torch.softmax(logits[torch.arange(len(suffixes)), torch.tensor(lens) - 1], -1).numpy()
                    batch_f.append(2 * np.sqrt(P).ravel())
                    batch_h.append(h[0, len(pre) - 1].numpy())
                    states.append(S[j, ts[j]])
                    c = np.bincount(X[j, : ts[j]], minlength=cfg.A) / ts[j]
                    last = np.zeros(2 * cfg.A)
                    last[X[j, ts[j] - 1]] = 1
                    last[cfg.A + X[j, ts[j] - 2]] = 1
                    inp.append(np.concatenate([c, last, [ts[j] / cfg.T]]))
                feats += batch_f
                hid += batch_h
        F_, H, Y, I_ = np.array(feats), np.array(hid), np.array(states), np.array(inp)
        half = cfg.n_test // 2
        tr, te = np.arange(half), np.arange(half, cfg.n_test)

        def acc(Z):
            sc = StandardScaler().fit(Z[tr])
            clf = LogisticRegression(C=1.0, max_iter=4000).fit(sc.transform(Z[tr]), Y[tr])
            return float((clf.predict(sc.transform(Z[te])) == Y[te]).mean())

        def r2(Z):
            mu = H[tr].mean(0)
            _, Sv, Vt = np.linalg.svd(H[tr] - mu, full_matrices=False)
            P_, w_ = (H - mu) @ Vt[:16].T, Sv[:16] ** 2 / (Sv ** 2).sum()
            sc = StandardScaler().fit(Z[tr])
            pred = RidgeCV(alphas=np.logspace(-2, 3, 11)).fit(sc.transform(Z[tr]), P_[tr]).predict(sc.transform(Z[te]))
            r = 1 - ((P_[te] - pred) ** 2).sum(0) / ((P_[te] - P_[te].mean(0)) ** 2).sum(0)
            return float((r * w_).sum() / w_.sum())

        # identifiability of the automaton state from the battery, computed on the automaton itself
        def true_psi(s):
            vec = []
            for sfx in suffixes:
                q = s
                for a in sfx:
                    q = world.delta[q, a]
                vec.append(np.round(world.emit[q], 6))
            return tuple(np.concatenate(vec))
        n_classes = len({true_psi(s) for s in range(world.n)})
        row = {"world": w, "n_states": world.n, "battery_classes": n_classes,
               "majority_acc": round(float(np.bincount(Y[te]).max() / len(te)), 4),
               "state_acc_greybox": round(acc(F_), 4), "state_acc_whitebox": round(acc(H), 4),
               "state_acc_input_only": round(acc(I_), 4),
               "hidden_R2_greybox": round(r2(F_), 4), "hidden_R2_input_only": round(r2(I_), 4),
               "seconds": round(time.time() - t0)}
        out.append(row)
        print(row, flush=True)
    summ = {k: [round(float(np.mean([r[k] for r in out])), 4), round(float(np.std([r[k] for r in out])), 4)]
            for k in ("state_acc_greybox", "state_acc_whitebox", "state_acc_input_only", "hidden_R2_greybox",
                      "hidden_R2_input_only", "majority_acc")}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "random_automata.json").write_text(json.dumps({"worlds": out, "mean_sd": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    sys.exit(main())
