"""Othello world (L1 of the ground-truth ladder): game simulator + Othello-GPT (Li et al. 2023, minGPT weights).

Tokens: 0 = pad; 1..60 = the 60 playable squares (0..63 without the centre 27, 28, 35, 36) in ascending order.
The model reads a sequence of moves and predicts the next legal move. Its latent "world model" (the board) is known
exactly, so it is a ground-truth test for black-box state recovery.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SQUARES = [s for s in range(64) if s not in (27, 28, 35, 36)]
TOK = {s: i + 1 for i, s in enumerate(SQUARES)}      # square -> token
SQ = {i + 1: s for i, s in enumerate(SQUARES)}       # token -> square
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
WEIGHTS = "/data/models/models--Baidicoot--Othello-GPT-Transformer-Lens/snapshots/30e9176f812fc31580671bc93ba9936bf2b7e08a/final.pth"


class Board:
    """1 = black (moves first), -1 = white, 0 = empty. Passes are implicit, as in the synthetic dataset."""

    def __init__(self):
        self.b = np.zeros(64, np.int8)
        self.b[[27, 36]] = -1
        self.b[[28, 35]] = 1
        self.player = 1

    def copy(self):
        o = Board.__new__(Board)
        o.b, o.player = self.b.copy(), self.player
        return o

    def flips(self, sq, player):
        if self.b[sq] != 0:
            return []
        r, c = divmod(sq, 8)
        out = []
        for dr, dc in DIRS:
            rr, cc, line = r + dr, c + dc, []
            while 0 <= rr < 8 and 0 <= cc < 8 and self.b[rr * 8 + cc] == -player:
                line.append(rr * 8 + cc)
                rr += dr
                cc += dc
            if line and 0 <= rr < 8 and 0 <= cc < 8 and self.b[rr * 8 + cc] == player:
                out += line
        return out

    def legal(self, player=None):
        p = self.player if player is None else player
        return [s for s in range(64) if self.b[s] == 0 and self.flips(s, p)]

    def play(self, sq):
        """Play sq for the side to move; pass automatically if that side has no legal move. Returns False if illegal."""
        if not self.legal():
            self.player = -self.player
        f = self.flips(sq, self.player)
        if not f:
            return False
        self.b[sq] = self.player
        self.b[f] = self.player
        self.player = -self.player
        if not self.legal():          # the next side must pass: the side to move flips back
            self.player = -self.player
        return True


def random_game(rng, max_len=60):
    b, moves = Board(), []
    while len(moves) < max_len:
        L = b.legal()
        if not L:
            break
        m = int(L[rng.integers(len(L))])
        b.play(m)
        moves.append(m)
    return moves


def boards_along(moves):
    """Board after each prefix (index t = after moves[:t+1]); also the side to move and the legal set."""
    b, out = Board(), []
    for m in moves:
        assert b.play(m), "illegal move in game"
        out.append((b.b.copy(), b.player, b.legal()))
    return out


# ------------------------------------------------------------------------------------------------ Othello-GPT (minGPT)
class Block(nn.Module):
    def __init__(self, d=512, h=8, dm=2048):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.Module()
        for k in ("key", "query", "value", "proj"):
            setattr(self.attn, k, nn.Linear(d, d))
        self.mlp = nn.Sequential(nn.Linear(d, dm), nn.GELU(), nn.Linear(dm, d))
        self.h = h

    def forward(self, x):
        B, T, D = x.shape
        a = self.ln1(x)
        q, k, v = (getattr(self.attn, n)(a).view(B, T, self.h, D // self.h).transpose(1, 2) for n in ("query", "key", "value"))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.attn.proj(y)
        return x + self.mlp(self.ln2(x))


class OthelloGPT(nn.Module):
    def __init__(self, n_layers=8, d=512, vocab=61, n_ctx=59):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_ctx, d))
        self.blocks = nn.ModuleList([Block(d) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, return_resid=False):
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.shape[1]]
        resid = [x]
        for b in self.blocks:
            x = b(x)
            resid.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, resid) if return_resid else logits

    def _heads(self, blk, a, name):
        B, T, D = a.shape
        return getattr(blk.attn, name)(a).view(B, T, blk.h, D // blk.h).transpose(1, 2)   # (B, h, T, dh)

    def forward_cache(self, idx):
        """Full causal pass over one game (idx: (1, T)); returns logits, residuals and per-layer K, V."""
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.shape[1]]
        resid, kv = [x], []
        for b in self.blocks:
            a = b.ln1(x)
            k, v = self._heads(b, a, "key"), self._heads(b, a, "value")
            kv.append((k[0], v[0]))
            x = b(x)
            resid.append(x)
        return self.head(self.ln_f(x)), resid, kv

    def extend(self, kv, pos, tok):
        """Logits after appending one token tok[i] at position pos[i]+1 to the prefix idx[:pos[i]+1] (batched)."""
        pos, tok = torch.as_tensor(pos), torch.as_tensor(tok)
        N = len(tok)
        x = self.tok_emb(tok) + self.pos_emb[0, pos + 1]                            # (N, D)
        T = kv[0][0].shape[1]
        mask = torch.arange(T)[None, :] <= pos[:, None]                            # (N, T)
        for b, (K, V) in zip(self.blocks, kv):
            a = b.ln1(x)[:, None]                                                   # (N, 1, D)
            q, k, v = (self._heads(b, a, n)[:, :, 0] for n in ("query", "key", "value"))   # (N, h, dh)
            dh = q.shape[-1]
            sp = torch.einsum("nhd,htd->nht", q, K) / math.sqrt(dh)                 # prefix scores
            sp = sp.masked_fill(~mask[:, None, :], float("-inf"))
            ss = (q * k).sum(-1, keepdim=True) / math.sqrt(dh)                     # self score
            w = torch.softmax(torch.cat([sp, ss], -1), -1)
            y = torch.einsum("nht,htd->nhd", w[..., :T], V) + w[..., T:] * v
            x = x + b.attn.proj(y.reshape(N, -1))
            x = x + b.mlp(b.ln2(x))
        return self.head(self.ln_f(x))

    def extend_multi(self, kv, pos, toks):
        """Logits at the last of k appended tokens: toks (N, k) appended after prefix idx[:pos[i]+1] (batched)."""
        pos, toks = torch.as_tensor(pos), torch.as_tensor(toks)
        N, k = toks.shape
        x = self.tok_emb(toks) + self.pos_emb[0, pos[:, None] + 1 + torch.arange(k)[None]]   # (N, k, D)
        T = kv[0][0].shape[1]
        pmask = torch.arange(T)[None, :] <= pos[:, None]                                   # (N, T)
        causal = torch.ones(k, k, dtype=torch.bool).tril()
        for b, (K, V) in zip(self.blocks, kv):
            a = b.ln1(x)
            q, ks, vs = (self._heads(b, a, n) for n in ("query", "key", "value"))          # (N, h, k, dh)
            dh = q.shape[-1]
            sp = torch.einsum("nhqd,htd->nhqt", q, K) / math.sqrt(dh)
            sp = sp.masked_fill(~pmask[:, None, None, :], float("-inf"))
            ss = torch.einsum("nhqd,nhkd->nhqk", q, ks) / math.sqrt(dh)
            ss = ss.masked_fill(~causal[None, None], float("-inf"))
            w = torch.softmax(torch.cat([sp, ss], -1), -1)
            y = torch.einsum("nhqt,htd->nhqd", w[..., :T], V) + torch.einsum("nhqk,nhkd->nhqd", w[..., T:], vs)
            x = x + b.attn.proj(y.transpose(1, 2).reshape(N, k, -1))
            x = x + b.mlp(b.ln2(x))
        return self.head(self.ln_f(x[:, -1]))

    @classmethod
    def load(cls, path=WEIGHTS, device="cpu"):
        m = cls()
        sd = {k: v for k, v in torch.load(path, map_location="cpu").items() if not k.endswith("attn.mask")}
        m.load_state_dict(sd)
        return m.to(device).eval()


def board_to_relative(b, player):
    """Per square: 0 empty, 1 'mine' (side to move), 2 'theirs' (Nanda's mine/theirs frame, linearly decodable)."""
    out = np.zeros(64, np.int64)
    out[b == player] = 1
    out[b == -player] = 2
    return out
