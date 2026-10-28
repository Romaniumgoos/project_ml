"""Последовательностные модели на PyTorch.

GRU-ячейка написана вручную (уравнения ниже), nn.GRU не используется:
    r_t = sigma(W_r x_t + U_r h_{t-1} + b_r)          — gate сброса
    z_t = sigma(W_z x_t + U_z h_{t-1} + b_z)          — gate обновления
    n_t = tanh(W_n x_t + r_t * (U_n h_{t-1} + b_hn))  — кандидат
    h_t = (1 - z_t) * n_t + z_t * h_{t-1}
Вход — последовательность нажатий (hold, dd, ud, отклонение от популяционной
латентности биграммы, ...), см. FeatureExtractor.sequence.

1) GRUClassifier — античит (человек / бот).
2) SiameseGRU — эмбеддинг стиля печати, обучается triplet loss с batch-hard
   майнингом; верификация = косинусное расстояние до центра шаблона пользователя.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUCellManual(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.x2h = nn.Linear(d_in, 3 * d_h)
        self.h2h = nn.Linear(d_h, 3 * d_h)
        self.d_h = d_h

    def forward(self, x, h):
        gx = self.x2h(x)
        gh = self.h2h(h)
        xr, xz, xn = gx.chunk(3, -1)
        hr, hz, hn = gh.chunk(3, -1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


class GRUEncoder(nn.Module):
    """Двунаправленный GRU + пулинг (среднее по маске + последнее состояние)."""

    def __init__(self, d_in, d_h=48, bidirectional=True):
        super().__init__()
        self.fw = GRUCellManual(d_in, d_h)
        self.bw = GRUCellManual(d_in, d_h) if bidirectional else None
        self.d_h = d_h
        self.out_dim = d_h * (4 if bidirectional else 2)

    def _run(self, cell, x, mask, reverse=False):
        B, T, _ = x.shape
        h = x.new_zeros(B, self.d_h)
        outs = [None] * T
        steps = range(T - 1, -1, -1) if reverse else range(T)
        for t in steps:
            h_new = cell(x[:, t], h)
            m = mask[:, t, None]
            h = m * h_new + (1 - m) * h        # паддинг не меняет состояние
            outs[t] = h
        H = torch.stack(outs, 1)
        mean = (H * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return torch.cat([mean, h], -1)

    def forward(self, x, mask):
        out = self._run(self.fw, x, mask)
        if self.bw is not None:
            out = torch.cat([out, self._run(self.bw, x, mask, reverse=True)], -1)
        return out


def pad(seqs: list[np.ndarray], max_len: int | None = None):
    T = max(len(s) for s in seqs) if max_len is None else max_len
    d = seqs[0].shape[1]
    X = np.zeros((len(seqs), T, d), dtype=np.float32)
    M = np.zeros((len(seqs), T), dtype=np.float32)
    for i, s in enumerate(seqs):
        s = s[:T]
        X[i, : len(s)] = s
        M[i, : len(s)] = 1
    return torch.from_numpy(X), torch.from_numpy(M)


# ============================================================ античит
class GRUClassifier(nn.Module):
    def __init__(self, d_in, d_h=48):
        super().__init__()
        self.enc = GRUEncoder(d_in, d_h)
        self.head = nn.Sequential(nn.Linear(self.enc.out_dim, 32), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, mask):
        return self.head(self.enc(x, mask))[:, 0]


def train_gru_classifier(seqs, y, epochs=25, batch=64, lr=2e-3, d_h=48, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.float32)
    model = GRUClassifier(seqs[0].shape[1], d_h)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_w = torch.tensor(float((1 - y.mean()) / max(float(y.mean()), 1e-6)), dtype=torch.float32)
    history = []
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(seqs))
        tot = 0.0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            X, M = pad([seqs[i] for i in idx])
            logits = model(X, M)
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(y[idx]), pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        history.append(tot / len(seqs))
        if verbose:
            print(f"  epoch {ep + 1}: loss {history[-1]:.4f}")
    model.history = history
    return model


@torch.no_grad()
def predict_gru(model, seqs, batch=256):
    model.eval()
    out = []
    for s in range(0, len(seqs), batch):
        X, M = pad(seqs[s:s + batch])
        out.append(torch.sigmoid(model(X, M)).numpy())
    return np.concatenate(out)


# ============================================================ биометрия (сиамская сеть)
class SiameseGRU(nn.Module):
    def __init__(self, d_in, d_h=48, d_emb=32):
        super().__init__()
        self.enc = GRUEncoder(d_in, d_h)
        self.proj = nn.Sequential(nn.Linear(self.enc.out_dim, 64), nn.ReLU(), nn.Linear(64, d_emb))

    def forward(self, x, mask):
        return F.normalize(self.proj(self.enc(x, mask)), dim=-1)


def batch_hard_triplet(emb, labels, margin=0.3):
    """Для каждого якоря: самый дальний «свой» и самый близкий «чужой»."""
    D = 1 - emb @ emb.T                       # косинусное расстояние
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool)
    pos = (D * (same & ~eye)).max(1).values
    neg = (D + 4.0 * same.float()).min(1).values
    return F.relu(pos - neg + margin).mean()


def train_siamese(seqs, users, epochs=30, P=8, K=6, steps_per_epoch=30, lr=2e-3,
                  margin=0.3, seed=0, verbose=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    users = np.asarray(users)
    by_user = {u: np.where(users == u)[0] for u in np.unique(users)}
    by_user = {u: ix for u, ix in by_user.items() if len(ix) >= 2}
    ulist = list(by_user)
    uid = {u: i for i, u in enumerate(ulist)}
    model = SiameseGRU(seqs[0].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            us = rng.choice(len(ulist), size=min(P, len(ulist)), replace=False)
            idx, lab = [], []
            for u in us:
                ix = by_user[ulist[u]]
                pick = rng.choice(ix, size=K, replace=len(ix) < K)
                idx += list(pick)
                lab += [u] * K
            X, M = pad([seqs[i] for i in idx])
            loss = batch_hard_triplet(model(X, M), torch.tensor(np.array(lab, dtype=np.int64)), margin)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        history.append(tot / steps_per_epoch)
        if verbose:
            print(f"  epoch {ep + 1}: triplet loss {history[-1]:.4f}")
    model.history = history
    return model


@torch.no_grad()
def embed(model, seqs, batch=256):
    model.eval()
    out = []
    for s in range(0, len(seqs), batch):
        X, M = pad(seqs[s:s + batch])
        out.append(model(X, M).numpy())
    return np.concatenate(out)
