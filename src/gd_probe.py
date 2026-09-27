"""Batched gradient-trained linear probes: L layers x C (lr, wd) configs trained in parallel.
Numerically matches torch.optim.Adam / AdamW on nn.Linear (checked to ~3e-8)."""
import numpy as np
import torch

DEV = "cuda"
BS = 32


def train_linear(X, Y, lrs, wds, epochs, decoupled, seed=0, bs=BS):
    """Train L x C independent linear probes y = xW + b with (Adam|AdamW), minibatch MSE.
    X: [L, N, d] (one feature matrix per layer), Y: [N, k]; lrs, wds: [C].
    Returns W [L, C, d, k], b [L, C, k]."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    L, N, d = X.shape
    k = Y.shape[1]
    C = len(lrs)
    lr = torch.tensor(lrs, device=DEV, dtype=torch.float32).view(1, C, 1, 1)
    wd = torch.tensor(wds, device=DEV, dtype=torch.float32).view(1, C, 1, 1)
    bound = 1 / np.sqrt(d)  # nn.Linear default init
    W = (torch.rand(L, C, d, k, device=DEV, generator=g) * 2 - 1) * bound
    b = (torch.rand(L, C, 1, k, device=DEV, generator=g) * 2 - 1) * bound
    params = [W, b]
    m = [torch.zeros_like(p) for p in params]
    v = [torch.zeros_like(p) for p in params]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    for _ in range(epochs):
        perm = torch.argsort(torch.rand(L, N, device=DEV, generator=g), dim=1)   # own order per layer
        lidx = torch.arange(L, device=DEV)[:, None]
        for s in range(0, N, bs):
            idx = perm[:, s:s + bs]                                      # [L,B]
            xb, yb = X[lidx, idx], Y[idx]                                # [L,B,d], [L,B,k]
            pred = torch.einsum("lbd,lcdk->lcbk", xb, W) + b              # [L,C,B,k]
            err = pred - yb[:, None]                                     # dMSE/dpred = 2 err / (B k)
            scale = 2.0 / (idx.shape[1] * k)
            gW = torch.einsum("lbd,lcbk->lcdk", xb, err) * scale
            gb = err.sum(2, keepdim=True) * scale
            t += 1
            for i, (p, gr) in enumerate(zip(params, (gW, gb))):
                if decoupled:
                    p.mul_(1 - lr * wd)
                else:
                    gr = gr + wd * p
                m[i].mul_(b1).add_(gr, alpha=1 - b1)
                v[i].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                p.sub_(lr * (m[i] / (1 - b1 ** t)) / ((v[i] / (1 - b2 ** t)).sqrt() + eps))
    return W, b[:, :, 0]


def predict(X, W, b):
    """X [L, N, d] -> [L, C, N, k]"""
    return torch.einsum("lnd,lcdk->lcnk", X, W) + b[:, :, None]


def r2_t(y, yhat):
    """y [N,k], yhat [..., N, k] -> R² averaged over outputs, shape [...]"""
    ss_res = ((yhat - y) ** 2).sum(-2)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return (1 - ss_res / ss_tot).mean(-1)


def to_dev(a):
    return torch.tensor(np.array(a, dtype=np.float32, copy=True), device=DEV)
