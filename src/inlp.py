"""Iterative nullspace probing ("orthogonal probe sequence", Joseph et al. App. C.11).

Features are z-scored once with train statistics; all subsequent geometry lives in that space.
Iteration k: fit ridge probe P_k on X_k = X (I - Q_<k Q_<k^T), take its weight columns W_k
(1 for scalars, 2 for sin/cos), orthogonalize against all earlier directions, append to Q,
and project them out. W_k lies in the span of the current data, so probe k reads only the
k-th block of directions.
"""
from dataclasses import dataclass, field

import numpy as np

from src.probes import fit_ridge, metrics


@dataclass
class INLPResult:
    mu: np.ndarray                     # z-score stats (train)
    sd: np.ndarray
    Q: np.ndarray                      # [d, m] orthonormal removed directions, in order
    probes: list = field(default_factory=list)   # (W [d,k], b [k]) per iteration, in z-space
    val: list = field(default_factory=list)      # metrics per iteration (probe k on held-out split)
    test: list = field(default_factory=list)

    def z(self, X):
        return (X - self.mu) / self.sd


def orth_against(W: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Orthonormal basis for the part of span(W) orthogonal to span(Q)."""
    if Q.shape[1]:
        W = W - Q @ (Q.T @ W)
        W = W - Q @ (Q.T @ W)  # re-orthogonalize for numerical safety
    q, r = np.linalg.qr(W)
    keep = np.abs(np.diag(r)) > 1e-8 * max(1.0, np.abs(r).max())
    return q[:, keep]


def run_inlp(name, X, y, idx, n_iter, direction_source="probe", seed=0, log=False,
             eval_splits=("val", "test")) -> INLPResult:
    """direction_source: 'probe' (INLP), 'random' (random orthonormal dirs, same count per step),
    'pca' (top principal components of train data, same count per step)."""
    Xtr = X[idx["train"]].astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = {k: (X[v].astype(np.float64) - mu) / sd for k, v in idx.items()}
    Y = {k: y[v] for k, v in idx.items()}
    d, k_per = X.shape[1], y.shape[1]
    Q = np.zeros((d, 0))
    res = INLPResult(mu, sd, Q)
    rng = np.random.default_rng(seed)
    if direction_source == "pca":
        _, _, Vt = np.linalg.svd(Z["train"] - Z["train"].mean(0), full_matrices=False)
        pcs = Vt.T
    for it in range(n_iter):
        P = {k: z - (z @ Q) @ Q.T for k, z in Z.items()}
        # alpha chosen on val each iteration (val is never used for final test numbers)
        pr = fit_ridge(P["train"], Y["train"], P["val"], Y["val"], standardize=False)
        res.probes.append((pr.W, pr.b - pr.mu @ pr.W))  # z-space affine readout: z @ W + b
        for s in eval_splits:
            getattr(res, s).append(metrics(name, Y[s], pr.predict(P[s]), log))
        if direction_source == "probe":
            W = pr.W
        elif direction_source == "random":
            W = rng.standard_normal((d, k_per))
        else:
            W = pcs[:, it * k_per:(it + 1) * k_per]
        Q = np.c_[Q, orth_against(W, Q)]
    res.Q = Q
    return res
