"""Multi-probe subspace steering (Joseph et al. App. C.12, Eq. 8), in INLP z-space.

With the first N INLP probes (W_k, b_k), V = orthonormal basis of span(W_1..W_N) = Q[:, :N*k].
For an activation z and target t:  c* = argmin_c sum_k ||W_k^T V c + b_k - t||^2,
z* = z - V V^T z + V c*   (replace in-subspace coordinates; out-of-subspace part untouched).
Because W_k is orthogonal to Q_<k, probe k reads W_k^T z directly.
"""
import numpy as np

from src.inlp import INLPResult


def target_vec(name: str, value: float) -> np.ndarray:
    if name == "direction":
        th = np.radians(value)
        return np.array([np.sin(th), np.cos(th)])
    return np.array([value])


def clamp_coords(res: INLPResult, n_probes: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (V [d, m], c* [m]) for steering to target t with the first n_probes probes."""
    k = len(t)
    V = res.Q[:, : n_probes * k]
    A = np.concatenate([W.T @ V for W, _ in res.probes[:n_probes]])        # [N*k, m]
    rhs = np.concatenate([t - b for _, b in res.probes[:n_probes]])        # [N*k]
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return V, c


def steer(z: np.ndarray, V: np.ndarray, c: np.ndarray) -> np.ndarray:
    """z [n, d] -> steered z* [n, d]."""
    return z - (z @ V) @ V.T + c[None] @ V.T
