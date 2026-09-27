"""Linear probes (closed-form ridge) and metrics.

Features are z-scored with train statistics. Direction is regressed onto (sin θ, cos θ)
and decoded with atan2; scalars are regressed directly (optionally in log space).
"""
from dataclasses import dataclass

import numpy as np
import torch

ALPHAS = np.logspace(-2, 5, 15)


def targets(name: str, df, log: bool = False) -> np.ndarray:
    if name == "direction":
        th = np.radians(df.theta_degrees.values)
        return np.c_[np.sin(th), np.cos(th)]
    y = (df.speed_mps if name == "speed" else df.acceleration_mps2).values[:, None]
    return np.log(y) if log else y


def angle_deg(y: np.ndarray) -> np.ndarray:
    """(sin, cos) predictions -> angle in [0, 360)."""
    return np.degrees(np.arctan2(y[:, 0], y[:, 1])) % 360


def circ_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs((a - b + 180) % 360 - 180)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """Mean over output columns of the coefficient of determination."""
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / ss_tot))


def metrics(name: str, y: np.ndarray, yhat: np.ndarray, log: bool = False) -> dict:
    out = {"r2": r2(y, yhat)}
    if name == "direction":
        out["circ_mae_deg"] = float(circ_err(angle_deg(yhat), angle_deg(y)).mean())
    else:
        t, p = (np.exp(y), np.exp(yhat)) if log else (y, yhat)
        out["mae"] = float(np.abs(t - p).mean())
    return out


@dataclass
class Ridge:
    """Ridge on standardized features. W: [d, k] in standardized space."""
    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray
    b: np.ndarray
    alpha: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.W + self.b

    @property
    def W_raw(self) -> np.ndarray:
        """Weights acting on raw (unstandardized) features: yhat = X @ W_raw + b_raw."""
        return self.W / self.sd[:, None]

    @property
    def b_raw(self) -> np.ndarray:
        return self.b - (self.mu / self.sd) @ self.W


def _solve_path(Xs: np.ndarray, yc: np.ndarray, alphas) -> list[np.ndarray]:
    """Ridge solutions for all alphas via one eigendecomposition (GPU, float64)."""
    X = torch.as_tensor(Xs, dtype=torch.float64, device="cuda")
    Y = torch.as_tensor(yc, dtype=torch.float64, device="cuda")
    n, d = X.shape
    if n < d:  # dual form
        evals, U = torch.linalg.eigh(X @ X.T)
        UtY = U.T @ Y
        return [(X.T @ (U @ (UtY / (evals + a)[:, None]))).cpu().numpy() for a in alphas]
    evals, V = torch.linalg.eigh(X.T @ X)
    VtXtY = V.T @ (X.T @ Y)
    return [(V @ (VtXtY / (evals + a)[:, None])).cpu().numpy() for a in alphas]


def fit_ridge(X, y, X_val=None, y_val=None, alphas=ALPHAS, alpha=None, standardize=True) -> Ridge:
    """Fit on (X, y); pick alpha maximizing val R² (or use the given alpha).

    standardize=False skips z-scoring (caller has already standardized); features are then
    only centered, so W lies in the row span of the centered data.
    """
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6 if standardize else np.ones(X.shape[1])
    Xs = (X - mu) / sd
    b = y.mean(0)
    if alpha is not None:
        alphas = [alpha]
    Ws = _solve_path(Xs, y - b, alphas)
    if len(alphas) == 1:
        return Ridge(mu, sd, Ws[0], b, alphas[0])
    Xv = (np.asarray(X_val, np.float64) - mu) / sd
    scores = [r2(y_val, Xv @ W + b) for W in Ws]
    i = int(np.argmax(scores))
    return Ridge(mu, sd, Ws[i], b, alphas[i])
