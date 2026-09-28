"""Activation manifolds for physical variables (after Wurgaft et al. §2.2, App. A.3), fit at one layer.

Pipeline: pooled activations -> centre -> PCA to k dims (train clips) -> one curve per PCA
coordinate as a function of the label's intrinsic coordinate u (θ in radians for direction, log speed,
acceleration). Two curve families:
  interp : cubic spline through the per-value centroids (periodic for direction) — the paper's choice
  lsq    : least-squares fit of a small basis to all training clips — Fourier harmonics for direction
           (periodic by construction), cubic B-spline with K interior knots otherwise; K chosen on val
Steering (paper Eq. 2, App. A.6): decode s(u_target) in PCA space, lift back, and replace the
in-subspace part of the clip's activation while keeping the orthogonal residual.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, CubicSpline


def coord(name: str, value):
    v = np.asarray(value, dtype=float)
    if name == "direction":
        return np.radians(v)
    if name == "speed":
        return np.log(v)
    return v


def uncoord(name: str, u):
    u = np.asarray(u, dtype=float)
    if name == "direction":
        return np.degrees(u) % 360
    if name == "speed":
        return np.exp(u)
    return u


class Curve:
    """Vector-valued curve u -> R^k."""

    def __init__(self, name, kind, K, lo, hi):
        self.name, self.kind, self.K, self.lo, self.hi = name, kind, K, lo, hi
        self.B = None          # lsq coefficients [n_basis, k]
        self.splines = None    # interp: list of CubicSpline

    def design(self, u):
        u = np.atleast_1d(np.asarray(u, float))
        if self.name == "direction":
            return np.column_stack([np.ones_like(u)] + [f(h * u) for h in range(1, self.K + 1) for f in (np.sin, np.cos)])
        t = np.r_[[self.lo] * 4, np.linspace(self.lo, self.hi, self.K + 2)[1:-1], [self.hi] * 4]
        return BSpline.design_matrix(np.clip(u, self.lo, self.hi), t, 3).toarray()

    def fit_lsq(self, u, Z):
        self.B = np.linalg.lstsq(self.design(u), Z, rcond=None)[0]
        return self

    def fit_interp(self, u, C):
        if self.name == "direction":
            o = np.argsort(u)
            uu, CC = np.r_[u[o], u[o][0] + 2 * np.pi], np.r_[C[o], C[o][:1]]
            self.splines = [CubicSpline(uu, CC[:, j], bc_type="periodic") for j in range(C.shape[1])]
        else:
            o = np.argsort(u)
            self.splines = [CubicSpline(u[o], C[o, j], bc_type="natural") for j in range(C.shape[1])]
        return self

    def __call__(self, u):
        u = np.atleast_1d(np.asarray(u, float))
        if self.name == "direction":
            u = u % (2 * np.pi)
        if self.kind == "lsq":
            return self.design(u) @ self.B
        return np.stack([s(u) for s in self.splines], -1)


@dataclass
class Manifold:
    name: str
    mu: np.ndarray            # [d]
    P: np.ndarray             # [d, k]
    values: np.ndarray        # label values with centroids
    C: np.ndarray             # [n_values, k] centroids in PCA coords
    curve: Curve
    kind: str
    K: int

    def project(self, X):
        return (X - self.mu) @ self.P

    def lift(self, Z):
        return Z @ self.P.T + self.mu

    def point(self, value):
        return self.curve(coord(self.name, value))

    def steer(self, X, value):
        """Replace the in-subspace component of X [n, d] with the curve point for `value` (scalar or [n])."""
        Z = self.project(X)
        T = self.point(value)
        T = np.broadcast_to(T, Z.shape) if T.shape[0] == 1 else T
        return X - self.lift(Z) + self.lift(T)

    def shift(self, X, value, v_hat):
        """Move along the curve: X + P [s(value) - s(v_hat)], v_hat the clip's current value (e.g. probe readout).
        Keeps everything in X except the step along the curve."""
        return X + (self.point(value) - self.point(v_hat)) @ self.P.T

    def span(self, n_dims=None, n=500):
        """Orthonormal basis (in PCA coords) of the subspace the curve moves in: top singular vectors of
        the centred curve samples. n_dims defaults to the basis size of the curve."""
        T = self.curve(self.u_grid(n))
        T = T - T.mean(0)
        U = np.linalg.svd(T, full_matrices=False)[2]
        k = n_dims or (2 * self.K if self.name == "direction" else self.K + 3)
        return U[:k].T                                                        # [k_pca, n_dims]

    def steer_span(self, X, value, n_dims=None):
        """Replace only the curve-span component of X with that of the curve point; everything else kept."""
        S = self.span(n_dims)
        Z = self.project(X)
        T = self.point(value)
        T = np.broadcast_to(T, Z.shape) if T.shape[0] == 1 else T
        Zs = Z - (Z @ S) @ S.T + (T @ S) @ S.T
        return X - self.lift(Z) + self.lift(Zs)

    def u_grid(self, n=2048):
        if self.name == "direction":
            return np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.linspace(self.curve.lo, self.curve.hi, n)

    def nearest_u(self, Z, n=2048):
        """Intrinsic coordinate of the nearest curve point for PCA coords Z [m, k]."""
        us = self.u_grid(n)
        pts = self.curve(us)
        d = ((Z[:, None, :] - pts[None]) ** 2).sum(-1)
        return us[d.argmin(1)], np.sqrt(d.min(1))


def fit_manifold(name, X_train, labels_train, n_pcs=64, kind="lsq", K=2) -> Manifold:
    mu = X_train.mean(0)
    _, _, Vt = np.linalg.svd(X_train - mu, full_matrices=False)
    P = Vt[:n_pcs].T
    lab = np.round(labels_train, 6)
    vals = np.unique(lab)
    Z = (X_train - mu) @ P
    C = np.stack([Z[lab == v].mean(0) for v in vals])
    u_vals = coord(name, vals)
    lo, hi = (0.0, 2 * np.pi) if name == "direction" else (u_vals.min(), u_vals.max())
    curve = Curve(name, kind, K, lo, hi)
    if kind == "lsq":
        curve.fit_lsq(coord(name, labels_train), Z)
    else:
        curve.fit_interp(u_vals, C)
    return Manifold(name, mu, P, vals, C, curve, kind, K)
