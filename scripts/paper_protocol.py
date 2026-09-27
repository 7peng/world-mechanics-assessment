"""Joseph et al. (arXiv 2602.07050) protocol, followed as literally as the paper allows.

Stated by the paper, and implemented here as written:
  App. B   linear probe f(h)=Wh+b on spatiotemporally mean-pooled residual stream, layers 0..n-1
           (= output of block 0..23 = our hidden_states[1..24]); gradient-trained;
           sweep lr {1e-4,3e-4,1e-3,3e-3,5e-3} x weight decay {0.01,0.1,0.4,0.8};
           "selecting the best model based on validation performance"; 5-fold grouped CV;
           mean ± std across folds; y-axis "Validation R²" (Fig. 2).
  C.11     orthogonal probe sequence: Adam lr 1e-3, weight decay 1e-4, 100 epochs (direction) /
           50 (speed); 80/20 split, fixed seed; Q_k = QR(W_k^T); X <- X - X Q_k Q_k^T;
           stop: direction R² < 0.1 or circ. MAE > 80°; speed R² < 0.05 or MAE > 90% of random
           baseline. (Fig. 22 caption instead uses direction R² < 0.3, speed R² < 0.1: both reported.)
  C.12     70/30 train/test split; orthogonal probes (25, until R² < 0.1) on train; held-out
           evaluation probe trained on *test* activations only; V = QR([W_1^T..W_K^T]);
           c = V^T x, x_perp = x - V c; c* by least squares s.t. all probes predict θ*;
           x* = V c* + x_perp; steer test clips to θ* = 90° with N in {1,2,3,5,10,15,20} probes;
           report MAE (deg) of the held-out probe to target and to true labels. Layer 8.
  Direction target (sin θ, cos θ), MSE; speed scalar, MSE. Fig. 2 R² for 2-output targets is taken
  as the uniform average over outputs (sklearn default).

NOT stated by the paper; our choices (flagged in outputs):
  - optimizer for App. B sweep: AdamW (decoupled decay; values up to 0.8 imply it); C.11 says "Adam
    with weight decay" -> torch Adam (L2 penalty)
  - epochs for App. B probes: same as C.11 (100 direction, 50 speed/acceleration); batch size 32;
    PyTorch nn.Linear default init; no feature normalization
  - CV group key: the (θ, motion type, magnitude) label combination
  - "validation performance" = R² on the held-out fold (the same fold that is reported); we also
    report a *nested* variant (config chosen on an inner split of the training folds)
  - acceleration uses the speed-probe settings and stopping rule
  - datasets: supplied direction / speed / acceleration sets (paper: its own Kubric sets)
  - input resolution: run at 256 (native checkpoint) and 224 (App. C.6 states 224x224)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features
from src.probes import angle_deg, circ_err, targets

DEV = "cuda"
LRS = [1e-4, 3e-4, 1e-3, 3e-3, 5e-3]
WDS = [0.01, 0.1, 0.4, 0.8]
EPOCHS = {"direction": 100, "speed": 50, "acceleration": 50}
BS = 32


# ---------------------------------------------------------------- batched gradient-trained probes
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
        perm = torch.randperm(N, device=DEV, generator=g)
        for s in range(0, N, bs):
            idx = perm[s:s + bs]
            xb, yb = X[:, idx], Y[idx]                                   # [L,B,d], [B,k]
            pred = torch.einsum("lbd,lcdk->lcbk", xb, W) + b              # [L,C,B,k]
            err = pred - yb                                              # dMSE/dpred = 2 err / (B k)
            scale = 2.0 / (len(idx) * k)
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
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=DEV)


# ---------------------------------------------------------------- App. B: layer-wise probes
def layerwise(name, F, y, groups):
    X = to_dev(F[:, 1:].transpose(1, 0, 2))           # [24, N, d], paper layers 0..23
    Y = to_dev(y)
    cfg_lr = [lr for lr in LRS for _ in WDS]
    cfg_wd = [wd for _ in LRS for wd in WDS]
    exact, nested, chosen = [], [], []
    rng = np.random.default_rng(0)
    for fold, (tr, va) in enumerate(GroupKFold(5).split(F, groups=groups)):
        W, b = train_linear(X[:, tr], Y[tr], cfg_lr, cfg_wd, EPOCHS[name], decoupled=True, seed=fold)
        sc = r2_t(Y[va], predict(X[:, va], W, b))       # [24, 20]
        best = sc.argmax(1)
        exact.append(sc.max(1).values.cpu().numpy())
        chosen.append(best.cpu().numpy())
        # nested: choose config on an inner 80/20 split of the training folds, report on the fold
        itr = rng.permutation(tr)
        iva, itr = itr[: len(itr) // 5], itr[len(itr) // 5:]
        W2, b2 = train_linear(X[:, itr], Y[itr], cfg_lr, cfg_wd, EPOCHS[name], decoupled=True, seed=100 + fold)
        bi = r2_t(Y[iva], predict(X[:, iva], W2, b2)).argmax(1)
        sc2 = r2_t(Y[va], predict(X[:, va], W2, b2))
        nested.append(sc2.gather(1, bi[:, None])[:, 0].cpu().numpy())
    exact, nested = np.stack(exact), np.stack(nested)
    return {"exact_mean": exact.mean(0).tolist(), "exact_std": exact.std(0).tolist(),
            "nested_mean": nested.mean(0).tolist(), "nested_std": nested.std(0).tolist(),
            "chosen_config_idx": np.stack(chosen).tolist()}


# ---------------------------------------------------------------- C.11: orthogonal probe sequence
def probe_metrics(name, y, yhat):
    """y [N,k] numpy; yhat [L, N, k] numpy -> dict of [L] arrays"""
    ss_res = ((yhat - y) ** 2).sum(1)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    out = {"r2": (1 - ss_res / ss_tot).mean(-1)}
    if name == "direction":
        e = np.stack([circ_err(angle_deg(p), angle_deg(y)) for p in yhat])
        out["circ_mae"] = e.mean(1)
        out["acc15"] = (e <= 15).mean(1)
    else:
        out["mae"] = np.abs(yhat[..., 0] - y[:, 0]).mean(1)
    return out


def orthogonal_sequence(name, Xl, y, tr, va, n_iter, seed=0):
    """Run C.11 on features Xl [L, N, d] (numpy) with fixed train/val indices.
    Returns per-iteration val metrics, and the probes (W_k [L,d,k], b_k [L,k]) and projections."""
    X = to_dev(Xl)
    Y = to_dev(y)
    hist, probes = [], []
    for it in range(n_iter):
        W, b = train_linear(X[:, tr], Y[tr], [1e-3], [1e-4], EPOCHS[name], decoupled=False, seed=seed + it)
        W, b = W[:, 0], b[:, 0]                                         # [L,d,k], [L,k]
        yhat = (torch.einsum("lnd,ldk->lnk", X[:, va], W) + b[:, None]).cpu().numpy()
        hist.append({k: v.tolist() for k, v in probe_metrics(name, y[va], yhat).items()})
        Q, _ = torch.linalg.qr(W)                                       # [L,d,k] orthonormal basis of W_k
        probes.append((W.cpu().numpy(), b.cpu().numpy(), Q.cpu().numpy()))
        X = X - torch.einsum("lnd,ldk,lek->lne", X, Q, Q)               # X <- X - X Q Q^T
    return hist, probes


def n_until(hist, key, thr, below=True):
    """number of probes trained before the metric first crosses thr, per layer (None if never)."""
    v = np.array([h[key] for h in hist])                                # [iters, L]
    hit = v < thr if below else v > thr
    return [int(np.argmax(hit[:, l])) if hit[:, l].any() else None for l in range(v.shape[1])]


# ---------------------------------------------------------------- C.12: steering
def steer_eval(F9, y, theta, n_list, target=90.0, seed=0, n_probes=25):
    """F9: [N, d] layer-8 (paper index) features of the direction set."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(F9))
    n_tr = int(round(0.7 * len(F9)))
    tr, te = perm[:n_tr], perm[n_tr:]
    # steering probes: orthogonal sequence on train activations (its own 80/20 split for stopping)
    p = rng.permutation(tr)
    itr, iva = p[: int(0.8 * len(p))], p[int(0.8 * len(p)):]
    Xtr = F9[None]  # [1, N, d]
    hist, probes = orthogonal_sequence("direction", Xtr, y, itr, iva, n_probes, seed=1000)
    r2s = [h["r2"][0] for h in hist]
    K = next((i for i, r in enumerate(r2s) if r < 0.1), n_probes)  # "25 probes until R² < 0.1"
    # held-out evaluation probe: trained on (clean) test activations only
    We, be = train_linear(to_dev(F9[te])[None], to_dev(y[te]), [1e-3], [1e-4], EPOCHS["direction"],
                          decoupled=False, seed=2000)
    We, be = We[0, 0].cpu().numpy(), be[0, 0].cpu().numpy()
    fit_r2 = float(probe_metrics("direction", y[te], (F9[te] @ We + be)[None])["r2"][0])
    ev = lambda X: angle_deg(X @ We + be)
    Xte = F9[te].astype(np.float64)
    t = np.array([np.sin(np.radians(target)), np.cos(np.radians(target))])
    base_to_truth = float(circ_err(ev(Xte), theta[te]).mean())
    base_to_target = float(circ_err(ev(Xte), target).mean())
    out = {"n_probes_trained": n_probes, "K_until_r2_0.1": K, "train_seq_val_r2": r2s,
           "eval_probe_fit_r2_on_test": fit_r2, "n_train": len(tr), "n_test": len(te),
           "baseline": {"to_truth": base_to_truth, "to_target": base_to_target}, "steer": []}
    for N in n_list:
        Ws = [pr[0][0].astype(np.float64) for pr in probes[:N]]        # W_k [d, 2]
        bs = [pr[1][0].astype(np.float64) for pr in probes[:N]]
        Qs = [pr[2][0].astype(np.float64) for pr in probes[:N]]
        V, _ = np.linalg.qr(np.concatenate(Ws, 1))                      # Eq. 8
        c = Xte @ V
        x_perp = Xte - c @ V.T
        # probe k acts on X^(k) = X P_1 ... P_{k-1};  P_j = I - Q_j Q_j^T
        rows, rhs_off = [], []
        for k in range(N):
            Mk = Ws[k]
            for j in reversed(range(k)):
                Mk = Mk - Qs[j] @ (Qs[j].T @ Mk)                        # P_{<k} W_k (P symmetric)
            rows.append(V.T @ Mk)                                        # [2N, 2]
            rhs_off.append((x_perp @ Mk) + bs[k])                        # [n, 2]
        A = np.concatenate(rows, 1).T                                    # [2N, 2N]
        R = np.concatenate([t[None] - o for o in rhs_off], 1)            # [n, 2N]
        cstar = np.linalg.lstsq(A, R.T, rcond=None)[0].T                 # [n, 2N]
        xs = cstar @ V.T + x_perp
        out["steer"].append({"N": N, "to_target": float(circ_err(ev(xs), target).mean()),
                             "to_truth": float(circ_err(ev(xs), theta[te]).mean())})
    return out


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="vjepa2", help="vjepa2 (256px) or vjepa2_224")
    ap.add_argument("--parts", default="layerwise,orth,steer")
    ap.add_argument("--orth-iter", type=int, default=100)
    args = ap.parse_args()
    parts = args.parts.split(",")
    res = {"features": args.features, "layer_index": "paper (0 = output of block 0)"}
    for name in DATASETS:
        df = load_manifest(name)
        F = load_features(name, args.features).astype(np.float32)
        y = targets(name, df)
        groups = df[["theta_degrees", "motion", "speed_mps", "acceleration_mps2"]].astype(str).agg("|".join, axis=1)
        r = res[name] = {}
        if "layerwise" in parts:
            r["layerwise"] = layerwise(name, F, y, groups.values)
            print(name, "layerwise exact R²:", " ".join(f"{v:.2f}" for v in r["layerwise"]["exact_mean"]), flush=True)
            print(name, "layerwise nested R²:", " ".join(f"{v:.2f}" for v in r["layerwise"]["nested_mean"]), flush=True)
        if "orth" in parts:
            rng = np.random.default_rng(0)
            perm = rng.permutation(len(F))
            tr, va = perm[: int(0.8 * len(F))], perm[int(0.8 * len(F)):]
            hist, _ = orthogonal_sequence(name, F[:, 1:].transpose(1, 0, 2), y, tr, va, args.orth_iter)
            o = r["orth"] = {"hist": hist}
            if name == "direction":
                o["n_stop_C11"] = [min(a, b) if a is not None and b is not None else (a if b is None else b)
                                   for a, b in zip(n_until(hist, "r2", 0.1), n_until(hist, "circ_mae", 80, below=False))]
                o["n_stop_fig22"] = n_until(hist, "r2", 0.3)
            else:
                base = float(np.abs(y[va, 0] - y[tr, 0].mean()).mean())      # constant-mean predictor
                a, b = n_until(hist, "r2", 0.05), n_until(hist, "mae", 0.9 * base, below=False)
                o["n_stop_C11"] = [min(x for x in (p, q) if x is not None) if (p is not None or q is not None) else None
                                   for p, q in zip(a, b)]
                o["n_stop_fig22"] = n_until(hist, "r2", 0.1)
            print(name, "orth probes until stop (C.11):", o["n_stop_C11"], flush=True)
            print(name, "orth probes until stop (Fig.22):", o["n_stop_fig22"], flush=True)
        if "steer" in parts and name == "direction":
            r["steer"] = steer_eval(F[:, 9], y, df.theta_degrees.values, [1, 2, 3, 5, 10, 15, 20])
            print("steer:", json.dumps({k: v for k, v in r["steer"].items() if k != "train_seq_val_r2"}), flush=True)
    (OUT / "results").mkdir(exist_ok=True)
    (OUT / "results" / f"paper_protocol_{args.features}.json").write_text(json.dumps(res, indent=1))
