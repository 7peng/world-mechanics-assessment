"""Part 2, step 1: fit activation manifolds at layer 12 and check them.

For each variable: PCA-64 on train; curves of two kinds (interp = paper's cubic through centroids,
lsq = least-squares basis fit to all train clips with basis size K chosen on val). Checks:
  - val mean squared distance of clips' projections to s(label): curve vs centroid vs global mean
  - held-out values: curve fit on 48 values; RMS distance of the 16 withheld centroids to it,
    in units of the mean neighbour-centroid spacing
  - nearest-point decoding of val clips (the manifold as a decoder)
  - direction ring: atan2 of the top-2 centroid PCs vs θ
Saves outputs/results/manifolds.json, pickles in outputs/manifolds/, figure supp_manifolds.png.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import DATASETS, OUT, load_manifest
from src.features import load_features, split_idx
from src.manifold import fit_manifold, uncoord
from src.splits import LABEL, load_splits

L, N_PCS = 12, 64
KGRID = {"direction": [1, 2, 3, 4, 6, 8], "speed": [0, 1, 2, 3, 4, 6, 8], "acceleration": [0, 1, 2, 3, 4, 6, 8]}
(OUT / "manifolds").mkdir(exist_ok=True)
res = {"layer": L, "n_pcs": N_PCS}
mans = {}
msd = lambda A, B: float(((A - B) ** 2).sum(1).mean())
for name in DATASETS:
    df = load_manifest(name)
    lab = df[LABEL[name]].values
    X = load_features(name)[:, L].astype(np.float64)
    idx = split_idx(name)
    r = res[name] = {}
    Xtr, ltr, Xva, lva = X[idx["train"]], lab[idx["train"]], X[idx["val"]], lab[idx["val"]]
    # K on val
    val_err = {}
    for K in KGRID[name]:
        m = fit_manifold(name, Xtr, ltr, N_PCS, "lsq", K)
        val_err[K] = msd(m.project(Xva), m.point(lva))
    K = min(val_err, key=val_err.get)
    m_lsq = fit_manifold(name, Xtr, ltr, N_PCS, "lsq", K)
    m_int = fit_manifold(name, Xtr, ltr, N_PCS, "interp", 0)
    mans[name] = {"lsq": m_lsq, "interp": m_int}
    Zv = m_lsq.project(Xva)
    r.update({"K_by_val": val_err, "K": K,
              "val_msd": {"lsq": msd(Zv, m_lsq.point(lva)), "interp": msd(Zv, m_int.point(lva)),
                          "centroid": msd(Zv, m_lsq.C[np.searchsorted(m_lsq.values, np.round(lva, 6))]),
                          "global_mean": msd(Zv, Zv.mean(0))}})
    # held-out values
    ih = split_idx(name, "value_heldout")
    held = np.array(load_splits()[name]["heldout_values"])
    r["heldout_rmsd_over_step"] = {}
    for kind, KK in (("lsq", K), ("interp", 0)):
        mh = fit_manifold(name, X[ih["train"]], lab[ih["train"]], N_PCS, kind, KK)
        Zh = mh.project(X[ih["heldout_values"]])
        labh = np.round(lab[ih["heldout_values"]], 6)
        Ch = np.stack([Zh[labh == v].mean(0) for v in held])
        step = np.linalg.norm(np.diff(mh.C, axis=0), axis=1).mean()
        r["heldout_rmsd_over_step"][kind] = float(np.sqrt((np.linalg.norm(Ch - mh.point(held), axis=1) ** 2).mean()) / step)
    # nearest-point decoding on val
    r["val_nearest_point_error"] = {}
    for kind, m in mans[name].items():
        u_hat, _ = m.nearest_u(m.project(Xva))
        pred = uncoord(name, u_hat)
        e = np.abs((pred - lva + 180) % 360 - 180) if name == "direction" else np.abs(pred - lva)
        r["val_nearest_point_error"][kind] = float(e.mean())
    if name == "direction":
        pc = m_lsq.C - m_lsq.C.mean(0)
        s = np.linalg.svd(pc, compute_uv=False)
        _, _, V2 = np.linalg.svd(pc, full_matrices=False)
        ang = np.degrees(np.arctan2(pc @ V2[1], pc @ V2[0])) % 360
        r["ring_2d_variance_share"] = float((s[:2] ** 2).sum() / (s ** 2).sum())
        r["ring_atan2_vs_theta_maxcorr"] = float(max(np.corrcoef(np.unwrap(np.radians(sg * ang + o)), np.radians(m_lsq.values))[0, 1]
                                                     for sg in (1, -1) for o in np.arange(0, 360, 2)))
    for kind, m in mans[name].items():
        with open(OUT / "manifolds" / f"{name}_L{L}_{kind}.pkl", "wb") as f:
            pickle.dump(m, f)
    print(name, json.dumps(r, default=lambda o: round(float(o), 3)))
(OUT / "results" / "manifolds.json").write_text(json.dumps(res, indent=1))

# ---- figure
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.linewidth": 0,
                     "axes.grid": True, "grid.color": "white", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.spines.left": False, "axes.spines.bottom": False, "xtick.major.size": 0, "ytick.major.size": 0,
                     "xtick.color": "#555", "ytick.color": "#555", "font.size": 9, "axes.titlesize": 10.5, "legend.frameon": False})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
for a, name in zip(axes, DATASETS):
    m, mi = mans[name]["lsq"], mans[name]["interp"]
    pc = m.C - m.C.mean(0)
    _, _, V = np.linalg.svd(pc, full_matrices=False)
    B = V[:2].T
    us = m.u_grid(400)
    a.plot(*((mi.curve(us) - m.C.mean(0)) @ B).T, color="#aaa", lw=1)
    a.plot(*((m.curve(us) - m.C.mean(0)) @ B).T, color="#333", lw=1.4)
    sc = a.scatter(*(pc @ B).T, c=m.values, cmap="twilight" if name == "direction" else "viridis", s=14, zorder=3)
    a.set_title(name); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(sc, ax=a, fraction=0.04, pad=0.02).set_label({"direction": "θ (°)", "speed": "m/s", "acceleration": "m/s²"}[name], fontsize=8)
axes[0].set_ylabel("top-2 PCs of the centroid cloud")
axes[-1].legend(handles=[plt.Line2D([], [], color="#333", label="least-squares curve (K on val)"),
                         plt.Line2D([], [], color="#aaa", label="interpolating cubic (paper)"),
                         plt.Line2D([], [], ls="none", marker="o", color="#555", label="per-value centroids")],
                loc="upper left", bbox_to_anchor=(1.3, 1.0), borderaxespad=0)
fig.tight_layout()
fig.savefig(OUT / "figures" / "report" / "supp_manifolds.png", dpi=150, bbox_inches="tight", pad_inches=0.1,
            bbox_extra_artists=[axes[-1].get_legend()] + fig.get_default_bbox_extra_artists())
