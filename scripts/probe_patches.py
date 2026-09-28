"""Per-patch direction probes (Joseph et al. App. C.5, Fig. 18), on time-averaged patch tokens.

(a) One ridge probe per spatial patch (256 per layer): test R² distribution vs layer, against the
    pooled probe.
(b) Cross-half generalisation: probe trained on left-half patch tokens (train clips), evaluated on
    left-half (in-distribution) and right-half (out-of-distribution) tokens of test clips.
Saves outputs/results/probe_patches.json and outputs/figures/report/f7_patches.png.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import OUT, load_manifest
from src.features import split_idx
from src.probes import fit_ridge, metrics, r2, targets

name = "direction"
df = load_manifest(name)
y = targets(name, df)
idx = split_idx(name)
pdir = OUT / "features" / "vjepa2_patch"
assert (np.load(pdir / f"{name}_ids.npy") == df.id.values).all()
layers = sorted(int(p.stem.split("_L")[1]) for p in pdir.glob(f"{name}_L*.npy"))
pooled = {i: m["r2"] for i, m in enumerate(json.loads((OUT / "results" / "probe_layers.json").read_text())[name]["raw"]["vjepa2"])}
left = np.array([p for p in range(256) if p % 16 < 8])
right = np.array([p for p in range(256) if p % 16 >= 8])

RES = OUT / "results" / "probe_patches.json"
if "--plot-only" in sys.argv and RES.exists():
    res = json.loads(RES.read_text())
    res = {k: ({int(a): b for a, b in v.items()} if isinstance(v, dict) else v) for k, v in res.items()}
    layers = res["layers"]
else:
    res = {"layers": layers, "per_patch_r2": {}, "pooled_r2": {}, "cross_half": {}}
for L in ([] if "--plot-only" in sys.argv else layers):
    F = np.load(pdir / f"{name}_L{L}.npy").astype(np.float32)          # [N, 256, d]
    rr = []
    for p in range(256):
        q = fit_ridge(F[idx["train"], p], y[idx["train"]], F[idx["val"], p], y[idx["val"]])
        rr.append(r2(y[idx["test"]], q.predict(F[idx["test"], p])))
    res["per_patch_r2"][L] = rr
    res["pooled_r2"][L] = pooled[L]
    # cross-half: tokens as samples
    tok = lambda rows, patches: F[rows][:, patches].reshape(-1, F.shape[-1])
    lab = lambda rows, patches: np.repeat(y[rows], len(patches), axis=0)
    q = fit_ridge(tok(idx["train"], left), lab(idx["train"], left), tok(idx["val"], left), lab(idx["val"], left))
    res["cross_half"][L] = {"train_left": r2(lab(idx["train"], left), q.predict(tok(idx["train"], left))),
                            "test_left": metrics(name, lab(idx["test"], left), q.predict(tok(idx["test"], left))),
                            "test_right": metrics(name, lab(idx["test"], right), q.predict(tok(idx["test"], right)))}
    print(f"L{L}: per-patch R² median {np.median(rr):.3f} max {max(rr):.3f} | pooled {pooled[L]:.3f} | "
          f"cross-half ID {res['cross_half'][L]['test_left']['r2']:.3f} OOD {res['cross_half'][L]['test_right']['r2']:.3f}", flush=True)
if "--plot-only" not in sys.argv:
    RES.write_text(json.dumps(res, indent=1))

# ---- figure
C, DARK, GRAY, LIGHT = "#3b74c4", "#333333", "#8c8c8c", "#c8c8c8"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titlelocation": "left", "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": LIGHT, "axes.linewidth": 0.8,
                     "xtick.color": GRAY, "ytick.color": GRAY, "axes.labelcolor": DARK, "text.color": DARK,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.5})
show = [l for l in (1, 4, 8, 9, 12, 24) if l in layers]
fig = plt.figure(figsize=(11, 5.8))
gs = fig.add_gridspec(2, len(show), height_ratios=[1.6, 1], hspace=0.55)
clip = lambda L: np.clip(res["per_patch_r2"][L], 0, 1)   # a few corner patches at layers 1–2 have R² << 0
a = fig.add_subplot(gs[0, : len(show) // 2])
vp = a.violinplot([clip(L) for L in layers], positions=layers, widths=1.2, showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor(C); b.set_alpha(0.35); b.set_edgecolor("none")
a.plot(layers, [np.median(res["per_patch_r2"][L]) for L in layers], color=C, marker="o", ms=3)
a.plot(layers, [res["pooled_r2"][L] for L in layers], color=DARK, ls=(0, (4, 2)), marker="o", ms=3)
a.annotate("per-patch, median", (layers[-1], np.median(res["per_patch_r2"][layers[-1]])), xytext=(4, -8), textcoords="offset points", fontsize=8, color=C)
a.annotate("pooled probe", (layers[-1], res["pooled_r2"][layers[-1]]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=DARK)
a.set_xlim(0, 31); a.set_ylim(-0.05, 1.03); a.set_xlabel("layer"); a.set_ylabel("test R²"); a.set_title("one probe per spatial patch (R² clipped at 0)")
b = fig.add_subplot(gs[0, len(show) // 2:])
for key, c, lab in (("test_left", C, "same half as training (ID)"), ("test_right", DARK, "other half (OOD)")):
    v = [res["cross_half"][L][key]["r2"] for L in layers]
    b.plot(layers, v, color=c, marker="o", ms=3)
    b.annotate(lab, (layers[-1], v[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=c, va="center")
b.set_xlim(0, 36); b.set_ylim(-0.05, 1.03); b.set_xlabel("layer"); b.set_title("probe trained on left-half patch tokens")
for k, L in enumerate(show):
    h = fig.add_subplot(gs[1, k])
    im = h.imshow(np.array(res["per_patch_r2"][L]).reshape(16, 16), vmin=0, vmax=1, cmap="Blues")
    h.set_title(f"layer {L}", fontsize=8.5); h.set_xticks([]); h.set_yticks([])
    for s in h.spines.values():
        s.set_visible(False)
fig.colorbar(im, ax=fig.axes[2:], fraction=0.015, pad=0.01).set_label("per-patch test R²", fontsize=8)
fig.suptitle("Per-patch direction probes", x=0.01, ha="left", fontsize=11)
fig.savefig(OUT / "figures" / "report" / "supp_patches.png", dpi=150, bbox_inches="tight", pad_inches=0.08)
