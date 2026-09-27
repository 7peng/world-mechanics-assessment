"""Summary figure: probe type x data regime, paper-protocol steering, and persistence of injected shifts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import OUT
from src.plotting import COLORS, INK2, NEUTRAL, plt

R = lambda f: json.loads((OUT / "results" / f).read_text())
PAPER = "#2c3140"
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
pc = R("probe_protocol_check.json")
L = pc["layers"]
sty = {"ridge_z": (COLORS["direction"], "-", "ridge, standardised (ours)"), "ridge_raw": (COLORS["direction"], "--", "ridge, raw"),
       "gd_paper": (PAPER, "-", "gradient probe, 100 epochs (paper)"), "gd_paper_long": (PAPER, ":", "gradient probe, 1000 epochs"),
       "gd_paper_z": (PAPER, "--", "gradient probe, standardised")}
for a, reg, t in ((ax[0, 0], "full", "Direction R², full set (1,500 clips)"),
                  (ax[0, 1], "paperlike", "Direction R², paper-like subset (96 clips)")):
    for p, (c, ls, lab) in sty.items():
        a.plot(L, [x["r2"] for x in pc[reg][p]], ls, color=c, marker="o", ms=3, label=lab)
    a.plot([1, 9], [0.22, 0.93], "o", mfc="none", mec=PAPER, ms=8, label="paper Fig. 2c (approx.)")
    a.set_ylim(0, 1.02); a.set_xlabel("our layer (paper = ours − 1)"); a.set_ylabel("5-fold CV R²"); a.set_title(t)
ax[0, 0].legend(fontsize=8)
for tag, ls in (("vjepa2", "-"), ("vjepa2_224", "--")):
    s = R(f"paper_protocol_{tag}.json")["direction"]["steer"]
    for key, c in (("block_output", PAPER), ("embedding_is_0", COLORS["direction"])):
        runs = s[key]["runs"]
        N = [x["N"] for x in runs[0]["steer"]]
        tt = np.array([[x["to_target"] for x in r["steer"]] for r in runs])
        px = "256" if tag == "vjepa2" else "224"
        ax[0, 2].plot(N, tt.mean(0), ls, color=c, marker="o", ms=3,
                      label=f"{px} px, layer 8 = {'block-8 output' if key == 'block_output' else 'hidden state 8'}")
        if tag == "vjepa2":
            ax[0, 2].fill_between(N, tt.min(0), tt.max(0), color=c, alpha=0.12, lw=0)
ax[0, 2].plot([1, 2, 3, 5, 10, 15, 20], [77, 66, 61, 51, 24, 14, 11.9], "o", mfc="none", mec=INK2, ms=8, label="paper Fig. 24 (approx.)")
ax[0, 2].set_xlabel("probes in steering subspace (N)"); ax[0, 2].set_ylabel("error to 90° target (°)")
ax[0, 2].set_title("Paper-protocol steering, 3 splits"); ax[0, 2].legend(fontsize=8)
d = R("diag_persistence.json")
for a, n in zip(ax[1], ["direction", "speed", "acceleration"]):
    P = d[n]
    a.plot(P["layers"], P["retained_frac"], "-o", color=COLORS[n], ms=3, label="steering edit (N=20)")
    a.plot(P["layers"], P["retained_frac_natural"], "--", color=PAPER, label="real clip-to-clip difference")
    a.plot(P["layers"], P["retained_frac_random"], "-", color=NEUTRAL, label="random edit, same norm")
    a.set_ylim(0, 1.02); a.set_xlabel("layer"); a.set_ylabel("fraction of Δ retained along Δ")
    a.set_title(f"{n}: injected shift, injected at layer 12")
ax[1, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "figures" / "summary.png", dpi=120)
