"""Assemble docs/report/kinematics-v-jepa2.html from docs/report/template.html.

Embeds the three figures from scripts/plot_figures.py and three example clips (test split, one per
set; re-encoded to H.264 because browsers do not play the supplied MPEG-4 Part 2) as data URIs.
"""
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import av
import numpy as np

from src.data import OUT, ROOT, load_manifest, read_video
from src.features import split_idx

FIG = OUT / "figures" / "report"


def h264_data_uri(path):
    frames = read_video(path)
    buf = io.BytesIO()
    out = av.open(buf, "w", format="mp4")
    st = out.add_stream("libx264", rate=24)
    st.width, st.height, st.pix_fmt = 256, 256, "yuv420p"
    st.options = {"crf": "18", "preset": "slow"}
    for fr in frames:
        for pk in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
            out.mux(pk)
    for pk in st.encode():
        out.mux(pk)
    out.close()
    return "data:video/mp4;base64," + base64.b64encode(buf.getvalue()).decode()


vids = []
for n, col, pick in [("direction", "theta_degrees", 225), ("speed", "speed_mps", 2.0), ("acceleration", "acceleration_mps2", 5.0)]:
    df = load_manifest(n)
    te = set(split_idx(n)["test"].tolist())
    i = [j for j in df.index[np.isclose(df[col], pick, atol=0.08)] if j in te and (n != "direction" or df.motion[j] == "velocity")][0]
    lab = {"direction": f"θ = {df.theta_degrees[i]:g}°", "speed": f"{df.speed_mps[i]:.2f} m/s",
           "acceleration": f"{df.acceleration_mps2[i]:.2f} m/s²"}[n]
    vids.append({"src": h264_data_uri(df.video[i]), "dataset": n, "label": lab})

t = (ROOT / "docs" / "report" / "template.html").read_text()
for k, name in enumerate(["f1_probes", "f2_nullspace", "f3_steering"], 1):
    uri = "data:image/png;base64," + base64.b64encode((FIG / f"{name}.png").read_bytes()).decode()
    t = t.replace(f"__F{k}__", uri)
t = t.replace("__VIDS__", json.dumps(vids, separators=(",", ":")).replace("</", "<\\/"))
(ROOT / "docs" / "report" / "kinematics-v-jepa2.html").write_text(t)
print("wrote docs/report/kinematics-v-jepa2.html", len(t) // 1024, "KB")
