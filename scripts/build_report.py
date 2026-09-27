"""Assemble docs/report/kinematics-v-jepa2.html from docs/report/template.html and outputs/results/*.json.

Example clips (test split) are re-encoded to H.264 (the supplied MPEG-4 Part 2 does not play in
browsers) and embedded as data URIs.
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

R = lambda f: json.loads((OUT / "results" / f).read_text())
VARS = ["direction", "speed", "acceleration"]
d = {}

# ---- Experiment 1 (our protocol)
pl = R("probe_layers.json")
d["layers"] = {}
for n in VARS:
    r, k = pl[n]["raw"], ("circ_mae_deg" if n == "direction" else "mae")
    e = d["layers"][n] = {
        "ridge_r2": [m["r2"] for m in r["vjepa2"]], "ridge_ci": [m["r2_ci"] for m in r["vjepa2"]],
        "ridge_err": [m[k] for m in r["vjepa2"]], "random_r2": [m["r2"] for m in r["vjepa2_random"]],
        "random_err": [m[k] for m in r["vjepa2_random"]], "tconcat_r2": [m["r2"] for m in r["vjepa2_tconcat"]],
        "pixels_r2": r["pixels"]["r2"], "centroid_r2": r["centroid_poly2"]["r2"],
        "pixels_err": r["pixels"][k], "centroid_err": r["centroid_poly2"][k]}
    if n == "direction":
        e["transfer_speed"] = [m["transfer_speed_circ_mae_deg"] for m in r["vjepa2"]]
        e["transfer_acc"] = [m["transfer_acceleration_circ_mae_deg"] for m in r["vjepa2"]]

# ---- paper protocol
d["paper"] = {}
for tag in ["vjepa2", "vjepa2_224"]:
    p = R(f"paper_protocol_{tag}.json")
    d["paper"][tag] = {}
    for n in VARS:
        q = p[n]
        e = d["paper"][tag][n] = {
            "exact": q["layerwise"]["exact_mean"], "exact_std": q["layerwise"]["exact_std"],
            "nested": q["layerwise"]["nested_mean"],
            "n_stop_C11": q["orth"]["n_stop_C11"], "n_stop_fig22": q["orth"]["n_stop_fig22"],
            "n_stop_C11_runs": q["orth"]["n_stop_C11_runs"], "n_stop_fig22_runs": q["orth"]["n_stop_fig22_runs"],
            "never": q["orth"]["never_reached_any_seed"]}
        if "steer" in q:
            e["steer"] = {}
            for key, v in q["steer"].items():
                runs = v["runs"]
                Ns = [x["N"] for x in runs[0]["steer"]]
                e["steer"][key] = {
                    "feature_index": v["feature_index"], "N": Ns,
                    "to_target": np.mean([[x["to_target"] for x in r_["steer"]] for r_ in runs], 0).tolist(),
                    "to_target_sd": np.std([[x["to_target"] for x in r_["steer"]] for r_ in runs], 0).tolist(),
                    "to_truth": np.mean([[x["to_truth"] for x in r_["steer"]] for r_ in runs], 0).tolist(),
                    "base_to_target": float(np.mean([r_["baseline"]["to_target"] for r_ in runs])),
                    "base_to_truth": float(np.mean([r_["baseline"]["to_truth"] for r_ in runs])),
                    "K": [r_["K_until_r2_0.1"] for r_ in runs],
                    "min_r2": [r_["min_train_seq_val_r2"] for r_ in runs],
                    "eval_fit_r2": [r_["eval_probe_fit_r2_on_test"] for r_ in runs]}
d["protocol_check"] = R("probe_protocol_check.json")

# ---- Experiment 2 (our protocol)
inl = R("inlp.json")
d["inlp"] = {"layer": inl["layer"], "sweep": inl["sweep"],
             "main": {n: {s: [m["r2"] for m in inl["main"][n][s]["test"]] for s in ["probe", "random", "pca"]}
                      for n in VARS}}

# ---- Experiment 3
sp = R("steer_pooled.json")
d["steer_pooled"] = {"layer": sp["layer"], "n_list": sp["n_list"]}
for n in VARS:
    s = sp[n]
    d["steer_pooled"][n] = {"clean": s["clean"], "targets": s["targets"],
                            "strict": [x["to_target"]["strict"] for x in s["steer"]],
                            "paper": [x["to_target"]["paper"] for x in s["steer"]],
                            "truth": [x["to_truth"]["strict"] for x in s["steer"]],
                            "random": [x["to_target"]["strict"] for x in s["random"]],
                            "theta": [x.get("theta_mae") for x in s["steer"]]}
d["steer_prop"] = {}
for n in VARS:
    s = R(f"steer_propagate_{n}.json")
    c = s["conditions"]
    d["steer_prop"][n] = {
        "layers": s["layers"], "n_list": s["n_list"], "clean_truth": c[0]["to_truth"],
        "steer": {str(N): np.mean([x["to_target"] for x in c if x["kind"] == "steer" and x["N"] == N], 0).tolist()
                  for N in s["n_list"]},
        "random": np.mean([x["to_target"] for x in c if x["kind"] == "random"], 0).tolist()}
d["persist"] = R("diag_persistence.json")

# ---- data checks
sc = R("sanity_checks.json")
d["sanity"] = sc
d["disp"] = {n: {"x": sc[n]["label_values"], "y": sc[n]["mean_displacement_px"]} for n in ["speed", "acceleration"]}


# ---- example clips
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
for n, col, picks in [("direction", "theta_degrees", [0, 90, 225]), ("speed", "speed_mps", [0.25, 2.0, 4.0]),
                      ("acceleration", "acceleration_mps2", [0.25, 5.0, 10.0])]:
    df = load_manifest(n)
    te = set(split_idx(n)["test"].tolist())
    for p in picks:
        i = [j for j in df.index[np.isclose(df[col], p, atol=0.08)]
             if j in te and (n != "direction" or df.motion[j] == "velocity")][0]
        lab = {"direction": f"θ = {df.theta_degrees[i]:g}°", "speed": f"{df.speed_mps[i]:.2f} m/s",
               "acceleration": f"{df.acceleration_mps2[i]:.2f} m/s²"}[n]
        meta = (f"{df.motion[i].replace('velocity', 'constant velocity')}, "
                + (f"{df.speed_mps[i]:g} m/s" if df.motion[i] == "velocity" else f"{df.acceleration_mps2[i]:g} m/s²")
                if n == "direction" else f"θ = {df.theta_degrees[i]:g}°")
        vids.append({"src": h264_data_uri(df.video[i]), "dataset": n, "label": lab, "meta": meta})

js = lambda o: json.dumps(o, separators=(",", ":")).replace("</", "<\\/")
t = (ROOT / "docs" / "report" / "template.html").read_text()
t = t.replace("__DATA__", js(d)).replace("__VIDS__", js(vids))
(ROOT / "docs" / "report" / "kinematics-v-jepa2.html").write_text(t)
print("wrote docs/report/kinematics-v-jepa2.html", len(t) // 1024, "KB")
