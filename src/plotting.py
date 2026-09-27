"""Shared figure style. Fixed categorical colors per physical variable."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"direction": "#2a78d6", "speed": "#eb6834", "acceleration": "#1baf7a"}
NEUTRAL = "#8a8983"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 5, "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
})
