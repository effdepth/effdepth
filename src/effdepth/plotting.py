"""Figure style for the manuscript: validated palette, thin marks, recessive axes."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# categorical slots, fixed order (validated: adjacent CVD and normal-vision gates pass)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# ordinal blue ramp, light to dark, lightest step clears 2:1 on the light surface
ORD = {3: ["#6da7ec", "#256abf", "#104281"],
       4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
       5: ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]}   # at most 5 visible steps
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e2e1dc"
FULL, HALF = 6.85, 3.35  # inches: 174 mm and 85 mm text widths


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
        "mathtext.fontset": "cm", "axes.linewidth": 0.6, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 4.5, "legend.frameon": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def ordinal(n: int) -> list[str]:
    """Ordered series colours; the blue ramp holds at most 5 visibly distinct steps."""
    if n not in ORD:
        raise ValueError("at most 5 ordered series: facet or drop series")
    return ORD[n]
