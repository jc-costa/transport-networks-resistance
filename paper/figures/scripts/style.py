"""Shared figure style for the TNANO paper: IEEE column widths, Times-like
serif text at 8 pt, thin marks, recessive grid. Every figure script imports
this module and writes a vector PDF to figures/.

The pipeline folder (this repository) is read-only input; its path comes from
the COLAB environment variable (default: the root of this repository).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB = os.environ.get("COLAB", os.path.normpath(os.path.join(HERE, "..", "..", "..")))
FIGDIR = os.path.normpath(os.path.join(HERE, ".."))
DATADIR = os.path.join(FIGDIR, "data")

ONE_COL = 3.5      # in
TWO_COL = 7.16     # in
R0 = 1000.0        # Ohm per unit length in the thesis; paper reports R in units of R0*L

# categorical slots 1-3 (validated all-pairs for up to three series), with
# markers as a second encoding so identity never rests on color alone
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
MARK = ["o", "s", "^"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e4e1"
RED = "#e34948"
GRAY = "#8a8a86"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.0, "lines.markersize": 3.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.5, which="major")
    ax.set_axisbelow(True)


def panel(ax, letter, x=-0.16, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, metadata={"Creator": None, "Producer": None})
    if os.environ.get("PREVIEW"):          # optional PNG copy for checking the layout
        fig.savefig(os.path.join(os.environ["PREVIEW"], os.path.basename(name).replace(".pdf", ".png")), dpi=300)
    plt.close(fig)
    print("wrote", path)
