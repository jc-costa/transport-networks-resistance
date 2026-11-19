"""Figure and numbers for the three placement rules of the sticks.

Draws one network under each rule with the project's own generator, then
measures how much stick length lies inside the film at each distance from the
left edge, averaged over many draws. Also prints the acceptance rate of the
rejection rule and the number of sticks that enter a film under the clipping
rule.

Rules (utils/point_generator.py):
  unbounded : boundary="reject", enforce_bounds=False. Origin uniform in the
              square, direction uniform, the second end may fall outside. This
              is the construction of Bottcher (2020) and of the undergraduate
              thesis. Only the part inside the film is kept for the profile.
  reject    : boundary="reject", enforce_bounds=True. Redraw until both ends
              are inside.
  clip      : boundary="clip". Centres in the film enlarged by a/2 on every
              side, every stick cut at the edge of the film.

Usage: python cluster/boundary_rules_figure.py OUT.png [NUMBERS.json]
Run from the repository root.
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, "utils")
from point_generator import clip_segment_to_box, generate_points  # noqa: E402

N, A, SEED, REPS, BINS = 40, 0.4, 7, 3000, 20
LO, HI = (0.0, 0.0), (1.0, 1.0)
RULES = [
    ("unbounded", "Unbounded (Böttcher, 2020; undergraduate thesis)",
     dict(boundary="reject", enforce_bounds=False)),
    ("reject", "Rejection", dict(boundary="reject", enforce_bounds=True)),
    ("clip", "Clipping (this thesis)", dict(boundary="clip")),
]
BLUE, GREY = "#2b6cb0", "#8a8a8a"


def draw(a, **kw):
    p1, p2, ang = [], [], []
    generate_points(N, p1, p2, ang, a, **kw)
    return np.array(p1), np.array(p2)


def profile(kw, a=A, reps=REPS, bins=BINS):
    """Stick length inside the film per unit area, divided by N, against x."""
    edges = np.linspace(0, 1, bins + 1)
    acc = np.zeros(bins)
    for _ in range(reps):
        p1, p2 = draw(a, **kw)
        for u, v in zip(p1, p2):
            c = clip_segment_to_box(tuple(u), tuple(v), LO, HI)
            if c is None:
                continue
            (x1, y1), (x2, y2) = c
            length = np.hypot(x2 - x1, y2 - y1)
            if length > 0:
                acc += length * np.histogram(np.linspace(x1, x2, 25), edges)[0] / 25
    return 0.5 * (edges[1:] + edges[:-1]), acc / reps * bins / N


def acceptance(a, m=2_000_000, seed=1):
    rng = np.random.default_rng(seed)
    x, y, th = rng.random(m), rng.random(m), rng.random(m) * 2 * np.pi
    x2, y2 = x + a * np.cos(th), y + a * np.sin(th)
    return float(((x2 >= 0) & (x2 <= 1) & (y2 >= 0) & (y2 <= 1)).mean())


def inside_fraction(a, m=2_000_000, seed=2):
    """Mean share of an unbounded-rule stick (length a) that lies in the film."""
    rng = np.random.default_rng(seed)
    x, y, th = rng.random(m), rng.random(m), rng.random(m) * 2 * np.pi
    lo, hi = np.zeros(m), np.ones(m)
    for p, d in ((x, a * np.cos(th)), (y, a * np.sin(th))):
        with np.errstate(divide="ignore", invalid="ignore"):
            s0, s1 = -p / d, (1 - p) / d
        lo = np.maximum(lo, np.minimum(s0, s1))
        hi = np.minimum(hi, np.maximum(s0, s1))
    return float(np.clip(hi - lo, 0, None).mean())


def main(out, out_json=None):
    fig, ax = plt.subplots(2, 3, figsize=(8.4, 5.4), gridspec_kw=dict(height_ratios=[3, 1.6]))
    numbers = {"N": N, "a": A, "reps": REPS, "profile": {}}
    for k, (key, title, kw) in enumerate(RULES):
        np.random.seed(SEED)
        p1, p2 = draw(A, **kw)
        top = ax[0, k]
        top.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.8, ec="black", zorder=5))
        if key == "clip":
            top.add_patch(Rectangle((-A / 2, -A / 2), 1 + A, 1 + A, fill=False, lw=1,
                                    ls="--", ec="#c05621", zorder=4))
        for u, v in zip(p1, p2):
            c = clip_segment_to_box(tuple(u), tuple(v), LO, HI)
            if c is not None:
                top.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], color=BLUE, lw=1.2, zorder=3)
            if key == "unbounded":
                top.plot([u[0], v[0]], [u[1], v[1]], color=GREY, lw=0.9, ls=":", zorder=2)
        if key == "unbounded":
            top.scatter(p1[:, 0], p1[:, 1], s=7, color="#c53030", zorder=6)
        top.set_title(f"{title}\n{len(p1)} sticks in this draw", fontsize=8.5)
        top.set_xlim(-A - .05, 1 + A + .05)
        top.set_ylim(-A - .05, 1 + A + .05)
        top.set_aspect("equal")
        top.axis("off")

        x, y = profile(kw)
        bot = ax[1, k]
        bot.bar(x, y, width=1 / BINS, color=BLUE, alpha=.85)
        bot.axhline(A, color="black", lw=1, ls="--")
        bot.set_xlim(0, 1)
        bot.set_ylim(0, 0.75)
        bot.set_xlabel("distance from the left edge")
        if k == 0:
            bot.set_ylabel("stick length in the film\nper unit area, over $N$")
        centre = float(y[(x > 0.4) & (x < 0.6)].mean())
        numbers["profile"][key] = {"edge_bin": float(y[0]), "centre": centre,
                                   "edge_over_centre": float(y[0]) / centre}

    fig.tight_layout()
    fig.savefig(out, dpi=200)

    numbers["acceptance"] = {}
    for a in (0.06, 0.2, 0.4, 0.8, 1.0, 1.1):
        p = acceptance(a)
        numbers["acceptance"][str(a)] = {
            "accept": p, "draws_per_stick_reject": 1 / p,
            "sticks_entering_clip_per_N": 1 + 4 * a / np.pi,
            "centres_drawn_per_N_clip": (1 + a) ** 2,
            "unbounded_inside_fraction": inside_fraction(a)}
    print(json.dumps(numbers, indent=1))
    if out_json:
        json.dump(numbers, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
