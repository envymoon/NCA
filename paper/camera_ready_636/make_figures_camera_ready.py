"""Final paper figures, generated ONLY from frozen audit rows (no GPU, no retraining).

Source: audit_full_population.json (row dump from full_population_audit.py; every pool
scene x grid x seed, censored scenes included with knee=None).

  Fig 1  ruler discrimination: (a) per-scene T*_tail vs geodesic / straight-line ruler,
         colored by tortuosity, G=32; (b) per-scene CV of T*/ruler by grid, mean+-std
         over 5 seeds.
  Fig 2  scaling + policy: (a) budget coefficient c1 vs grid size with a D log D
         reference; (b) unconditional cost-coverage frontier for flat/geo/free/euc.

Run: python3 make_figures.py            (writes figs/fig1_ruler.{pdf,png} etc.)
"""
from pathlib import Path
import sys
EXP = Path(__file__).resolve().parents[2] / "experiments"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import json
import math
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy_discrete import eval_policy

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

# colorblind-safe, consistent across both figures: geo blue / Chebyshev red /
# Euclidean orange / flat grey / aggregate-estimand green
C_GEO, C_FREE, C_EUC, C_FLAT = "#2166ac", "#b2182b", "#e08214", "#4d4d4d"
C_AGG = "#1b7837"

aud = json.load(open(EXP / "audit_full_population.json"))
rows = [r for r in aud["rows"] if not r["clamped"]]
grids = sorted({r["grid"] for r in rows})
seeds = sorted({r["seed"] for r in rows})
KNEE = "knee_tail"


def unit_cv(G, s, ruler):
    conv = [r for r in rows if r["grid"] == G and r["seed"] == s
            and r[KNEE] is not None]
    ratios = [r[KNEE] / max(r[ruler], 1.0) for r in conv]
    return st.stdev(ratios) / st.mean(ratios)


# ---------------------------------------------------------------- figure 1
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.10),
                         gridspec_kw={"width_ratios": [1, 1.22, 1.10], "wspace": 0.65})

conv32 = [r for r in rows if r["grid"] == 32 and r[KNEE] is not None]
taus = [r["tau"] for r in conv32]
norm = matplotlib.colors.Normalize(vmin=1.0, vmax=3.0)
# one shared axis limit for (a) and (b): the wall-blind ruler visibly compresses
# the x-axis and smears vertically, and shared limits keep that comparison honest
lim = max(max(r[k] for r in conv32) for k in ("dgeo", "dfree", KNEE)) * 1.05
for ax, ruler, label in ((axes[0], "dgeo", r"$D_\mathrm{geo}$"),
                         (axes[1], "dfree", r"$D_\mathrm{free}$")):
    xs = [r[ruler] for r in conv32]
    ys = [r[KNEE] for r in conv32]
    sc = ax.scatter(xs, ys, c=taus, cmap="viridis", norm=norm, s=4.5, alpha=0.5,
                    linewidths=0, rasterized=False)
    cmed = st.median(r[KNEE] / max(r[ruler], 1.0) for r in conv32)
    ax.plot([0, lim], [0, cmed * lim], color="k", lw=0.8, ls="--",
            label=rf"$T^*={cmed:.2f}\,${label}")
    ax.set_xlim(0, lim); ax.set_ylim(0, max(ys) * 1.08)
    ax.set_xlabel(label + "  (steps)")
    ax.legend(loc="upper left", frameon=False, handlelength=1.4)
    ucv = st.mean(unit_cv(32, s, ruler) for s in seeds)
    ax.text(0.97, 0.05, f"per-scene CV {100 * ucv:.0f}%", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color="0.25")
axes[0].set_ylabel(r"per-scene tail knee $T^*$  (steps)")
axes[0].set_title("(a) geodesic ruler")
axes[1].set_title("(b) straight-line ruler")
cb = fig.colorbar(sc, ax=axes[1], pad=0.03, fraction=0.06)
cb.solids.set_rasterized(False)
cb.ax.set_title(r"$\tau$", fontsize=9)
cb.ax.tick_params(labelsize=8)

x = range(len(grids))
w = 0.26
axes[2].grid(axis="y")
axes[2].set_axisbelow(True)
for off, ruler, color, name, hatch in ((-w, "dgeo", C_GEO, "geodesic", ""),
                                (0.0, "dfree", C_FREE, "Chebyshev", "//"),
                                (w, "deuc", C_EUC, "Euclidean", "..")):
    means = [st.mean(unit_cv(G, s, ruler) for s in seeds) for G in grids]
    stds = [st.stdev(unit_cv(G, s, ruler) for s in seeds) for G in grids]
    axes[2].bar([xi + off for xi in x], [100 * m for m in means], w,
                yerr=[100 * sd for sd in stds], color=color, label=name,
                edgecolor="white", linewidth=0.4, hatch=hatch,
                error_kw={"lw": 0.7}, capsize=2)
axes[2].set_xticks(list(x)); axes[2].set_xticklabels([str(g) for g in grids])
axes[2].set_xlabel("grid size $G$")
axes[2].set_ylabel(r"CV of $T^*\!/D$  (%)")
axes[2].set_title("(c) dispersion across scenes")
axes[2].set_ylim(0, 135)
axes[2].set_yticks([0, 30, 60, 90])
axes[2].legend(frameon=False, loc="upper right")
fig.savefig(OUT / "fig1_ruler.pdf"); fig.savefig(OUT / "fig1_ruler.png")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.30), gridspec_kw={"wspace": 0.32})

# (a) c1 vs grid: per-scene tail estimand (headline) + aggregate mean-knee estimand
# (the two are kept separate per the measurement protocol; do not average them)
ax = axes[0]
c1m, c1s = [], []
for G in grids:
    per_seed = []
    for s in seeds:
        conv = [r for r in rows if r["grid"] == G and r["seed"] == s
                and r[KNEE] is not None]
        per_seed.append(st.mean(r[KNEE] / max(r["dgeo"], 1.0) for r in conv))
    c1m.append(st.mean(per_seed)); c1s.append(st.stdev(per_seed))
ax.errorbar(grids, c1m, yerr=c1s, color=C_GEO, marker="o", ms=4, capsize=2,
            label="tail, per scene")
E = json.load(open(EXP / Path(aud["source"]).name))
agg = E["c1_geo_by_grid"]
ax.errorbar(grids, [agg[str(G)]["mean"] for G in grids],
            yerr=[agg[str(G)]["std"] for G in grids], color=C_AGG,
            marker="s", ms=4, capsize=2,
            label="mean, aggregate")
dbar = {G: st.mean(r["dgeo"] for r in rows if r["grid"] == G) for G in grids}
ref = [c1m[0] * math.log(dbar[G]) / math.log(dbar[grids[0]]) for G in grids]
# reference line labeled in place (a legend row here used to collide with the line)
ax.plot(grids, ref, color="k", ls=":", lw=1)
ax.text(grids[-1] - 0.3, ref[-1] + 0.015, r"$D\log D$ reference",
        ha="right", va="bottom", fontsize=8, color="0.15")
ax.axhline(1.0, color="0.75", lw=0.7)
ax.set_xticks(grids); ax.set_xlabel("grid size $G$")
ax.set_ylim(top=max(ref[-1], max(c1m)) + 0.12)
ax.set_ylabel(r"budget coefficient $c_1$")
ax.set_title("(a) finite-range coefficients")
ax.legend(frameon=True, framealpha=0.9, edgecolor="none", loc="upper left",
          borderpad=0.3)

# (b) unconditional cost-coverage frontier
ax = axes[1]
pop = rows
n = len(pop)


def frontier(ruler):
    if ruler == "flat":
        pts = []
        for T in range(4, 153, 2):
            served = sum(1 for r in pop if r[KNEE] is not None and r[KNEE] <= T)
            pts.append((float(T), served / n))
        return pts
    ratios = sorted(r[KNEE] / max(r[ruler], 1.0) for r in pop if r[KNEE] is not None)
    cmax = math.ceil(100 * max(ratios))
    cs = [i / 100 for i in range(1, cmax + 1)]
    pts = []
    for c in cs:
        cost, coverage = eval_policy(pop, ruler, c, KNEE)
        pts.append((cost, coverage))
    return pts


ceiling = sum(1 for r in pop if r[KNEE] is not None) / n
for ruler, color, name in (("flat", C_FLAT, "flat $T$"),
                           ("dgeo", C_GEO, r"$T=\lceil c D_\mathrm{geo}\rceil$"),
                           ("dfree", C_FREE, r"$T=\lceil c D_\mathrm{free}\rceil$"),
                           ("deuc", C_EUC, r"$T=\lceil c D_\mathrm{euc}\rceil$")):
    pts = sorted(frontier(ruler))
    ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=color,
            label=name, lw=1.3 if ruler in ("dgeo", "flat") else 1.0,
            ls={"flat": "--", "dgeo": "-", "dfree": "-.", "deuc": ":"}[ruler])
ax.grid(axis="y")
ax.set_axisbelow(True)
ax.axhline(100 * ceiling, color="0.6", lw=0.7, ls=":")
ax.text(97, 100 * ceiling + 0.4, f"attainment ceiling {100*ceiling:.1f}%",
        fontsize=8, color="0.35", ha="right")
for f, marker in ((0.95, "o"),):
    rec = next(p for p in aud["policy_full"] if abs(p["f"] - f) < 1e-9)
    geo_c, geo_v = rec["geo_cost"], 100 * rec["geo_cov"]
    ax.plot(geo_c, geo_v, marker, color=C_GEO, ms=5)
    ax.plot(rec["flat_T"], 100 * rec["flat_cov"], marker, color=C_FLAT, ms=5)
    ax.annotate(f"fitted $f$={f:.2f}", (geo_c, geo_v),
                textcoords="offset points", xytext=(-2, -15), fontsize=8)
    # the headline saving, drawn where it happens: same coverage, fewer steps
    ax.annotate("", xy=(rec["flat_T"], geo_v), xytext=(geo_c, geo_v),
                arrowprops={"arrowstyle": "<->", "lw": 0.7, "color": "0.3"})
    sav = 100 * (1 - geo_c / rec["flat_T"])
    ax.text((geo_c + rec["flat_T"]) / 2 - 1.2, geo_v + 1.3, f"$-${sav:.0f}%",
            ha="center", fontsize=8, color="0.2")
ax.set_xlim(10, 100); ax.set_ylim(55, 97)
ax.set_xlabel("mean provisioned steps per scene")
ax.set_ylabel("knee-attainment coverage (%)")
ax.set_title("(b) pooled cost and coverage")
ax.legend(frameon=False, loc="lower right")
fig.savefig(OUT / "fig2_policy.pdf"); fig.savefig(OUT / "fig2_policy.png")
plt.close(fig)

print("wrote figs/fig1_ruler.{pdf,png} figs/fig2_policy.{pdf,png}")
print(f"convergence ceiling {100*ceiling:.2f}%  rows {n}")
