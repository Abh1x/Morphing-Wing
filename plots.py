import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from airfoil import shape, spec, thickness_dist
NAVY, TEAL, ORG, GREY = "#1E3A8A", "#0EA5A4", "#F97316", "#94A3B8"
COLS = [TEAL, NAVY, ORG, "#7C3AED", "#DC2626", "#059669"]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

def fig_shapes(cfg, results, path, tag=""):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(*shape(cfg, 0, 2.5, 0).T, color="k", lw=2, label=f"Baseline {cfg['airfoil'].get('name', spec(cfg))} (fixed)")
    for r, c in zip(results, COLS):
        ax.plot(*shape(cfg, r["delta"], r["p"], r["tau"]).T, color=c, lw=1.6,
                label=f"{r['name']} (TE {r['delta']*100:+.1f}% c, thick {r['tau']*100:+.0f}%)")
    xs = cfg["morphing"]["x_start"]
    ax.axvline(xs, color=GREY, ls=":", lw=1); ax.text(xs + .01, -.075, "camber morph zone →", color=GREY, fontsize=9)
    ax.set_aspect("equal"); ax.set_xlabel("x/c"); ax.set_ylabel("y/c"); ax.set_ylim(-0.09, 0.11)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title("Optimized morphed shapes per flight condition" + tag, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig_thickness(cfg, results, path, tag=""):
    x = np.linspace(0.005, 0.995, 300)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(x, thickness_dist(shape(cfg, 0, 2.5, 0), x) * 100, "k", lw=2, label="Baseline")
    for r, c in zip(results, COLS):
        ax.plot(x, thickness_dist(shape(cfg, r["delta"], r["p"], r["tau"]), x) * 100, color=c, lw=1.5,
                label=f"{r['name']} (tau {r['tau']*100:+.0f}%)")
    z0, z1 = cfg["constraints"].get("hardware_zone", [0.25, 0.55])
    ax.axvspan(z0, z1, color=GREY, alpha=0.15, lw=0); ax.text((z0 + z1) / 2, ax.get_ylim()[0], "hardware zone", ha="center", va="bottom", fontsize=8, color=GREY)
    mt = cfg["constraints"].get("min_tc_in_zone", 0)
    if mt > 0: ax.axhline(mt * 100, color="r", ls="--", lw=1, label="min t/c in zone")
    ax.set_xlabel("x/c"); ax.set_ylabel("local thickness t/c (%)"); ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title("Thickness distribution, fixed vs morphing" + tag, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig_ld(results, path, tag=""):
    names = [r["name"] for r in results]; x = np.arange(len(names)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 3.6))
    b1 = ax.bar(x - w / 2, [r["LD_fixed"] for r in results], w, color=GREY, label="Fixed airfoil")
    b2 = ax.bar(x + w / 2, [r["LD"] for r in results], w, color=TEAL, label="Morphing (optimized)")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + .3, f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9); ax.set_ylabel("Section L/D")
    ax.legend(frameon=False, fontsize=9); ax.set_title("L/D at required CL, fixed vs morphing" + tag, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

LAB = {"delta": ("TE displacement  δ  (% chord)", 100), "p": ("shape exponent p", 1), "tau": ("thickness change τ (%)", 100)}
def fig_space(A, B, LD, feas, best, axes, path, title=""):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    (la, ka), (lb, kb) = LAB[axes[0]], LAB[axes[1]]
    cs = ax.contourf(A * ka, B * kb, np.ma.masked_invalid(LD), 18, cmap="viridis")
    if feas.any() and (~feas).any(): ax.contour(A * ka, B * kb, feas.astype(float), [0.5], colors="w", linestyles="--")
    ax.plot(best[axes[0]] * ka, best[axes[1]] * kb, "r*", ms=14)
    fig.colorbar(cs, label="L/D"); ax.set_xlabel(la); ax.set_ylabel(lb)
    ax.set_title(title, fontsize=10, loc="left"); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig_polars(cfg, base_pol, morph_pols, path, tag=""):
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))
    ax[0].plot(base_pol["CD"], base_pol["CL"], "k", lw=2, label="Fixed")
    for (n, p), c in zip(morph_pols.items(), COLS): ax[0].plot(p["CD"], p["CL"], color=c, lw=1.4, label=n)
    ax[0].set_xlabel("CD"); ax[0].set_ylabel("CL"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(base_pol["alpha"], base_pol["CL"], "k", lw=2)
    for (n, p), c in zip(morph_pols.items(), COLS): ax[1].plot(p["alpha"], p["CL"], color=c, lw=1.4)
    ax[1].set_xlabel("alpha (deg)"); ax[1].set_ylabel("CL"); fig.suptitle("Drag polar & lift curve" + tag, fontsize=11, x=0.05, ha="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
