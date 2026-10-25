"""Draw the morph-parametrization figure (slide 6) for YOUR airfoil:  py make_param_figure.py"""
import yaml, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from airfoil import shape
cfg = yaml.safe_load(open("config.yaml")); m, t = cfg["morphing"], cfg.get("thickness", {})
on = t.get("enabled", False)
fig, axs = plt.subplots(2 if on else 1, 1, figsize=(7.2, 5.2 if on else 3.0), dpi=200)
axs = [axs] if not on else list(axs)
ax = axs[0]; ax.axvspan(m["x_start"], 1.0, color="#E0F2FE", alpha=0.6, lw=0)
ax.plot(*shape(cfg).T, color="#0F172A", lw=2.2, label="Baseline")
for d, c in [(m["delta_min"], "#F97316"), (0.5 * m["delta_max"], "#0EA5A4"), (m["delta_max"], "#1E3A8A")]:
    ax.plot(*shape(cfg, d, 2.5, 0).T, color=c, lw=1.7, label=f"camber: delta {d*100:+.1f} % c")
ax.set_title("Camber morph", fontsize=9, loc="left")
if on:
    ax = axs[1]
    if t.get("mode") == "bump": ax.axvspan(t["x_start"], t["x_end"], color="#FEF3C7", alpha=0.7, lw=0)
    ax.plot(*shape(cfg).T, color="#0F172A", lw=2.2, label="Baseline")
    for tau, c in [(t["tau_min"], "#F97316"), (t["tau_max"], "#1E3A8A")]:
        ax.plot(*shape(cfg, 0, 2.5, tau).T, color=c, lw=1.7, label=f"thickness: tau {tau*100:+.0f} %")
    ax.set_title("Thickness morph", fontsize=9, loc="left")
for a in axs:
    a.set_aspect("equal"); a.tick_params(labelsize=8); a.set_xlim(-0.03, 1.05)
    for s_ in ("top", "right"): a.spines[s_].set_visible(False)
    a.legend(fontsize=7.5, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
axs[-1].set_xlabel("x/c", fontsize=9)
fig.tight_layout(); fig.savefig("figures/parametrization.png", bbox_inches="tight"); print("saved figures/parametrization.png")
