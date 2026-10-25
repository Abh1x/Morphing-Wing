"""Run the full pipeline:  py run_all.py [--config config.yaml] [--fast]"""
import argparse, json, os, yaml
import numpy as np, pandas as pd
from atmosphere import conditions_table
from airfoil import shape, save_dat
from aero import get_backend
from optimize import optimize_condition, evaluate, design_space, thickness_on
import plots

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--fast", action="store_true")
args = ap.parse_args()
cfg = yaml.safe_load(open(args.config))
rd, fd = cfg["output"]["results_dir"], cfg["output"]["figures_dir"]
os.makedirs(rd, exist_ok=True); os.makedirs(fd, exist_ok=True)
name, backend = get_backend(cfg["project"]["aero_backend"], cfg["project"]["xfoil_path"])
tag = "" if name != "surrogate" else "  [SURROGATE - NOT REAL RESULTS]"
print(f"Aero backend: {name}   |   thickness morphing: {'ON' if thickness_on(cfg) else 'OFF'}")
if name == "surrogate":
    print("!! WARNING: surrogate backend is for pipeline testing only. Install neuralfoil or XFOIL for real results.")

geo, conds = conditions_table(cfg)
print("Wing:", {k: round(v, 3) for k, v in geo.items()})
rows, morph_pols = [], {}
for c in conds:
    fixed = evaluate((0.0, 2.5, 0.0), c["CL_req"], c["Re"], cfg, backend)
    best = optimize_condition(c["CL_req"], c["Re"], cfg, backend, maxiter=8 if args.fast else 25)
    keys = ("LD", "delta", "p", "tau", "alpha", "CM", "strain", "stroke_mm", "stroke_t_mm", "te_angle", "tc_max", "tc_zone_min", "dt_max", "ok")
    c.update(LD_fixed=fixed.get("LD", np.nan), **{k: best.get(k) for k in keys})
    c["gain_pct"] = 100 * (c["LD"] / c["LD_fixed"] - 1) if c["LD_fixed"] else np.nan
    rows.append(c)
    coords = shape(cfg, best["delta"], best["p"], best["tau"])
    morph_pols[c["name"]] = backend(coords, c["Re"])
    print(f"{c['name']:<16} CLreq={c['CL_req']:.2f} L/D {c['LD_fixed']:.1f}->{c['LD']:.1f} ({c['gain_pct']:+.1f}%) "
          f"delta={c['delta']*100:+.2f}%c p={c['p']:.2f} tau={c['tau']*100:+.1f}% | strain={c['strain']*100:.2f}% "
          f"stroke={c['stroke_mm']:+.1f}mm thk-stroke={c['stroke_t_mm']:.1f}mm t/c={c['tc_max']*100:.1f}% feasible={c['ok']}")
    save_dat(coords, f"{rd}/airfoil_{c['name'].split()[0].lower()}.dat", c["name"])

pd.DataFrame(rows).to_csv(f"{rd}/optimal_configurations.csv", index=False)
json.dump(dict(backend=name, wing=geo, thickness_enabled=thickness_on(cfg)), open(f"{rd}/run_info.json", "w"), indent=2)
plots.fig_shapes(cfg, rows, f"{fd}/morphed_shapes.png", tag)
plots.fig_ld(rows, f"{fd}/ld_comparison.png", tag)
plots.fig_polars(cfg, backend(shape(cfg, 0, 2.5, 0), conds[1]["Re"]), morph_pols, f"{fd}/polars.png", tag)
if thickness_on(cfg): plots.fig_thickness(cfg, rows, f"{fd}/thickness_distribution.png", tag)
ref = rows[1] if len(rows) > 1 else rows[0]; n = 11 if args.fast else 17
A, B, LD, F = design_space(ref["CL_req"], ref["Re"], cfg, backend, ref, ("delta", "p"), n)
plots.fig_space(A, B, LD, F, ref, ("delta", "p"), f"{fd}/design_space_delta_p.png", f"{ref['name']}: L/D vs (δ, p){tag}")
if thickness_on(cfg):
    A, B, LD, F = design_space(ref["CL_req"], ref["Re"], cfg, backend, ref, ("delta", "tau"), n)
    plots.fig_space(A, B, LD, F, ref, ("delta", "tau"), f"{fd}/design_space_delta_tau.png", f"{ref['name']}: L/D vs (δ, τ){tag}")
print("Saved results/ and figures/")
