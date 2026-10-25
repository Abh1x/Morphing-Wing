"""Per-condition optimization: maximise L/D at the required CL over the morphing variables
(delta, p [, tau]) subject to skin strain, camber-actuator stroke, thickness-actuator stroke,
pitching moment, alpha range and thickness-envelope (hardware) limits."""
import numpy as np
from scipy.optimize import differential_evolution
from airfoil import shape, skin_strain, te_angle_deg, thickness_metrics
from aero import at_cl
from mechanism import stroke_mm, thickness_stroke_mm

NAMES = ["delta", "p", "tau"]

def thickness_on(cfg):
    return bool(cfg.get("thickness", {}).get("enabled", False))

def bounds(cfg):
    m, t = cfg["morphing"], cfg.get("thickness", {})
    b = [(m["delta_min"], m["delta_max"]), (m["p_min"], m["p_max"])]
    if thickness_on(cfg): b.append((t["tau_min"], t["tau_max"]))
    return b

def evaluate(x, cl_req, Re, cfg, backend_fn):
    delta, p = float(x[0]), float(x[1]); tau = float(x[2]) if len(x) > 2 else 0.0
    c = cfg["constraints"]
    coords = shape(cfg, delta, p, tau)
    try:
        pol = backend_fn(coords, Re)
    except Exception as e:
        return dict(LD=0.0, ok=False, why=str(e))
    r = at_cl(pol, cl_req, c["alpha_range_deg"])
    if r is None:
        return dict(LD=0.0, ok=False, why="CL not reachable")
    eps = skin_strain(cfg, delta, p, tau)[0]
    th = thickness_metrics(cfg, delta, p, tau)
    st_t = thickness_stroke_mm(th["dt_max"], cfg)
    g = dict(strain=eps - c["max_skin_strain"],
             stroke=abs(stroke_mm(delta, cfg)) - cfg["mechanism"]["actuator_stroke_mm"],
             cm=abs(r["CM"]) - c["max_cm_magnitude"])
    if thickness_on(cfg):
        g["stroke_t"] = st_t - cfg["mechanism"]["thickness_actuator_stroke_mm"]
        if c.get("max_tc", 0) > 0: g["tc_max"] = th["tc_max"] - c["max_tc"]
        if c.get("min_tc_in_zone", 0) > 0: g["tc_zone"] = c["min_tc_in_zone"] - th["tc_zone_min"]
    return dict(LD=cl_req / r["CD"], CD=r["CD"], alpha=r["alpha"], CM=r["CM"], strain=eps, g=g,
                ok=all(v <= 0 for v in g.values()), delta=delta, p=p, tau=tau,
                stroke_mm=stroke_mm(delta, cfg), stroke_t_mm=st_t, te_angle=te_angle_deg(delta, p, cfg["morphing"]["x_start"]),
                tc_max=th["tc_max"], tc_zone_min=th["tc_zone_min"], dt_max=th["dt_max"])

def optimize_condition(cl_req, Re, cfg, backend_fn, seed=1, maxiter=25, popsize=12):
    def obj(x):
        r = evaluate(x, cl_req, Re, cfg, backend_fn)
        if "g" not in r: return 1e3
        return -r["LD"] + sum(max(v, 0) for v in r["g"].values()) * 1e3
    res = differential_evolution(obj, bounds(cfg), seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-4, polish=False)
    return evaluate(res.x, cl_req, Re, cfg, backend_fn)

def design_space(cl_req, Re, cfg, backend_fn, best, axes=("delta", "p"), n=17):
    """2-D slice of the design space; the remaining variable is held at its optimum value."""
    b = dict(zip(NAMES, bounds(cfg) + [(0, 0)] * (3 - len(bounds(cfg)))))
    ia, ib = NAMES.index(axes[0]), NAMES.index(axes[1])
    A, B = np.linspace(*b[axes[0]], n), np.linspace(*b[axes[1]], max(5, n // 2))
    LD = np.full((len(B), len(A)), np.nan); feas = np.zeros_like(LD, dtype=bool)
    for i, vb in enumerate(B):
        for j, va in enumerate(A):
            x = [best["delta"], best["p"], best["tau"]]; x[ia], x[ib] = va, vb
            r = evaluate(x[: 3 if thickness_on(cfg) else 2], cl_req, Re, cfg, backend_fn)
            LD[i, j] = r["LD"] if r.get("LD", 0) > 0 else np.nan; feas[i, j] = r["ok"]
    return A, B, LD, feas
