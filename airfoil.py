"""Baseline airfoil (NACA 4-digit OR any .dat file, e.g. GOE 239) + morphing of CAMBER and THICKNESS.

Camber morph (variables delta, p):
    yc(x) -> yc(x) - delta * ((x - x_s)/(1 - x_s))**p            for x > x_s       (delta>0 = TE moves DOWN)
Thickness morph (variable tau):
    h(x)  -> h(x) * (1 + tau * w(x))                              h = half-thickness
    w(x) = 1                                   mode "global"  (uniform scaling)
    w(x) = sin^2(pi (x-x0)/(x1-x0)) in [x0,x1]  mode "bump"    (local, zero at LE and TE side of the window)
Coordinates are returned TE(upper) -> LE -> TE(lower) (Selig/XFOIL order).
"""
import os
import numpy as np

# ------------------------------------------------------------------ NACA 4-digit
def naca4_camber(code, x):
    m, p, t = int(code[0]) / 100, int(code[1]) / 10, int(code[2:]) / 100
    yc = np.zeros_like(x); dyc = np.zeros_like(x)
    if m > 0 and p > 0:
        f = x < p
        yc[f] = m / p**2 * (2 * p * x[f] - x[f] ** 2); dyc[f] = 2 * m / p**2 * (p - x[f])
        yc[~f] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x[~f] - x[~f] ** 2)
        dyc[~f] = 2 * m / (1 - p) ** 2 * (p - x[~f])
    return yc, dyc, t

def naca4_thickness(t, x):
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)

# ------------------------------------------------------------------ .dat files
def load_dat(path):
    """Read Selig- or Lednicer-format .dat -> coords in Selig order, chord normalised to x in [0, 1]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Airfoil file not found: {path}\n"
                                "Download the Selig-format .dat for your airfoil and put it in the airfoils/ folder.")
    pts = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        parts = line.replace(",", " ").split()
        if len(parts) == 2:
            try: pts.append((float(parts[0]), float(parts[1])))
            except ValueError: pass
    pts = np.array(pts)
    if len(pts) < 20: raise ValueError("Could not read enough coordinate rows from " + path)
    if pts[0, 0] > 1.5 and pts[0, 1] > 1.5:                      # Lednicer: first row = point counts
        nu, nl = int(pts[0, 0]), int(pts[0, 1]); body = pts[1:]
        up, lo = body[:nu], body[nu:nu + nl]
        pts = np.vstack([up[::-1], lo[1:]])
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    pts = pts.copy(); pts[:, 0] = (pts[:, 0] - xmin) / (xmax - xmin); pts[:, 1] = pts[:, 1] / (xmax - xmin)
    return pts

def spec(cfg):
    a = cfg["airfoil"]
    return a["dat_file"] if a.get("dat_file") else a["naca"]

def _is_file(code):
    return isinstance(code, str) and (code.lower().endswith(".dat") or os.path.sep in code or "/" in code)

def _split(coords):
    i = int(np.argmin(coords[:, 0]))
    return coords[: i + 1][::-1], coords[i:]                     # upper LE->TE, lower LE->TE

def _surf_y(surf, x):
    return np.interp(np.sqrt(x), np.sqrt(np.clip(surf[:, 0], 0, None)), surf[:, 1])   # sqrt(x): smooth at LE

# ------------------------------------------------------------------ morphing
def thickness_window(x, tcfg):
    if not tcfg or not tcfg.get("enabled", False): return np.zeros_like(x)
    if tcfg.get("mode", "bump") == "global": return np.ones_like(x)
    x0, x1 = tcfg["x_start"], tcfg["x_end"]
    w = np.zeros_like(x); m = (x > x0) & (x < x1)
    w[m] = np.sin(np.pi * (x[m] - x0) / (x1 - x0)) ** 2
    return w

def make_airfoil(code="2412", delta=0.0, p=2.5, x_start=0.6, n=161, tau=0.0, tcfg=None):
    beta = np.linspace(0, np.pi, n)
    x = 0.5 * (1 - np.cos(beta))                                 # cosine spacing
    xi = np.clip((x - x_start) / (1 - x_start), 0.0, 1.0)
    w = thickness_window(x, tcfg)
    if _is_file(code):
        up, lo = _split(load_dat(code))
        yu0, yl0 = _surf_y(up, x), _surf_y(lo, x)
        yc, h = 0.5 * (yu0 + yl0), 0.5 * (yu0 - yl0)
        yc = yc - delta * xi**p; h = h * (1 + tau * w)           # thickness kept vertical
        xu, yu, xl, yl = x, yc + h, x, yc - h
    else:
        yc, dyc, t = naca4_camber(code, x)
        dyc = dyc - delta * p * xi ** (p - 1) / (1 - x_start) * (x > x_start)
        yc = yc - delta * xi**p
        yt = naca4_thickness(t, x) * (1 + tau * w)
        th = np.arctan(dyc)
        xu, yu, xl, yl = x - yt * np.sin(th), yc + yt * np.cos(th), x + yt * np.sin(th), yc - yt * np.cos(th)
    return np.vstack([np.column_stack([xu[::-1], yu[::-1]]), np.column_stack([xl[1:], yl[1:]])])

def shape(cfg, delta=0.0, p=2.5, tau=0.0):
    """Build coordinates for given morph variables straight from the config."""
    return make_airfoil(spec(cfg), delta, p, cfg["morphing"]["x_start"], cfg["airfoil"]["n_points"],
                        tau, cfg.get("thickness"))

# ------------------------------------------------------------------ measurements
def thickness_dist(coords, x):
    up, lo = _split(coords)
    return _surf_y(up, x) - _surf_y(lo, x)

def thickness_metrics(cfg, delta, p, tau):
    """Max t/c, thickness change vs same camber with tau=0, and min local t/c in the hardware zone."""
    x = np.linspace(0.005, 0.995, 400)
    t1 = thickness_dist(shape(cfg, delta, p, tau), x)
    t0 = thickness_dist(shape(cfg, delta, p, 0.0), x)
    z0, z1 = cfg["constraints"].get("hardware_zone", [0.25, 0.55])
    zone = (x >= z0) & (x <= z1)
    return dict(tc_max=float(t1.max()), dt_max=float(np.abs(t1 - t0).max()), tc_zone_min=float(t1[zone].min()))

def skin_strain(cfg, delta, p, tau=0.0):
    """Average engineering strain of upper/lower skin over the whole morphing zone vs the undeformed shape.
    Zone = from the earliest start of the camber or thickness morph to the trailing edge. Positive = stretch."""
    t = cfg.get("thickness", {}); x_lo = cfg["morphing"]["x_start"]
    if t.get("enabled"): x_lo = 0.0 if t.get("mode") == "global" else min(x_lo, t["x_start"])
    def zone_len(c, side):
        up, lo = _split(c); s = up if side == "u" else lo
        s = s[s[:, 0] >= x_lo]
        return np.sum(np.hypot(*np.diff(s, axis=0).T))
    base, mor = shape(cfg, 0.0, p, 0.0), shape(cfg, delta, p, tau)
    eu, el = zone_len(mor, "u") / zone_len(base, "u") - 1, zone_len(mor, "l") / zone_len(base, "l") - 1
    return max(abs(eu), abs(el)), eu, el

def te_angle_deg(delta, p, x_start):
    return float(np.degrees(np.arctan(delta * p / (1 - x_start))))

def clean_name(name):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_") or "MORPH"

def save_dat(coords, path, name="MORPH"):
    """XFLR5/XFOIL-safe Selig file: plain-ASCII name line, 'x y' rows, Windows line endings."""
    with open(path, "w", newline="\r\n") as f:
        f.write(clean_name(name) + "\n")
        for x, y in coords: f.write(f"{x:.6f} {y:.6f}\n")
