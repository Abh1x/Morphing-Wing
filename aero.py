"""Aerodynamic back-ends. All return a polar dict {alpha, CL, CD, CM} for given coords & Re.

  xfoil       : runs the XFOIL executable (install XFOIL or use the one bundled with XFLR5). [UNTESTED in
                the build sandbox - verify on your machine against one XFLR5 run]
  neuralfoil  : pip install neuralfoil  (fast XFOIL-like neural surrogate, good for optimization loops)
  surrogate   : built-in thin-airfoil lift + empirical drag. PIPELINE TESTING ONLY - NOT FOR RESULTS.
"""
import os, shutil, subprocess, tempfile
import numpy as np
_trapz = getattr(np, "trapezoid", None) or np.trapz
from airfoil import save_dat

ALPHAS = np.arange(-10.0, 20.0, 0.5)

def _surrogate(coords, Re, alphas=ALPHAS):
    n = (len(coords) + 1) // 2
    xu, yu = coords[:n][::-1].T; xl, yl = coords[n - 1:].T
    xc = np.linspace(0.0, 1.0, 200)
    yc = 0.5 * (np.interp(xc, xu, yu) + np.interp(xc, xl, yl))
    t = np.max(np.interp(xc, xu, yu) - np.interp(xc, xl, yl))
    dyc = np.gradient(yc, xc)
    th = np.arccos(1 - 2 * xc)
    a0 = -_trapz(dyc * (np.cos(th) - 1), th) / np.pi            # zero-lift angle (rad), thin airfoil
    a_rad = np.radians(alphas)
    CL = 2 * np.pi * (a_rad - a0) * 0.92                              # 0.92: viscous lift-slope knockdown
    cl_des = 0.25 + 0.5 * (-a0 * 2 * np.pi)                           # crude ideal-CL proxy
    Cf = 0.074 / Re**0.2 * 0.6                                        # mixed lam/turb flat-plate
    cd0 = 2 * Cf * (1 + 2 * t + 60 * t**4)
    CD = cd0 + 0.0075 * (CL - np.clip(cl_des, 0.1, 0.9)) ** 2 + 0.0015 * np.clip(alphas - 8, 0, None) ** 2
    CM = np.full_like(CL, -0.05 + 0.5 * a0)                           # crude
    return dict(alpha=alphas, CL=CL, CD=CD, CM=CM)

def _neuralfoil(coords, Re, alphas=ALPHAS):
    import neuralfoil as nf
    r = nf.get_aero_from_coordinates(coordinates=coords, alpha=alphas, Re=Re, model_size="large")
    return dict(alpha=alphas, CL=np.asarray(r["CL"]), CD=np.asarray(r["CD"]), CM=np.asarray(r["CM"]))

def _xfoil(coords, Re, alphas=ALPHAS, exe="xfoil"):
    exe = shutil.which(exe) or exe
    with tempfile.TemporaryDirectory() as d:
        dat, pol = os.path.join(d, "a.dat"), os.path.join(d, "p.txt")
        save_dat(coords, dat)
        cmds = f"LOAD {dat}\nA\nPANE\nOPER\nVISC {Re:.0f}\nMACH 0\nITER 300\nPACC\n{pol}\n\n" \
               f"ASEQ {alphas[0]} {alphas[-1]} {alphas[1]-alphas[0]}\nPACC\n\nQUIT\n"
        subprocess.run([exe], input=cmds, text=True, capture_output=True, timeout=120, cwd=d)
        if not os.path.exists(pol):
            raise RuntimeError("XFOIL produced no polar")
        rows = [l.split() for l in open(pol).read().splitlines()[12:] if len(l.split()) >= 5]
        a = np.array([[float(v) for v in r[:5]] for r in rows])
        return dict(alpha=a[:, 0], CL=a[:, 1], CD=a[:, 2], CM=a[:, 4])

def get_backend(name="auto", xfoil_path="xfoil"):
    if name == "auto":
        try:
            import neuralfoil; name = "neuralfoil"
        except ImportError:
            name = "xfoil" if shutil.which(xfoil_path) else "surrogate"
    fn = {"neuralfoil": _neuralfoil, "surrogate": _surrogate,
          "xfoil": lambda c, Re, alphas=ALPHAS: _xfoil(c, Re, alphas, xfoil_path)}[name]
    return name, fn

def at_cl(polar, cl_target, alpha_range=(-4, 12)):
    """Interpolate alpha, CD, CM at a target CL. Returns None if CL not reachable in range."""
    a, cl, cd, cm = polar["alpha"], polar["CL"], polar["CD"], polar["CM"]
    ok = np.isfinite(cl) & np.isfinite(cd) & (a >= alpha_range[0]) & (a <= alpha_range[1])
    a, cl, cd, cm = a[ok], cl[ok], cd[ok], cm[ok]
    if len(a) < 3:
        return None
    i = np.argsort(cl); cl, a, cd, cm = cl[i], a[i], cd[i], cm[i]
    if cl_target < cl[0] or cl_target > cl[-1]:
        return None
    return dict(alpha=float(np.interp(cl_target, cl, a)), CD=float(np.interp(cl_target, cl, cd)),
                CM=float(np.interp(cl_target, cl, cm)))
