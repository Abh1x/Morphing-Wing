"""Check an airfoil .dat file and (optionally) write an XFLR5-safe copy.
    py check_dat.py airfoils\\goe239.dat
    py check_dat.py results\\airfoil_cruise.dat --fix      -> writes results\\airfoil_cruise_xflr5.dat
"""
import sys, os
import numpy as np
from airfoil import load_dat, save_dat, clean_name

def main(path, fix=False):
    ok = True
    def bad(msg):
        nonlocal ok; ok = False; print("  [PROBLEM] " + msg)
    print(f"Checking {path}")
    if not os.path.exists(path): print("  File not found - check the path / spelling."); return
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or b"\x00" in raw[:200]: bad("file is UTF-16 / binary, not plain text. Re-save as ANSI/UTF-8 text.")
    text = raw.decode("utf-8", errors="ignore").splitlines()
    if not text: bad("file is empty"); return
    print(f"  first line (name): {text[0].strip()!r}")
    if "," in "".join(text[1:5]): bad("commas found in coordinate rows - XFLR5 needs spaces between x and y.")
    nonnum = [i + 1 for i, l in enumerate(text[1:], 1) if l.strip() and not all(_isnum(t) for t in l.replace(",", " ").split())]
    if nonnum: print(f"  note: non-numeric lines at {nonnum[:6]} (headers/comments are skipped by this checker but may upset XFLR5)")
    try:
        pts = load_dat(path)
    except Exception as e:
        bad(str(e)); return
    n = len(pts); i = int(np.argmin(pts[:, 0])); up, lo = pts[: i + 1][::-1], pts[i:]
    print(f"  points: {n}   leading edge at row {i + 1} (x={pts[i,0]:.4f})   TE gap: {abs(pts[0,1]-pts[-1,1]):.5f}")
    if n < 30: bad("very few points (<30); XFLR5 may reject or give poor results.")
    if abs(pts[0, 0] - 1) > 0.02 or abs(pts[-1, 0] - 1) > 0.02: bad("file does not start and end at the trailing edge (x~1). Expected Selig order: TE -> upper -> LE -> lower -> TE.")
    if np.any(np.diff(up[:, 0]) < -1e-9) or np.any(np.diff(lo[:, 0]) < -1e-9): bad("x is not monotonic along a surface (wrong point order or two airfoils in one file).")
    if np.any(np.diff(pts, axis=0).__abs__().sum(axis=1) < 1e-9): bad("duplicate consecutive points (XFLR5 can fail on these).")
    x = np.linspace(0.02, 0.98, 200)
    gap = np.interp(x, up[:, 0], up[:, 1]) - np.interp(x, lo[:, 0], lo[:, 1])
    if gap.min() < -1e-5: bad(f"upper surface goes below lower surface (min gap {gap.min():.4f}) - self-intersecting shape.")
    print(f"  max thickness ≈ {gap.max()*100:.2f} % chord")
    out = os.path.splitext(path)[0] + "_xflr5.dat"
    if fix:
        save_dat(pts, out, clean_name(os.path.splitext(os.path.basename(path))[0]))
        print(f"  wrote clean copy: {out}")
    print("  RESULT:", "file looks valid" if ok else "problems found (see above) - try --fix, or re-download the .dat")

def _isnum(t):
    try: float(t); return True
    except ValueError: return False

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args: print(__doc__)
    for p in args: main(p, "--fix" in sys.argv)
