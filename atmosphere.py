"""ISA atmosphere (troposphere) and flight-condition helpers."""
import numpy as np

def isa(h_m=0.0):
    T0, p0, L, R, g = 288.15, 101325.0, 0.0065, 287.058, 9.80665
    T = T0 - L * h_m
    p = p0 * (T / T0) ** (g / (R * L))
    rho = p / (R * T)
    mu = 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)  # Sutherland
    return rho, mu, T

def required_cl(mass_kg, S_m2, V, n=1.0, h_m=0.0):
    rho, _, _ = isa(h_m)
    q = 0.5 * rho * V**2
    return n * mass_kg * 9.80665 / (q * S_m2)

def reynolds(V, chord_m, h_m=0.0):
    rho, mu, _ = isa(h_m)
    return rho * V * chord_m / mu

def wing_geometry(ac):
    b, cr, ct = ac["span_m"], ac["chord_root_m"], ac["chord_tip_m"]
    S = 0.5 * (cr + ct) * b
    lam = ct / cr
    mac = 2.0 / 3.0 * cr * (1 + lam + lam**2) / (1 + lam)
    return dict(S=S, AR=b**2 / S, MAC=mac, taper=lam, wing_loading=ac["mtow_kg"] * 9.80665 / S)

def conditions_table(cfg):
    ac = cfg["aircraft"]; g = wing_geometry(ac); rows = []
    for fc in cfg["flight_conditions"]:
        cl = required_cl(ac["mtow_kg"], g["S"], fc["V"], fc.get("n", 1.0), ac["altitude_m"])
        re = reynolds(fc["V"], g["MAC"], ac["altitude_m"])
        rows.append(dict(name=fc["name"], V=fc["V"], n=fc.get("n", 1.0), CL_req=cl, Re=re))
    return g, rows

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    g, rows = conditions_table(cfg)
    print({k: round(v, 3) for k, v in g.items()})
    for r in rows:
        print(f"{r['name']:<18} V={r['V']:5.1f} n={r['n']:.1f}  CL_req={r['CL_req']:.3f}  Re={r['Re']:.3g}")
