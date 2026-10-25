"""Links between design variables and actuator commands.
REPLACE the placeholders with your CAD kinematics or measured calibration tables."""

def stroke_mm(delta, cfg):
    """Camber actuator stroke for TE displacement delta (fraction of chord)."""
    k = cfg["mechanism"]["k_stroke_per_delta"]
    chord_mm = cfg["mechanism"]["rib_chord_m"] * 1000
    return delta * chord_mm if k == 0 else k * delta * chord_mm      # k = 0 -> placeholder 1:1

def thickness_stroke_mm(dt_max_c, cfg):
    """Thickness actuator stroke for the largest local thickness change dt_max (fraction of chord)."""
    k = cfg["mechanism"].get("k_thickness_stroke", 0.0)
    chord_mm = cfg["mechanism"]["rib_chord_m"] * 1000
    return dt_max_c * chord_mm if k == 0 else k * dt_max_c * chord_mm  # k = 0 -> placeholder 1:1
