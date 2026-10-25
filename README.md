# Morphing Rib - Optimization Framework (template)

Pipeline: flight condition -> required CL -> optimize (TE displacement `delta`, shape exponent `p`, thickness change `tau`)
-> feasible morphed airfoil -> actuator stroke command.  Outputs: `results/*.csv`, `results/*.dat`
(import into CAD/XFLR5), `figures/*.png`.

## Setup
    pip install -r requirements.txt
    pip install neuralfoil        # recommended aero backend (or install XFOIL and set xfoil_path)

## Use
1. Edit `config.yaml` - every `# TODO` is yours to fill (aircraft, airfoil, morph zone, actuator, skin strain limit).
2. `python atmosphere.py`  prints wing geometry + required CL / Re for each flight condition.
3. `python run_all.py`  (add `--fast` for a quick test).

## IMPORTANT
* Backend `surrogate` is a crude built-in model for testing the pipeline ONLY. Figures it makes are watermarked
  and must NOT go into the submission. Use `neuralfoil` or `xfoil`.
* The XFOIL wrapper was not executable in the build sandbox: validate it on your machine by comparing one
  polar against XFLR5 before trusting it.
* `mechanism.py` is a placeholder 1:1 stroke map. Replace with your CAD kinematics / measured calibration.
* Morph model (`airfoil.py`) is a smooth camber-line deformation. Replace with FEA/CAD-derived shapes when available.
* 2D section analysis only; XFOIL is reliable to roughly 8 deg AoA (Li et al. 2018 review). Do 3D checks in XFLR5.

## Thickness morphing (3rd variable)
`tau` scales the local half-thickness: h -> h*(1 + tau*w(x)). Mode `bump` changes thickness only inside
[x_start, x_end] (LE and TE untouched); mode `global` scales it everywhere. Set `thickness.enabled: false` for
camber-only runs. Extra checks per candidate: thickness-actuator stroke, max t/c, and minimum local t/c inside
`hardware_zone`; skin strain now includes the surface-length change from thickness. Replace the placeholder stroke
map for thickness in `mechanism.py` with your own kinematics.
