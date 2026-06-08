# Exploratory — forward morphology lab

This directory is the **unsealed, forward-looking** area of the T0 Morphology
Furnace. The rest of the repo (root core, `tests/`, `experiments/t0-sealed/`,
`paper/`) is the **sealed ACE/T0 archive** and stays frozen; new morphology
candidates incubate here first and graduate into the sealed core only after a
pre-registered, gated run.

Not collected by the sealed test suite: `pytest.ini` sets `testpaths = tests`,
so files here are run explicitly (each is standalone with a `main()` runner),
never as part of the sealed gate.

## Contents

- `shadow-ambiguity/` — research-candidate #10. A `W_u`-**using** readout
  morphology statistic (effective rank / spectral entropy / off-top
  pseudo-volume of the centered softmax-Fisher `F_c = W_uᵀ(diag(p) − p pᵀ) W_u`),
  the deliberate complement to ACE's `W_u`-**free** attention morphology — both
  read the model's geometry at the commitment instant.
  `test_shadow_ambiguity.py` is the numpy identity/contract suite (7/7 green);
  check 6 cross-checks the reference spectrum (eigenvalues **and** null-ratio)
  against the inherited centered-Fisher readout core's `kl_discharged_and_centered`.

## Shadow-Ambiguity Verdict

Current verdict (2026-06-07): **H1 NO-GO**. RPV is confidence-independent
(`+0.102` random-effects AUROC increment over `{surprise}` across 26
model-benchmark pairs), but it is redundant with the prior `null_ratio` detector
once the fair deployment baseline includes `{surprise, null_ratio, p_max}`
(`+0.011`, below the pre-registered `+0.02` practical-effect bar).

H2 remains useful as a narrow backstop story: RPV adds most where `null_ratio`
collapses (weighted slope `+0.080`, Qwen3-8B standout). Do not frame this as a
new universal hallucination detector; frame it as another reason the production
surface is per-model, per-distribution calibration with safety rails.

## Running

    .venv/bin/python exploratory/shadow-ambiguity/test_shadow_ambiguity.py

Any numpy-capable Python works for the pure contracts; the repo `.venv` is
needed for the production cross-check import.

Paper figures and the compact verdict table are regenerated from the checked-in
`comprehensive_outputs/` JSONs, with no model re-tracing:

    exploratory/shadow-ambiguity/paper/figures/build_all.sh

Rendered PDFs/PNGs and `table1_summary.tex` land in
`exploratory/shadow-ambiguity/paper/figures/out/`.
