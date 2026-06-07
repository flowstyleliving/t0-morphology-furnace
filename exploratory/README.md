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

## Running

    python3 exploratory/shadow-ambiguity/test_shadow_ambiguity.py

Any numpy-capable Python (scipy optional).
