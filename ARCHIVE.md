# Archive Scope

This repository is the canonical t=0 ACE archive for the `t=0` morphology result.

## Included

Core sealed specification:

- `T0_ACE_PRE_REGISTRATION_PLAN.md`
- `README.md`
- `REPRODUCE.md`

Runtime and calibration code:

- `pri_calibrator.py`
- `pri_detector.py`
- `pri_runtime.py`
- `pri_v2_io_plugins.py`
- `pri_v2_mlx_pipeline.py`
- `model_adapters.py`
- `attention_contribution.py`
- `hidden_state_collector.py`
- helper modules needed by imports and tests

ACE/t=0 scripts:

- `scripts/run_t0_sealed_sweep.sh`
- `scripts/run_t0_step1_pipeline.sh`
- `scripts/run_t0_step2_rauq.sh`
- `scripts/run_t0_step2_sinkprobe.sh`
- `scripts/diagnose_inter_head_disagreement.py`
- `scripts/rauq_at_commit.py`
- `scripts/sinkprobe_baseline.py`
- `scripts/build_t0_coverage_matrix.py`

Artifacts:

- `experiments/t0-sealed/2026-05-26/data/`
- `experiments/t0-sealed/2026-05-26/profiles/`
- `experiments/t0-sealed/2026-05-26/smoke/`
- `paper/t0/figures/`
- `paper/t0/figures/source/head_to_head.csv` — minimal secondary baseline source for regenerating Table 3.

Tests:

- ACE attention-cell tests
- t0 head-to-head table tests
- baseline helper tests
- calibrator/detector schema tests

## Intentionally Excluded

Exploratory branches and artifacts after the T0 archive:

- v5 residual friction
- v6 projection veto
- v7 attention route
- v8 ACE route override
- residual-friction / projection-veto / ace-route-override experiment outputs

Older v3 experiment sweeps are not part of this archive except where small compatibility helpers remain necessary for imports.

## Source

Imported from:

```text
Repository: flowstyleliving/PRI_at_commitment
Commit: ad17f54
Tag to create here: t0-ace-sealed-2026-05-26
```

## Archive Principle

This repo should answer one question cleanly:

> What exactly was the sealed ACE `t=0` attention-morphology result, and how do we reproduce or inspect it?

Changes after import should preserve that shape. New exploratory candidates belong elsewhere.

Historical prep scripts may remain for audit context, but only `scripts/run_t0_sealed_sweep.sh` is the sealed reproduction runner.
