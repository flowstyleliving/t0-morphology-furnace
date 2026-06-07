# T0 Morphology Furnace

Canonical private archive for **ACE — Attention Commitment Estimator**, the T0 Furnace result: a sealed `t=0` attention-morphology instrument for reading YES/NO commitment before the first generated token.

ACE asks whether the attention channel at the prefill last position already carries the model's commitment. It does not use the output head as the metric. Instead it calibrates over attention morphology cells: cross-head JS disagreement, BOS/sink mass, and value-norm-weighted attention reductions across `final`, `mid`, and `last_minus_1` block depths.

## Sealed Result

The frozen T0 run used:

```bash
--t0-commit
--attention-with-v-norms
--n-bootstrap 1000
```

Primary ANLI R1 gate:

- Dataset: `experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl`
- Panel: 9 MLX 4-bit models x 21 ACE cells
- Verdict: **7/9 PASS** with OOB CI lower bound above chance

Cross-task TriviaQA:

- Dataset: `experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl`
- Verdict: stronger descriptive discriminability, **8/9** with OOB CI lower bound above chance
- Exact ANLI to TriviaQA cell transfer: **3/9**, triggering the pre-registered partial-transfer framing

The durable claim is method-level generalization, not universal-cell transfer: ACE works as a per-model, per-distribution calibrator, and the winning cell must be calibrated for the deployment setting.

## Repository Map

- `T0_ACE_PRE_REGISTRATION_PLAN.md` — frozen ACE/t=0 pre-registration plus post-seal prose clarification.
- `pri_calibrator.py` — schema v1.2 calibrator with nested out-of-bag winner selection.
- `pri_detector.py` — deployment-time scorer for calibrated profiles.
- `pri_runtime.py`, `model_adapters.py`, `pri_v2_io_plugins.py` — model runtime, architecture adapters, and prompt/answer parsing.
- `scripts/run_t0_sealed_sweep.sh` — sealed ACE sweep runner.
- `scripts/diagnose_inter_head_disagreement.py` — attention and value-norm capture helpers used by ACE.
- `scripts/rauq_at_commit.py`, `scripts/sinkprobe_baseline.py`, `scripts/build_t0_coverage_matrix.py` — baseline and table helpers.
- `experiments/t0-sealed/2026-05-26/` — sealed data, profiles, and run logs.
- `paper/t0/figures/` — figure/table builders and rendered artifacts for the ACE paper track.
- `tests/` — fast tests for ACE attention cells, baseline helpers, calibrator/detector schema behavior.

## Quick Check

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

.venv/bin/pytest tests/test_attention_cells.py tests/test_t0_head_to_head.py -q -m "not slow"
```

See `REPRODUCE.md` for the sealed sweep and `ARCHIVE.md` for what is intentionally included/excluded.
