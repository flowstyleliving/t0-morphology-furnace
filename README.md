# Readout Pseudo-Volume (RPV)

Primary repository for the **Readout Pseudo-Volume (RPV)** study: a readout-geometry commitment signal tested across 13 open-weight models and two benchmarks. RPV reads the eigen-spectrum of the centered softmax-Fisher pullback of the unembedding at the commitment instant, then asks whether that geometry adds signal beyond confidence and the earlier `null_ratio` detector.

The headline result is deliberately narrow: RPV generalizes at the cross-model meta level as a confidence-independent commitment signal, but it is largely absorbed by `null_ratio`. The result is evidence for shared commitment geometry, not a new universal hallucination detector.

## RPV Result

RPV was evaluated in `exploratory/shadow-ambiguity/` on 26 model-benchmark pairs:

- 13 models x 2 benchmarks: ANLI R1 and paired TriviaQA
- Base-A result over `{surprise}`: random-effects mean `+0.102` AUROC, 95% CI `[+0.065, +0.140]`
- Base-B result over `{surprise, null_ratio, p_max}`: `+0.011`, below the pre-registered `+0.02` practical-effect bar
- Verdict: **H1 NO-GO** as a new detector; useful as a narrow backstop candidate where `null_ratio` collapses

The paper figures and table are generated from checked-in JSON outputs:

```bash
exploratory/shadow-ambiguity/paper/figures/build_all.sh
```

The core contract suite is:

```bash
.venv/bin/python exploratory/shadow-ambiguity/test_shadow_ambiguity.py
```

## Archived ACE Result

The sealed **ACE — Attention Commitment Estimator** result is frozen on the `archive` branch. ACE asks whether the attention channel at the prefill last position already carries the model's YES/NO commitment before the first generated token. It does not use the output head as the metric. Instead it calibrates over attention morphology cells: cross-head JS disagreement, BOS/sink mass, and value-norm-weighted attention reductions across `final`, `mid`, and `last_minus_1` block depths.

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

- `exploratory/shadow-ambiguity/` — RPV harness, pre-registration draft, 26-pair JSON outputs, paper figure builder, and contract tests.
- `exploratory/shadow-ambiguity/paper/figures/` — RPV figure/table builder and rendered workshop figures.
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

See `exploratory/README.md` for the RPV handoff, `REPRODUCE.md` for the sealed ACE sweep, and `ARCHIVE.md` for what is intentionally included/excluded.

The `archive` branch is the frozen ACE archive. `main` now centers the RPV handoff while preserving ACE provenance.
