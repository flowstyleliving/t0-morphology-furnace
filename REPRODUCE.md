# Reproduce ACE t=0

This repo archives the sealed `t=0` ACE run from `PRI_at_commitment`, imported from source commit `ad17f54`.

## Environment

Apple Silicon with MLX:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If gated Hugging Face access is needed:

```bash
hf auth login
```

## Sealed Data

The sealed runner checks file hashes before scoring.

```text
ANLI R1 n=200:
experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl
d1a3aed5e86af05c4b7bd459bb5938bbcca7ab6c758c855e1bce3f938b62f48e

TriviaQA paired n=100:
experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl
f2f870a7e2feb2c711b2a782f6aa6040233c8915bb066e87421e85f4778b3149
```

## Sealed Sweep

The sealed command path is:

```bash
bash scripts/run_t0_sealed_sweep.sh
```

To run one dataset:

```bash
DATASET=anli bash scripts/run_t0_sealed_sweep.sh
DATASET=triviaqa bash scripts/run_t0_sealed_sweep.sh
```

The important flags, fixed by the pre-registration, are:

```text
--t0-commit
--attention-with-v-norms
--n-bootstrap 1000
```

`--t0-commit` uses the prefill last-position attention slice, `captures[layer][0]`, before generated-token attention. `--attention-with-v-norms` selects the 21-cell ACE panel:

```text
{final, mid, last_minus_1}
x
{js, js_kv_groups, js_no_bos, bos_mass, v_norm_bos, v_norm_max, v_norm_lastq_weighted}
```

## Existing Sealed Artifacts

The archived profile JSONs and logs live at:

```text
experiments/t0-sealed/2026-05-26/profiles/anli/
experiments/t0-sealed/2026-05-26/profiles/triviaqa/
```

Each profile records:

- winning ACE cell
- sign
- in-sample AUROC
- nested out-of-bag AUROC median and interval
- winner stability
- provenance hashes for score-critical files

## Fast Validation

No model load:

```bash
.venv/bin/pytest tests/test_attention_cells.py tests/test_t0_head_to_head.py -q -m "not slow"
```

Expected for the narrow README smoke slice:

```text
58 passed, 2 deselected
```

The full non-slow archive suite is:

```bash
.venv/bin/pytest -q -m "not slow"
```

Expected on the current archive state:

```text
154 passed, 12 deselected
```

## Caveat

Re-running the sealed sweep today may produce provenance-hash drift if runtime files differ from the checked-in sealed profiles or if upstream model snapshots have changed. The archived profile JSONs are the sealed artifacts; a fresh run is a reproduction attempt, not a rewrite of the sealed verdict.
