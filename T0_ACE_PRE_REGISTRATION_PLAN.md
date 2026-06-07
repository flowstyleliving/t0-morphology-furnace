# PRI T0 Pre-Registration Plan — ACE: Attention Commitment Estimator at the Commit Step

**Status:** `frozen 2026-05-26` — datasets generated; implementation verified (109/109 fast tests pass); Phi-3.5-mini gate decision INCLUDED. Do NOT re-specify sealed parameters after this line.

**Paper method name (locked 2026-05-30, post-seal presentation-only):** **ACE — Attention Commitment Estimator.** This is the paper-facing name for the `t=0` sealed instrument (attention-channel calibrators at the `t=0` commit step). It does NOT modify any sealed parameter, panel cell, gate threshold, or analysis-plane choice below. All references to "attention-channel calibrators at the commit step," "T0 calibrator," or "`t=0` attention calibrators" in this document refer to ACE.

**Feeds:** `wiki/paper/t0-scope-2026-05-26.md` (paper-scope decision memo, Step 5.2)
**Mirrors structure of:** `PRI_V3_PRE_REGISTRATION_PLAN.md`

---

## Amendments / post-seal clarifications

No pre-data amendments were filed.

**2026-06-06 post-seal prose clarification (no parameter change):** cleaned stale wording that still described the older generation-phase panel as `gen_step=1` / `layer=final`. The sealed block, runner, and checked-in profiles all use `--t0-commit`, `ATTENTION_PANEL_T0_WITH_V_NORMS`, `detector.gen_step=0`, and the three block-depth prefixes `{final, mid, last_minus_1}`. This clarification only aligns the narrative prose with the already-frozen implementation and artifacts; it does not change datasets, thresholds, panel cells, gates, bootstrap settings, or outcomes.

---

## One-line thesis

Per-model **ACE** (Attention Commitment Estimator) calibrators at the t=0 commit step discriminate YES/NO reliably across architectures and task domains. The analysis locus is fixed at the prefill-last-position (`t=0`); the winning metric, sign, and block-depth prefix are selected per model and dataset, with exact cell transfer tested separately.

---

## What T0 gives us that v3 cannot

1. **Commit-step localization.** v3 measured `null_ratio_post_rank1` at gen_step=1 (generation phase). T0 measures at **t=0** — the prefix/generation boundary, the actual first-token logit. This is the honest commit moment.
2. **Attention-channel discriminability.** Attention entropy and JS divergence at the commit layer provide a different observable family from Fisher-pullback geometry — more interpretable, lighter to compute, directly deployable.
3. **Cross-task generalization test.** v3 was single-task (ANLI synthetic). T0 adds TriviaQA (factual-disagreement domain) as a cross-task hold-out, directly testing whether the cell that works on ANLI transfers to a real-world task.
4. **Prior-art comparison.** T0 reports head-to-head against RAUQ and SinkProbe at the same t=0 commit step — the first honest same-instrument comparison in this line.

---

## Panel specification

### Models (9 panel models, all MLX 4-bit)

| Slug (short) | MLX community slug |
|---|---|
| Llama-3.2-3B | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Mistral-7B | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Mistral-Nemo | `mlx-community/Mistral-Nemo-Instruct-2407-4bit` |
| Phi-3.5-mini | `mlx-community/Phi-3.5-mini-instruct-4bit` |
| Phi-4-mini | `mlx-community/Phi-4-mini-instruct-4bit` |
| Qwen2.5-7B | `mlx-community/Qwen2.5-7B-Instruct-4bit` |
| Qwen3-1.7B | `mlx-community/Qwen3-1.7B-4bit` |
| Qwen3-8B | `mlx-community/Qwen3-8B-4bit` |
| Gemma-3-4B | `mlx-community/gemma-3-4b-it-4bit` |

**Phi-3.5-mini note:** This model shows real low-decidedness (Step 0 neighborhood audit 2026-05-17). Its attention discriminability is reported with that caveat. The pre-run gate decision below fixes how it enters the primary denominator.

> **Pre-run Phi-3.5-mini gate decision:** `INCLUDED` — real low-decidedness confirmed but attention features remain valid; counts in gate if OOB CI_lo > 0.50, does not count against if it fails. Denominator = 9. Set 2026-05-26 before freeze.

### Attention metric panel (21 cells per model)

All cells are measured at **t=0** via `--t0-commit`: the prefill last-position attention slice `captures[layer][0]`, before any generated-token attention query. The 21 cells are:

| Block-depth prefix | Metrics (7 each) |
|---|---|
| `final_` (last transformer block) | `js`, `js_kv_groups`, `js_no_bos`, `bos_mass`, `v_norm_bos`, `v_norm_max`, `v_norm_lastq_weighted` |
| `mid_` (middle transformer block) | same 7 |
| `last_minus_1_` (second-to-last block) | same 7 |

Invoked via `pri_calibrator.py --t0-commit --attention-with-v-norms`. Sign locked from calibration-set OOB winner direction (no post-hoc sign-fitting on test data).

### Datasets

| Role | Dataset | n | Seed | data_hash_sha256 | Path |
|---|---|---|---|---|---|
| Primary (ANLI) | ANLI R1 | 200 | **20260526** | `d1a3aed5e86af05c4b7bd459bb5938bbcca7ab6c758c855e1bce3f938b62f48e` | `experiments/t0-sealed/2026-05-26/data/anli_R1_seed20260526_n200.jsonl` |
| Secondary (TriviaQA) | TriviaQA paired-prompt | 100 | **20260526** | `f2f870a7e2feb2c711b2a782f6aa6040233c8915bb066e87421e85f4778b3149` | `experiments/t0-sealed/2026-05-26/data/triviaqa_paired_seed20260526_n100.jsonl` |

Datasets generated 2026-05-26 before freeze. Hashes computed by `sha256(file_bytes)`. Any sweep that uses different data hashes is NOT part of this pre-reg.

---

## Sealed block

**Filed 2026-05-26. Parameters below are locked. No post-hoc re-specification.**

### Primary instrument

> **t=0 first-token logit** — the prefix-phase logit for the first generated token, computed at the prefix/generation boundary (last prefix position). This is the commit step. NOT gen_step=1 generation-phase logit. Per panel-run-design decision 2026-05-25.
>
> The sealed sweep uses the `--t0-commit` flag (or equivalent data format that captures the prefix-phase hidden state). All attention metrics are extracted from the t=0 position.

### Analysis plane

- **Step**: step 0 attention capture (`captures[layer][0]`) at the t=0 commit locus — prefill last-position, before generated-token attention.
- **Block-depth prefixes**: `final`, `mid`, and `last_minus_1`; each is crossed with the seven attention metrics above.
- **Calibrator**: per-model OOB bootstrap AUROC, 1000 resamples, schema v1.2
- **Winner selection**: OOB bootstrap mean AUROC; sign locked from calibration-set direction; no pooling across models
- **Calibration data**: same dataset as evaluation (ANLI R1 n=200 or TriviaQA n=100); OOB bootstrap provides honest selection-bias-free CIs

### Primary gate threshold (E_A1)

**≥ 7/9 models** (or ≥ 7/8 if Phi-3.5-mini excluded pre-run) show OOB CI_lo > 0.50 on at least ONE of the 21 attention cells.

If Phi-3.5-mini is included and fires the low-decidedness caveat: count it only if CI_lo > 0.50; it does not count against the gate if it fails.

### Cell-transfer bars (E_A2)

Transfer = exact match on `(metric_label, block_depth_prefix, sign)` between the ANLI-winner cell and the TriviaQA-winner cell for the same model.

| Transfer count | Verdict |
|---|---|
| ≤ 1/9 | "No transfer" — Candidate A confirmed |
| 2/9 | "No transfer" — marginal but below partial-transfer bar |
| ≥ 3/9 | "Partial transfer" — paper reframes to "limited transfer; per-task recalibration still required for remaining models" |
| ≥ 5/9 | "Substantial transfer" — Candidate A headline collapses; paper repositions |

**Block-depth-only transfer** (metric differs, prefix matches): reported descriptively as a secondary observation. Does not satisfy or falsify the E_A2 gate.

### Baseline comparison (E_B1 — secondary, not gating)

RAUQ and SinkProbe are run at the same t=0 commit step on the same panel.

- **RAUQ**: reported as best-single-layer AUROC (not native max-over-layers aggregate). Aggregate sandbagging must be reported explicitly as a methodological note.
- **SinkProbe**: reported as ‖V‖-weighted column-sum (not last-query approximation).
- **Comparison**: per-model OOB AUROC. Win/loss per model (ours vs each baseline). No pooling.
- **OOB-trustworthy models only**: baseline wins/losses reported only on models where OUR calibrator's winner_stability ≥ 0.70. Flagged-ours models are reported separately as "no trustworthy comparison."

E_B1 is **supporting evidence**, not a primary gate. Baseline comparison outcome does not alter the E_A1 discriminability verdict.

### No-post-hoc re-specification

Once the first sealed-run data file lands (any model, any dataset), the thresholds, transfer bars, win/loss criteria, and Phi gate decision above are **frozen**. Any deviation must be filed as a NEW exploratory variant in the Amendments section, not a revised sealed spec.

The only exception: the Phi-3.5-mini gate decision (`[TBD]` above) **must be filled in before the first model's sealed data is generated**. If it is not filled in before data lands, the decision defaults to EXCLUDED (8-model denominator).

---

## Pre-registered experiments

### E_A1 — Attention discriminability at t=0

**Hypothesis:** Attention-channel calibrators at the t=0 commit step discriminate YES/NO conditions across architectures.

**Acceptance criterion:** ≥ 7/9 models (or ≥ 7/8 if Phi excluded) with OOB CI_lo > 0.50 on at least one of the 21 attention cells. On ANLI R1 n=200.

**Falsification:** < 7 qualifying models. Paper repositions to "attention features are insufficient for commit-step discrimination."

**Pilot evidence (gen_step=1, pre-sealed, not confirmatory):** All 9 models showed OOB AUROC 0.706–0.949 (TriviaQA pilot), all with CI_lo > 0.50. Step 1 ANLI sweep (n=75–100 per model) also showed 9/9 qualifying models. These motivate the ≥ 7/9 bar — room for t=0 instrument shift to lose signal on ≤ 2 models without falsifying the headline.

### E_A2 — Cell-transfer failure (the "no universal cell" test)

**Hypothesis:** No single attention cell (metric + block-depth + sign) transfers across ANLI R1 and TriviaQA for the same model.

**Primary measure:** Exact (metric_label, block_depth_prefix, sign) match count across 9 models.

**Acceptance criterion for Candidate A headline:** ≤ 2/9 cell matches (< partial-transfer bar of 3/9).

**Reframe trigger:** ≥ 3/9 matches → paper pivots to "partial transfer exists; layer+sign stability characterizes it."

**Collapse trigger:** ≥ 5/9 matches → "no universal cell" claim collapses.

**Pilot evidence (gen_step=1, pre-sealed):** 1/9 cell match (gemma-3-4b only). ANLI R1↔R2↔R3 inter-round comparison also showed per-model cell instability (different winning cells across rounds). Expect ≤ 2/9 at t=0.

**Layer+step stability (descriptive companion):** Report whether the block-depth prefix (final/mid/last_minus_1) is stable across tasks even when metric and sign are not. This is descriptive — it cannot satisfy or falsify E_A2.

### E_B1 — Baseline comparison (secondary)

**Hypothesis:** Our per-model OOB calibrator is competitive with RAUQ and SinkProbe at the same t=0 locus.

**Measure:** Per-model win/loss/draw on OOB AUROC vs best-single-layer RAUQ and ‖V‖-weighted SinkProbe. Reported only on OOB-trustworthy models (winner_stability ≥ 0.70).

**No sealed threshold.** E_B1 is reported descriptively; it does not gate the paper. Outcome shapes the supporting-evidence section, not the headline.

**Pilot evidence (Step 2, gen_step=1):** On 4 OOB-trustworthy models — 2 decisive wins (Phi-3.5, Qwen2.5), 1 loss to RAUQ (Llama-3.2-3B), 1 dead-heat with SinkProbe (Qwen3-8B). Baselines disagree in direction on Mistral-Nemo and Qwen3-8B (RAUQ lo, SinkProbe hi) — this baseline-disagreement observation is a genuine finding to preserve regardless of t=0 re-measurement outcome.

---

## Falsification conditions

**Structural rule (mirrors v3):** Only outcomes of confirmatory pre-registered experiments (E_A1, E_A2) can project-falsify T0. E_B1 is secondary — a bad baseline-comparison result reshapes the paper framing, but does not falsify the attention-discriminability claim.

### Confirmatory blockers (project-falsifying)

T0's "no universal cell, but discrimination works everywhere" claim is falsified if:

1. **E_A1 fails**: < 7 qualifying models at t=0 → attention features at the commit step do not reliably discriminate. Paper repositions to "commit-step discriminability is architecture-dependent."
2. **E_A2 collapses (≥ 5/9 transfer)**: a substantial universal cell exists → paper repositions to "attention-calibrator cell is near-universal; previous gen_step=1 finding was an artifact of the wrong locus."

### Diagnostic (not project-falsifying, reshapes claims)

- **Winner instability (winner_stability < 0.70) on most models**: calibration needs larger n. Paper adds a calibration-n requirement section.
- **E_B1: our calibrator loses to RAUQ on ≥ 6/9 models**: the baseline-comparison section shifts from "competitive" to "RAUQ is stronger baseline; our contribution is the cell-transfer finding."
- **Block-depth prefix stable but metric unstable**: layer-stable, metric-volatile finding is a descriptive contribution even if E_A2 shows 0/9 exact transfer.

---

## Pre-run checklist (must complete before sealing)

All boxes must be checked before the `[DRAFT]` status changes to `frozen <DATE>`:

- [x] Phi-3.5-mini gate decision filled in — INCLUDED, denominator 9 (2026-05-26)
- [x] `--t0-commit` flag implemented (2026-05-26): `ATTENTION_STEPS_T0=(0,)`, `step<0` guard, `ATTENTION_PANEL_T0_WITH_V_NORMS`; 109/109 fast tests green
- [x] ANLI R1 n=200 dataset generated — seed 20260526, hash `d1a3aed5...`
- [x] TriviaQA n=100 dataset generated — seed 20260526, hash `f2f870a7...`
- [x] Bootstrap n already defaults to 1000 in calibrator (`--n-bootstrap 1000` is the default since schema v1.1 update)
- [x] t=0 instrument: `captures[layer][0]` (prefill last-position); `--t0-commit` uses step=0 throughout
- [ ] RAUQ and SinkProbe implementations confirmed at t=0 locus — deferred to E_B1 run (secondary, non-blocking)
- [x] Winner-selection locks sign from calibration-set OOB direction (existing behavior, no change)
- [x] Repo git SHA recorded in every profile JSON (existing `provenance.calibrated_at_iso` + hash fields)
- [x] `wiki/paper/t0-scope-2026-05-26.md` Step 5.3 handoff note consistent with this spec

---

## Gap analysis (from scope memo Step 5.2)

The following gaps **must be closed** before the sealed run. All are blocking for the primary Candidate A paper:

| Gap | Severity | What fills it |
|---|---|---|
| t=0 attention sweep on ANLI R1 (n=200) | 🔴 blocking | Implement `--t0-commit` flag; re-run calibrator sweep on all 9 models |
| t=0 attention sweep on TriviaQA (n=100) | 🔴 blocking | Same flag, same panel, paired-prompt dataset at seed 20260526 |
| RAUQ + SinkProbe at t=0 locus | 🟡 important (for E_B1) | Adapt baseline scripts to t=0 commit step |
| Phi-3.5-mini gate decision | 🟡 important (for gate denominator) | Decide before first data lands |
| n_bootstrap=1000 in sealed calibrator config | 🟡 important (for CI reliability) | Config change + re-run |

---

## Relationship to v3 and the paper arc

v3 established that `null_ratio_post_rank1` at gen_step=1 discriminates YES/NO reliably (sealed E18 passes 3/3). T0 shifts the instrument (t=0, not gen_step=1), shifts the observable family (attention channels, not Fisher geometry), and adds a cross-task test.

If both v3 (Fisher) and T0 (attention) pass their respective discriminability gates at t=0, the paper argues these are complementary observables: attention geometry is deployable without `W_u`; Fisher geometry provides the mechanistic interpretation of WHY attention diverges at commitment. The causal probe pilot (Step 4) is the bridge — it shows the Fisher direction `v_top` causally modulates commitment, suggesting attention-divergence at the commit step is a *consequence* of Fisher-space geometry.

v3's sealed claims are **not modified** by T0. T0 is a separate paper-worthy contribution, not a re-run of v3.
