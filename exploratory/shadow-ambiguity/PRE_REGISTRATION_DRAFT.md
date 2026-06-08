# Pre-registration Draft v2: Shadow-Ambiguity Readout Morphology

**Status:** Draft v2, not sealed. This document is binding only for fresh-seed runs launched after v2 is written.
**Scope:** Forward morphology lab only (`exploratory/shadow-ambiguity/`). The existing Qwen3-8B ANLI R1 n=200 pilot result is exploratory: its +0.13 increment was measured over a degraded `{surprise, null_ratio}` base and fails the fair-base bar over `{surprise}` alone.

## 1. Lesson From the Pilot

The pilot's Qwen3-8B increment was inflated by degraded-base behavior: `{surprise, null_ratio}` AUROC was 0.602 and the null-ratio marginal was below chance at 0.456. Reanalysis over the fair `{surprise}` base leaves positive point estimates but CIs include zero for the named candidates (`fisher_eff_rank` +0.044 [-0.025, 0.111]; `shadow_logvol_r1` +0.076 [-0.007, 0.158]). The robust pilot evidence is the oriented partial association beyond confidence (+0.28 for `fisher_eff_rank`, -0.37 for `shadow_logvol_r1`), not a decisive incremental AUROC win.

The reframed hypothesis is therefore not a single-model anecdote. It is: shadow statistics add over confidence broadly, while the null-ratio baseline may subsume them where it works. That claim is tested by a cross-(model, benchmark) meta-analysis.

## 2. Hypotheses

**H1, primary.** The pre-specified oriented primary statistic adds incremental AUROC over both fair bases in a cross-(model, benchmark) random-effects meta-analysis:

- base A: `{surprise}`;
- base B: `{surprise, null_ratio_post_rank1, p_max}`.

The primary effect must survive multiplicity and brittleness gating and must meet a minimum practical effect: random-effects mean increment >= 0.02. CI exclusion of zero is necessary but not sufficient.

**H2, secondary complementarity.** The increment over `{surprise, null_ratio_post_rank1}` is larger where null-ratio is weak. Weakness is defined as `1 - AUROC(marginal null_ratio_post_rank1, train-locked)`, so the predictor measures null-ratio's own oriented marginal power rather than the combined `{surprise, null_ratio}` base. This regime interaction is tested across the full cohort, not as a single-model story.

**Null.** The candidate is falsified if the meta-CI over fair base A includes 0, or the primary effect is only present at high brittleness, or the primary fails multiplicity, or the meta effect is below +0.02.

## 3. Statistics and Orientation

Exactly one primary is registered:

- `fisher_eff_rank`, oriented higher -> contradiction.

Secondary statistics, corrected as one family:

- `neg_shadow_logvol_r1 = -shadow_logvol_r1`, oriented higher -> contradiction because lower raw log-volume predicted contradiction in the pilot;
- `spectral_entropy`, oriented higher -> contradiction and treated as monotone-equivalent/near-duplicate evidence for effective rank.

Orientation is locked from pre-registration or train folds only. Test-fold labels must never be used to pick a sign. Every output records the orientation source and asserts that orientation was not learned from test folds.

## 4. Pinned Layer Aggregation

No best-layer selection is allowed.

For a model with `B` transformer blocks indexed `0..B-1`, the late-window block count is `ceil(B / 4)`, start index is `B - ceil(B / 4)`, and the pinned block set is `[start, start+1, ..., B-1]`. The aggregate statistic is the arithmetic mean over those late-window block logit-lens statistics plus the final readout statistic at `gen_step = 1`.

Example: for 36 blocks, `ceil(36/4)=9`, so blocks `27..35` plus readout are used. Exact indices are recorded per model.

## 5. Design and Analysis

- Data: all available sealed benchmark JSONL files under `experiments/t0-sealed/*/data/`, including ANLI R1 n=200 and TriviaQA paired n=100; ANLI R2/R3 are included if present.
- Models: all cached `mlx-community/*` models discoverable under `~/.cache/huggingface/hub`, with known-failing models recorded in coverage. `gpt-oss-20b` is expected to be too heavy; `gemma-3-1b` and `dolphin-nemo` are expected harness-gate risks.
- Commit instant: `gen_step = 1`.
- Features: `label`, `surprise`, `p_max`, `null_ratio_post_rank1`, `fisher_eff_rank`, `neg_shadow_logvol_r1`, `spectral_entropy`.
- CV: repeated out-of-fold logistic regression, 5 folds x at least 10 repeats when class counts allow. OOF predictions are averaged across repeats before AUROC estimation. The same fold assignments are used for base and augmented models across all statistics. If a base feature set is identical, its AUROC must be identical in every reported comparison. Reports include across-repeat AUROC and increment variability as a split-instability diagnostic.
- Uncertainty: paired bootstrap CIs over shared averaged-OOF predictions; oriented partial correlations with bootstrap CIs. The repeated-CV diagnostic is reported separately and does not change the registered averaged-OOF endpoint.

## 6. Multiplicity

There is exactly one uncorrected primary test: primary statistic x pinned late-window-plus-readout aggregate x fair base A x random-effects meta-rule.

All other tests are secondary and corrected by permutation/familywise logic or Holm correction over the declared family. The shuffled-label max-stat control uses 1000 permutations by default. Reports must include the total test count and the empirical p-value resolution.

## 7. Brittleness Gate

For the aggregated primary statistic, report:

- `corr(stat, p_max)` with bootstrap CI;
- `corr(stat, surprise)` with bootstrap CI.

The primary must also beat base B `{surprise, p_max, null_ratio_post_rank1}`. A primary claim fails if either brittleness upper CI is >= 0.75. This is stricter than the pilot's 0.9 threshold and applies to the exact aggregate used in the primary test.

## 8. Mandatory Controls

- Shuffled-label control with the same analysis machinery.
- Temperature-matched/confidence control: report whether the statistic still adds after `p_max`, and compare against `{surprise, p_max}` confidence-only augmentation.
- Random-rotation control: because the registered statistics are spectral, an orthogonal hidden-space rotation is an invariance control, not a destruction control. The harness applies a deterministic random Householder reflection in hidden coordinates. The transformed-spectrum values must match numerical tolerance.
- Degraded-base flag: report `{surprise, null_ratio_post_rank1}` versus `{surprise}`. Warn if the former is lower than the latter.
- Coverage report: record every discovered model/benchmark, attempted pair, skip reason, drops, and non-finite counts.

## 9. Family-Confound Rule

A general claim requires positive primary evidence in at least two collapse-regime models spanning at least two architecture families. A per-pair positive for this family rule must clear both base A and base B with increment >= 0.02 and lower CI > 0. If only Qwen-family models pass, the permitted claim is "Qwen-family null-ratio rescue," not a general shadow-ambiguity result.

## 10. Confound Register

- Degraded-base inflation: lesson #1 from the pilot.
- Post-hoc layer selection: prevented by the final-25%-plus-readout rule.
- Multiplicity at scale: controlled by one primary and corrected secondary tests.
- Qwen-family confound: handled by the family-spanning verdict rule.
- Per-model underpower: addressed by cross-(model, benchmark) random-effects meta-analysis rather than one model. A single completed pair (`k = 1`) is reported without a selected meta CI. For small meta-analytic cohorts (`2 <= k < 10`), the selected random-effects CI is a modified Knapp-Hartung/t interval; otherwise the selected CI is the normal DerSimonian-Laird interval, with both methods recorded.

## 11. Provenance

Fresh runs use a seed distinct from the pilot seed `20260607`. Record fresh seed, code hashes, data hashes, model inventory, benchmark inventory, exact window indices, non-finite/drop coverage, and environment versions.
