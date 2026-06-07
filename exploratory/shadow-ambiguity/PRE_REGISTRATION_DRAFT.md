# Pre-registration (DRAFT, unnamed) — Readout shadow-ambiguity as a late-layer commitment-morphology detector

**Status:** DRAFT — not sealed, not named. Falsification criteria below are binding once a fresh-seed run is launched.
**Lineage:** forward morphology lab (`exploratory/`); `W_u`-using readout-morphology sibling of ACE (the `W_u`-free attention-morphology line). Branch `shadow-ambiguity`.

---

## 1. Background (evidence in hand)

A readout statistic family reads the spectrum of the centered softmax-Fisher pulled back to hidden space, `F_c = W_uᵀ (diag(p) − p pᵀ) W_u`, at the commitment instant:
- `fisher_eff_rank` = exp(Shannon entropy of the active spectrum) [Roy–Vetterli],
- `spectral_entropy` (monotone-equivalent to eff_rank → identical AUROC),
- `shadow_logvol` = per-direction mean off-top log pseudo-volume.

Three exploratory results so far (all on the morphology lab, not sealed):
- **Contract suite** (numpy identities, 7/7) — math is correct.
- **Temperature pre-check** (label-free, 4 models) — the stats are not *pure* confidence proxies on generic text (|Spearman ρ vs surprise| 0.62–0.87 across commits at T=1).
- **Labeled pilot** (ANLI R1 n=200, 4 models) — **complementary to the rank-1 readout null-ratio baseline (`null_ratio_post_rank1`)**: subsumed where the baseline works (Mistral-7B, Qwen2.5-7B, Llama-3.2-3B; incremental ≈ 0), but adds **control-clean incremental AUROC where the baseline collapses** — Qwen3-8B (baseline AUROC 0.456 = below chance): eff_rank/shadow_logvol incremental +0.13 over {surprise, null-ratio} (paired CI excludes 0), partial r +0.28 / −0.37 beyond surprise+null-ratio (CIs exclude 0), shuffled-label control flat.
- **Depth audit** (Qwen3-8B logit-lens, all 36 layers) — **no single crossover**; the readout shadow signal is **confidence-coupled at early/mid layers** (`pearson(p_max, shadow_logvol)` ~0.97–0.99) and **decouples only in late layers** (≳ block 22, → 0.1–0.5). The genuine beyond-confidence complementary signal is a **late-layer phenomenon** (band ~L24–28 on Qwen3-8B).

## 2. Hypothesis (pre-registered)

**H1.** In the regime where the sealed rank-1 readout null-ratio baseline collapses (baseline AUROC ≤ 0.55), a **late-layer** readout shadow statistic (`fisher_eff_rank` and/or `shadow_logvol`) carries **incremental** discriminative signal for the commitment label, *after controlling for both token surprise and the null-ratio baseline*, and this signal is **not** attributable to confidence-coupling (it persists where per-commit brittleness is low).

**H0 (null).** Any incremental signal is (a) absent (paired CI includes 0) across the baseline-collapse regime, OR (b) confined to high-brittleness (confidence-coupled) layers/commits.

## 3. Design (pinned before the run)

- **Statistics (carry all):** `fisher_eff_rank`, `spectral_entropy`, `shadow_logvol` (reference impls in `test_shadow_ambiguity.py`). Headline = the one with the larger *labeled* partial-r in the pilot (currently `shadow_logvol`); report all.
- **Baselines:** `surprise`; the rank-1 readout null-ratio metric `null_ratio_post_rank1` (computed by the inherited centered-Fisher readout core).
- **Layer window (pinned from the depth audit):** a **late-layer window** — the final 25% of blocks (e.g. Qwen3-8B blocks 27–35), plus the actual readout. NO early/mid layers in the primary (they are confidence-coupled). Pin the exact block indices per model before the run.
- **Commit instant:** `gen_step = 1` (the sealed commitment plane), final-layer readout `p_t` + the chosen late-layer hidden states via logit lens.
- **Data:** sealed ANLI R1 n=200 (balanced); label = contradiction (1) vs entailed (0). Replicate on a second distribution if available (e.g. the sealed TriviaQA paired set).
- **Models — expand the baseline-collapse regime (the decisive cohort):** Qwen3-8B (confirmed collapse), **Qwen3-1.7B** and at least one more high-confidence / Qwen-family model to test whether the rescue generalizes. **Complementarity controls (baseline works):** Mistral-7B, Qwen2.5-7B, Llama-3.2-3B (expect subsumed).

## 4. Primary endpoint (the decisive bar)

For each model in the baseline-collapse cohort, a headline shadow statistic must show **positive incremental AUROC** under k-fold nested-OOB logistic regression, with a **paired CI on the difference excluding 0**, against **two** base models:
1. `{surprise}` alone — the regularized confidence base (NOT only the dead-baseline base, which inflated the pilot's +0.13);
2. `{surprise, null_ratio_post_rank1}` — confidence + the sealed baseline.

Plus the **partial correlation** of the statistic with the label, controlling for both `surprise` and `null_ratio` (CI excludes 0).

**Success:** incremental CI > 0 (over base 1 *and* base 2) and partial-r CI > 0 on **≥ 2** baseline-collapse models.

## 5. Mandatory controls (a finding only counts if all pass)

- **Brittleness gate.** Report `pearson(p_max, statistic)` per layer/commit. A claimed incremental result is **discarded** if it sits where brittleness ≥ 0.9 (confidence-in-disguise). The primary must come from a low-brittleness late-layer window.
- **Shuffled-label control.** Incremental AUROC must vanish (CI includes 0).
- **Temperature-matched control.** Beat "just surprise at this T".
- **Random-rotation control.** Rotating the off-top subspace must destroy `shadow_logvol`'s signal (proves it reads spectrum shape, not a basis artifact).
- **No silent caps.** Report n per model, drops, and any model that fails to run (load-bound, etc.).

## 6. Falsification

H0 stands (candidate retired or reframed) if, across the baseline-collapse cohort, the incremental CI includes 0 over base 1, OR the only positive results are at brittleness ≥ 0.9, OR the partial-r CI includes 0. A pilot-style single-model positive that does not replicate on a second collapse-regime model is **not** sufficient.

## 7. Confound register (lessons already paid for)

- **Degraded-base inflation** — a dead baseline as a feature drags the base AUROC down and inflates "incremental"; mitigated by requiring incremental over `{surprise}` alone with a regularized base.
- **Confidence-coupling by depth** — early/mid-layer shadow AUROC is ~confidence; mitigated by the late-layer window + brittleness gate.
- **Single-model fragility** — Qwen3-8B alone is suggestive, not conclusive; mitigated by the ≥2-model cohort.
- **Logit-lens ≠ readout** — the depth audit's per-layer null-ratio (block-Δh, lens-p) is not the readout baseline; the primary uses the readout/late-window consistently for all statistics and baselines.

## 8. Provenance / reproducibility

Pin and record: data sha256, core-code sha256, seed, env (model revisions, mlx versions), and the exact per-model layer-window indices. Fresh seed distinct from all pilot seeds.

---

*Naming + any rebranding of inherited-core references are deferred to the separate artifact-audit pass; this draft uses functional names for the baseline and compute core.*
