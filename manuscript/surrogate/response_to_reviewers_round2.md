---
title: "Response to the referee, second round"
subtitle: "Equilibrium-projected graph neural network surrogates for linear frame analysis: uncertainty-aware FE referral and residual-guided adaptive learning"
date: "1 November 2026"
geometry: margin=2.3cm
fontsize: 11pt
---

We thank the referee for a second careful report and for checking each of our earlier answers against the manuscript. We agree with the remaining concerns. The most important one, the calibration of the referral rule, has been fixed by re-running the experiment with separate calibration data rather than by changing the text. As before, all new results come from experiments run for this revision. Section, table and figure numbers refer to the revised manuscript.

# Summary of the changes

1. **Referral calibration (Comment 1).** The domain threshold and the conformal quantile now come from two separate sets of 1,000 new in-distribution frames, generated with their own seeds and used for nothing else. The Mahalanobis model is fitted on the training frames and the validation frames are no longer used. The procedure was repeated for all five independently trained ensembles. Section 3.8 states exactly what is guaranteed and what is only observed.
2. **Scope of the referral guarantee (Comment 2).** We took the referee's Option B. In addition, a new table (Table 18) reports how often accepted frames exceed the tolerance for nine other response quantities. The result supports the referee's concern, as explained under Comment 2.
3. **Engineering applicability (Comment 3).** The discussion, conclusions and abstract now distinguish member-force screening, kinematic screening and complete structural analysis. The claim that the surrogate can "replace most repeated FE analyses" has been removed. The axial-only compatibility residual is now reported separately.
4. **DES benchmark (Comment 4).** It is renamed "design-informed synthetic frame benchmark". The new Appendix C gives the full generation procedure, including the sizing equations, the realised storey and bay counts, and the clipping and rejection rules.
5. **Feature ablation (Comment 5).** We trained the missing variant (dimensionless scaling and load-case separation, but no statics features) with three seeds and added the A–D table the referee proposed (Table 11).
6. **Reproducibility (Comment 6).** The release now includes:
   - the input parameters of every frame (`release/frame_parameters.json.gz`);
   - a standalone cross-solver verification script that reads them;
   - a calibration script.

   The quick script is now described as report regeneration only. Merging into the default branch, the release tag and the Zenodo DOI are covered under Comment 6.
7. **Editorial (Comments 7 to 10).**
   - The definition of $H$ is now consistent (it was wrong in our first letter).
   - Section 3 is renamed.
   - Confidence intervals for the tabular baselines are now in Appendix D.
   - The text states that all runs share one fixed data split.
   - The timing scope is now stated explicitly.

# Point-by-point responses

## Comment 1: formal validity of the split-conformal referral

> *The same 400 validation frames set the domain threshold, select the in-domain frames and supply the conformal scores ... Separate the calibration stages (A–D).*

**Done, as proposed, and re-run (Sections 2, 3.8 and 6.8, Tables 16 and 17, Figs. 9 and 10).** The referee is right that the earlier procedure did not establish the finite-sample guarantee. The rule now follows the referee's Stages A to D exactly:

- **A.** The descriptor model (standardisation and Ledoit–Wolf Mahalanobis distance on 15 geometry and load-pattern descriptors) is fitted on the 3,000 training frames.
- **B.** The domain threshold $D^\ast$ is the conformal 99% quantile of the distances of an independent domain-calibration set: 1,000 new ID frames with seed family 6, analysed in PyNite and checked against statics.
- **C.** With $D^\ast$ fixed, the in-domain frames of a second independent conformal-calibration set are selected: 996 of 1,000 new ID frames, seed family 7. The quantile $\hat a$ of $e_M/s$ is computed on them ($\alpha=0.1$). This step is repeated for each of the five independently trained ensembles, giving $\hat a = 1.40$ to $1.51$.
- **D.** Both thresholds are applied to the untouched test sets.

The 400 validation frames, which were used for early stopping and model selection, are no longer used by the rule.

Section 3.8 now distinguishes three things:

1. **What the theory guarantees.** Because the domain rule is fixed independently of the conformal-calibration frames, selection preserves exchangeability. For a new in-distribution frame that passes the domain check, $P(e_M \le \hat a s) \ge 0.9$. It follows that $P(\text{accepted and } e_M > \tau) \le 0.1$.
2. **What it does not guarantee.** The violation rate *among accepted frames* is not bounded by the theory.
3. **What is only observed.** All rates on OOD-G, OOD-L and DES are empirical, because exchangeability does not hold for them.

Results (mean over the five runs):

| | ID | OOD-G | OOD-L | DES |
|---|---|---|---|---|
| Flagged by domain check | 0.3% | 49% | 71% | 0% |
| Coverage of $e_M \le \hat a s$ (in-domain) | 90.7% | 95% | 84% | 98% |
| Referred in total, $\tau=10\%$ | 0.9% | 73% | 71% | 0% |
| Accepted frames with $e_M>10\%$ | 1.4% | 1.1% | 1.7% | 0% |
| Accepted and $e_M>10\%$ (joint) | 1.4% | 0.3% | 0.5% | 0% |

The in-distribution coverage of 90.7% matches the guarantee. OOD-L shows that coverage fails under load-pattern shift, and the text keeps that qualification throughout.

The new protocol is less aggressive in Stage B than the old one, and we report this openly in Section 6.8. With 1,000 frames, the 99% quantile reaches further into the heavy tail of the in-distribution distances, so the domain check now flags 49% of OOD-G instead of 79%, and no DES frames instead of 20%. The conformal stage absorbs the difference: the violation rate among accepted frames stays at or below 1.7% on every test set.

The earlier rule's results are kept in the repository (`results/revision/ood.json`) for comparison, marked as superseded.

## Comment 2: the referral protects only the end-moment field

**Option B adopted, with additional evidence (Sections 3.8 and 6.8, new Table 18; Discussion; Conclusions; Abstract).** Wherever the rule is described, the manuscript now says that it is calibrated for the relative $L_2$ error of the end-moment field. Statements that could suggest general structural safety assurance have been removed.

To show what this restriction means in practice, Table 18 reports, for frames accepted at $\tau=10\%$, the share whose error in each of nine other quantities exceeds 10%. The table is explicitly labelled as empirical, with no guarantee. The result confirms the referee's concern, but only for the kinematic quantities:

- **Force quantities** behave like the calibrated field. The largest end moment exceeds a 10% error in at most 0.7% of accepted frames, and axial forces, shears and reactions in at most 0.6%.
- **Kinematic quantities** do not. The largest inter-storey drift exceeds a 10% error in 19% to 38% of accepted frames, rotations in 6% to 23%, and horizontal displacements in 9% to 36%.

The paper therefore states that the referral rule covers the end-moment field. The other member forces are empirically well behaved among accepted frames, but carry no guarantee. Displacement, rotation and drift predictions are separately validated estimates that the rule does not protect, and would need their own calibrated score. This finding also feeds into Comment 3.

## Comment 3: force–displacement compatibility and engineering applicability

**Done (Discussion, "Answer to the research question"; Conclusions; Abstract; Section 3.6 and Table 13).** The sentence saying that the surrogate can replace most repeated first-order analyses has been removed. The discussion now separates the three uses the referee proposed:

1. **Member-force screening.** The directly predicted, equilibrium-projected forces are the most accurate output and the one the referral rule covers.
2. **Kinematic screening.** Displacements, rotations and drifts are validated separately and are less accurate. The referral rule does not protect them (Comment 2). They are suitable for ranking and trend studies, not for checking a drift limit near its threshold.
3. **Complete structural analysis.** This use is not established, because the projected forces are not compatible with the predicted displacements.

FE analysis is retained for final verification and for referred frames. The final sentence of the conclusions now describes the framework as "an equilibrium-enforced, uncertainty-aware surrogate for selected structural-response screening tasks ... not a replacement for a complete FE analysis".

Table 13 and Eq. (6) now report the axial-only residual $r_{\mathrm{compat},N}$ alongside the full and moment-only residuals. For the raw ensemble, the median axial-only residual is 48% on ID (88% on OOD-G, 60% on OOD-L and 15% on DES), practically equal to the full residual, against a moment-only residual of 11%. This confirms that the full residual comes mainly from the axial stiffness amplifying small displacement errors. We agree that a mixed formulation treating axial and flexural compatibility separately belongs in a follow-up article, and the outlook says so.

## Comment 4: positioning and reproducibility of the DES set

**Done (Section 2, new Appendix C).** The set is now called a *design-informed synthetic frame benchmark* throughout, including the tables and figure labels. Section 2 lists its limitations explicitly:

- the section properties are clipped to the training ranges (28% of the frames have at least one clipped column section);
- it shares the training topology family;
- it uses simplified, non-code-compliant sizing rules;
- 29% of the frames exceed the $h/400$ drift target.

Appendix C gives everything needed to reproduce the 300 frames from the paper alone:

- seeds;
- storey and bay distributions, with the realised counts;
- bay widths and storey heights;
- the gravity-load and wind-load generation;
- the beam-sizing equations ($L/360$ live-load deflection and simple-span moment with $f_y = 250$ MPa);
- the column-sizing equation (shear-building drift $h/400$ with a stiffness factor of 0.5, one section per two storeys, $I_c \ge 0.3 I_b$);
- the clipping bounds and the share of clipped frames;
- the radii of gyration;
- base fixity;
- the statement that no frame is rejected.

The parameters of all 300 frames are also released in machine-readable form (Comment 6).

## Comment 5: isolating the mechanics-feature contributions

**Done (Section 5, new Table 11, Section 6.4, Discussion).** We trained the one missing variant, B, with three seeds: dimensionless scaling and separate load cases, with all ten statics-derived features removed (storey shear, overturning moment, tributary gravity load, stiffness ratios, shear-building drift and displacement). The other three variants already existed. Table 11 reports $e_{u_x}$ and $e_M$ (mean ± s.d. over three seeds) on all four test sets for:

| Variant | Dimensionless scaling | Statics features | Separate load cases |
|---|---|---|---|
| A (conventional GNN) | No | No | No |
| B (new) | Yes | No | Yes |
| C | Yes | Yes | No |
| D (proposed) | Yes | Yes | Yes |

Median errors, mean over three seeds, in %:

| Variant | ID $e_{u_x}$ | ID $e_M$ | OOD-G $e_{u_x}$ | OOD-G $e_M$ | OOD-L $e_{u_x}$ | OOD-L $e_M$ | DES $e_{u_x}$ | DES $e_M$ |
|---|---|---|---|---|---|---|---|---|
| A | 14.3 | 5.4 | 34.2 | 16.9 | 22.2 | 8.4 | 27.9 | 4.3 |
| B | 6.9 | 3.9 | 14.9 | 11.8 | 12.0 | 9.3 | 8.4 | 3.9 |
| C | 12.5 | 5.5 | 26.0 | 14.8 | 15.5 | 10.0 | 34.9 | 4.2 |
| D | 6.7 | 3.7 | 14.1 | 11.3 | 8.1 | 10.0 | 7.2 | 3.3 |

The result changes our interpretation, and we have revised the text accordingly:

- **Most of the gain comes from scaling and load-case separation.** Going from A to B, which keeps scaling and load-case separation but has no statics features, halves the displacement errors on every set. It also lowers the moment errors on ID and OOD-G.
- **The statics features add much less (B to D).** On ID, OOD-G and DES they lower the errors by 0.2 to 1.2 points. Under unseen load patterns they lower the displacement error from 12.0% to 8.1%, but the moment error rises from 9.3% to 10.0%.
- **Removing the load-case split hurts (C).** Without it, the displacement errors roughly double even with the statics features.

The manuscript's earlier statement that "the mechanics-informed inputs matter even more" was not supported, and has been replaced. Section 6.4, the discussion and conclusion 3 now credit the gain to the combined use of dimensionless scaling and load-case separation, with a smaller, mainly displacement-related contribution from the statics features.

The comparison of A with B still changes three things at once, namely physical versus dimensionless inputs, physical versus dimensionless targets, and combined versus separate cases. We therefore attribute the A-to-B gain to the combination of scaling and load-case separation, not to either one alone, and the discussion says so.

## Comment 6: reproducibility

**Partly done; the remaining steps need the authors' accounts or approval.** In this revision:

- **Verification scripts and input data (items 6 and 7).** The direct-stiffness solver (`solvers.py`) and the OpenSeesPy verification (`experiment_validation.py`) were already in the release. We have added `release/frame_parameters.json.gz`, which holds the input parameters of all 9,500 frames. We have also added `verify_from_parameters.py`, which runs PyNite, the DSM solver and OpenSeesPy directly from that file, with no dataset generation. On a 20-frame DES spot check it reproduces agreement at the 1e-13 level.
- **Quick versus full reproduction.** The referee is right. `reproduce.sh --quick` is now described in the script, the README and the manuscript as *report regeneration only*: it reads the multi-run statistics, graph baselines, adaptive curves and the five-run calibration from committed JSON. The full run (`reproduce.sh`, about 11 to 13 h on 4 CPU cores) is named as the definitive route, and it now also runs the new calibration and ablation experiments.
- **Merge, release tag, Zenodo DOI and commit hash (items 1 to 5).** These steps need the authors. The research folder becomes visible on the default branch once the pull request that adds it is merged, and the release tag, Zenodo archive and DOI must be created from the authors' GitHub and Zenodo accounts. The data availability statement keeps clearly marked placeholders for the DOI, tag and commit hash until these steps are complete.

## Comment 7: inconsistent definition of $H$

**Corrected.** The manuscript defines $H$ consistently as the **mean storey height**, $H = \frac{1}{n_s}\sum_s h_s$. This is the definition in Section 3.2, Table 4, the feature dictionary and the code (`features.py`). Our first response letter wrongly called it the total frame height; that was an error in the letter only. We checked every use of $H$ (normalisation of lengths, displacements, rotations, moments, the compatibility residual and the equilibrium rows) against the code, and all use the mean storey height.

## Comment 8: "mechanics-constrained" terminology

**Corrected.** Section 3 is now titled "Equilibrium-projected graph neural network surrogate". The remaining definition of "mechanics-constrained" in the introduction has been replaced by a sentence explaining that the surrogate is called equilibrium-projected because static equilibrium (with exact linear scaling) is the only mechanical condition it enforces. The term no longer appears in the manuscript.

## Comment 9: statistics of the baselines and the fixed data split

**Done (new Appendix D, Section 5).** Appendix D (Table D.22) gives the median $e_{u_x}$ and $e_M$ of all five tabular baselines with 95% frame-bootstrap intervals (2,000 resamples) on all four test sets. These intervals were computed in the previous revision but not shown.

The statistics paragraph of Section 5 now states that all runs share one training, validation and test split. The across-run intervals therefore reflect initialisation and batch-order variability, and the bootstrap intervals reflect test-frame sampling. Neither covers the variability from a different training sample or a different design space.

## Comment 10: scope of the timing comparison

**Clarified (Section 6.10).** All three FE timings have the same scope. Each starts from the frame parameters and includes model construction, assembly, solution, and extraction of nodal displacements and member end forces for both load cases. PyNite builds one model with two load cases. OpenSeesPy and the DSM solver build and solve once per case, so their timings are if anything slightly pessimistic. The surrogate timing also starts from the parameters and includes graph and feature construction, ensemble evaluation and projection. The conclusion that an optimised linear FE solver is faster than the surrogate for these frames stays prominent in the cost section, discussion and conclusions.

## Submission information

Author names, affiliations, the CRediT statement and the Zenodo DOI remain placeholders, to be completed by the authors.
