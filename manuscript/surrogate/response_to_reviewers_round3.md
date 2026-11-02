---
title: "Response to the referee, third round"
subtitle: "Equilibrium-projected graph neural network surrogates for linear frame analysis: uncertainty-aware FE referral and residual-guided adaptive learning"
date: "2 November 2026"
geometry: margin=2.3cm
fontsize: 11pt
---

We thank the referee for a third careful assessment and for the positive recommendation. All remaining comments concern precision of wording, the literature and the public release, and we have addressed each one. As the referee suggested, no new experimental programme was started. The one new piece of work is a reproduction test from a fresh installation, reported under Comment 2. Section, table and page numbers refer to the third-round manuscript.

# Point-by-point responses

## Comment 1: differentiability of the complete pipeline

**Option B adopted (Sections 3.5 and 6.10).** The referee is right that the claim was not supported. The features and the equilibrium matrix $\mathbf{C}$ are assembled in NumPy, and the projection is solved there with LAPACK. All of this happens outside PyTorch's automatic-differentiation graph. The domain check, the conformal acceptance and the FE referral are also discrete decisions.

We made two changes:

- **Section 6.10.** The phrase "a differentiable, batched map ... useful for gradient-based design" has been removed. The surrogate is now described as "a batched, equilibrium-projected surrogate of the member forces with a calibrated referral rule for response screening". A new sentence states that end-to-end differentiability is not claimed, gives the reasons above, and notes that gradient-based design would need a differentiable implementation of these steps and a check of its gradients, which this study does not provide.
- **Section 3.5.** After the description of the LAPACK solve, a sentence now states that the implemented pipeline is not differentiable end to end with respect to the frame parameters.

The word "differentiable" now appears only in these two statements that the pipeline is not differentiable.

## Comment 2: public reproducibility

**Manuscript wording corrected. Release steps prepared, but waiting for the authors' approval.** We agree that the paper must not describe the package as released before it is. We made three changes:

- **Contribution 4.** The sentence "Code, data generator and trained weights are released" now reads: "The code, data generator, frame parameters and trained weights have been prepared as a reproducibility package (see Data and code availability); its archived public release is pending."
- **Data and code availability.** The statement now says that the folder is on a development branch, which it names, and that it has not yet been merged into the default branch. It also says that no versioned release has been published yet. A marked placeholder states that the authors will merge the folder, publish a versioned release, archive it with the dataset on Zenodo, and insert the release tag, commit hash and DOI before publication.
- **Other mentions.** Two sentences that referred to "the released scripts" (Section 6 and the AI-use declaration) now say "the scripts of the reproducibility package".

The distinction between `reproduce.sh --quick` (report regeneration) and the full reproduction is unchanged.

**Reproduction from a fresh installation (item 5).** We cloned the repository at the submitted commit into an empty directory, created a new Python 3.13 virtual environment, and ran `bash reproduce.sh --quick` on a 4-core CPU with no GPU. The script installed PyNite from the repository and the pinned dependencies, and finished without errors in about 19 minutes, of which about 8 minutes were spent installing dependencies. In this run:

- all 7,200 dataset frames and the 300 DES frames were regenerated with PyNite;
- the cross-solver check agreed with PyNite to at most $1.3\times10^{-11}$ (relative) for the direct-stiffness solver and OpenSeesPy, both on the regenerated data and from the parameter file;
- the closed-form benchmarks were reproduced;
- the projection variants, the $\boldsymbol{\Sigma}$ sensitivity, the warm-start study and the tabular baselines were recomputed from the bundled weights;
- every number in these recomputed results agreed with the committed values to within floating-point round-off (about $10^{-15}$ relative), and all tables of the manuscript were regenerated unchanged.

Only machine-dependent wall-clock times differed, as expected.

The test also exposed two defects in the quick mode, which we have fixed:

1. The quick mode does not regenerate the two calibration sets, and it was rewriting the frame-parameter file without them. It now keeps them.
2. The quick mode overwrote the committed timing file with the local measurement, and it lost three training-time entries in a table because it read them from model files that only the full run creates. It now writes the local timing to a separate file, and the training times are stored in the committed results.

After the fixes, the quick mode leaves every committed input and table unchanged.

We repeat that the quick mode does not retrain the networks and does not rerun the multi-run, adaptive-sampling or calibration experiments. We did not repeat the full reproduction (about 11 to 13 hours) in a fresh environment for this round.

**Merge, release, Zenodo and DOI (items 1 to 4).** These steps publish material under the authors' accounts, so they are done only with the authors' explicit approval and are not yet complete. The manuscript does not claim them.

## Comment 3: kinematic screening described too broadly

**Corrected (Section 7, Conclusions, Abstract).** We adopted the referee's wording:

- **Member-force screening.** The section now states that the calibrated referral procedure supports this use, subject to the end-moment guarantee. That guarantee is $P(e_M \le \hat a s) \ge 0.9$, and hence $P(\text{accepted and } e_M > \tau) \le 0.1$, for in-distribution frames.
- **Kinematic screening.** The discussion now says the model "supports exploratory ranking, trend studies and preliminary estimation as supplementary screening estimates". It adds that the current framework "provides no calibrated decision mechanism for accepting displacement or drift predictions in place of FE analysis".
- **The summary sentence.** It now reads: "The current referral framework therefore directly supports member-force screening inside a validated domain, where it can reduce the number of repeated FE analyses; the kinematic outputs are supplementary screening estimates without a calibrated acceptance rule. FE analysis is retained for any final structural verification, for the frames that the rule refers and for any decision that depends on displacements or drifts."
- **The abstract.** Its final sentence now says that the referral framework supports member-force screening and that the kinematic outputs serve only as supplementary screening estimates.

## Comment 4: statistical interpretation in the abstract and conclusions

**Corrected (Abstract, Section 6.8, Section 7, Conclusion 5).** Wherever the 1.4% figure appears, it is now called an observed, empirical conditional violation rate on the test set, not a guarantee. It is reported separately from the marginal coverage:

- **Abstract.** "The referral rule reached the guaranteed marginal coverage of the end-moment error on in-distribution test frames (observed 90.7% for a nominal 90%); it accepted 99% of them, with an observed, not guaranteed, 1.4% of the accepted frames above a 10% end-moment error."
- **Conclusion 5.** The 90.7% coverage is described as "consistent with the guaranteed marginal coverage of at least 90%". The 1.4% and the OOD rates of 1.1% to 1.7% are described as "observed (empirical, not guaranteed)". For the OOD rates, the conclusion adds that no guarantee applies because those frames are not exchangeable with the calibration data.
- **Section 6.8.** A sentence after the ID result now states what the theory guarantees: the marginal coverage, and the 10% bound on the joint rate of accepted violations. It also states that none of the OOD-G, OOD-L or DES rates is guaranteed.
- **Section 3.8.** The explicit distinction between joint and conditional rates is kept unchanged.

## Literature: PeTIGN (2026) and hybrid local-global attention GNNs (2026)

**Added (Section 1.1).** We checked both references against the publisher's metadata (Crossref and ScienceDirect):

- Liang Z, Li J, Deng L, Kong X. PeTIGN: A physics-encoded graph network for dynamic response computation of large-scale structures. *Engineering Structures* 364 (2026) 123094. doi:10.1016/j.engstruct.2026.123094.
- Patrignani L, Pinho ST. Graph neural networks with hybrid local-global attention for effective prediction of mechanical response in structures. *Computer Methods in Applied Mechanics and Engineering* 452 (2026) 118753. doi:10.1016/j.cma.2026.118753.

A new paragraph in Section 1.1 introduces both papers:

- **PeTIGN** computes the dynamic response of large-scale structures with a physics-encoded graph network.
- **The hybrid-attention network** combines local neighbourhood attention with periodic global attention in a graph transformer. It predicts strain fields in open-hole composite plates and woven composites.

The paragraph explains why we do not benchmark against them: their response quantities and structural domains differ. It also notes that they share the trend towards physical priors and long-range information exchange, which motivates our global-pooling and StructGNN-style comparisons. The gap statement that follows now says that these gaps also remain open in this broader literature. It ends by stating what distinguishes the present work for linear static planar frames: exact equilibrium projection, a referral rule whose two thresholds are calibrated on separate data, and residual-based adaptive FE acquisition.

The two papers were not added to Table 1, because we could not verify every column of that table for them (equilibrium and compatibility enforcement, uncertainty, adaptive sampling, OOD testing) from the material available to us.

## Minor editorial corrections (Section 6 of the report)

| Location | Action |
|---|---|
| Abstract | 1.4% identified as an observed rate (Comment 4). |
| Section 1.1 | 2026 literature added. |
| Section 3.5 | NumPy projection reconciled; non-differentiability stated (Comment 1). |
| Section 3.8 | Conditional versus joint distinction kept. |
| Section 6.8 | Empirical OOD results separated from the ID guarantee. |
| Section 6.10 | Gradient-based design statement removed. |
| Section 7 | Kinematic-screening claim narrowed (Comment 3). |
| Conclusion 5 | Conditional violation rate described as empirical. |
| Data availability | Release status stated accurately; tag, commit and DOI placeholders kept until release (Comment 2). |
| Contribution 4 | No longer says the package is released. |
| CRediT, front page | Left as marked placeholders for the authors to complete. |
| GitHub PR | Description updated: 9,500 analyses, the 60-page third-round manuscript, the referral results with separately calibrated thresholds, and the statement that no release has yet been published. |

We are grateful for the referee's guidance over three rounds. It has made the paper narrower in its claims and stronger in its evidence.
