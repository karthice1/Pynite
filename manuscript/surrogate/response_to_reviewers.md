---
title: "Response to the referee"
subtitle: "Revised manuscript: Equilibrium-projected graph neural network surrogates for linear frame analysis: uncertainty-aware FE referral and residual-guided adaptive learning"
date: "30 October 2026"
geometry: margin=2.3cm
fontsize: 11pt
---

We thank the referee for a careful, constructive and very specific report. We agree with the main assessment: the earlier version showed that the forces were statically admissible, but it did not show how far the predicted response is mechanically credible, how the method compares with recent structural graph networks, or how stable the results are. The revision addresses every point. All new results come from experiments run for this revision. Nothing is extrapolated or taken from other papers. Where a result is unfavourable to the proposed method, we report it and have toned down the related claims.

Section, table and figure numbers refer to the revised manuscript. The previous title has been replaced by the title suggested by the referee.

# Summary of the main changes

1. **Novelty and positioning.** There is a new Section 1.1 on recent mechanics-informed graph learning for structural analysis. It discusses Parisi et al. (2024), Chou et al. (2024, StructGNN), Leng et al. (2026) and Kuo et al. (2025). Table 1 compares these studies on the eight attributes the referee proposed. We checked each of these references (journal, volume, article number, DOI) against the publisher's page. The contributions are now stated as the combination named by the referee, not as a new architecture.
2. **Independent FE validation (Section 2.3, Table 3).** All 2,100 test frames were re-analysed with OpenSeesPy and with a separate direct-stiffness solver. The quantities compared are horizontal and vertical displacements, rotations, reactions, column and beam end moments, shears and axial forces, and all differences are below 1.3e-11. Three closed-form benchmarks were also added.
3. **Compatibility (Sections 3.6 and 6.6, Table 12).** A compatibility residual is defined and reported for the four force fields the referee asked for, on all test sets.
4. **Graph baselines and depth ablation (Sections 5 and 6.4, Tables 9 and 10, Fig. 7).** The new baselines are:
   - a StructGNN-style network;
   - a conventional GNN;
   - a soft equilibrium penalty;
   - global pooling;
   - a displacement-only network with force recovery;
   - a Leng-style network with a stiffness-residual loss.

   There is also an ablation of message-passing depth at 4, 8, 12 and 16 steps. Every model was trained three times.
5. **Statistics (Section 6.3, Table 8).** The full ensemble pipeline was trained five times from scratch. Results are given as mean ± std with t-intervals, plus bootstrap intervals for the median and 95th percentile. The claim about the projection's effect on moment accuracy has been weakened.
6. **Two-stage OOD rule (Section 3.8, Tables 15 and 16, Fig. 10).** Stage 1 is a domain check on geometry and load-pattern descriptors. Stage 2 is a split-conformal threshold on the ensemble spread. Both are calibrated on the validation set and evaluated on untouched test sets.
7. **Expanded metrics (Section 5, Table 7).** We now report the 11 response errors and two governing-member agreement rates the referee asked for, and all metrics are defined by formula.
8. **Realistic benchmark.** A new design-driven set (DES) contains 300 steel frames with correlated parameters (Section 2).
9. **Adaptive learning (Section 6.9, Table 17, Fig. 11).** The study now uses five seeds and adds a fourth, diversity-aware rule.
10. **Reproducibility.** The release now includes:
    - pinned requirements;
    - a one-command `reproduce.sh` script;
    - pretrained weights;
    - machine-readable JSON results for every table.
11. **Mathematical and editorial corrections.** These cover:
    - E scaling and the definitions of the scaled variables, with a dimensional table (Table 4);
    - the classification of the tributary gravity feature as approximate;
    - regularisation and conditioning of the projection, with a sensitivity table (Table 13);
    - a sign-convention figure (Fig. 1);
    - a feature dictionary and the sampling algorithm (Appendices A and B);
    - the full computing environment;
    - enlarged figures and a split Table 2.

# Point-by-point responses

## Critical issue 1: novelty against recent literature

> *Add Section 1.1 discussing Parisi et al. (2024), Chou et al. (2024) and Leng et al. (2026), with a comparative table.*

**Done.** Section 1.1 covers four studies:

- Parisi et al., Structures 59 (2024) 105712, doi:10.1016/j.istruc.2023.105712. This work applies mechanics-informed GNNs to 2D and 3D trusses.
- Chou et al., StructGNN, Computers & Structures 299 (2024) 107385, doi:10.1016/j.compstruc.2024.107385. This work uses storey pseudo nodes and extrapolates to taller frames.
- Leng et al., Buildings 16(13) (2026) 2530, doi:10.3390/buildings16132530. This work uses a mechanically constrained GNN for planar frames with member loads and recovers member forces from displacements.
- Kuo et al., EG-ICE 2025 (an additional related study we found).

Table 1 compares these with earlier physics-informed surrogates. The columns are those the referee proposed: structural system, architecture, predicted quantities, equilibrium enforcement, compatibility enforcement, uncertainty quantification, adaptive sampling and OOD testing.

The section then states what remains open. None of these studies does the following:

- enforces nodal equilibrium of the predicted forces exactly at inference, independently of the training loss;
- uses the predictive uncertainty to decide which outputs the correction may change;
- tells the user when a prediction should be referred back to the FE solver;
- selects FE analyses for training on the basis of a mechanics residual.

The contribution list ("Objectives and contributions") now consists of exactly the four components identified by the referee. It states explicitly that the GNN architecture and mechanics-informed graph modelling are not claimed as new. The experiments support this framing: a StructGNN-style network reaches similar accuracy to ours (see Comment 6).

## Critical issue 2: independent FE validation

> *Cross-validate PyNite against OpenSeesPy, SAP2000 or another independent solver; include classical analytical benchmarks.*

**Done, with OpenSeesPy and a second independent solver instead of SAP2000.** SAP2000 is commercial, Windows-only software and we could not use it in our computing environment. Instead, we used two solvers that share no code with PyNite:

- OpenSeesPy 3.8 with `elasticBeamColumn` elements, `beamUniform` loads and UMFPACK.
- A direct-stiffness planar frame solver written for this study, with consistent fixed-end forces and a sparse direct solve.

We re-analysed all 2,100 test frames (ID, OOD-G, OOD-L and DES), not just a subset. Together these cover:

- fixed and pinned bases;
- 1 to 8 storeys and 1 to 6 bays;
- stiffness ratios across the whole sampling range;
- uniform and pattern gravity loading;
- four lateral profiles.

Table 3 reports, per quantity, the largest relative L2 difference over every frame of each set. The quantities are:

- $u_x$ and $u_y$;
- $\theta_z$;
- reactions;
- column and beam end moments;
- shears;
- axial forces.

The largest difference is 1.3e-11.

Three closed-form benchmarks were also run in PyNite:

- a portal frame with a rigid beam and fixed bases ($Ph^3/24EI$, $Ph/4$);
- the same portal frame with pinned bases ($Ph^3/6EI$, $Ph/2$);
- a beam with fixed ends ($wL^2/12$).

PyNite agrees with these to within 9e-6. The remaining difference comes from the large but finite stiffness used to model the rigid beam.

> *Realistic structural configurations.*

**Done.** We added a design-driven test set (DES) of 300 steel moment frames (Section 2). Its main features are:

- E = 200 GPa;
- standard bays of 6.0, 7.5 and 9.0 m;
- a taller first storey;
- gravity loads from 1.2D + 1.0L area loads;
- wind from facade pressures;
- beams sized for L/360 deflection and moment demand;
- columns sized for an h/400 drift, with one section per two storeys;
- 80% fixed bases.

The surrogate was not retrained on this set. On DES its median moment error is 2.0%, axial error 0.3% and horizontal displacement error 4.4%, all at least as good as on the ID test set. The domain check flags 20% of the DES frames, and none of the frames it accepts exceed the 10% tolerance (Table 16).

## Comment 1: compatibility

> *Introduce a compatibility residual, evaluate it on all test sets for raw, Euclidean-projected, variance-weighted and displacement-recovered forces.*

**Done (Sections 3.6 and 6.6, Table 12).** For the predicted displacements $\hat u$ we compute member basic forces with the element stiffness relations, including fixed-end forces: $q_\mathrm{c}(\hat u)$. The sign convention is shown in Fig. 1. The residual is

$$r_\mathrm{compat}=\lVert S\,(q-q_\mathrm{c}(\hat u))\rVert_2\,/\,\lVert S\,q_\mathrm{c}(\hat u)\rVert_2,$$

where $S$ scales the end moments by $1/H$ so that $N$ and $M$ are commensurate. We also report a version restricted to the moments. Median values (%) are:

| Force field | ID | OOD-G | OOD-L | DES |
|---|---|---|---|---|
| Raw ensemble | 47.2 | 87.5 | 59.0 | 15.1 |
| Euclidean projection | 47.0 | 87.6 | 59.0 | 14.9 |
| Variance-weighted projection | 47.0 | 87.6 | 59.0 | 14.9 |
| Recovered from $\hat u$ | 0 | 0 | 0 | 0 |

Moment-only medians are 4 to 40% for the network forces. The projection slightly increases the moment-only incompatibility, by 0.1 to 1.1 percentage points.

The full residual is dominated by the axial terms. Columns are axially very stiff, so a small error in the predicted vertical displacements produces a large error in the axial forces derived from them.

Forces recovered from the predicted displacements are compatible by construction, but they are far less accurate. Their median moment error is 11.2% on ID (against 2.4%) and their axial error is 53%. They are also not in equilibrium: the out-of-balance force is 1.7 times the reference scale on ID.

We also tried using the predicted displacements to warm-start a conjugate-gradient solve of the full stiffness system. It gave no saving in iterations (Section 6.6).

Accordingly, we no longer describe the result as "mechanics-constrained" or "physically consistent". The title, abstract, method and conclusions now say "statically admissible" or "in static equilibrium" throughout. The discussion states that enforcing both conditions exactly amounts to solving the FE problem, and names a mixed (flexibility-based) formulation as future work.

## Comment 2: Young's modulus statement

**Corrected (Section 3.2).** The paragraph now distinguishes the two kinds of response. Under a uniform scaling of $E$, displacements and rotations scale as $1/E$, while member forces are invariant ($E^0$), because the relative stiffness distribution is unchanged. Table 4 lists the scaling of every output.

## Comment 3: reference rigidity and dimensionless variables

**Done (Section 3.2, Table 4).** The scaled variables are now defined as follows:

- $H$ is the total frame height.
- $I_\mathrm{ref}$ is the mean column second moment of area over all storeys, which allows for the taper.
- $\lambda_\mathrm{W}=\sum_k F_k$ is the signed resultant of the lateral forces.
- $\lambda_\mathrm{G}=\sum_e |w_e|L_e$ is the total gravity load.

Table 4 lists every dimensionless group with its dimensions and its scaling in $E$ and in the loads. The handling of a vanishing scale is stated explicitly. In the generator $|\lambda_\mathrm{W}|\ge5$ kN and $\lambda_\mathrm{G}>0$, so neither scale can vanish. In deployment a load case with $\lambda=0$ is simply omitted from the superposition. A lateral pattern whose floor forces cancel ($\sum F\approx0$) would make $\lambda_\mathrm{W}$ ill-conditioned; it should then be scaled by $\sum|F|$, and the domain check flags such a pattern as unseen.

## Comment 4: tributary gravity feature

**Corrected (Section 3.3 and Appendix A).** The tributary column-line gravity load is now called a classical approximation, exact only for simply supported beams. Storey shear and overturning moment are classed as exact. The feature dictionary (Table A.19) marks each feature as exact, approximate, or geometric/topological.

## Comment 5: numerical stability of the projection

**Done (Section 3.5, Eq. 5, Section 6.7, Table 13).** The projection details are now stated in full:

- **Metric.** The regularised metric is $\Sigma_\mathrm{reg}=\mathrm{diag}(\sigma_i^2+\epsilon\bar\sigma^2)$ with $\epsilon=10^{-6}$. It is relative to the mean variance, so it is unit-free and stays positive definite even when all members agree exactly.
- **Solver.** The system with $C\Sigma C^T$ is solved by dense LU with partial pivoting (LAPACK `gesv`), so there is no iterative tolerance. The resulting relative equilibrium residual is below $10^{-12}$ for every frame.

Table 13 compares floors from $10^{-8}$ to 1, in max form and additive form, with the Euclidean metric. It reports the median moment error and the median condition number of $C\Sigma C^T$ on every test set. The results are unchanged for $\epsilon$ from $10^{-8}$ to $10^{-4}$. The largest median condition number is 4e5, far from ill-conditioning in double precision. Larger floors drift towards the Euclidean result.

The text now states that a diagonal deep-ensemble variance is a heuristic uncertainty model, not a calibrated posterior covariance. The Bayesian reading is offered only as motivation.

## Comment 6: graph-based baselines

**Done (Sections 5 and 6.4, Tables 9 and 10).** All models below use the same data, the same splits and the same error definitions. Each was trained three times, and Tables 9 and 10 report mean ± std. The tables cover displacement, moment, axial and equilibrium errors on all four test sets, plus training time.

- **StructGNN-style** (after Chou et al.): storey pseudo-nodes, shared weights, a number of steps tied to the number of storeys, and an MAE loss. We gave it our inputs and targets, so that the comparison isolates the architecture. On ID it reaches $u_x$ 6.3% and M 4.2%, against our single network's 6.7% and 3.7%. On OOD-G its $u_x$ is 12.3% against 14.1%. It is therefore competitive, better for displacements on taller frames and slightly worse for moments, at about seven times our training cost. We state this plainly, and it is why the paper no longer claims a better architecture.
- **Conventional GNN without mechanics** (physical inputs and targets, same architecture): ID $u_x$ 14.3%, about twice our error.
- **Soft equilibrium penalty:** accuracy similar to ours, but the penalty barely reduces the out-of-balance force (median $6.2\times10^{-2}$ on ID against $6.7\times10^{-2}$ without it). The hard projection reaches $10^{-12}$.
- **Global pooling:** a small displacement gain on OOD-G (11.1% against 14.1%).
- **Displacement-only GNN with force recovery:** moment error 20% on ID.
- **Leng-style network** (displacement output, force recovery, stiffness-residual loss): residual weights $10^{-2}$ and $10^{-4}$ performed worse than without the loss.

We say explicitly that the StructGNN-style and Leng-style models are re-implementations from the published descriptions and not the authors' codes. The adaptations are listed in Section 5. The poor result of our Leng-style variant may reflect our scaling of the residual term rather than the original method.

## Comment 7: message-passing depth

**Done (Table 9, Fig. 7).** We trained depths 4, 8, 12 and 16 (the depth-8 case is our main network). Median $u_x$ errors were:

| Depth | ID $u_x$ (%) | OOD-G $u_x$ (%) |
|---|---|---|
| 4 | 9.1 | 20.1 |
| 8 | 6.7 | 14.1 |
| 12 | 6.3 | 13.1 |
| 16 | 5.8 | 11.8 |

Moment errors hardly change beyond depth 8. The OOD-G graphs have diameters up to 14, so part of the OOD-G degradation is due to limited communication, and part remains at depth 16. Depth 16, global pooling and the StructGNN-style network, which all give long-range communication, recover a similar share. We conclude that both factors contribute and we say so in Section 6.4. The main model is unchanged so that the other results stay comparable. Table 9 shows the deeper variants for readers who prioritise extrapolation.

## Comment 8: more response quantities

**Done (Section 5 metric definitions, Table 7).** For the ensemble, Table 7 reports medians and 95th percentiles (mean ± std over five runs) for the following errors:

- horizontal and vertical displacement;
- rotation;
- maximum inter-storey drift;
- maximum beam deflection;
- end moments, axial forces and shears;
- support reactions;
- largest moment and largest axial force.

It also reports how often the governing member (largest |M|) is identified correctly as the top-1 and within the top-3.

On ID the governing member is top-1 in 94% of frames and in the top 3 in 99.9%. On the shifted sets these are 86% and 98 to 99.5%.

The weak points are now visible too. The drift error has a heavy tail on OOD-G (p95 97%) because drifts are differences of small displacements, and the rotation error is 10% on OOD-L. Both are discussed.

## Critical issue 3 and Comment 9: statistics

**Done (Section 6.3, Table 8).** The whole pipeline was repeated five times, each with a different seed for initialisation and data order:

- training a five-member ensemble;
- projection;
- evaluation.

Table 8 reports:

- median moment errors with bootstrap 95% intervals (2,000 resamples of frames);
- 95th percentiles with bootstrap intervals;
- the across-run mean ± std;
- the raw-minus-projected difference, with both a bootstrap interval and a t-interval across runs.

Five runs fit within the computing budget available for this revision, so ten were not run.

On ID the projection lowers the median moment error by 0.10 percentage points. The across-run interval is [0.05, 0.14], and the bootstrap interval over frames for a single run is [-0.01, 0.20]. We now describe this as negligible. The text no longer presents 2.5 $\rightarrow$ 2.4% as an accuracy improvement. The projection is presented as an admissibility device that does not harm accuracy, provided the variance-weighted metric is used. The Euclidean metric raises the error to 4.0%.

The clearer accuracy effects of the projection are:

- axial forces and reactions, whose errors fall by factors of 2 to 5;
- moments on OOD-G, where the reduction is 0.53 pp [0.33, 0.72].

## Comment 10: OOD uncertainty and two-stage referral

**Done (Section 3.8, Section 6.8, Tables 14 to 16, Fig. 10).**

*Stage 1, domain check.* We compute a Mahalanobis distance, with Ledoit–Wolf covariance, on 15 descriptors known before any analysis:

- ten geometry and stiffness descriptors;
- five load-pattern descriptors (shape of the lateral profile, and the spread of gravity load between bays).

The threshold is the 99th percentile on the validation set. The check flags 1.7% of ID, 79% of OOD-G, 75% of OOD-L and 20% of DES frames. The load-pattern part alone flags 67% of OOD-L and 9% of OOD-G. This is the mechanism that the ensemble spread lacked.

*Stage 2, conformal threshold.* For frames that pass Stage 1, a split-conformal quantile of the ratio of moment error to ensemble spread is calibrated on the 400 validation frames ($\alpha=0.1$, $\hat q=1.41$). A frame is accepted if $\hat q\,s$ is within a tolerance $\tau$.

Thresholds are fixed on the validation set, and the test sets are used only for evaluation. The resulting rates are:

| Tolerance | Set | Referred | Accepted frames above $\tau$ |
|---|---|---|---|
| $\tau$ = 10% | ID | 2.2% | 1.4% |
| $\tau$ = 10% | OOD-G | 86% | 1.2% |
| $\tau$ = 10% | OOD-L | 75% | 1.4% |
| $\tau$ = 10% | DES | 20% | 0% |
| $\tau$ = 5% | ID | 28% | 3.5% |

The test-set risk–coverage curves (Fig. 10) are now shown only as diagnostics, with the two calibrated operating points marked on them. We state that the conformal guarantee assumes exchangeability with the calibration frames. It holds for ID, and only approximately for the in-domain part of shifted data.

## Comment 11: adaptive acquisition

**Done (Sections 4 and 6.9, Table 17, Fig. 11; abstract).** The abstract, workflow and conclusions now say explicitly that two different indicators are used:

- ensemble spread, for deployment-time referral;
- the equilibrium residual of the raw prediction, a label-free mechanics indicator and not an uncertainty estimate, for training-time acquisition.

The study now uses five seeds and four rules: random, ensemble variance, equilibrium residual, and equilibrium residual with diversity. The diversity rule takes the top 3B candidates by residual, groups them into B k-means clusters of standardised frame descriptors, and picks the highest-residual frame in each cluster.

After 1,200 analyses, the median OOD-G moment error (mean ± std over five seeds) is:

| Rule | OOD-G moment error (%) |
|---|---|
| Random | 14.8 ± 0.2 |
| Ensemble variance | 19.0 ± 1.1 |
| Equilibrium residual | 12.1 ± 0.7 |
| Residual + diversity | 12.1 ± 0.2 |

Residual acquisition reaches random sampling's final OOD-G accuracy with 600 analyses, in every seed. On ID and OOD-L the gains are smaller: about 0.3 and 1.0 percentage points. Diversity did not improve the median, but it reduced the spread between seeds.

We also explain why variance acquisition fails. It favours small frames (2.6 to 2.7 storeys, half with pinned bases), whose dimensionless outputs vary most between ensemble members but which teach little about larger frames. Residual acquisition favours taller, more flexible frames, which are closest to the unseen geometries.

## Section 7 of the report: computational efficiency

**Done (Sections 5 and 6.10, Table 18).**

*Environment.* The full environment is now listed:

- four cores of an Intel Xeon at 2.1 GHz, 15 GB RAM, no GPU;
- Linux, Python 3.13;
- PyTorch 2.15, NumPy 2.5, SciPy 1.18, scikit-learn 1.9, OpenSeesPy 3.8 and PyNite 3.2.0, with versions pinned in `requirements.txt`;
- timing runs on one thread with batch sizes stated;
- five repetitions, reported as mean ± std.

*Optimised solvers.* As the referee asked, we also timed two faster pipelines:

- OpenSeesPy, at 1.2 ms (ID);
- our vectorised sparse direct-stiffness solver, at 1.5 ms (ID).

PyNite takes 33.6 ms. The batched surrogate (2.7 ms per frame) is 12 to 14 times faster than PyNite, but only 0.34 to 0.65 times as fast as the sparse direct-stiffness solver and 0.40 to 0.50 times as fast as OpenSeesPy. In other words, for frames of this size an optimised linear solver is faster than the surrogate.

The cost section, discussion and conclusions now say this plainly. The value of the framework lies in cases where FE is expensive or unoptimised, in its differentiable, batched predictions, and in the admissibility and referral layer, not in raw speed for small linear frames. The break-even against PyNite (about 8e4 ID frames without referral, 1.1e5 with 20% referral) is still reported; against an optimised solver there is none.

## Critical issue 4: reproducibility

**Partly done. Two steps need the authors' accounts.** The `research/surrogate` folder now contains:

- the data generator with all seeds;
- the PyNite version pin;
- the projection, uncertainty, OOD and acquisition modules;
- all five tabular and all graph baselines;
- training configurations;
- JSON results for every table and figure;
- the plotting scripts;
- pinned `requirements.txt`;
- pretrained checkpoints for one ensemble (`weights/`), with a loader in `export_weights.py` whose round trip has been verified;
- a one-command script, `reproduce.sh`.

`./reproduce.sh` regenerates the data and all results and figures (about 10 to 12 h on a 4-core CPU). `./reproduce.sh --quick` regenerates the dataset, rebuilds the reported ensemble from the bundled weights without training any network, and recomputes every table and figure that depends on it (including the main accuracy tables and the triage figure); the multi-run statistics, graph baselines and adaptive curves are taken from the committed JSON summaries.

The referee could not see this folder because it was on a development branch, not the repository's default branch. The material becomes visible on the default branch once that branch is merged. A Zenodo DOI must be created from the authors' own Zenodo account, so the data availability statement contains a marked placeholder for the DOI and the pinned release tag. These will be filled in before submission.

## Editorial table of the report

| Item | Action |
|---|---|
| Authors and affiliations, CRediT | Left as marked placeholders for the authors to complete. |
| Abstract on active learning | Rewritten; it now separates spread (referral) from residual (acquisition) and reports the equilibrium residual separately from ensemble spread. |
| Outdated literature | Section 1.1 and Table 1 added, with eleven new references. |
| Sampling algorithm | Appendix B gives a step-by-step procedure with distributions and probabilities. |
| Sign conventions | New Fig. 1 (local axes, end forces, positive directions). |
| E scaling | Corrected (Comment 2). |
| Tributary gravity feature | Reclassified as approximate (Comment 4). |
| Feature dictionary | Appendix A, Table A.19. |
| Projection implementation | Regularisation, solver and tolerance added; sensitivity in Table 13. |
| Uncertainty calibration | Conformal calibration on a separate set (Section 3.8). |
| Metric formulas | Section 5 gives the exact expression for every metric. |
| Ablations fully tabulated | Tables 9 and 10 give all metrics discussed. |
| OOD failure | Domain detection added (Comment 10). |
| Hardware | Full environment given (Section 7 of the report). |
| Compatibility | Quantified (Comment 1). |
| Data availability | Release contents listed; DOI placeholder (Critical issue 4). |
| Figures 3, 7, 8, 9 | Enlarged to full text width as multi-panel 2-by-2 or 2-by-4 layouts. |
| Table 2 too dense | Split into Tables 5 (ID, DES) and 6 (OOD-G, OOD-L); the extended metrics are in Table 7. |
| AI disclosure | Retained. The authors must still independently check the code, numbers and interpretations before submission. |
| Terminology | "Static equilibrium" or "statically admissible" is used throughout; "mechanically consistent" has been removed. |

# Limitations stated in the revision

So that the referee can see where we stopped, these are the main limitations stated in the revised manuscript:

- compatibility is quantified but not enforced;
- a StructGNN-style architecture is competitive with ours;
- the projection does not change moment accuracy in any meaningful way;
- an optimised sparse FE solver is faster than the surrogate for these frames;
- the domain check refers most shifted frames rather than predicting them well;
- the graph baselines are re-implementations;
- SAP2000 was replaced by OpenSeesPy and an independent solver;
- five independent runs, not ten.
