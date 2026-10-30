# Equilibrium-projected GNN surrogate for frame analysis (PyNite)

Code behind the manuscript in `manuscript/surrogate/`. Every reference FE analysis is run with
PyNite (`Pynite.FEModel3D`, first-order linear analysis) and cross-checked against OpenSeesPy and
an independent direct-stiffness solver. Units: kN, m.

| File | Purpose |
|---|---|
| `frames.py` | Parametric planar moment frames (random and design-driven), PyNite model building/analysis, independent frame statics (equilibrium matrix) |
| `generate_dataset.py` | Generates 7,200 PyNite analyses (pool, validation, three test sets) and checks every result against statics |
| `experiment_validation.py` | Cross-solver verification (own DSM and OpenSeesPy) of all test frames, closed-form benchmarks, design-driven (DES) set |
| `solvers.py` | Direct-stiffness and OpenSeesPy solvers, member forces from displacements |
| `features.py` | Dimensionless graph features (and raw physical features for the conventional GNN), load-case superposition scaling |
| `gnn.py` | Proposed graph network (optionally with global pooling), StructGNN-style network, deep ensemble, equilibrium projection, soft-penalty / stiffness-residual losses |
| `baselines.py` | Ridge (linear/quadratic), random forest, gradient boosting, MLP |
| `metrics.py` | Error metrics (definitions in the module docstring), equilibrium and compatibility residuals |
| `experiment_main.py` | Ensemble 0, tabular baselines, projection variants, ablations |
| `experiment_adaptive.py` | Adaptive FE sampling: random, ensemble variance, equilibrium residual, residual + diversity (`python experiment_adaptive.py <threads> <seed>`) |
| `experiment_revision.py` | Ensembles 1-4, graph baselines, depth ablation, adaptive seeds 2-4 and diversity rule |
| `analysis_revision.py` | Projection variants, compatibility, multi-run statistics, Σ sensitivity, OOD domain check + conformal referral, CG warm start -> `results/revision/*.json` |
| `experiment_timing.py` | Timing of PyNite, DSM, OpenSeesPy and the surrogate on an idle machine (5 repeats) |
| `export_weights.py` | Writes / loads the pretrained ensemble checkpoints in `weights/` |
| `make_figures_rev.py` | Writes all figures and tables of the manuscript |
| `reproduce.sh` | One-command reproduction (see below) |
| `make_figures.py` | Figures of the first submission (superseded, kept for reference) |

## Reproduce

```bash
cd research/surrogate
bash reproduce.sh            # everything from scratch, about 10-12 h on a 4-core CPU, no GPU
bash reproduce.sh --quick    # no training: rebuilds the reported ensemble from weights/ and
                             # recomputes every table and figure; multi-run statistics, graph
                             # baselines and adaptive curves come from the committed JSON
```

Dependencies are pinned in `requirements.txt` (Python 3.13). `data/`, `results/*.pkl` and
`results/rev/*.pkl` are not committed: they are regenerated deterministically from the seeds. The
JSON files in `results/`, `results/revision/` and `results/adaptive/` hold the numbers reported in
the paper. `results/timing.json` is the timing used in the manuscript; `results/timing_v1.json` is
the measurement of the first submission.
