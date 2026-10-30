# Mechanics-constrained surrogate for frame response (PyNite)

Code behind the manuscript in `manuscript/surrogate/`. Every FE analysis is run with PyNite
(`Pynite.FEModel3D`, first-order linear analysis). Units: kN, m.

| File | Purpose |
|---|---|
| `frames.py` | Parametric planar moment frames, PyNite model building/analysis, independent frame statics (equilibrium matrix) |
| `generate_dataset.py` | Generates 7,200 PyNite analyses (pool, validation, three test sets) and checks every result against statics |
| `features.py` | Dimensionless graph features, exact statics features, load-case superposition scaling |
| `gnn.py` | Graph network, deep ensemble, equilibrium projection |
| `baselines.py` | Ridge (linear/quadratic), random forest, gradient boosting, MLP |
| `metrics.py` | Engineering error metrics |
| `experiment_main.py` | Benchmark, projection variants, ablations, baselines |
| `experiment_adaptive.py` | Uncertainty-driven adaptive FE sampling (`python experiment_adaptive.py <threads> <seed>`) |
| `experiment_timing.py` | Serial timing of PyNite vs. the surrogate on an idle machine |
| `make_figures.py` | Writes all figures and tables of the manuscript and `results/uq.json`, `results/adaptive_summary.json`, `results/dataset_stats.json` |

## Reproduce

```bash
pip install -e .  scipy scikit-learn matplotlib torch   # from the repository root
cd research/surrogate
python generate_dataset.py          # ~1-2 min on 4 cores, writes data/frames.pkl
python experiment_main.py 4         # ~1.5-2 h on a 4-core CPU
python experiment_adaptive.py 4 0   # ~1 h per seed
python experiment_adaptive.py 4 1
python experiment_timing.py         # run alone on an idle machine
python make_figures.py              # figures/tables -> ../../manuscript/surrogate/
```

`data/` and the model/prediction pickles in `results/` are not committed (they are regenerated
deterministically from the seeds); the JSON summaries in `results/` are the numbers reported in
the paper. The timing numbers inside `results/main.json` were measured while other jobs shared
the CPU; the manuscript uses `results/timing.json`, measured serially. The ensemble training
time in `main.json` is `null` because that run was interrupted after training and resumed from
the saved ensemble; `timing.json` holds a clean single-network training time.
