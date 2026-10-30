#!/usr/bin/env bash
# One-command reproduction of every number, table and figure in the manuscript.
#   bash reproduce.sh            # full run (about 10-12 h on a 4-core CPU, no GPU needed)
#   bash reproduce.sh --quick    # train no network: rebuild the reported ensemble from weights/,
#                                # recompute every analysis that needs only that ensemble, and take
#                                # the multi-run statistics, graph baselines and adaptive curves from
#                                # the committed JSON summaries in results/
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -e ../.. -r requirements.txt
export OMP_NUM_THREADS=1
if [[ "${1:-}" != "--quick" ]]; then
  python generate_dataset.py              # 7,200 PyNite analyses -> data/frames.pkl
  python experiment_validation.py         # DSM + OpenSeesPy cross-check, closed form, data/realistic.pkl
  python experiment_main.py 4             # ensemble 0, tabular baselines, ablations (seed 0)
  for s in 0 1; do python experiment_adaptive.py 4 $s; done
  python experiment_revision.py 4         # ensembles 1-4, graph baselines, depth ablation, adaptive seeds 2-4 + diversity rule
  python export_weights.py                # weights/ from the trained models
else
  python generate_dataset.py
  python experiment_validation.py
  python export_weights.py --load         # rebuild results/ensemble.pkl from weights/
fi
if [[ "${1:-}" != "--quick" ]]; then
  python analysis_revision.py           # results/revision/*.json
else
  python analysis_revision.py variants sensitivity ood pcg baselines   # runs/graphs JSON are committed
fi
python experiment_timing.py --skip-train  # run on an otherwise idle machine
python make_figures_rev.py              # figures and tables -> ../../manuscript/surrogate/
