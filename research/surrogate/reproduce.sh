#!/usr/bin/env bash
# Reproduction of every number, table and figure in the manuscript.
#   bash reproduce.sh            # FULL REPRODUCTION (the definitive route): regenerates all data,
#                                # trains every network and reruns every experiment
#                                # (about 11-13 h on a 4-core CPU, no GPU needed)
#   bash reproduce.sh --quick    # REPORT REGENERATION ONLY, not an independent reproduction:
#                                # regenerates the data, rebuilds ensemble 0 from weights/ and
#                                # recomputes the analyses that need only that ensemble; the
#                                # multi-run statistics, graph baselines, adaptive curves and the
#                                # five-run referral calibration are read from the committed JSON
#                                # summaries in results/
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -e ../.. -r requirements.txt
export OMP_NUM_THREADS=1
python generate_dataset.py                # 7,200 PyNite analyses -> data/frames.pkl
python experiment_validation.py           # DSM + OpenSeesPy cross-check, closed form, data/realistic.pkl (DES)
if [[ "${1:-}" != "--quick" ]]; then
  python experiment_main.py 4             # ensemble 0, tabular baselines, ablations (seed 0)
  for s in 0 1; do python experiment_adaptive.py 4 $s; done
  python experiment_revision.py 4         # ensembles 1-4, graph baselines, feature and depth ablations,
                                          # adaptive seeds 2-4 + diversity rule
  python export_weights.py                # weights/ from the trained models
  python analysis_revision.py             # results/revision/*.json
  python experiment_calibration.py        # 2 x 1,000 calibration frames, two-stage referral, 5 runs
else
  python export_weights.py --load         # rebuild results/ensemble.pkl from weights/
  python analysis_revision.py variants sensitivity pcg baselines   # runs/graphs/calibration JSON are committed
fi
python export_parameters.py               # release/frame_parameters.json.gz
python verify_from_parameters.py          # cross-solver check from the released parameters
if [[ "${1:-}" != "--quick" ]]; then
  python experiment_timing.py --skip-train  # run on an otherwise idle machine -> results/timing.json
else
  python experiment_timing.py --skip-train --out results/timing_rerun.json   # keeps the paper's timing
fi
python make_figures_rev.py                # figures and tables -> ../../manuscript/surrogate/
