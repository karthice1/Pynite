#!/bin/sh
# One-command reproduction of every number, figure and table of the paper.
#   ./reproduce.sh            (about 10-12 h on 4 cores; NPROC sets the number of workers)
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
./run_all.sh jobs.txt                     # all optimisation runs and enumerations
python run_experiments.py closed_form
python run_experiments.py verify          # 10^6-sample verification of every optimum
python run_experiments.py opensees
python run_experiments.py fullmc
for k in base_DDO base_RBDO-C S3_RBDO-C S9_RBDO-C B6_RBDO-C B10_RBDO-C W08_RBDO-C W16_RBDO-C \
         S3_RBDO-C_lcb S3_RBDO-C_bu3.21_bs1.5_lcb; do
    echo "validate $k"; echo "validateC $k"
done | xargs -P "${NPROC:-4}" -L 1 python run_experiments.py
python run_experiments.py driftbias base_RBDO-C,S3_RBDO-C,S9_RBDO-C,B6_RBDO-C,B10_RBDO-C,W08_RBDO-C,W16_RBDO-C,base_RBDO-C_lcb,S3_RBDO-C_lcb,B6_RBDO-C_lcb,W16_RBDO-C_lcb
python run_experiments.py modelunc
python run_experiments.py sensitivity
python run_experiments.py convergence
python make_figures.py
cd paper && pdflatex -interaction=batchmode main && bibtex main && pdflatex -interaction=batchmode main && pdflatex -interaction=batchmode main
