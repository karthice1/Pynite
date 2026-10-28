#!/bin/sh
# Run all optimisation jobs of jobs.txt on 4 worker processes.
cd "$(dirname "$0")"
mkdir -p logs results
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
cat jobs.txt | xargs -P "${NPROC:-4}" -L 1 sh -c 'python run_experiments.py "$@" > "logs/$(echo "$@" | tr " ," "__").log" 2>&1' _
echo ALLDONE > logs/done.flag
