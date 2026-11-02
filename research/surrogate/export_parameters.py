"""Write the input parameters of every frame of the study to a portable file.

    python export_parameters.py    # -> release/frame_parameters.json.gz

The file lets the cross-solver verification (experiment_validation.py, solvers.py) and
any other solver be run on exactly the frames of the paper without regenerating the
dataset. Each entry holds n_st, n_bay, bays, heights, E, Ic, Ib, Ac, Ab, fixed_base,
w (floor x bay line loads, kN/m) and F (floor lateral forces, kN). Units: kN, m.
"""

import gzip
import json
import os
import pickle

import numpy as np

KEYS = ('n_st', 'n_bay', 'bays', 'heights', 'E', 'Ic', 'Ib', 'Ac', 'Ab', 'fixed_base', 'w', 'F')


def _clean(p):
    return {k: (np.asarray(p[k]).tolist() if isinstance(p[k], np.ndarray) else
                (bool(p[k]) if k == 'fixed_base' else p[k])) for k in KEYS}


def main():
    D = pickle.load(open('data/frames.pkl', 'rb'))
    sets = {s: D[s] for s in ('pool', 'val', 'test_id', 'test_oodG', 'test_oodL')}
    sets['test_des'] = pickle.load(open('data/realistic.pkl', 'rb'))
    if os.path.exists('data/calibration.pkl'):
        sets.update(pickle.load(open('data/calibration.pkl', 'rb')))
    out = {s: [_clean(r['p']) for r in recs] for s, recs in sets.items()}
    # reproduce.sh --quick does not regenerate the two calibration sets: keep the ones in the
    # existing file instead of dropping them
    if 'cal_domain' not in out and os.path.exists('release/frame_parameters.json.gz'):
        with gzip.open('release/frame_parameters.json.gz', 'rt') as fh:
            old = json.load(fh)
        out.update({s: old[s] for s in ('cal_domain', 'cal_conf') if s in old})
    os.makedirs('release', exist_ok=True)
    with gzip.open('release/frame_parameters.json.gz', 'wt') as fh:
        json.dump(out, fh)
    print({s: len(v) for s, v in out.items()})


if __name__ == '__main__':
    main()
