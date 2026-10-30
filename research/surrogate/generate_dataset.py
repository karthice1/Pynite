"""Generate the PyNite frame dataset used in the study.

    python generate_dataset.py            # writes data/frames.pkl

Each record holds the sampled parameters, PyNite nodal displacements, member
basic forces and the measured FE wall-clock time. Every record is also checked
for nodal equilibrium using the independent statics module (frames.py).
"""

import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

import frames as F

SPLITS = {          # name: (scenario, count, seed)
    'pool': ('train', 5000, 1),
    'val': ('train', 400, 2),
    'test_id': ('train', 600, 3),
    'test_oodG': ('oodG', 600, 4),
    'test_oodL': ('oodL', 600, 5),
}


def _one(args):
    scenario, seed = args
    rng = np.random.default_rng(seed)
    p = F.sample_frame(rng, scenario)
    r = F.run_fe(p)
    t = F.topology(p)
    scale = max(1.0, np.abs(t['P']).max(), np.abs(t['wg'] * t['L']).max())
    for case in ('W', 'G'):
        res = F.equilibrium_residual(F.case_topology(t, case), r['q_' + case])
        assert np.abs(res).max() < 1e-6 * scale, 'PyNite result fails statics check'
    return {k: r[k] for k in ('u_W', 'q_W', 'u_G', 'q_G', 't_fe')} | dict(p=p)


def main():
    os.makedirs('data', exist_ok=True)
    out = {}
    with Pool(4) as pool:
        for name, (scenario, count, seed) in SPLITS.items():
            t0 = time.time()
            seeds = [(scenario, seed * 1_000_000 + k) for k in range(count)]
            out[name] = pool.map(_one, seeds, chunksize=20)
            print(f'{name:10s} {count:5d} frames  {time.time() - t0:6.1f} s')
    with open('data/frames.pkl', 'wb') as fh:
        pickle.dump(out, fh)


if __name__ == '__main__':
    main()
