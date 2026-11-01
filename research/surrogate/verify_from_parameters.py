"""Cross-check PyNite against the direct-stiffness solver and OpenSeesPy from the released parameters.

    python verify_from_parameters.py [set] [n]     # e.g. test_des 50; default: every frame of every test set

Reads release/frame_parameters.json.gz (no dataset generation needed) and prints, per set,
the largest relative L2 difference of displacements and member basic forces between
each solver and PyNite over both load cases.
"""

import gzip
import json
import sys

import numpy as np

import frames as F
import solvers as S


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def main():
    P = json.load(gzip.open('release/frame_parameters.json.gz', 'rt'))
    sets = [sys.argv[1]] if len(sys.argv) > 1 else ['test_id', 'test_oodG', 'test_oodL', 'test_des']
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for s in sets:
        worst = dict(dsm_u=0.0, dsm_q=0.0, ops_u=0.0, ops_q=0.0)
        for p in P[s][:n]:
            p = {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in p.items()}
            r = F.run_fe(p)
            for case in ('W', 'G'):
                for tag, fn in (('dsm', S.dsm_solve), ('ops', S.opensees_solve)):
                    o = fn(p, case)
                    worst[f'{tag}_u'] = max(worst[f'{tag}_u'], rel(o['u'], r['u_' + case]))
                    worst[f'{tag}_q'] = max(worst[f'{tag}_q'], rel(o['q'], r['q_' + case]))
        print(s, len(P[s][:n]), 'frames, largest relative difference to PyNite:', {k: f'{v:.1e}' for k, v in worst.items()})


if __name__ == '__main__':
    main()
