"""Export / import the weights of the reported surrogate (ensemble 0).

    python export_weights.py          # results/ensemble.pkl -> weights/ensemble0_m<k>.pt
    python export_weights.py --load   # weights/ -> results/ensemble.pkl (no training needed)

Each file holds the state dict, the normaliser statistics and the architecture
arguments of one network; the five networks share one normaliser.
"""

import os
import pickle
import sys

import numpy as np
import torch

import gnn

ARCH = dict(nx=25, ne=10, hidden=96, steps=8, n_out=6)


def export():
    ens, _ = pickle.load(open('results/ensemble.pkl', 'rb'))
    os.makedirs('weights', exist_ok=True)
    for k, (m, nrm) in enumerate(ens.members):
        torch.save(dict(state_dict=m.state_dict(), nrm={a: np.asarray(v) for a, v in nrm.__dict__.items()},
                        arch=ARCH), f'weights/ensemble0_m{k}.pt')


def load():
    members = []
    for k in range(5):
        d = torch.load(f'weights/ensemble0_m{k}.pt', weights_only=False)
        m = gnn.FrameGNN(**d['arch'])
        m.load_state_dict(d['state_dict'])
        nrm = gnn.Normaliser.__new__(gnn.Normaliser)
        nrm.__dict__.update(d['nrm'])
        members.append((m, nrm))
    os.makedirs('results', exist_ok=True)
    pickle.dump((gnn.Ensemble(members), None), open('results/ensemble.pkl', 'wb'))


if __name__ == '__main__':
    load() if '--load' in sys.argv else export()
