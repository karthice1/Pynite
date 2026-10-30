"""Uncertainty-driven adaptive FE data generation (pool-based active learning).

    python experiment_adaptive.py [threads] [seed]   # writes results/adaptive_seed<seed>.json

All candidate frames are drawn from the training design space and labelled
by PyNite in advance; a label is only *revealed* to the learner when the
acquisition strategy requests that FE analysis, so the experiment is
retrospective but the FE budget accounting is exact (the recorded PyNite
wall-clock time of each acquired frame is summed).
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import torch

import features as FT
import gnn
import metrics as MT

N0, BATCH, ROUNDS = 200, 200, 5          # 200 -> 1200 FE analyses
N_ENS, EPOCHS = 3, 100
STRATEGIES = ['random', 'ensemble variance', 'equilibrium residual']
TESTS = ['test_id', 'test_oodG', 'test_oodL']


def frame_scores(graphs, P, nrm):
    """Acquisition scores: normalised ensemble variance and raw equilibrium residual."""
    var, eq = [], []
    for g, p in zip(graphs, P):
        m = np.tile(~g['restr'], len(g['cases']))
        v = (p['u_var'] / nrm.us ** 2)[m].mean() + (p['q_var'] / nrm.qs ** 2).mean()
        var.append(v)
        eq.append(MT.equilibrium_residual(g, p['q']))
    return np.array(var), np.array(eq)


def main(threads=4, seed=0):
    torch.set_num_threads(threads)
    D = pickle.load(open('data/frames.pkl', 'rb'))
    pool = [FT.graph(r) for r in D['pool']]
    t_fe = np.array([r['t_fe'] for r in D['pool']])
    tests = {s: [FT.graph(r) for r in D[s]] for s in TESTS}
    out = {}
    for strat in STRATEGIES:
        rng = np.random.default_rng(seed)
        labelled = list(rng.choice(len(pool), N0, replace=False))
        hist = []
        for rnd in range(ROUNDS + 1):
            t0 = time.time()
            ens = gnn.Ensemble.train([pool[i] for i in labelled], n_models=N_ENS, epochs=EPOCHS,
                                     seed0=100 * rnd + 10_000 * seed)
            t_train = time.time() - t0
            rec = dict(n_fe=len(labelled), fe_time_s=float(t_fe[labelled].sum()), train_s=t_train)
            for s in TESTS:
                P = ens.predict(tests[s])
                rows = [MT.frame_metrics(g, p['u'], gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var'])))
                        for g, p in zip(tests[s], P)]
                sm = MT.summarise(rows)
                rec[s] = {k: sm[k] for k in ('e_dx_med', 'e_dx_p95', 'e_M_med', 'e_M_p95',
                                            'e_drift_med', 'e_Mmax_med', 'e_N_med')}
            hist.append(rec)
            print(strat, rnd, len(labelled), f'{t_train:.0f}s',
                  {s: (round(rec[s]['e_dx_med'], 4), round(rec[s]['e_M_med'], 4)) for s in TESTS}, flush=True)
            if rnd == ROUNDS:
                break
            cand = np.setdiff1d(np.arange(len(pool)), labelled)
            if strat == 'random':
                new = rng.choice(cand, BATCH, replace=False)
            else:
                P = ens.predict([pool[i] for i in cand])
                var, eq = frame_scores([pool[i] for i in cand], P, ens.members[0][1])
                score = var if strat == 'ensemble variance' else eq
                new = cand[np.argsort(-score)[:BATCH]]
            labelled += list(new)
        out[strat] = dict(history=hist, final_labelled=[int(i) for i in labelled])
        json.dump(dict(results=out, N0=N0, BATCH=BATCH, ROUNDS=ROUNDS, N_ENS=N_ENS, EPOCHS=EPOCHS, seed=seed),
                  open(f'results/adaptive_seed{seed}.json', 'w'), indent=1)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
