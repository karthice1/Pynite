"""Uncertainty-driven adaptive FE data generation (pool-based active learning).

    python experiment_adaptive.py [threads] [seed] [strategy]
        # writes results/adaptive/seed<seed>_<strategy>.json (all strategies if omitted)

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
STRATEGIES = ['random', 'ensemble variance', 'equilibrium residual', 'diverse equilibrium residual']
N_SHORTLIST = 3                           # diversity rule: cluster the top 3 x BATCH candidates
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


def descriptor(g):
    """Frame-level design descriptor: mean and max of the dimensionless node features."""
    return np.concatenate([g['x'].mean(0), g['x'].max(0)])


def diverse_top(cand, score, desc, batch, seed):
    """Shortlist the N_SHORTLIST * batch highest scores, cluster the shortlist
    into `batch` groups (k-means on standardised descriptors) and take the
    highest-scoring frame of each cluster."""
    from sklearn.cluster import KMeans
    short = np.argsort(-score)[:N_SHORTLIST * batch]
    Z = desc[short]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    lab = KMeans(batch, n_init=1, random_state=seed).fit_predict(Z)
    pick = [short[lab == c][np.argmax(score[short][lab == c])] for c in range(batch)]
    return cand[np.array(pick)]


def run(seed, strat, D=None):
    D = D or pickle.load(open('data/frames.pkl', 'rb'))
    pool = [FT.graph(r) for r in D['pool']]
    t_fe = np.array([r['t_fe'] for r in D['pool']])
    tests = {s: [FT.graph(r) for r in D[s]] for s in TESTS}
    desc = np.array([descriptor(g) for g in pool])
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
            rows = [MT.frame_metrics(g, p['u'], gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var'])),
                                     compat=False) for g, p in zip(tests[s], P)]
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
            if strat == 'diverse equilibrium residual':
                new = diverse_top(cand, score, desc[cand], BATCH, seed + rnd)
            else:
                new = cand[np.argsort(-score)[:BATCH]]
        labelled += list(new)
    out = dict(history=hist, final_labelled=[int(i) for i in labelled])
    os.makedirs('results/adaptive', exist_ok=True)
    json.dump(dict(results={strat: out}, N0=N0, BATCH=BATCH, ROUNDS=ROUNDS, N_ENS=N_ENS, EPOCHS=EPOCHS, seed=seed),
              open(f'results/adaptive/seed{seed}_{strat.replace(" ", "_")}.json', 'w'), indent=1)


if __name__ == '__main__':
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for strat in ([sys.argv[3]] if len(sys.argv) > 3 else STRATEGIES):
        run(seed, strat)
