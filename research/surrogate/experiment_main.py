"""Main benchmark: surrogates vs. baselines, constraint ablations, uncertainty.

    python experiment_main.py      # writes results/main.json, results/main_preds.pkl

Training data: the first N_TRAIN frames of the pool split. Test sets: test_id
(in-distribution), test_oodG (unseen geometries), test_oodL (unseen loading).
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import torch

import baselines as BL
import features as FT
import gnn
import metrics as MT

N_TRAIN = 3000
EPOCHS = 150
N_ENS = 5
TESTS = ['test_id', 'test_oodG', 'test_oodL']


def evaluate(graphs, preds):
    rows = [MT.frame_metrics(g, p['u'], p['q']) for g, p in zip(graphs, preds)]
    return MT.summarise(rows), rows


def main(threads=4):
    torch.set_num_threads(threads)
    os.makedirs('results', exist_ok=True)
    D = pickle.load(open('data/frames.pkl', 'rb'))
    G = {k: [FT.graph(r) for r in v] for k, v in D.items()}
    train = G['pool'][:N_TRAIN]
    results, preds, times = {}, {}, {}

    # --- Mechanics-constrained graph ensemble ---------------------------------
    if os.path.exists('results/ensemble.pkl'):      # reuse a finished training run
        ens, times['train_gnn_ensemble_s'] = pickle.load(open('results/ensemble.pkl', 'rb'))
    else:
        t0 = time.time()
        ens = gnn.Ensemble.train(train, n_models=N_ENS, epochs=EPOCHS)
        times['train_gnn_ensemble_s'] = time.time() - t0
        with open('results/ensemble.pkl', 'wb') as fh:
            pickle.dump((ens, times['train_gnn_ensemble_s']), fh)
    for split in TESTS + ['val']:
        P = ens.predict(G[split])
        single = gnn.predict(*ens.members[0], G[split])
        variants = {
            'GNN (single, raw)': [dict(u=u, q=q) for u, q in zip(*single)],
            'GNN ensemble (raw)': [dict(u=p['u'], q=p['q']) for p in P],
            'GNN ensemble + Euclidean projection': [dict(u=p['u'], q=gnn.project(g, p['q']))
                                                    for g, p in zip(G[split], P)],
            'GNN ensemble + variance-weighted projection':
                [dict(u=p['u'], q=gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var'])))
                 for g, p in zip(G[split], P)],
        }
        for name, pr in variants.items():
            results.setdefault(name, {})[split], _ = evaluate(G[split], pr)
            preds.setdefault(name, {})[split] = pr
        preds.setdefault('_ens_var', {})[split] = [dict(u_var=p['u_var'], q_var=p['q_var']) for p in P]
        print('GNN', split, {k: round(v, 4) for k, v in results['GNN ensemble + variance-weighted projection'][split].items() if k.endswith('_med')}, flush=True)

    # --- Ablations (single models, seed 0, same training set) -----------------
    abl = {
        'Ablation: soft equilibrium penalty': dict(graphs=train, kw=dict(eq_weight=1.0)),
        'Ablation: no load-case superposition': dict(graphs=[FT.graph(r, 'total') for r in D['pool'][:N_TRAIN]], kw={}),
    }
    for name, a in abl.items():
        cache = f"results/{name.split(': ')[1].replace(' ', '_')}.pkl"
        if os.path.exists(cache):
            m, nrm, times['train_' + name] = pickle.load(open(cache, 'rb'))
        else:
            t0 = time.time()
            m, nrm = gnn.train_model(a['graphs'], seed=0, epochs=EPOCHS, **a['kw'])
            times['train_' + name] = time.time() - t0
            pickle.dump((m, nrm, times['train_' + name]), open(cache, 'wb'))
        for split in TESTS:
            gs = G[split] if 'superposition' not in name else [FT.graph(r, 'total') for r in D[split]]
            U, Q = gnn.predict(m, nrm, gs)
            results.setdefault(name, {})[split], _ = evaluate(gs, [dict(u=u, q=q) for u, q in zip(U, Q)])
            results.setdefault(name + ' + Euclidean projection', {})[split], _ = evaluate(
                gs, [dict(u=u, q=gnn.project(g, q)) for g, u, q in zip(gs, U, Q)])
        print(name, {s: round(results[name][s]['e_M_med'], 4) for s in TESTS}, flush=True)

    # --- Baselines ---------------------------------------------------------------
    for name in BL.BASELINES:
        t0 = time.time()
        mdl = BL.TabularSurrogate(name).fit(train)
        times['train_' + name] = time.time() - t0
        for split in TESTS:
            P = mdl.predict(G[split])
            results.setdefault(name, {})[split], _ = evaluate(G[split], P)
            results.setdefault(name + ' + Euclidean projection', {})[split], _ = evaluate(
                G[split], [dict(u=p['u'], q=gnn.project(g, p['q'])) for g, p in zip(G[split], P)])
            preds.setdefault(name, {})[split] = P
        print(name, f'{times["train_" + name]:.0f}s',
              {s: round(results[name][s]['e_M_med'], 4) for s in TESTS}, flush=True)

    # --- Timing ------------------------------------------------------------------
    timing = {}
    for split in TESTS:
        recs = D[split]
        timing[split] = dict(fe_mean_s=float(np.mean([r['t_fe'] for r in recs])),
                             fe_total_s=float(np.sum([r['t_fe'] for r in recs])))
        t0 = time.perf_counter(); gs = [FT.graph(r) for r in recs]; t_feat = time.perf_counter() - t0
        t0 = time.perf_counter(); P = ens.predict(gs); t_inf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for g, p in zip(gs, P):
            gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var']))
        t_proj = time.perf_counter() - t0
        # Single-frame latency (no batching)
        t0 = time.perf_counter()
        for g in gs[:50]:
            ens.predict([g])
        t_single = (time.perf_counter() - t0) / 50
        timing[split].update(features_total_s=t_feat, ensemble_inference_total_s=t_inf,
                             projection_total_s=t_proj, ensemble_single_frame_s=t_single,
                             n_frames=len(recs))
    times['timing'] = timing
    json.dump(dict(results=results, times=times, n_train=N_TRAIN, epochs=EPOCHS, n_ens=N_ENS),
              open('results/main.json', 'w'), indent=1)
    with open('results/main_preds.pkl', 'wb') as fh:
        pickle.dump(preds, fh)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
