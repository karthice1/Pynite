"""Revision experiments: repeated training runs, graph baselines, depth ablation,
and the extended adaptive-sampling study, run as one parallel job queue.

    python experiment_revision.py [n_workers]     # default 4 single-thread workers

Each job trains one network on the same 3,000 training frames as
experiment_main.py (150 epochs) and stores dimensionless predictions on the
validation split and all test splits in results/rev/<job>.pkl; network
weights go to results/rev/weights/<job>.pt. Adaptive-sampling jobs write
results/adaptive/seed<seed>_<strategy>.json. Finished jobs are skipped, so the
queue can be resumed.

Jobs
----
ens{r}_m{k}     independent ensembles r = 1..4 (members k = 0..4, seed 1000 r + k);
                ensemble r = 0 is results/ensemble.pkl
depth{d}_s{k}   message-passing depth d = 4, 12, 16 (seeds 0..2; depth 8 = ensemble 0)
raw_s{k}        conventional GNN: physical inputs/targets, no statics features
pool_s{k}       proposed GNN with global mean pooling in every step
story_s{k}      StructGNN-style storey pseudo-node network (Chou et al., 2024)
disp_s{k}       displacement-only GNN, forces recovered from displacements
dispK_s{k}      as disp, plus the stiffness residual loss (Leng et al., 2026 style)
soft_s{k}       soft equilibrium penalty (seeds 1, 2; seed 0 in experiment_main)
nosup_s{k}      no load-case superposition (seeds 1, 2; seed 0 in experiment_main)
"""

import os
import pickle
import sys
import time
from multiprocessing import Pool

import torch

import features as FT
import gnn

N_TRAIN, EPOCHS = 3000, 150
SPLITS = ['val', 'test_id', 'test_oodG', 'test_oodL', 'test_real']
K_WEIGHT = 0.01            # stiffness-residual weight, chosen on a short validation sweep


def jobs():
    J = []
    J += [(f'story_s{k}', dict(arch='story', hidden=256, loss_type='mae'), 'story', k) for k in range(3)]
    J += [('adaptive', None, s, st) for s in (2, 3, 4)
          for st in ('random', 'ensemble variance', 'equilibrium residual')]
    J += [('adaptive', None, s, 'diverse equilibrium residual') for s in range(5)]
    J += [(f'depth16_s{k}', dict(steps=16), 'split', k) for k in range(3)]
    J += [(f'depth12_s{k}', dict(steps=12), 'split', k) for k in range(3)]
    J += [(f'dispK_s{k}', dict(disp_only=True, k_weight=K_WEIGHT), 'split', k) for k in range(3)]
    J += [(f'soft_s{k}', dict(eq_weight=1.0), 'split', k) for k in (1, 2)]
    J += [(f'pool_s{k}', dict(arch='pool'), 'split', k) for k in range(3)]
    J += [(f'ens{r}_m{k}', {}, 'split', 1000 * r + k) for r in range(1, 5) for k in range(5)]
    J += [(f'nosup_s{k}', {}, 'total', k) for k in (1, 2)]
    J += [(f'raw_s{k}', {}, 'raw', k) for k in range(3)]
    J += [(f'disp_s{k}', dict(disp_only=True), 'split', k) for k in range(3)]
    J += [(f'depth4_s{k}', dict(steps=4), 'split', k) for k in range(3)]
    return J


def load_graphs(mode):
    D = pickle.load(open('data/frames.pkl', 'rb'))
    D['test_real'] = pickle.load(open('data/realistic.pkl', 'rb'))
    m = 'split' if mode == 'story' else mode
    G = {s: [FT.graph(r, m) for r in (D['pool'][:N_TRAIN] if s == 'train' else D[s])]
         for s in ['train'] + SPLITS}
    if mode == 'story':
        G = {s: [gnn.story_graph(g) for g in v] for s, v in G.items()}
    return G


def run_job(job):
    torch.set_num_threads(1)
    name, kw, mode, seed = job
    if name == 'adaptive':
        import experiment_adaptive as EA
        out = f'results/adaptive/seed{mode}_{seed.replace(" ", "_")}.json'
        if not os.path.exists(out):
            t0 = time.time()
            EA.run(mode, seed)
            print(f'done adaptive seed {mode} {seed} {time.time() - t0:.0f}s', flush=True)
        return
    out = f'results/rev/{name}.pkl'
    if os.path.exists(out):
        return
    G = load_graphs(mode)
    t0 = time.time()
    model, nrm = gnn.train_model(G['train'], seed=seed, epochs=EPOCHS, **kw)
    t_train = time.time() - t0
    preds = {s: gnn.predict(model, nrm, G[s]) for s in SPLITS}
    torch.save(dict(state_dict=model.state_dict(), nrm=nrm.__dict__, kw=kw, mode=mode, seed=seed),
               f'results/rev/weights/{name}.pt')
    pickle.dump(dict(preds=preds, train_s=t_train, kw=kw, mode=mode, seed=seed), open(out, 'wb'))
    print(f'done {name} {t_train:.0f}s', flush=True)


def main(workers=4):
    os.makedirs('results/rev/weights', exist_ok=True)
    with Pool(workers, maxtasksperchild=1) as pool:
        for _ in pool.imap_unordered(run_job, jobs()):
            pass


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
