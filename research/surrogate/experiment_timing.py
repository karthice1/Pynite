"""Wall-clock cost of PyNite vs. the surrogate, measured serially on an idle machine.

    python experiment_timing.py [--skip-train]     # writes results/timing.json

Every per-frame time is measured REPEATS times; the mean and the standard
deviation over the repetitions are reported. Besides PyNite, two other
solvers are timed for the same frames: the vectorised sparse direct-stiffness
solver of solvers.py (an optimised minimal FE pipeline: assembly + sparse LU
for the W and G cases) and OpenSeesPy.

Run this alone: the timings recorded during dataset generation and in
experiment_main.py were taken while other jobs shared the CPU.
"""

import json
import pickle
import sys
import time

import numpy as np
import torch

import features as FT
import frames as F
import gnn
import solvers as S

TESTS = ['test_id', 'test_oodG', 'test_oodL']
N_FRAMES = 200
REPEATS = 5


def per_frame(fn, items):
    t0 = time.perf_counter()
    for it in items:
        fn(it)
    return (time.perf_counter() - t0) / len(items)


def stats_of(v):
    return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), runs=[float(x) for x in v])


def main():
    torch.set_num_threads(4)
    D = pickle.load(open('data/frames.pkl', 'rb'))
    ens, _ = pickle.load(open('results/ensemble.pkl', 'rb'))
    out = {}
    for s in TESTS:
        recs = D[s][:N_FRAMES]
        gs = [FT.graph(r) for r in recs]
        ens.predict(gs[:10])                                     # warm-up
        P = ens.predict(gs)
        T = {k: [] for k in ('fe', 'dsm', 'opensees', 'features', 'inference', 'projection', 'single')}
        for _ in range(REPEATS):
            T['fe'].append(per_frame(lambda r: F.run_fe(r['p']), recs))
            T['dsm'].append(per_frame(lambda r: (S.dsm_solve(r['p'], 'W'), S.dsm_solve(r['p'], 'G')), recs))
            T['opensees'].append(per_frame(lambda r: (S.opensees_solve(r['p'], 'W'), S.opensees_solve(r['p'], 'G')), recs))
            T['features'].append(per_frame(FT.graph, recs))
            t0 = time.perf_counter(); ens.predict(gs); T['inference'].append((time.perf_counter() - t0) / len(gs))
            T['projection'].append(per_frame(lambda gp: gnn.project(gp[0], gp[1]['q'], gp[1]['q_var'] + 1e-6 * np.mean(gp[1]['q_var'])),
                                             list(zip(gs, P))))

            def single(r):
                g = FT.graph(r); p = ens.predict([g])[0]
                gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var']))
            T['single'].append(per_frame(single, recs[:50]))
        out[s] = {k + '_per_frame_s': stats_of(v) for k, v in T.items()}
        out[s]['n_frames'] = len(recs)
        bat = np.array(T['features']) + np.array(T['inference']) + np.array(T['projection'])
        out[s]['batched_per_frame_s'] = stats_of(bat)
        # flat keys used by make_figures / uq analysis
        out[s].update(fe_mean_s=float(np.mean(T['fe'])), batched_per_frame_s_mean=float(bat.mean()))
        print(s, {k: (round(v['mean'] * 1e3, 3), round(v['std'] * 1e3, 3)) for k, v in out[s].items()
                  if isinstance(v, dict)}, flush=True)

    # Training cost of one network on the main training set (the ensemble trains five)
    if '--skip-train' in sys.argv:
        out['train'] = json.load(open('results/timing.json'))['train']
        json.dump(out, open('results/timing.json', 'w'), indent=1)
        return
    train = [FT.graph(r) for r in D['pool'][:3000]]
    t0 = time.perf_counter(); gnn.train_model(train, seed=0, epochs=150); t_net = time.perf_counter() - t0
    t0 = time.perf_counter()
    for r in D['pool'][:200]:
        F.run_fe(r['p'])
    fe_train = (time.perf_counter() - t0) / 200 * 3000
    out['train'] = dict(one_network_s=t_net, ensemble_s_est=5 * t_net, fe_training_set_s=fe_train, threads=4)
    print(out['train'])
    json.dump(out, open('results/timing.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
