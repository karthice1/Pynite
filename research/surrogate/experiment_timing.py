"""Wall-clock cost of PyNite vs. the surrogate, measured serially on an idle machine.

    python experiment_timing.py      # writes results/timing.json

Run this alone: the timings recorded during dataset generation and in
experiment_main.py were taken while other jobs shared the CPU.
"""

import json
import pickle
import time

import numpy as np
import torch

import features as FT
import frames as F
import gnn

TESTS = ['test_id', 'test_oodG', 'test_oodL']
N_FRAMES = 200


def main():
    torch.set_num_threads(4)
    D = pickle.load(open('data/frames.pkl', 'rb'))
    ens, _ = pickle.load(open('results/ensemble.pkl', 'rb'))
    out = {}
    for s in TESTS:
        recs = D[s][:N_FRAMES]
        t_fe = []
        for r in recs:
            t0 = time.perf_counter(); F.run_fe(r['p']); t_fe.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); gs = [FT.graph(r) for r in recs]; t_feat = time.perf_counter() - t0
        ens.predict(gs[:10])                                     # warm-up
        t0 = time.perf_counter(); P = ens.predict(gs); t_inf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for g, p in zip(gs, P):
            gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var']))
        t_proj = time.perf_counter() - t0
        t_single = []
        for r in recs[:50]:
            t0 = time.perf_counter()
            g = FT.graph(r); p = ens.predict([g])[0]
            gnn.project(g, p['q'], p['q_var'] + 1e-6 * np.mean(p['q_var']))
            t_single.append(time.perf_counter() - t0)
        out[s] = dict(n_frames=len(recs), fe_mean_s=float(np.mean(t_fe)), fe_median_s=float(np.median(t_fe)),
                      features_per_frame_s=t_feat / len(recs), inference_per_frame_s=t_inf / len(recs),
                      projection_per_frame_s=t_proj / len(recs), single_frame_end_to_end_s=float(np.mean(t_single)))
        out[s]['batched_per_frame_s'] = (out[s]['features_per_frame_s'] + out[s]['inference_per_frame_s']
                                         + out[s]['projection_per_frame_s'])
        print(s, {k: round(v * 1e3, 3) if k != 'n_frames' else v for k, v in out[s].items()}, flush=True)

    # Training cost of one network on the main training set (the ensemble trains five)
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
