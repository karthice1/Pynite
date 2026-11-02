"""Two-stage FE referral with separate calibration sets (second revision).

    python experiment_calibration.py      # writes data/calibration.pkl, results/revision/calibration.json

Protocol (the validation split, used for early stopping and model selection, is not used):
  A  the Mahalanobis model of the frame descriptors is fitted on the training frames
  B  the domain threshold is the conformal 99 % quantile of the distances of an
     independent domain-calibration set (1,000 new ID frames, seed 6)
  C  the conformal quantile of a = e_M / s is computed on the in-domain frames of a
     second independent set (1,000 new ID frames, seed 7), selected with the
     threshold fixed in B
  D  both thresholds are applied to the untouched test sets
The protocol is repeated for the five independently trained ensembles.
"""

import json
import sys
import os
import pickle
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

import analysis_revision as AR
import features as FT
import gnn
import metrics as MT
from generate_dataset import _one

CAL = {'cal_domain': ('train', 1000, 6), 'cal_conf': ('train', 1000, 7)}
ALPHA_DOM, ALPHA_CONF = 0.01, 0.10
TAUS = (0.05, 0.10)
OTHER = ['e_Mmax', 'e_Nmax', 'e_drift', 'e_defl', 'e_R', 'e_rz', 'e_dx', 'e_N', 'e_V']


def calibration_data():
    path = 'data/calibration.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    out = {}
    with Pool(4) as pool:
        for name, (scenario, count, seed) in CAL.items():
            out[name] = pool.map(_one, [(scenario, seed * 1_000_000 + k) for k in range(count)], chunksize=20)
    pickle.dump(out, open(path, 'wb'))
    return out


def members(r):
    if r == 0:
        ens, _ = pickle.load(open('results/ensemble.pkl', 'rb'))
        return ens.members
    out = []
    for k in range(5):
        d = torch.load(f'results/rev/weights/ens{r}_m{k}.pt', weights_only=False)
        m = gnn.FrameGNN(25, 10, 96, 8, 6)
        m.load_state_dict(d['state_dict'])
        nrm = gnn.Normaliser.__new__(gnn.Normaliser)
        nrm.__dict__.update(d['nrm'])
        out.append((m, nrm))
    return out


def conformal_quantile(x, alpha):
    n = len(x)
    return float(np.quantile(x, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method='higher'))


def spread_and_metrics(G, P, full=True):
    s, rows = [], []
    for g, p in zip(G, P):
        u, q = AR.proposed(g, p, 'variance-weighted')
        sdM = np.sqrt(sum(p['q_var'][:, 3 * k + 1:3 * k + 3] * g['qsc'][k][1] ** 2 for k in range(len(g['cases']))))
        s.append(np.linalg.norm(sdM) / (np.linalg.norm(q[:, 1:]) + 1e-12))
        rows.append(MT.frame_metrics_phys(g, u, q, compat=False) if full else dict(e_M=MT._rel(q[:, 1:], g['q_true'][:, 1:])))
    return np.array(s), {k: np.array([r[k] for r in rows]) for k in rows[0]}


def main():
    torch.set_num_threads(4)
    C = calibration_data()
    D = AR.data()
    Gc = {s: [FT.graph(r, 'split') for r in C[s]] for s in CAL}
    G = AR.graphs()
    # Stage A: descriptor model from the training frames
    X = np.array([np.concatenate(AR.descriptors(r['p'])) for r in D['pool'][:3000]])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    lw = LedoitWolf().fit((X - mu) / sd)
    dist = lambda recs: lw.mahalanobis((np.array([np.concatenate(AR.descriptors(r['p'])) for r in recs]) - mu) / sd)
    dcal = dist(C['cal_domain'])
    dconf = dist(C['cal_conf'])
    dtest = {s: dist(D[s]) for s in AR.TESTS}
    # Stage B: fixed domain threshold
    thr = conformal_quantile(dcal, ALPHA_DOM)
    in_conf = dconf <= thr
    out = dict(alpha_domain=ALPHA_DOM, alpha_conformal=ALPHA_CONF, n_cal_domain=len(dcal), n_cal_conf=len(dconf),
               threshold=thr, flagged={s: float(np.mean(dtest[s] > thr)) for s in AR.TESTS},
               flagged_cal_conf=float(np.mean(~in_conf)), n_cal_conf_in=int(in_conf.sum()), runs=[])
    print('threshold', thr, out['flagged'], flush=True)
    for r in range(5):
        mem = members(r)
        Pc = gnn.Ensemble(mem).predict(Gc['cal_conf'])
        P = AR.ensemble_preds(r)
        # Stage C: conformal quantile on the in-domain frames of the second calibration set
        sc, mc = spread_and_metrics(Gc['cal_conf'], Pc, full=False)
        qhat = conformal_quantile(mc['e_M'][in_conf] / sc[in_conf], ALPHA_CONF)
        run = dict(qhat=qhat, policy={})
        # Stage D: untouched test sets
        for s in AR.TESTS:
            st, mt = spread_and_metrics(G[s], P[s])
            ind = dtest[s] <= thr
            e = mt['e_M']
            bound = qhat * st
            run.setdefault('coverage_in_domain', {})[s] = float(np.mean(e[ind] <= bound[ind]))
            for tau in TAUS:
                acc = ind & (bound <= tau)
                pol = dict(referred_domain=float(np.mean(~ind)), referred_spread=float(np.mean(ind & (bound > tau))),
                           referred_total=float(np.mean(~acc)), accepted=int(acc.sum()),
                           accepted_violation=float(np.mean(e[acc] > tau)) if acc.any() else None,
                           joint_violation=float(np.mean(acc & (e > tau))),
                           accepted_p95=float(np.percentile(e[acc], 95)) if acc.any() else None,
                           other={k: float(np.mean(mt[k][acc] > tau)) if acc.any() else None for k in OTHER},
                           other_p95={k: float(np.percentile(mt[k][acc], 95)) if acc.any() else None for k in OTHER})
                run['policy'].setdefault(str(tau), {})[s] = pol
            print(r, s, round(qhat, 3), run['coverage_in_domain'][s],
                  {t: (round(run['policy'][str(t)][s]['referred_total'], 3), run['policy'][str(t)][s]['accepted_violation']) for t in TAUS},
                  flush=True)
        out['runs'].append(run)
    out['groups'] = domain_groups(C)
    json.dump(out, open(f'{AR.OUT}/calibration.json', 'w'), indent=1)


def domain_groups(C):
    """Flag rates of the geometry-only and load-only distances under the same protocol (A, B)."""
    D = AR.data()
    res = {}
    for grp, idx in (('geometry', 0), ('load', 1), ('combined', None)):
        pick = lambda recs: np.array([np.concatenate(AR.descriptors(r['p'])) if idx is None else AR.descriptors(r['p'])[idx] for r in recs])
        X = pick(D['pool'][:3000])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        lw = LedoitWolf().fit((X - mu) / sd)
        dist = lambda recs: lw.mahalanobis((pick(recs) - mu) / sd)
        thr = conformal_quantile(dist(C['cal_domain']), ALPHA_DOM)
        res[grp] = dict(threshold=float(thr), flagged={s: float(np.mean(dist(r) > thr)) for s, r in
                                                       [('cal_conf', C['cal_conf'])] + [(t, D[t]) for t in AR.TESTS]})
    return res


if __name__ == '__main__':
    if 'groups' in sys.argv:
        path = f'{AR.OUT}/calibration.json'
        out = json.load(open(path))
        out['groups'] = domain_groups(calibration_data())
        json.dump(out, open(path, 'w'), indent=1)
        print(out['groups'])
    else:
        main()
