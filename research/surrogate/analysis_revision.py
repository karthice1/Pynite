"""Analyses added in the revision (run after experiment_revision.py).

    python analysis_revision.py [stage ...]     # default: all stages

Stages (each writes results/revision/<stage>.json):
  variants     proposed surrogate (ensemble 0): raw / Euclidean / variance-weighted
               projection / forces recovered from displacements; extended metrics
               and compatibility residuals on every split
  runs         five independent ensembles: mean, std and t-based 95 % CI of the
               per-run median / p95 errors; frame-bootstrap CIs for ensemble 0
  graphs       graph-learning baselines and depth ablation (3 seeds each)
  baselines    tabular baselines refitted, all splits, bootstrap CIs
  sensitivity  projection floor eps and condition number of C Sigma C^T
  ood          two-stage referral: Mahalanobis domain check + split-conformal
               spread threshold, both calibrated on the validation split
  pcg          conjugate-gradient iterations with the surrogate as warm start
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import stats

import features as FT
import gnn
import metrics as MT

SPLITS = ['val', 'test_id', 'test_oodG', 'test_oodL', 'test_real']
TESTS = SPLITS[1:]
OUT = 'results/revision'
KEYS = ['e_dx', 'e_dy', 'e_rz', 'e_drift', 'e_defl', 'e_M', 'e_N', 'e_V', 'e_R', 'e_Mmax', 'e_Nmax',
        'gov_top1', 'gov_top3', 'eq_res', 'compat', 'compat_M']
N_BOOT = 2000
_cache = {}


def data():
    if 'D' not in _cache:
        D = pickle.load(open('data/frames.pkl', 'rb'))
        D['test_real'] = pickle.load(open('data/realistic.pkl', 'rb'))
        _cache['D'] = D
    return _cache['D']


def graphs(mode='split'):
    if mode not in _cache:
        D = data()
        _cache[mode] = {s: [FT.graph(r, mode) for r in D[s]] for s in SPLITS}
    return _cache[mode]


def varw(p):
    return p['q_var'] + 1e-6 * np.mean(p['q_var'])


def ens_from_members(preds):
    """preds: list over members of (U, Q) lists -> per-frame dicts with mean / var."""
    out = []
    for i in range(len(preds[0][0])):
        Us = np.stack([m[0][i] for m in preds]); Qs = np.stack([m[1][i] for m in preds])
        out.append(dict(u=Us.mean(0), u_var=Us.var(0), q=Qs.mean(0), q_var=Qs.var(0)))
    return out


def ensemble_preds(r):
    """Ensemble r (0..4) predictions on all splits."""
    path = f'results/rev/ens{r}_preds.pkl'
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    G = graphs()
    if r == 0:
        ens, _ = pickle.load(open('results/ensemble.pkl', 'rb'))
        members = [{s: gnn.predict(m, n, G[s]) for s in SPLITS} for m, n in ens.members]
    else:
        members = [pickle.load(open(f'results/rev/ens{r}_m{k}.pkl', 'rb'))['preds'] for k in range(5)]
    P = {s: ens_from_members([m[s] for m in members]) for s in SPLITS}
    P['_members'] = members
    pickle.dump(P, open(path, 'wb'))
    return P


def proposed(g, p, how):
    """Physical (u, q) of the proposed ensemble for a force-correction variant."""
    if how == 'raw':
        return FT.physical(g, p['u'], p['q'])
    if how == 'euclidean':
        return FT.physical(g, p['u'], gnn.project(g, p['q']))
    if how == 'variance-weighted':
        return FT.physical(g, p['u'], gnn.project(g, p['q'], varw(p)))
    if how == 'recovered':
        u, _ = FT.physical(g, p['u'], p['q'])
        return u, MT.forces_from_displacements(g, u)
    raise ValueError(how)


def rows_for(G, P, how):
    return [MT.frame_metrics_phys(g, *proposed(g, p, how)) for g, p in zip(G, P)]


def summ(rows):
    return {k: dict(mean=float(np.mean([r[k] for r in rows])),
                    med=float(np.median([r[k] for r in rows])),
                    p95=float(np.percentile([r[k] for r in rows], 95))) for k in KEYS if k in rows[0]}


def boot_ci(x, fn, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    b = np.array([fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def t_ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)), ci95=[float(v.mean() - h), float(v.mean() + h)], n=len(v))


# ---------------------------------------------------------------------------

def stage_variants():
    G = graphs()
    P = ensemble_preds(0)
    out = {}
    for how in ('raw', 'euclidean', 'variance-weighted', 'recovered'):
        out[how] = {}
        for s in SPLITS:
            rows = rows_for(G[s], P[s], how)
            out[how][s] = summ(rows)
            _cache[('rows', how, s)] = rows
        print(how, {s: round(out[how][s]['e_M']['med'], 4) for s in SPLITS},
              'compat', {s: round(out[how][s]['compat']['med'], 4) for s in SPLITS}, flush=True)
    # Does projection increase or decrease the compatibility error? (paired, per frame)
    out['compat_change'] = {}
    for how in ('euclidean', 'variance-weighted'):
        out['compat_change'][how] = {}
        for s in SPLITS:
            a = np.array([r['compat'] for r in _cache[('rows', 'raw', s)]])
            b = np.array([r['compat'] for r in _cache[('rows', how, s)]])
            am = np.array([r['compat_M'] for r in _cache[('rows', 'raw', s)]])
            bm = np.array([r['compat_M'] for r in _cache[('rows', how, s)]])
            out['compat_change'][how][s] = dict(frac_increase=float(np.mean(b > a)),
                                                median_ratio=float(np.median(b / a)),
                                                frac_increase_M=float(np.mean(bm > am)),
                                                median_ratio_M=float(np.median(bm / am)))
    return out


def stage_runs():
    G = graphs()
    per_run = {s: {k: dict(med=[], p95=[]) for k in KEYS} for s in TESTS}
    raw_vs_proj = {s: [] for s in TESTS}
    for r in range(5):
        P = ensemble_preds(r)
        for s in TESTS:
            rows = rows_for(G[s], P[s], 'variance-weighted')
            rows_raw = rows_for(G[s], P[s], 'raw')
            for k in KEYS:
                v = [x[k] for x in rows]
                per_run[s][k]['med'].append(float(np.median(v)))
                per_run[s][k]['p95'].append(float(np.percentile(v, 95)))
            raw_vs_proj[s].append(float(np.median([x['e_M'] for x in rows_raw]) - np.median([x['e_M'] for x in rows])))
            if r == 0:
                _cache[('rows0', s)] = rows; _cache[('rows0raw', s)] = rows_raw
        print('run', r, {s: round(per_run[s]['e_M']['med'][-1], 4) for s in TESTS}, flush=True)
    out = dict(per_run=per_run,
               across_runs={s: {k: dict(med=t_ci(per_run[s][k]['med']), p95=t_ci(per_run[s][k]['p95']))
                                for k in KEYS} for s in TESTS},
               raw_minus_projected_median_eM=dict(per_run=raw_vs_proj,
                                                 across_runs={s: t_ci(v) for s, v in raw_vs_proj.items()}))
    boot = {}
    for s in TESTS:
        rows, rows_raw = _cache[('rows0', s)], _cache[('rows0raw', s)]
        boot[s] = {}
        for k in ('e_dx', 'e_dy', 'e_rz', 'e_drift', 'e_defl', 'e_M', 'e_N', 'e_V', 'e_R', 'e_Mmax'):
            v = [x[k] for x in rows]
            boot[s][k] = dict(med=float(np.median(v)), med_ci=boot_ci(v, np.median),
                              p95=float(np.percentile(v, 95)), p95_ci=boot_ci(v, lambda a: np.percentile(a, 95)))
        # paired bootstrap of median(raw) - median(projected) moment error
        a = np.array([x['e_M'] for x in rows_raw]); b = np.array([x['e_M'] for x in rows])
        rng = np.random.default_rng(1)
        d = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(a), len(a))
            d.append(np.median(a[i]) - np.median(b[i]))
        boot[s]['eM_raw_minus_proj'] = dict(est=float(np.median(a) - np.median(b)),
                                            ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    out['bootstrap_ens0'] = boot
    return out


GRAPH_VARIANTS = {
    'Proposed, single network (depth 8)': ('ens0', 'split', [0, 1, 2]),
    'Depth 4': ('depth4', 'split', [0, 1, 2]),
    'Depth 12': ('depth12', 'split', [0, 1, 2]),
    'Depth 16': ('depth16', 'split', [0, 1, 2]),
    'Global pooling': ('pool', 'split', [0, 1, 2]),
    'Conventional GNN (physical inputs)': ('raw', 'raw', [0, 1, 2]),
    'StructGNN-style (storey pseudo nodes)': ('story', 'split', [0, 1, 2]),
    'Displacement-only + force recovery': ('disp', 'split', [0, 1, 2]),
    'Displacement-only + stiffness residual (Leng-style)': ('dispK', 'split', [0, 1, 2]),
    'Leng-style, residual weight 1e-4': ('dispK4', 'split', [0, 1, 2]),
    'Soft equilibrium penalty': ('soft', 'split', [0, 1, 2]),
    'No load-case superposition': ('nosup', 'total', [0, 1, 2]),
}


def _single_preds(tag, seed):
    if tag == 'ens0':
        return {s: (v[0], v[1]) for s, v in ensemble_preds(0)['_members'][seed].items()}
    if tag in ('soft', 'nosup') and seed == 0:
        import torch  # noqa: F401  (unpickling torch modules)
        name = 'soft_equilibrium_penalty' if tag == 'soft' else 'no_load-case_superposition'
        m, nrm, _ = pickle.load(open(f'results/{name}.pkl', 'rb'))
        G = graphs('split' if tag == 'soft' else 'total')
        return {s: gnn.predict(m, nrm, G[s]) for s in SPLITS}
    return pickle.load(open(f'results/rev/{tag}_s{seed}.pkl', 'rb'))['preds']


def stage_graphs():
    out = {}
    for name, (tag, mode, seeds) in GRAPH_VARIANTS.items():
        G = graphs(mode)
        Gs = graphs('split')
        res = {'raw': {}, 'projected': {}}
        try:
            per_seed = [_single_preds(tag, k) for k in seeds]
        except FileNotFoundError as err:
            print('missing', name, err); continue
        for s in TESTS:
            for corr in res:
                meds = {k: [] for k in KEYS}; p95 = {k: [] for k in KEYS}
                for pr in per_seed:
                    rows = []
                    for g, gs, u, q in zip(G[s], Gs[s], *pr[s]):
                        uh, qh = FT.physical(g, u, q)
                        if tag.startswith('disp'):
                            qh = MT.forces_from_displacements(gs, uh)
                            if corr == 'projected':      # Euclidean projection of the recovered forces
                                qh = _project_total(gs, qh)
                        elif corr == 'projected':
                            _, qh = FT.physical(g, u, gnn.project(g, q))
                        rows.append(MT.frame_metrics_phys(gs, uh, qh))
                    for k in KEYS:
                        v = [r[k] for r in rows]
                        meds[k].append(float(np.median(v))); p95[k].append(float(np.percentile(v, 95)))
                res[corr][s] = {k: dict(med=meds[k], p95=p95[k], med_mean=float(np.mean(meds[k])),
                                        med_std=float(np.std(meds[k], ddof=1)) if len(meds[k]) > 1 else 0.0,
                                        p95_mean=float(np.mean(p95[k])))
                                for k in KEYS}
        if tag not in ('ens0', 'soft', 'nosup'):
            res['train_s'] = [pickle.load(open(f'results/rev/{tag}_s{k}.pkl', 'rb'))['train_s'] for k in seeds]
        out[name] = res
        print(name, {s: (round(res['raw'][s]['e_dx']['med_mean'], 4), round(res['raw'][s]['e_M']['med_mean'], 4))
                     for s in TESTS}, flush=True)
    return out


def _project_total(g, q):
    """Euclidean projection (dimensionless metric of the first case) of total physical forces q."""
    lam = [sc[0] for sc in g['qsc']]
    d = sum(l * dd for l, dd in zip(lam, g['d'])) / lam[0]
    C, qv = g['C'], (q / g['qsc'][0]).ravel()
    return (qv + C.T @ np.linalg.solve(C @ C.T, d - C @ qv)).reshape(-1, 3) * g['qsc'][0]


def stage_baselines():
    import baselines as BL
    D = data()
    G = graphs()
    train = [FT.graph(r) for r in D['pool'][:3000]]
    out, preds = {}, {}
    for name in BL.BASELINES:
        t0 = time.time()
        mdl = BL.TabularSurrogate(name).fit(train)
        t_fit = time.time() - t0
        out[name] = dict(train_s=t_fit)
        for s in TESTS:
            P = mdl.predict(G[s])
            preds.setdefault(name, {})[s] = P
            rows = [MT.frame_metrics(g, p['u'], p['q']) for g, p in zip(G[s], P)]
            out[name][s] = summ(rows)
            for k in ('e_dx', 'e_M'):
                v = [x[k] for x in rows]
                out[name][s][k]['med_ci'] = boot_ci(v, np.median)
                out[name][s][k]['p95_ci'] = boot_ci(v, lambda a: np.percentile(a, 95))
            rows = [MT.frame_metrics(g, p['u'], gnn.project(g, p['q']), compat=False) for g, p in zip(G[s], P)]
            out[name][s]['projected'] = summ(rows)
        print(name, f'{t_fit:.0f}s', {s: round(out[name][s]['e_M']['med'], 4) for s in TESTS}, flush=True)
        pickle.dump(preds, open('results/rev/baseline_preds.pkl', 'wb'))
    return out


def stage_sensitivity():
    G = graphs()
    P = ensemble_preds(0)
    out = {}
    for eps in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 'additive 1e-6', 'euclidean']:
        out[str(eps)] = {}
        for s in TESTS:
            eM, eN, conds = [], [], []
            for g, p in zip(G[s], P[s]):
                if eps == 'euclidean':
                    W = np.ones_like(p['q_var'])
                elif eps == 'additive 1e-6':
                    W = varw(p)
                else:
                    W = np.maximum(p['q_var'], eps * np.mean(p['q_var']))
                qs = gnn.project(g, p['q'], W)
                u, q = FT.physical(g, p['u'], qs)
                eM.append(MT._rel(q[:, 1:], g['q_true'][:, 1:])); eN.append(MT._rel(q[:, 0], g['q_true'][:, 0]))
                conds.append(max(np.linalg.cond((g['C'] * W[:, 3 * k:3 * k + 3].ravel()[None, :]) @ g['C'].T)
                                 for k in range(len(g['cases']))))
            out[str(eps)][s] = dict(e_M_med=float(np.median(eM)), e_M_p95=float(np.percentile(eM, 95)),
                                    e_N_med=float(np.median(eN)), cond_med=float(np.median(conds)),
                                    cond_max=float(np.max(conds)))
        print(eps, {s: (round(v['e_M_med'], 4), f"{v['cond_max']:.1e}") for s, v in out[str(eps)].items()}, flush=True)
    return out


def descriptors(p):
    """Frame-level geometry and load-pattern descriptors (no FE solve needed)."""
    h, b, w, F = np.asarray(p['heights']), np.asarray(p['bays']), np.asarray(p['w']), np.asarray(p['F'])
    elev = np.cumsum(h)
    aF = np.abs(F)
    geo = [p['n_st'], p['n_bay'], h.max(), h[0] / h.mean(), b.mean(), b.max() / b.min(), elev[-1] / b.sum(),
           np.log10(p['E']), np.log10(np.mean(p['Ib']) / np.mean(p['Ic'])), float(p['fixed_base'])]
    load = [np.sum(aF * elev) / (aF.sum() * elev[-1]),          # height of the lateral-load centroid
            aF[-1] / aF.sum(),                                    # roof share of the lateral load
            np.mean(w / w.max(axis=1, keepdims=True)),            # bay-to-bay gravity uniformity
            np.mean(w.std(axis=1) / w.mean(axis=1)),              # coefficient of variation within a floor
            np.log10(w.mean() * b.mean() / aF.sum())]             # gravity-to-lateral intensity
    return np.array(geo, float), np.array(load, float)


def stage_ood():
    from sklearn.covariance import LedoitWolf
    D = data()
    G = graphs()
    P = ensemble_preds(0)
    train = D['pool'][:3000]
    desc = {s: [descriptors(r['p']) for r in D[s]] for s in SPLITS}
    dtr = [descriptors(r['p']) for r in train]
    out = dict(alpha_domain=0.01, alpha_conformal=0.10)
    scores = {}
    for grp, idx in (('geometry', 0), ('load', 1), ('combined', None)):
        X = np.array([np.concatenate(d) if idx is None else d[idx] for d in dtr])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        lw = LedoitWolf().fit((X - mu) / sd)
        f = lambda s: lw.mahalanobis((np.array([np.concatenate(d) if idx is None else d[idx] for d in desc[s]]) - mu) / sd)
        scores[grp] = {s: f(s) for s in SPLITS}
        thr = float(np.quantile(scores[grp]['val'], 1 - out['alpha_domain']))
        out[grp] = dict(threshold=thr, flagged={s: float(np.mean(scores[grp][s] > thr)) for s in SPLITS})
        print(grp, out[grp]['flagged'], flush=True)

    # Stage 2: split-conformal bound on the moment error from the ensemble spread
    def spread_err(s):
        sp_, e = [], []
        for g, p in zip(G[s], P[s]):
            u, q = proposed(g, p, 'variance-weighted')
            sdM = np.sqrt(sum(p['q_var'][:, 3 * k + 1:3 * k + 3] * g['qsc'][k][1] ** 2 for k in range(len(g['cases']))))
            sp_.append(np.linalg.norm(sdM) / (np.linalg.norm(q[:, 1:]) + 1e-12))
            e.append(MT._rel(q[:, 1:], g['q_true'][:, 1:]))
        return np.array(sp_), np.array(e)
    SE = {s: spread_err(s) for s in SPLITS}
    dom = scores['combined']
    sv, ev = SE['val']
    ok = dom['val'] <= out['combined']['threshold']
    ncs = ev[ok] / sv[ok]
    n = len(ncs)
    qhat = float(np.quantile(ncs, min(1.0, np.ceil((n + 1) * (1 - out['alpha_conformal'])) / n), method='higher'))
    out['conformal_qhat'] = qhat
    out['policy'] = {}
    for tau in (0.05, 0.10):
        out['policy'][str(tau)] = {}
        for s in TESTS:
            sp_, e = SE[s]
            in_dom = dom[s] <= out['combined']['threshold']
            bound = qhat * sp_
            acc = in_dom & (bound <= tau)
            out['policy'][str(tau)][s] = dict(
                referred_domain=float(np.mean(~in_dom)), referred_spread=float(np.mean(in_dom & (bound > tau))),
                referred_total=float(np.mean(~acc)),
                coverage_in_domain=float(np.mean(e[in_dom] <= bound[in_dom])) if in_dom.any() else None,
                accepted_violation=float(np.mean(e[acc] > tau)) if acc.any() else None,
                accepted_p95=float(np.percentile(e[acc], 95)) if acc.any() else None,
                accepted_max=float(np.max(e[acc])) if acc.any() else None,
                all_p95=float(np.percentile(e, 95)))
        print(tau, {s: (round(v['referred_total'], 3), v['accepted_violation']) for s, v in out['policy'][str(tau)].items()}, flush=True)
    return out


def stage_pcg(n_frames=100):
    G = graphs()
    P = ensemble_preds(0)
    out = {}
    for s in TESTS:
        it0, it1 = [], []
        for g, p in list(zip(G[s], P[s]))[:n_frames]:
            Kd, fs, fdof = gnn.stiffness_system(g)
            K = Kd.tocsr()
            # total load, in units of the first case (same dimensionless K for all cases)
            lam = [sc[0] for sc in g['qsc']]
            f = sum(l * fc for l, fc in zip(lam, fs)) / lam[0]
            u, _ = FT.physical(g, p['u'], p['q'])
            y0 = (u / g['usc'][0]).ravel()[fdof]
            Mi = sp.diags(1 / K.diagonal())
            for x0, acc in ((None, it0), (y0, it1)):
                cnt = [0]
                spla.cg(K, f, x0=x0, rtol=1e-8, maxiter=20000, M=Mi, callback=lambda xk: cnt.__setitem__(0, cnt[0] + 1))
                acc.append(cnt[0])
        out[s] = dict(iters_cold_med=float(np.median(it0)), iters_warm_med=float(np.median(it1)),
                      ratio_med=float(np.median(np.array(it1) / np.array(it0))))
        print(s, out[s], flush=True)
    return out


STAGES = dict(variants=stage_variants, runs=stage_runs, graphs=stage_graphs, baselines=stage_baselines,
              sensitivity=stage_sensitivity, ood=stage_ood, pcg=stage_pcg)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for st in (sys.argv[1:] or list(STAGES)):
        t0 = time.time()
        res = STAGES[st]()
        json.dump(res, open(f'{OUT}/{st}.json', 'w'), indent=1)
        print(f'== {st} done in {time.time() - t0:.0f}s', flush=True)
