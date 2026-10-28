"""Run the numerical experiments of the paper.

Usage: python run_experiments.py STAGE [ARGS]

Stages (each writes JSON into results/):
  closed_form              analytical benchmarks of PyNite
  opt FRAME MODE SEEDS     GA optimisation (MODE = DDO | RBDO), e.g. ``opt base RBDO 1,2,3``
  pareto FRAME             NSGA-II weight / beta_ULS front
  verify                   independent large-sample MC of every optimised design
  opensees                 PyNite vs OpenSeesPy (linear unit cases and P-Delta)
  fullmc                   surrogate vs full P-Delta re-analysis Monte Carlo
  sensitivity              probabilistic-model sensitivity of the base designs
  convergence              MC convergence of beta for the base designs
"""
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np

from rbo.frames import FrameSpec, analyse
from rbo.limit_states import MemberTable, deterministic_check
from rbo.optimize import Problem, ga, nsga2
from rbo.reliability import ProbModel, draw, evaluate_samples, estimate
from rbo import validation

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)

BASE = FrameSpec(name='base')
FRAMES = {
    'base': BASE,
    'S3': replace(BASE, n_storeys=3, name='S3'),
    'S9': replace(BASE, n_storeys=9, group_size=3, name='S9'),
    'B6': replace(BASE, bay_width=6.0, name='B6'),
    'B10': replace(BASE, bay_width=10.0, name='B10'),
    'W08': replace(BASE, p_wind=0.8, name='W08'),
    'W16': replace(BASE, p_wind=1.6, name='W16'),
}
PM = ProbModel()
BETA_ULS, BETA_SLS = 3.0, 1.5
N_MC_OPT, N_SCREEN = 50000, 10000
N_VERIFY = 1_000_000
SEED_OPT, SEED_VERIFY = 12345, 987654


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save(name, obj):
    with open(os.path.join(RES, name), 'w') as f:
        json.dump(_clean(obj), f, indent=1)


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def log(msg):
    print(time.strftime('%H:%M:%S'), msg, flush=True)


# ---------------------------------------------------------------- stages
def stage_closed_form():
    save('closed_form.json', {'static': validation.run_closed_form(),
                              'pdelta_cantilever': validation.bench_pdelta_cantilever()})


def stage_opt(frame, mode, seeds, beta_uls=BETA_ULS, beta_sls=BETA_SLS):
    spec = FRAMES[frame]
    tag = '' if (beta_uls, beta_sls) == (BETA_ULS, BETA_SLS) else f'_bu{beta_uls}_bs{beta_sls}'
    runs = []
    for seed in seeds:
        t0 = time.time()
        prob = Problem(spec, 'DDO' if mode.startswith('DDO') else mode, PM, beta_uls, beta_sls,
                       N_MC_OPT, N_SCREEN, SEED_OPT)
        if mode == 'DDO-h500':   # deterministic design with a stricter drift limit
            prob.ddo_drift_limit = 1 / 500
        out = ga(prob, seed=seed, log=lambda s: log(f'{frame} {mode} s{seed} {s}'))
        b = out['best']
        runs.append({'seed': seed, 'x': b['x'], 'weight': b['weight'], 'feasible': b['feasible'],
                     'info': b['info'], 'ga_weight': out['ga_best']['weight'],
                     'history': out['history'], 'n_eval': out['n_eval'],
                     'time_s': time.time() - t0, 't_analysis': prob.t_analysis, 't_mc': prob.t_mc,
                     'design': spec.describe_design(b['x'])})
        log(f'{frame} {mode} seed {seed}: {b["weight"]:.0f} kg feasible={b["feasible"]}')
        save(f'opt_{frame}_{mode}{tag}.json', {'frame': frame, 'spec': asdict(spec), 'mode': mode,
                                               'beta_uls': beta_uls, 'beta_sls': beta_sls, 'runs': runs})


def stage_pareto(frame, three=False):
    spec = FRAMES[frame]
    prob = Problem(spec, 'RBDO', PM, BETA_ULS, BETA_SLS, N_MC_OPT, N_SCREEN, SEED_OPT)
    if three:
        # screen only designs far below the region of interest of the three-objective front
        prob.screen = (1.0, -1.0, BETA_SLS - 0.4)
    out = nsga2(prob, pop_size=60 if three else 48, n_gen=60, seed=3, drift_objective=three,
                log=lambda s: log(f'pareto {s}'))
    front = [{'x': p['x'], 'weight': p['weight'], 'info': p['info'],
              'design': spec.describe_design(p['x'])} for p in out['front']]
    arch = [{'x': p['x'], 'weight': p['weight'], 'beta_uls': p['info']['beta_uls'],
             'beta_drift': p['info']['beta_drift']} for p in out['archive']]
    save(f'pareto{"3" if three else ""}_{frame}.json', {'front': front, 'archive': arch, 'n_eval': out['n_eval']})


def _best(frame, mode):
    d = load(f'opt_{frame}_{mode}.json')
    feas = [r for r in d['runs'] if r['feasible']] or d['runs']
    return min(feas, key=lambda r: r['weight'])


def verify_design(spec, x, n=N_VERIFY, seed=SEED_VERIFY, pm=PM, detail=False):
    r = analyse(spec, x)
    t = MemberTable(r)
    s = draw(pm, n, len(t.A), seed)
    ind = evaluate_samples(r, s, t, detail=detail)
    est = estimate(ind)
    out = {'est': est, 'det': {k: v for k, v in deterministic_check(r, t).items() if k != 'U_members'},
           'weight': r.weight}
    if detail:
        fail = ind['U'] >= 1
        Um = ind['U_members']
        gov = np.argmax(Um[fail], axis=1) if fail.any() else np.array([], int)
        out['governing_member_counts'] = {t.names[k]: int(np.sum(gov == k)) for k in np.unique(gov)}
        out['U_quantiles'] = np.quantile(ind['U'], [0.5, 0.9, 0.99, 0.999]).tolist()
        out['drift_quantiles'] = np.quantile(ind['drift'], [0.5, 0.9, 0.99]).tolist()
        out['U_member_p999'] = np.quantile(Um, 0.999, axis=0).tolist()
        out['U_member_median'] = np.median(Um, axis=0).tolist()
        out['member_names'] = t.names
        out['det_U_members'] = deterministic_check(r, t)['U_members'].tolist()
        # histograms for plotting
        out['hist_U'] = np.histogram(ind['U'], bins=120, range=(0, 1.2))[0].tolist()
        out['hist_drift'] = np.histogram(ind['drift'], bins=120, range=(0, 2.4))[0].tolist()
    return out


def stage_verify():
    res = {}
    for frame in FRAMES:
        for mode in ('DDO', 'RBDO', 'DDO-h500'):
            try:
                b = _best(frame, mode)
            except FileNotFoundError:
                continue
            log(f'verify {frame} {mode}')
            res[f'{frame}_{mode}'] = {'x': b['x'], 'design': b['design'],
                                      **verify_design(FRAMES[frame], b['x'], detail=(frame == 'base'))}
    # RBDO with alternative reliability targets (base frame)
    import glob
    for fn in sorted(glob.glob(os.path.join(RES, 'opt_base_RBDO_bu*.json'))):
        d = load(os.path.basename(fn))
        b = min([r for r in d['runs'] if r['feasible']] or d['runs'], key=lambda r: r['weight'])
        key = f"base_RBDO_bu{d['beta_uls']}_bs{d['beta_sls']}"
        log(f'verify {key}')
        res[key] = {'x': b['x'], 'design': b['design'], 'beta_uls_T': d['beta_uls'],
                    'beta_sls_T': d['beta_sls'], **verify_design(BASE, b['x'])}
    # Pareto designs (re-verified with independent samples, smaller n)
    try:
        p = load('pareto_base.json')
        res['pareto'] = [{'x': f['x'], 'weight': f['weight'],
                          'est': verify_design(BASE, f['x'], n=400_000)['est']} for f in p['front']]
    except FileNotFoundError:
        pass
    save('verify.json', res)


def stage_opensees():
    res = {}
    for mode in ('DDO', 'RBDO'):
        b = _best('base', mode)
        combo = {'D': 1.2, 'L': 0.5, 'W': 1.6, 'ND': 1.2 * 0.001, 'NL': 0.5 * 0.001}
        rows, pd = validation.compare_opensees(BASE, b['x'], combo)
        res[mode] = {'unit': rows, 'pdelta': pd, 'combo': combo}
    for frame in ('S3', 'S9'):
        b = _best(frame, 'DDO')
        combo = {'D': 1.2, 'L': 0.5, 'W': 1.6, 'ND': 1.2 * 0.001, 'NL': 0.5 * 0.001}
        rows, pd = validation.compare_opensees(FRAMES[frame], b['x'], combo)
        res[f'{frame}_DDO'] = {'unit': rows, 'pdelta': pd, 'combo': combo}
    save('opensees.json', res)


def stage_fullmc(n=1000):
    res = {}
    b = _best('base', 'RBDO')
    for ws in (1.0, 2.0):
        log(f'full MC wind scale {ws}')
        out = validation.full_reanalysis_mc(BASE, b['x'], PM, n, seed=55, wind_scale=ws,
                                            log=lambda s: log(s))
        res[f'ws{ws}'] = out
    save('fullmc.json', res)


def stage_sensitivity():
    variants = {
        'baseline': PM,
        'W50 cov 0.30': replace(PM, W50_cov=0.30),
        'W50 cov 0.45': replace(PM, W50_cov=0.45),
        'W50 bias 0.90': replace(PM, W50_bias=0.90),
        'Fy bias 1.05': replace(PM, Fy_bias=1.05),
        'Fy cov 0.10': replace(PM, Fy_cov=0.10),
        'Fy corr 0': replace(PM, Fy_corr=0.0),
        'Fy corr 1': replace(PM, Fy_corr=1.0),
        'sway sd 0': replace(PM, sway_sd=0.0),
        'sway sd 1/500': replace(PM, sway_sd=1 / 500),
        'fab cov 0': replace(PM, fab_cov=0.0),
        'E cov 0': replace(PM, E_cov=0.0),
        'L50 cov 0.35': replace(PM, L50_cov=0.35),
    }
    res = {}
    for mode in ('DDO', 'RBDO'):
        b = _best('base', mode)
        res[mode] = {}
        for k, pm in variants.items():
            log(f'sensitivity {mode} {k}')
            res[mode][k] = verify_design(BASE, b['x'], n=400_000, pm=pm)['est']
    save('sensitivity.json', res)


def stage_convergence():
    res = {}
    Ns = [5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    for mode in ('DDO', 'RBDO'):
        b = _best('base', mode)
        r = analyse(BASE, b['x'])
        t = MemberTable(r)
        res[mode] = []
        for rep in range(5):
            s = draw(PM, Ns[-1], len(t.A), 1000 + rep)
            ind = evaluate_samples(r, s, t)
            for n in Ns:
                e = estimate({k: v[:n] for k, v in ind.items()})
                res[mode].append({'rep': rep, 'n': n, 'beta_uls': e['beta_uls'], 'beta_drift': e['beta_drift'],
                                  'nfail_uls': e['nfail_uls']})
            log(f'convergence {mode} rep {rep}')
    save('convergence.json', res)


if __name__ == '__main__':
    st = sys.argv[1]
    if st == 'opt':
        stage_opt(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(',')],
                  *[float(v) for v in sys.argv[5:7]])
    elif st == 'pareto':
        stage_pareto(sys.argv[2])
    elif st == 'pareto3':
        stage_pareto(sys.argv[2], three=True)
    else:
        globals()[f'stage_{st}']()
