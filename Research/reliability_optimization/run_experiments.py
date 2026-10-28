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
    # reduced design space (one section group, 3 variables) for exhaustive enumeration
    'R1': replace(BASE, group_size=6, name='R1'),
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


def opt_tag(beta_uls=BETA_ULS, beta_sls=BETA_SLS, n_mc=N_MC_OPT, saa_seed=SEED_OPT, accept='point'):
    """File-name suffix of the non-default optimisation settings."""
    tag = '' if (beta_uls, beta_sls) == (BETA_ULS, BETA_SLS) else f'_bu{beta_uls}_bs{beta_sls}'
    if n_mc != N_MC_OPT:
        tag += f'_n{n_mc}'
    if saa_seed != SEED_OPT:
        tag += f'_saa{saa_seed}'
    if accept != 'point':
        tag += f'_{accept}'
    return tag


def make_problem(spec, mode, beta_uls=BETA_ULS, beta_sls=BETA_SLS, n_mc=N_MC_OPT, saa_seed=SEED_OPT,
                 accept='point', cache=None):
    """mode: DDO | DDO-hNNN (drift limit h/NNN) | RBDO | RBDO-C (RBDO + all LRFD/SLS checks)."""
    base = 'DDO' if mode.startswith('DDO') else mode
    prob = Problem(spec, base, PM, beta_uls, beta_sls, n_mc, min(N_SCREEN, n_mc), saa_seed,
                   accept=accept, cache=cache)
    if mode.startswith('DDO-h'):
        prob.ddo_drift_limit = 1 / float(mode[5:])
    return prob


def stage_opt(frame, mode, seeds, **opts):
    """GA runs; all seeds run in one process and share the (deterministic) evaluation cache."""
    spec = FRAMES[frame]
    opts = {k: (float(v) if k in ('beta_uls', 'beta_sls') else int(v) if k in ('n_mc', 'saa_seed') else v)
            for k, v in opts.items()}
    fname = f'opt_{frame}_{mode}{opt_tag(**opts)}.json'
    try:
        old = load(fname)
        runs = [r for r in old['runs'] if r['seed'] not in seeds]
    except FileNotFoundError:
        runs = []
    cache = {}
    for seed in seeds:
        t0 = time.time()
        prob = make_problem(spec, mode, cache=cache, **opts)
        out = ga(prob, seed=seed, log=lambda s: log(f'{frame} {mode} s{seed} {s}'))
        b = out['best']
        runs.append({'seed': seed, 'x': b['x'], 'weight': b['weight'], 'feasible': b['feasible'],
                     'info': b['info'], 'ga_weight': out['ga_best']['weight'],
                     'history': out['history'], 'n_eval': out['n_eval'],
                     'time_s': time.time() - t0, 't_analysis': prob.t_analysis, 't_mc': prob.t_mc,
                     'design': spec.describe_design(b['x'])})
        log(f'{frame} {mode} seed {seed}: {b["weight"]:.0f} kg feasible={b["feasible"]}')
        runs.sort(key=lambda r: r['seed'])
        save(fname, {'frame': frame, 'spec': asdict(spec), 'mode': mode,
                     'beta_uls': prob.beta_uls, 'beta_sls': prob.beta_sls, 'n_mc': prob.n_mc,
                     'saa_seed': prob.seed, 'accept': prob.accept, 'runs': runs})


def design_weights(spec, X):
    """Steel mass (kg) of the designs in X (n, n_vars) without structural analysis."""
    from rbo.sections import COLUMN_SHAPES, BEAM_SHAPES
    X = np.asarray(X)
    w = np.zeros(len(X))
    h = spec.heights
    for st in range(spec.n_storeys):
        g = spec.group_of_storey(st)
        mc_e = np.array([sh.mass_per_m for sh in COLUMN_SHAPES])[X[:, 3 * g]]
        mc_i = np.array([sh.mass_per_m for sh in COLUMN_SHAPES])[X[:, 3 * g + 1]]
        mb = np.array([sh.mass_per_m for sh in BEAM_SHAPES])[X[:, 3 * g + 2]]
        w += h[st] * (2 * mc_e + (spec.n_bays - 1) * mc_i) + spec.n_bays * spec.bay_width * mb
    return w


def stage_enumerate(mode, frame='R1', seeds=tuple(range(1, 11))):
    """Exact optimum of the reduced problem by mass-ordered enumeration, and GA runs.

    The designs of the full catalogue product are sorted by mass and evaluated in
    that order; the first feasible design is the exact optimum (of the SAA problem
    for the reliability modes). The GA is then run with several seeds on the same
    problem (shared evaluation cache) and compared with the exact optimum."""
    spec = FRAMES[frame]
    ub = spec.var_bounds()
    X = np.array(np.meshgrid(*[np.arange(u) for u in ub], indexing='ij')).reshape(len(ub), -1).T
    w = design_weights(spec, X)
    r0 = analyse(spec, X[0])
    assert abs(r0.weight - w[0]) < 1e-6 * r0.weight
    order = np.argsort(w, kind='stable')
    cache = {}
    prob = make_problem(spec, mode, cache=cache)
    t0 = time.time()
    n_infeasible_lighter = 0
    for k, i in enumerate(order):
        res = prob.evaluate(X[i])
        if res['feasible']:
            break
        n_infeasible_lighter += 1
        if k % 200 == 0:
            log(f'enumerate {mode}: {k} designs, mass {w[i]:.0f} kg')
    opt = {'x': res['x'], 'weight': res['weight'], 'info': res['info'], 'design': spec.describe_design(res['x']),
           'rank_in_mass_order': int(k), 'time_s': time.time() - t0}
    log(f'enumerate {mode}: optimum {res["weight"]:.0f} kg after {k + 1} evaluations')
    runs = []
    for seed in seeds:
        p = make_problem(spec, mode, cache=cache)
        out = ga(p, seed=seed)
        b = out['best']
        runs.append({'seed': seed, 'x': b['x'], 'weight': b['weight'], 'feasible': b['feasible'],
                     'ga_weight': out['ga_best']['weight'], 'ga_x': out['ga_best']['x'],
                     'n_eval': out['n_eval'], 'n_gen': len(out['history'])})
        log(f'enumerate {mode} GA seed {seed}: {b["weight"]:.0f} kg')
    save(f'enum_{frame}_{mode}.json', {'frame': frame, 'mode': mode, 'n_designs': len(X),
                                       'optimum': opt, 'ga_runs': runs})


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
        out['fail_S1_only'] = int(np.sum(fail & (ind['U1'] >= 1) & (ind['U2'] < 1)))
        out['fail_S2_only'] = int(np.sum(fail & (ind['U2'] >= 1) & (ind['U1'] < 1)))
        out['fail_both'] = int(np.sum(fail & (ind['U1'] >= 1) & (ind['U2'] >= 1)))
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


def stage_verify(part=0, n_parts=1):
    """Independent 10^6-sample verification of the best design of every optimisation
    file, and of every distinct design found by the individual seeds. With
    ``n_parts`` > 1 only every n-th file is processed (merge with verify_merge)."""
    import glob
    part, n_parts = int(part), int(n_parts)
    res = {}
    done = {}

    def ver(frame, x, detail=False):
        key = (frame, tuple(x), detail)
        if key not in done:
            done[key] = verify_design(FRAMES[frame], x, detail=detail)
        return done[key]

    for i, fn in enumerate(sorted(glob.glob(os.path.join(RES, 'opt_*.json')))):
        if i % n_parts != part:
            continue
        d = load(os.path.basename(fn))
        frame = d['frame']
        key = os.path.basename(fn)[4:-5]
        feas = [r for r in d['runs'] if r['feasible']] or d['runs']
        b = min(feas, key=lambda r: r['weight'])
        detail = d['mode'] in ('DDO', 'RBDO-C') and key == f"{frame}_{d['mode']}"
        log(f'verify {key}')
        res[key] = {'x': b['x'], 'design': b['design'], 'mode': d['mode'],
                    'beta_uls_T': d.get('beta_uls', BETA_ULS), 'beta_sls_T': d.get('beta_sls', BETA_SLS),
                    'n_mc': d.get('n_mc', N_MC_OPT), 'accept': d.get('accept', 'point'),
                    'saa': {k: b['info'][k] for k in b['info'] if k.startswith(('beta', 'nfail'))},
                    **ver(frame, b['x'], detail)}
        res[key]['runs'] = [{'seed': r['seed'], 'weight': r['weight'], 'x': r['x'], 'feasible': r['feasible'],
                             'est': ver(frame, r['x'])['est']} for r in d['runs']]
    try:
        if part != 0:
            raise FileNotFoundError
        p = load('pareto_base.json')
        res['pareto'] = []
        for f in p['front']:
            vd = verify_design(BASE, f['x'], n=400_000)
            res['pareto'].append({'x': f['x'], 'weight': f['weight'], 'saa': f['info'], 'est': vd['est'],
                                  'lrfd': vd['det']['strength'], 'drift_det': vd['det']['drift']})
    except FileNotFoundError:
        pass
    save('verify.json' if n_parts == 1 else f'verify_part{part}.json', res)


def stage_verify_merge():
    import glob
    res = {}
    for fn in sorted(glob.glob(os.path.join(RES, 'verify_part*.json'))):
        res.update(load(os.path.basename(fn)))
    save('verify.json', res)


def stage_opensees():
    res = {}
    for mode in ('DDO', 'RBDO-C'):
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
    b = _best('base', 'RBDO-C')
    for ws in (1.0, 2.0):
        log(f'full MC wind scale {ws}')
        out = validation.full_reanalysis_mc(BASE, b['x'], PM, n, seed=55, wind_scale=ws,
                                            log=lambda s: log(s))
        res[f'ws{ws}'] = out
    save('fullmc.json', res)


def final_designs():
    """Designs used in the revised paper: DDO and code-compliant RBDO of every frame."""
    out = {}
    for frame in [f for f in FRAMES if f != 'R1']:
        for mode in ('DDO', 'RBDO-C'):
            try:
                out[f'{frame}_{mode}'] = (frame, _best(frame, mode))
            except FileNotFoundError:
                pass
    return out


def stage_validate(key):
    """Broad surrogate validation of one final design (see validation.broad_validation)."""
    frame, b = final_designs()[key]
    out = validation.broad_validation(FRAMES[frame], b['x'], PM, log=lambda s: log(f'{key} {s}'))
    out['x'] = b['x']
    save(f'validate_{key}.json', out)


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
        'model unc. 0.05': replace(PM, thR_cov=0.05, thE_cov=0.05, thD_cov=0.05),
        'model unc. 0.10': replace(PM, thR_cov=0.10, thE_cov=0.10, thD_cov=0.10),
    }
    res = {}
    for mode in ('DDO', 'RBDO-C'):
        b = _best('base', mode)
        res[mode] = {}
        for k, pm in variants.items():
            log(f'sensitivity {mode} {k}')
            res[mode][k] = verify_design(BASE, b['x'], n=400_000, pm=pm)['est']
    save('sensitivity.json', res)


MODEL_UNC = {'none': PM,
             'MU-a': replace(PM, thR_cov=0.05, thE_cov=0.05, thD_cov=0.05),
             'MU-b': replace(PM, thR_cov=0.10, thE_cov=0.10, thD_cov=0.10),
             'MU-R': replace(PM, thR_cov=0.10),
             'MU-E': replace(PM, thE_cov=0.10),
             'MU-D': replace(PM, thD_cov=0.10)}


def stage_modelunc():
    """Reliability of the final designs with frame-wide model-uncertainty factors
    (assumed lognormal, unit mean) on resistance, load effect and drift."""
    res = {}
    for key, (frame, b) in final_designs().items():
        res[key] = {}
        for k, pm in MODEL_UNC.items():
            log(f'model uncertainty {key} {k}')
            res[key][k] = verify_design(FRAMES[frame], b['x'], pm=pm)['est']
    save('modelunc.json', res)


def stage_convergence():
    res = {}
    Ns = [5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
    for mode in ('DDO', 'RBDO-C'):
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
        # opt FRAME MODE SEEDS [beta_uls=.. beta_sls=.. n_mc=.. saa_seed=.. accept=point|lcb]
        stage_opt(sys.argv[2], sys.argv[3], [int(s) for s in sys.argv[4].split(',')],
                  **dict(a.split('=') for a in sys.argv[5:]))
    elif st == 'validate':
        stage_validate(sys.argv[2])
    elif st == 'enumerate':
        stage_enumerate(sys.argv[2])
    elif st == 'pareto':
        stage_pareto(sys.argv[2])
    elif st == 'pareto3':
        stage_pareto(sys.argv[2], three=True)
    else:
        globals()[f'stage_{st}'](*sys.argv[2:])
