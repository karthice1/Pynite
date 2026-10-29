"""Discrete design optimisation: constrained GA (single objective) and NSGA-II.

Design variables are integer indices into the W-shape catalogues (sorted by
area, so index steps are monotonic in weight). Constraints are handled with
Deb's feasibility rules; a greedy discrete local search polishes the best GA
design (memetic step).
"""
import time
import numpy as np

from .frames import analyse
from .limit_states import MemberTable, deterministic_check
from .reliability import ProbModel, draw, evaluate_samples, estimate
from .sections import COLUMN_SHAPES


class Problem:
    """Weight minimisation of a frame under deterministic or reliability constraints.

    mode = 'DDO'    : LRFD strength <= 1, drift <= h/400, deflection <= L/360
    mode = 'RBDO'   : beta_ULS >= beta_uls, beta_drift >= beta_sls, beta_defl >= beta_sls
    mode = 'RBDO-C' : both sets of constraints (code-compliant RBDO)
    All modes add the constructability constraint that column areas do not
    increase with height.

    ``accept = 'point'`` imposes the reliability targets on the SAA point estimates
    of beta; ``accept = 'lcb'`` imposes them on the one-sided 95 % lower confidence
    bounds (exact binomial), i.e. a target is met only when the sample shows it with
    95 % confidence. ``cache`` may be shared between problems with identical
    constraints and samples (the evaluation is a deterministic function of x).
    """

    def __init__(self, spec, mode='DDO', pm=None, beta_uls=3.0, beta_sls=1.5,
                 n_mc=50000, n_screen=10000, seed=12345, accept='point', cache=None):
        self.spec = spec
        self.mode = mode
        self.pm = pm or ProbModel()
        self.beta_uls = beta_uls
        self.beta_sls = beta_sls
        self.ub = spec.var_bounds()
        self.cache = {} if cache is None else cache
        self.accept = accept
        self.requested = set()   # distinct designs requested by this problem instance
        self.n_eval = 0
        self.n_mc = n_mc
        self.n_screen = n_screen
        self.samples = None
        self.seed = seed
        self.screen = None   # optional (uls, drift, defl) screening thresholds
        self.ddo_drift_limit = 1 / 400
        self.t_analysis = 0.0
        self.t_mc = 0.0

    # --------------------------------------------------------------
    def _monotonic_violation(self, x):
        ng = self.spec.n_groups
        v = 0.0
        for g in range(ng - 1):
            for k in (0, 1):
                a_lo = COLUMN_SHAPES[x[3 * g + k]].A
                a_hi = COLUMN_SHAPES[x[3 * (g + 1) + k]].A
                v += max(0.0, a_hi / a_lo - 1.0)
        return v

    def evaluate(self, x):
        x = tuple(int(v) for v in x)
        if x not in self.requested:
            self.requested.add(x)
            self.n_eval += 1
        if x in self.cache:
            return self.cache[x]
        t0 = time.time()
        r = analyse(self.spec, x)
        t = MemberTable(r)
        self.t_analysis += time.time() - t0
        mono = self._monotonic_violation(x)
        info = {'weight': r.weight}
        g = []
        if self.mode in ('DDO', 'RBDO-C'):
            chk = deterministic_check(r, t, self.ddo_drift_limit)
            g += [chk['strength'] - 1, chk['drift'] - 1, chk['deflection'] - 1]
            info.update({k: chk[k] for k in ('strength', 'drift', 'deflection', 'max_B2')})
        if self.mode in ('RBDO', 'RBDO-C'):
            t1 = time.time()
            if self.samples is None:
                self.samples = draw(self.pm, self.n_mc, len(t.A), self.seed)
            s = self.samples
            ind = evaluate_samples(r, s.chunk(0, self.n_screen), t)
            est = estimate(ind)
            # skip the full sample when the screening sample is clearly infeasible
            su, sd, sf = self.screen or (self.beta_uls - 0.4, self.beta_sls - 0.4, self.beta_sls - 0.4)
            clear = est['beta_uls'] < su or est['beta_drift'] < sd or est['beta_defl'] < sf
            if not clear and self.n_mc > self.n_screen:
                ind2 = evaluate_samples(r, s.chunk(self.n_screen, self.n_mc), t)
                ind = {k: np.concatenate([ind[k], ind2[k]]) for k in ind}
                est = estimate(ind)
            self.t_mc += time.time() - t1
            b = 'beta_lcb_' if self.accept == 'lcb' else 'beta_'
            g = [(self.beta_uls - est[b + 'uls']) / self.beta_uls,
                 (self.beta_sls - est[b + 'drift']) / self.beta_sls,
                 (self.beta_sls - est[b + 'defl']) / self.beta_sls] + g
            info.update(est)
            info['screened'] = clear
        g = np.array(g + [mono])
        viol = float(np.sum(np.maximum(g, 0)))
        res = {'x': x, 'weight': r.weight, 'g': g, 'viol': viol, 'feasible': viol <= 0, 'info': info}
        self.cache[x] = res
        return res


def _better(a, b):
    """Deb's feasibility rules: True if a is preferred to b."""
    if a['feasible'] and b['feasible']:
        return a['weight'] < b['weight']
    if a['feasible'] != b['feasible']:
        return a['feasible']
    return a['viol'] < b['viol']


def _rank_key(res):
    return (0, res['weight']) if res['feasible'] else (1, res['viol'])


def _mutate(rng, x, ub, pm):
    y = x.copy()
    for i in range(len(y)):
        if rng.random() < pm:
            if rng.random() < 0.7:
                y[i] = np.clip(y[i] + rng.choice([-3, -2, -1, 1, 2, 3]), 0, ub[i] - 1)
            else:
                y[i] = rng.integers(0, ub[i])
    return y


def _crossover(rng, a, b, pc):
    if rng.random() > pc:
        return a.copy(), b.copy()
    mask = rng.random(len(a)) < 0.5
    return np.where(mask, a, b), np.where(mask, b, a)


def local_search(prob, x, max_rounds=20):
    """Greedy discrete descent: single down-sizing and down/up-sizing swaps."""
    best = prob.evaluate(x)
    if not best['feasible']:
        return best
    n = len(x)
    for _ in range(max_rounds):
        improved = False
        cands = []
        for i in range(n):
            if best['x'][i] > 0:
                y = list(best['x']); y[i] -= 1
                cands.append(y)
        for i in range(n):
            for j in range(n):
                if i != j and best['x'][i] > 0 and best['x'][j] < prob.ub[j] - 1:
                    y = list(best['x']); y[i] -= 1; y[j] += 1
                    cands.append(y)
        for y in cands:
            r = prob.evaluate(y)
            if r['feasible'] and r['weight'] < best['weight'] - 1e-9:
                best = r
                improved = True
                break
        if not improved:
            break
    return best


def ga(prob, pop_size=40, n_gen=60, pc=0.9, seed=0, patience=15, polish=True, log=None):
    """Single-objective constrained GA with elitist (mu + lambda) replacement."""
    rng = np.random.default_rng(seed)
    ub = prob.ub
    n = len(ub)
    pm = 1.5 / n
    pop = [np.array([rng.integers(0, u) for u in ub]) for _ in range(pop_size)]
    evals = [prob.evaluate(p) for p in pop]
    history = []
    best = min(evals, key=_rank_key)
    stall = 0
    for gen in range(n_gen):
        children = []
        while len(children) < pop_size:
            i, j = rng.integers(0, pop_size, 2)
            a = evals[i] if _better(evals[i], evals[j]) else evals[j]
            i, j = rng.integers(0, pop_size, 2)
            b = evals[i] if _better(evals[i], evals[j]) else evals[j]
            c1, c2 = _crossover(rng, np.array(a['x']), np.array(b['x']), pc)
            children += [_mutate(rng, c1, ub, pm), _mutate(rng, c2, ub, pm)]
        ch_eval = [prob.evaluate(c) for c in children]
        merged = {e['x']: e for e in evals + ch_eval}
        evals = sorted(merged.values(), key=_rank_key)[:pop_size]
        while len(evals) < pop_size:   # refill if duplicates collapsed the population
            evals.append(prob.evaluate(np.array([rng.integers(0, u) for u in ub])))
        cur = evals[0]
        if _rank_key(cur) < _rank_key(best):
            best, stall = cur, 0
        else:
            stall += 1
        history.append({'gen': gen, 'best_weight': best['weight'], 'feasible': best['feasible'],
                        'n_eval': prob.n_eval,
                        'n_feasible': sum(e['feasible'] for e in evals)})
        if log:
            log(f'gen {gen:3d} best {best["weight"]:.0f} kg feas={best["feasible"]} evals={prob.n_eval}')
        if stall >= patience:
            break
    ga_best = best
    if polish:
        best = local_search(prob, list(best['x']))
    return {'best': best, 'ga_best': ga_best, 'history': history, 'n_eval': prob.n_eval}


# ------------------------------------------------------------------ NSGA-II
def _dominates(a, b):
    """Constrained domination (Deb et al. 2002) on objectives (weight, -beta)."""
    if a['feasible'] and not b['feasible']:
        return True
    if not a['feasible'] and b['feasible']:
        return False
    if not a['feasible']:
        return a['viol'] < b['viol']
    fa, fb = a['obj'], b['obj']
    return np.all(fa <= fb) and np.any(fa < fb)


def _fronts(pop):
    n = len(pop)
    S = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p != q:
                if _dominates(pop[p], pop[q]):
                    S[p].append(q)
                elif _dominates(pop[q], pop[p]):
                    cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(pop, idx):
    d = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        return {i: np.inf for i in idx}
    objs = np.array([pop[i]['obj'] for i in idx])
    for k in range(objs.shape[1]):
        order = np.argsort(objs[:, k])
        span = objs[order[-1], k] - objs[order[0], k] or 1.0
        d[idx[order[0]]] = d[idx[order[-1]]] = np.inf
        for m in range(1, len(idx) - 1):
            d[idx[order[m]]] += (objs[order[m + 1], k] - objs[order[m - 1], k]) / span
    return d


def nsga2(prob, pop_size=48, n_gen=60, pc=0.9, seed=0, beta_min=1.0, log=None, drift_objective=False):
    """NSGA-II: minimise weight and maximise beta_ULS subject to the SLS reliability
    and column-monotonicity constraints (ULS target not imposed). With
    ``drift_objective`` the drift reliability becomes a third objective (maximised)
    and its constraint is relaxed to beta_drift >= 0."""
    rng = np.random.default_rng(seed)
    ub = prob.ub
    n = len(ub)
    pm = 1.5 / n

    def ev(x):
        r = dict(prob.evaluate(x))
        g = r['g'].copy()
        g[0] = (beta_min - r['info']['beta_uls']) / beta_min  # only a floor on beta_ULS
        if drift_objective:
            g[1] = -r['info']['beta_drift']
        r['viol'] = float(np.sum(np.maximum(g, 0)))
        r['feasible'] = r['viol'] <= 0
        obj = [r['weight'], -r['info']['beta_uls']]
        if drift_objective:
            obj.append(-r['info']['beta_drift'])
        r['obj'] = np.array(obj)
        return r

    pop = [ev(np.array([rng.integers(0, u) for u in ub])) for _ in range(pop_size)]
    archive = {}

    def select(pop, rank, crowd):
        i, j = rng.integers(0, len(pop), 2)
        if rank[i] != rank[j]:
            return pop[i] if rank[i] < rank[j] else pop[j]
        return pop[i] if crowd[i] > crowd[j] else pop[j]

    for gen in range(n_gen):
        fr = _fronts(pop)
        rank = np.zeros(len(pop), int)
        crowd = np.zeros(len(pop))
        for k, f in enumerate(fr):
            cd = _crowding(pop, f)
            for i in f:
                rank[i] = k
                crowd[i] = cd[i]
        children = []
        while len(children) < pop_size:
            a, b = select(pop, rank, crowd), select(pop, rank, crowd)
            c1, c2 = _crossover(rng, np.array(a['x']), np.array(b['x']), pc)
            children += [ev(_mutate(rng, c1, ub, pm)), ev(_mutate(rng, c2, ub, pm))]
        merged = {p['x']: p for p in pop + children}
        allp = list(merged.values())
        fr = _fronts(allp)
        newpop = []
        for f in fr:
            if len(newpop) + len(f) <= pop_size:
                newpop += [allp[i] for i in f]
            else:
                cd = _crowding(allp, f)
                rest = sorted(f, key=lambda i: -cd[i])[:pop_size - len(newpop)]
                newpop += [allp[i] for i in rest]
                break
        while len(newpop) < pop_size:
            newpop.append(ev(np.array([rng.integers(0, u) for u in ub])))
        pop = newpop
        for p in pop:
            if p['feasible']:
                archive[p['x']] = p
        if log:
            nf = sum(p['feasible'] for p in pop)
            log(f'gen {gen:3d} feasible {nf} evals {prob.n_eval}')
    # non-dominated archive
    arch = list(archive.values())
    front = [a for a in arch if not any(_dominates(b, a) for b in arch if b is not a)]
    front.sort(key=lambda p: p['weight'])
    return {'front': front, 'archive': arch, 'n_eval': prob.n_eval}
