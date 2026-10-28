"""Probabilistic models and Monte Carlo reliability analysis of a frame design.

Load combination follows Turkstra's rule with two scenarios over the 50-year
reference period:

* S1 (gravity):  D + L_50 (+ sway imperfection)
* S2 (wind):     D + L_apt + W_50 (+ sway imperfection)

The system ultimate limit state (ULS) fails when any member in either scenario
violates the AISC H1-1 interaction (or shear) using nominal resistances evaluated
with the random yield stress, modulus and fabrication factor, or when a storey
becomes unstable (B2 -> infinity). Two serviceability limit states (SLS) are
tracked: inter-storey drift > h/400 in S2 and beam live-load deflection
> span/360 under L_50.

Sample average approximation: an optimisation run uses a fixed set of samples
(common random numbers), which makes the reliability constraint a deterministic
function of the design; final designs are re-verified with independent samples.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from scipy import stats

from .frames import E_STEEL, FY_NOM
from .limit_states import (MemberTable, second_order_demands, utilisation, capacities,
                           DRIFT_LIMIT, DEFL_LIMIT)

EULER = 0.5772156649015329


@dataclass(frozen=True)
class ProbModel:
    """Statistics of the basic random variables (bias = mean / nominal)."""
    D_bias: float = 1.05
    D_cov: float = 0.10
    L50_bias: float = 1.00
    L50_cov: float = 0.25
    Lapt_bias: float = 0.24
    Lapt_cov: float = 0.60
    W50_bias: float = 0.78
    W50_cov: float = 0.37
    E_cov: float = 0.03
    Fy_bias: float = 1.10
    Fy_cov: float = 0.07
    Fy_corr: float = 0.5        # equicorrelation of ln(Fy) between members
    fab_cov: float = 0.05
    sway_sd: float = 1 / 1000   # rad, initial out-of-plumbness (mean zero)
    # model (epistemic) uncertainty, frame-wide lognormal factors with unit mean;
    # zero by default, i.e. the indices are conditional on the structural model
    thR_cov: float = 0.0        # resistance model (AISC H1-1 / E3 / F2 / G2 equations)
    thE_cov: float = 0.0        # load-effect model (second-order elastic analysis)
    thD_cov: float = 0.0        # drift model

    def table(self):
        return [
            ('Dead load $D$', 'Normal', self.D_bias, self.D_cov),
            ('50-yr max. live load $L_{50}$', 'Gumbel', self.L50_bias, self.L50_cov),
            ('Arbitrary-point-in-time live load $L_{apt}$', 'Gamma', self.Lapt_bias, self.Lapt_cov),
            ('50-yr max. wind load $W_{50}$', 'Gumbel', self.W50_bias, self.W50_cov),
            ('Elastic modulus $E$', 'Lognormal', 1.0, self.E_cov),
            ('Yield stress $F_y$ (per member)', 'Lognormal', self.Fy_bias, self.Fy_cov),
            ('Fabrication factor $F$ (per member)', 'Normal', 1.0, self.fab_cov),
            ('Out-of-plumbness $\\theta$ (rad)', 'Normal', 0.0, self.sway_sd),
        ]

    @property
    def has_model_uncertainty(self):
        return max(self.thR_cov, self.thE_cov, self.thD_cov) > 0


def _gumbel(rng, mean, cov, n):
    if cov == 0:
        return np.full(n, mean)
    scale = mean * cov * np.sqrt(6) / np.pi
    return rng.gumbel(mean - EULER * scale, scale, n)


def _lognormal(z, mean, cov):
    s = np.sqrt(np.log(1 + cov ** 2))
    return mean * np.exp(s * z - s ** 2 / 2)


@dataclass
class Samples:
    d: np.ndarray
    l50: np.ndarray
    lapt: np.ndarray
    w50: np.ndarray
    e: np.ndarray
    fy: np.ndarray       # (n, nmembers) factor on FY_NOM
    fab: np.ndarray      # (n, nmembers)
    th: np.ndarray
    thU: np.ndarray = None   # model-uncertainty factor on utilisation, theta_E / theta_R
    thD: np.ndarray = None   # model-uncertainty factor on drift

    @property
    def n(self):
        return len(self.d)

    def chunk(self, a, b):
        return Samples(*(None if v is None else v[a:b]
                         for v in (self.d, self.l50, self.lapt, self.w50, self.e,
                                   self.fy, self.fab, self.th, self.thU, self.thD)))


def draw(pm, n, n_members, seed):
    rng = np.random.default_rng(seed)
    d = pm.D_bias * (1 + pm.D_cov * rng.standard_normal(n))
    l50 = _gumbel(rng, pm.L50_bias, pm.L50_cov, n)
    if pm.Lapt_cov > 0:
        k = 1 / pm.Lapt_cov ** 2
        lapt = rng.gamma(k, pm.Lapt_bias / k, n)
    else:
        lapt = np.full(n, pm.Lapt_bias)
    w50 = _gumbel(rng, pm.W50_bias, pm.W50_cov, n)
    e = _lognormal(rng.standard_normal(n), 1.0, pm.E_cov)
    z0 = rng.standard_normal((n, 1))
    zi = rng.standard_normal((n, n_members))
    zf = np.sqrt(pm.Fy_corr) * z0 + np.sqrt(1 - pm.Fy_corr) * zi
    fy = _lognormal(zf, pm.Fy_bias, pm.Fy_cov)
    fab = 1 + pm.fab_cov * rng.standard_normal((n, n_members))
    th = pm.sway_sd * rng.standard_normal(n)
    thU = thD = None
    if pm.has_model_uncertainty:
        # separate stream, so the physical variables are identical with and without it
        rm = np.random.default_rng([seed, 1])
        thR = _lognormal(rm.standard_normal(n), 1.0, pm.thR_cov)
        thE = _lognormal(rm.standard_normal(n), 1.0, pm.thE_cov)
        thU = thE / thR
        thD = _lognormal(rm.standard_normal(n), 1.0, pm.thD_cov)
    return Samples(d, l50, lapt, w50, e, fy, fab, th, thU, thD)


def evaluate_samples(r, s, t=None, chunk=20000, detail=False):
    """Limit-state indicators for every sample.

    Returns dict of arrays: U (max system utilisation over both scenarios), U1 and
    U2 (per scenario), drift (max drift ratio / limit in S2), defl (max beam
    deflection / limit under L50). With ``detail`` also the per-member utilisations
    of the governing scenario. H1-1 and the shear check are homogeneous of degree
    one in the demands and of degree minus one in the resistances, so the
    model-uncertainty factors act on U exactly as theta_E / theta_R.
    """
    t = t or MemberTable(r)
    span = np.array([r.members[i].L for i in r.beam_idx])
    defl_unit = np.max(np.abs(r.beam_defl_L) / span) / DEFL_LIMIT
    U_all, U1_all, U2_all, drift_all, Umem = [], [], [], [], []
    for a in range(0, s.n, chunk):
        c = s.chunk(a, a + chunk)
        E = E_STEEL * c.e[:, None]
        Fy = FY_NOM * c.fy
        cap = capacities(t, Fy, E, c.fab)
        dem1 = second_order_demands(r, t, c.d, c.l50, np.zeros(c.n), c.th, c.e)
        U1, _ = utilisation(t, dem1, Fy, E, c.fab, cap=cap)
        dem2 = second_order_demands(r, t, c.d, c.lapt, c.w50, c.th, c.e)
        U2, _ = utilisation(t, dem2, Fy, E, c.fab, cap=cap)
        if c.thU is not None:
            U1, U2 = U1 * c.thU[:, None], U2 * c.thU[:, None]
        Um = np.maximum(U1, U2)
        U_all.append(np.max(Um, axis=1))
        U1_all.append(np.max(U1, axis=1))
        U2_all.append(np.max(U2, axis=1))
        dr = np.max(dem2['drift_ratio'], axis=1) / DRIFT_LIMIT
        drift_all.append(dr if c.thD is None else dr * c.thD)
        if detail:
            Umem.append(Um.astype(np.float32))
    out = {'U': np.concatenate(U_all), 'U1': np.concatenate(U1_all), 'U2': np.concatenate(U2_all),
           'drift': np.concatenate(drift_all), 'defl': defl_unit * s.l50 / s.e}
    if detail:
        out['U_members'] = np.concatenate(Umem)
    return out


def beta_from_pf(pf):
    return -stats.norm.ppf(pf)


def clopper_pearson(k, N, conf=0.95):
    """Exact two-sided binomial confidence interval (lo, hi) of a probability."""
    a = 1 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a / 2, k, N - k + 1))
    hi = 1.0 if k == N else float(stats.beta.ppf(1 - a / 2, k + 1, N - k))
    return lo, hi


def pf_upper(k, N, conf=0.95):
    """Exact one-sided upper confidence bound of a probability (k failures in N)."""
    return 1.0 if k == N else float(stats.beta.ppf(conf, k + 1, N - k))


def estimate(ind, n=None, conf=0.95):
    """Failure probabilities, reliability indices, MC c.o.v. and confidence intervals.

    The point estimate is k / N. Only to keep a finite number when no failure is
    observed, k = 0 is replaced by 0.5 in ``pf_*``/``beta_*``; this is a reporting
    convention, not a bound. The statistically meaningful quantities are the exact
    (Clopper-Pearson) interval ``ci_pf_*``, the corresponding interval of beta
    ``ci_beta_*`` and the one-sided lower confidence bound ``beta_lcb_*`` (from the
    one-sided upper bound of pf at the same confidence level)."""
    res = {}
    for key, fail in (('uls', ind['U'] >= 1.0), ('drift', ind['drift'] >= 1.0),
                      ('defl', ind['defl'] >= 1.0)):
        N = len(fail)
        k = int(np.sum(fail))
        pf = max(k, 0.5) / N
        res[f'pf_{key}'] = pf
        res[f'beta_{key}'] = float(beta_from_pf(pf))
        res[f'nfail_{key}'] = k
        res[f'n_{key}'] = N
        res[f'cov_{key}'] = float(np.sqrt((1 - pf) / (N * pf)))
        lo, hi = clopper_pearson(k, N, conf)
        res[f'ci_pf_{key}'] = [lo, hi]
        res[f'ci_beta_{key}'] = [float(beta_from_pf(hi)), float(beta_from_pf(lo)) if lo > 0 else float('inf')]
        res[f'beta_lcb_{key}'] = float(beta_from_pf(pf_upper(k, N, conf)))
    sls = (ind['drift'] >= 1.0) | (ind['defl'] >= 1.0)
    pf = max(int(np.sum(sls)), 0.5) / len(sls)
    res['pf_sls'] = pf
    res['beta_sls'] = float(beta_from_pf(pf))
    return res
