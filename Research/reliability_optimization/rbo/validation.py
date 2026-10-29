"""Verification of the PyNite models and of the reliability surrogate.

1. Closed-form benchmarks (cantilever, fixed-fixed beam, fixed-base portal frame,
   second-order cantilever beam-column).
2. Independent solver: the same frames rebuilt in OpenSeesPy (linear and P-Delta).
3. Surrogate check: Monte Carlo with full PyNite P-Delta re-analysis per sample
   (member-by-member random stiffness and section properties) versus the
   superposition + B1-B2 surrogate used inside the optimiser.
"""
import numpy as np
from Pynite import FEModel3D

from .frames import build_model, analyse, CASES, E_STEEL, FY_NOM, GAMMA_STEEL, N_BEAM_PTS
from .limit_states import MemberTable, second_order_demands, utilisation, DRIFT_LIMIT
from .reliability import draw


# ------------------------------------------------------------ closed form
def _plane_supports(m, free_nodes):
    for n in free_nodes:
        m.def_support(n, False, False, True, True, True, False)


def bench_cantilever(L=4.0, P=50.0, E=E_STEEL, I=1e-4, A=1e-2):
    m = FEModel3D()
    m.add_node('A', 0, 0, 0); m.add_node('B', L, 0, 0)
    m.def_support('A', True, True, True, True, True, True)
    _plane_supports(m, ['B'])
    m.add_material('S', E, 0.385 * E, 0.3, 0)
    m.add_section('s', A, I, I, I)
    m.add_member('M', 'A', 'B', 'S', 's')
    m.add_node_load('B', 'FY', -P)
    m.analyze_linear()
    return [('Cantilever tip deflection', abs(m.nodes['B'].DY['Combo 1']), P * L ** 3 / (3 * E * I), 'm'),
            ('Cantilever fixed-end moment', abs(m.members['M'].moment('Mz', 0)), P * L, 'kN m')]


def bench_fixed_beam(L=8.0, w=40.0, E=E_STEEL, I=2e-4, A=1e-2, n_el=4):
    m = FEModel3D()
    for k in range(n_el + 1):
        m.add_node(f'N{k}', k * L / n_el, 0, 0)
    m.def_support('N0', True, True, True, True, True, True)
    m.def_support(f'N{n_el}', True, True, True, True, True, True)
    _plane_supports(m, [f'N{k}' for k in range(1, n_el)])
    m.add_material('S', E, 0.385 * E, 0.3, 0)
    m.add_section('s', A, I, I, I)
    for k in range(n_el):
        m.add_member(f'M{k}', f'N{k}', f'N{k + 1}', 'S', 's')
        m.add_member_dist_load(f'M{k}', 'FY', -w, -w)
    m.analyze_linear()
    mid = f'N{n_el // 2}'
    return [('Fixed beam end moment', abs(m.members['M0'].moment('Mz', 0)), w * L ** 2 / 12, 'kN m'),
            ('Fixed beam mid-span moment', abs(m.members[f'M{n_el // 2}'].moment('Mz', 0)), w * L ** 2 / 24, 'kN m'),
            ('Fixed beam mid-span deflection', abs(m.nodes[mid].DY['Combo 1']), w * L ** 4 / (384 * E * I), 'm')]


def portal_closed_form(H, h, L, E, Ic, Ib):
    """Fixed-base portal, lateral load H at beam level, inextensible members
    (slope-deflection solution)."""
    k = (Ib / L) / (Ic / h)
    delta = H * h ** 3 * (2 + 3 * k) / (12 * E * Ic * (1 + 6 * k))
    M_base = H * h * (1 + 3 * k) / (2 * (1 + 6 * k))
    M_top = H * h * 3 * k / (2 * (1 + 6 * k))
    return delta, M_base, M_top


def bench_portal(H=100.0, h=4.0, L=8.0, E=E_STEEL, Ic=1.5e-4, Ib=3e-4, A=10.0):
    m = FEModel3D()
    m.add_node('A', 0, 0, 0); m.add_node('B', 0, h, 0); m.add_node('C', L, h, 0); m.add_node('D', L, 0, 0)
    for n in 'AD':
        m.def_support(n, True, True, True, True, True, True)
    _plane_supports(m, ['B', 'C'])
    m.add_material('S', E, 0.385 * E, 0.3, 0)
    m.add_section('c', A, Ic, Ic, Ic)
    m.add_section('b', A, Ib, Ib, Ib)
    m.add_member('C1', 'A', 'B', 'S', 'c'); m.add_member('BM', 'B', 'C', 'S', 'b'); m.add_member('C2', 'D', 'C', 'S', 'c')
    m.add_node_load('B', 'FX', H)
    m.analyze_linear()
    d, Mb, Mt = portal_closed_form(H, h, L, E, Ic, Ib)
    return [('Portal sway', m.nodes['B'].DX['Combo 1'], d, 'm'),
            ('Portal column base moment', abs(m.members['C1'].moment('Mz', 0)), Mb, 'kN m'),
            ('Portal column top moment', abs(m.members['C1'].moment('Mz', h)), Mt, 'kN m')]


def beam_column_exact(P, H, L, E, I):
    """Exact second-order tip deflection of a cantilever under axial P and tip H."""
    k = np.sqrt(P / (E * I))
    return H / (P * k) * (np.tan(k * L) - k * L)


def bench_pdelta_cantilever(L=5.0, E=E_STEEL, I=1e-4, A=1e-2, H=10.0, P_ratios=(0.2, 0.4, 0.6),
                            n_els=(1, 2, 4, 8)):
    """Second-order cantilever: PyNite P-Delta with n sub-elements vs exact."""
    Pcr = np.pi ** 2 * E * I / (4 * L ** 2)
    rows = []
    for pr in P_ratios:
        P = pr * Pcr
        exact = beam_column_exact(P, H, L, E, I)
        first = H * L ** 3 / (3 * E * I)
        row = {'P/Pcr': pr, 'exact': exact, 'first_order': first}
        for n in n_els:
            m = FEModel3D()
            for k in range(n + 1):
                m.add_node(f'N{k}', 0, k * L / n, 0)
            m.def_support('N0', True, True, True, True, True, True)
            _plane_supports(m, [f'N{k}' for k in range(1, n + 1)])
            m.add_material('S', E, 0.385 * E, 0.3, 0)
            m.add_section('s', A, I, I, I)
            for k in range(n):
                m.add_member(f'M{k}', f'N{k}', f'N{k + 1}', 'S', 's')
            m.add_node_load(f'N{n}', 'FX', H)
            m.add_node_load(f'N{n}', 'FY', -P)
            m.analyze_PDelta()
            row[f'pynite_{n}'] = m.nodes[f'N{n}'].DX['Combo 1']
        # AISC B2 amplifier on the first-order sway (R_M = 1 for a single cantilever)
        row['B2'] = first / (1 - P / (3 * E * I / L ** 3 * L))
        rows.append(row)
    return rows


def run_closed_form():
    rows = bench_cantilever() + bench_fixed_beam() + bench_portal()
    return [{'case': c, 'pynite': float(p), 'analytical': float(a), 'unit': u,
             'rel_err': float(abs(p - a) / abs(a))} for c, p, a, u in rows]


# --------------------------------------------------------------- OpenSees
def opensees_unit_cases(spec, x, pdelta_combo=None):
    """Rebuild design ``x`` in OpenSeesPy and solve each unit case (linear).

    If ``pdelta_combo`` (dict case -> factor) is given, also solve that combination
    with the P-Delta geometric transformation (load-controlled Newton)."""
    import openseespy.opensees as ops
    _, members, info = build_model(spec, x)
    nb, ns = spec.n_bays, spec.n_storeys
    lv = spec.levels
    tag = lambda i, j: 1000 * j + i + 1
    floor_nodes = lambda j: [tag(i, j) for i in range(nb + 1)]

    def model(transf):
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        for j in range(ns + 1):
            for i in range(nb + 1):
                ops.node(tag(i, j), i * spec.bay_width, lv[j])
                if j == 0:
                    ops.fix(tag(i, j), 1, 1, 1)
        ops.geomTransf(transf, 1)
        for k, mm in enumerate(members):
            ni = [int(v) for v in mm.i_node[1:].split('_')]
            nj = [int(v) for v in mm.j_node[1:].split('_')]
            ops.element('elasticBeamColumn', k + 1, tag(*ni), tag(*nj), mm.shape.A, E_STEEL, mm.shape.Ix, 1)

    def apply(case, factor, pat):
        ops.timeSeries('Constant', pat)
        ops.pattern('Plain', pat, pat)
        t = spec.trib_width
        if case in ('D', 'L'):
            for k, mm in enumerate(members):
                sw = GAMMA_STEEL * mm.shape.A
                if mm.kind == 'beam':
                    roof = mm.storey == ns - 1
                    if case == 'D':
                        q = (spec.q_dead_roof if roof else spec.q_dead) * t + sw
                    else:
                        q = (spec.q_live_roof if roof else spec.q_live) * t
                    ops.eleLoad('-ele', k + 1, '-type', '-beamUniform', -q * factor, 0.0)
                elif case == 'D':
                    ops.eleLoad('-ele', k + 1, '-type', '-beamUniform', 0.0, -sw * factor)
        else:
            key = {'W': 'wind', 'ND': 'floor_D', 'NL': 'floor_L'}[case]
            for j in range(ns):
                for n in floor_nodes(j + 1):
                    ops.load(n, info[key][j] / (nb + 1) * factor, 0.0, 0.0)

    def solve(nonlinear=False):
        ops.system('BandGeneral'); ops.numberer('RCM'); ops.constraints('Plain')
        if nonlinear:
            ops.test('NormDispIncr', 1e-10, 50)
            ops.algorithm('Newton')
            ops.integrator('LoadControl', 0.1)
            ops.analysis('Static')
            return ops.analyze(10)
        ops.algorithm('Linear'); ops.integrator('LoadControl', 1.0); ops.analysis('Static')
        return ops.analyze(1)

    def extract():
        cols = [k for k, mm in enumerate(members) if mm.kind == 'col']
        beams = [k for k, mm in enumerate(members) if mm.kind == 'beam']
        colM = np.array([[ops.eleResponse(k + 1, 'localForce')[2], -ops.eleResponse(k + 1, 'localForce')[5]] for k in cols])
        colP = np.array([ops.eleResponse(k + 1, 'localForce')[0] for k in cols])
        beamM = []
        for k in beams:
            f = ops.eleResponse(k + 1, 'localForce')
            beamM.append([f[2], -f[5]])
        dx = np.array([np.mean([ops.nodeDisp(n, 1) for n in floor_nodes(j)]) for j in range(ns + 1)])
        return {'col_M': colM, 'col_P': colP, 'beam_end_M': np.array(beamM), 'drift': np.diff(dx)}

    out = {}
    for case in CASES:
        model('Linear')
        apply(case, 1.0, 1)
        solve()
        out[case] = extract()
    if pdelta_combo:
        model('PDelta')
        for p, (case, fac) in enumerate(pdelta_combo.items()):
            apply(case, fac, p + 1)
        ok = solve(nonlinear=True)
        out['pdelta'] = extract()
        out['pdelta']['ok'] = ok
    ops.wipe()
    return out


def compare_opensees(spec, x, combo):
    """Relative differences PyNite vs OpenSees for unit cases and a P-Delta combo."""
    r = analyse(spec, x)
    os_ = opensees_unit_cases(spec, x, pdelta_combo=combo)
    rows = []

    def rel(a, b):
        a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
        scale = np.max(np.abs(b))
        return float(np.max(np.abs(a - b)) / scale) if scale > 0 else 0.0

    for c, case in enumerate(CASES):
        o = os_[case]
        rows.append({'case': case,
                     'drift': rel(r.drift[c], o['drift']),
                     'col_M': rel(r.col_M[c], o['col_M']),
                     'col_P': rel(r.col_P[c], o['col_P']),
                     'beam_M': rel(r.beam_M[c][:, [0, -1]], o['beam_end_M'])})
    # P-Delta: PyNite second-order solver and B1-B2 surrogate vs OpenSees
    m, _, _ = build_model(spec, x)
    m.add_load_combo('PD', combo)
    m.analyze_PDelta(check_stability=False, combo_tags=None)
    ns, nb = spec.n_storeys, spec.n_bays
    dx = np.array([np.mean([m.nodes[f'N{i}_{j}'].DX['PD'] for i in range(nb + 1)]) for j in range(ns + 1)])
    pn_drift = np.diff(dx)
    cols = [mm for mm in r.members if mm.kind == 'col']
    pn_colM = np.array([[m.members[mm.name].f('PD').ravel()[5], -m.members[mm.name].f('PD').ravel()[11]] for mm in cols])
    t = MemberTable(r)
    d, l, w = combo.get('D', 0), combo.get('L', 0), combo.get('W', 0)
    th = combo.get('ND', 0) / d if d else 0.0
    dem = second_order_demands(r, t, [d], [l], [w], [th], [1.0])
    surrogate_drift = dem['drift_ratio'][0] * spec.heights
    first_drift = np.abs(d * r.drift[0] + l * r.drift[1] + w * r.drift[2] + th * (d * r.drift[3] + l * r.drift[4]))
    o = os_['pdelta']
    pd = {'opensees_drift': np.abs(o['drift']), 'pynite_drift': np.abs(pn_drift),
          'b2_drift': surrogate_drift, 'first_order_drift': first_drift,
          'opensees_base_M': np.abs(o['col_M'][:nb + 1, 0]), 'pynite_base_M': np.abs(pn_colM[:nb + 1, 0]),
          'b2_base_M': dem['Mr'][0, :nb + 1],
          'converged': o['ok'] == 0}
    return rows, pd


# --------------------------------------------------- full re-analysis MC
def full_reanalysis_mc(spec, x, pm, n, seed=7, wind_scale=1.0, member_E_cov=0.03, log=None):
    """Scenario-S2 Monte Carlo with a PyNite P-Delta re-analysis per sample.

    Each member receives its own modulus factor and section property (fabrication)
    factor, so the uniform-modulus and nominal-stiffness assumptions of the
    surrogate are removed. Returns per-sample (U, drift) from the full analysis and
    from the surrogate evaluated on the same samples (mean member modulus factor).
    """
    r = analyse(spec, x)
    t = MemberTable(r)
    s = draw(pm, n, len(t.A), seed)
    rng = np.random.default_rng(seed + 1)
    names = t.names
    s_E = np.exp(np.sqrt(np.log(1 + member_E_cov ** 2)) * rng.standard_normal((n, len(names)))
                 - np.log(1 + member_E_cov ** 2) / 2)
    nb, ns = spec.n_bays, spec.n_storeys
    U_full, D_full = np.zeros(n), np.zeros(n)
    Pr = np.zeros((n, len(names))); Mr = np.zeros_like(Pr); Vr = np.zeros_like(Pr)
    for k in range(n):
        mE = dict(zip(names, s_E[k]))
        mP = dict(zip(names, s.fab[k]))
        m, members, info = build_model(spec, x, member_E=mE, member_prop_factor=mP)
        combo = {'D': s.d[k], 'L': s.lapt[k], 'W': s.w50[k] * wind_scale,
                 'ND': s.th[k] * s.d[k], 'NL': s.th[k] * s.lapt[k]}
        m.add_load_combo('S2', combo)
        m.analyze_PDelta(check_stability=False, combo_tags=None)
        for q, nm in enumerate(names):
            mem = m.members[nm]
            L = mem.L()
            xs = np.linspace(0, L, N_BEAM_PTS)
            Mr[k, q] = np.max(np.abs(mem.moment_array('Mz', N_BEAM_PTS, 'S2', x_array=xs)[1]))
            f = mem.f('S2').ravel()
            Pr[k, q] = f[0] if nm.startswith('C') else mem.axial(L / 2, 'S2')
            Vr[k, q] = max(abs(f[1]), abs(f[7]))
        dx = np.array([np.mean([m.nodes[f'N{i}_{j}'].DX['S2'] for i in range(nb + 1)]) for j in range(ns + 1)])
        D_full[k] = np.max(np.abs(np.diff(dx)) / spec.heights) / DRIFT_LIMIT
        if log and k % 50 == 0:
            log(f'full MC sample {k}/{n}')
    # resistances use each member's own modulus (buckling, LTB)
    E_mem = E_STEEL * s_E
    U_full, _ = utilisation(t, {'Pr': Pr, 'Mr': Mr, 'Vr': Vr}, FY_NOM * s.fy, E_mem, s.fab)
    U_full = np.max(U_full, axis=1)
    e_bar = np.mean(s_E, axis=1)
    dem = second_order_demands(r, t, s.d, s.lapt, s.w50 * wind_scale, s.th, e_bar)
    U_sur, _ = utilisation(t, dem, FY_NOM * s.fy, E_mem, s.fab)
    U_sur = np.max(U_sur, axis=1)
    D_sur = np.max(dem['drift_ratio'], axis=1) / DRIFT_LIMIT
    return {'U_full': U_full, 'U_sur': U_sur, 'D_full': D_full, 'D_sur': D_sur}


# ------------------------------------------- broad, tail-focused validation
def _member_E_factors(rng, n, nm, cov):
    s = np.sqrt(np.log(1 + cov ** 2))
    return np.exp(s * rng.standard_normal((n, nm)) - s ** 2 / 2)


def _full_sample(spec, x, t, s, s_E, k, scenario, wind_scale):
    """PyNite P-Delta re-analysis of sample k in scenario S1 (D + L50) or S2
    (D + Lapt + W50), each member with its own modulus and property factor.
    Returns member demands (Pr, Mr, Vr) and the maximum drift ratio / limit."""
    names = t.names
    m, _, _ = build_model(spec, x, member_E=dict(zip(names, s_E[k])),
                          member_prop_factor=dict(zip(names, s.fab[k])))
    l = s.l50[k] if scenario == 'S1' else s.lapt[k]
    w = 0.0 if scenario == 'S1' else s.w50[k] * wind_scale
    m.add_load_combo('C', {'D': s.d[k], 'L': l, 'W': w, 'ND': s.th[k] * s.d[k], 'NL': s.th[k] * l})
    m.analyze_PDelta(check_stability=False, combo_tags=None)
    nm = len(names)
    Pr, Mr, Vr = np.zeros(nm), np.zeros(nm), np.zeros(nm)
    for q, nme in enumerate(names):
        mem = m.members[nme]
        L = mem.L()
        xs = np.linspace(0, L, N_BEAM_PTS)
        Mr[q] = np.max(np.abs(mem.moment_array('Mz', N_BEAM_PTS, 'C', x_array=xs)[1]))
        f = mem.f('C').ravel()
        Pr[q] = f[0] if nme.startswith('C') else mem.axial(L / 2, 'C')
        Vr[q] = max(abs(f[1]), abs(f[7]))
    nb, ns = spec.n_bays, spec.n_storeys
    dx = np.array([np.mean([m.nodes[f'N{i}_{j}'].DX['C'] for i in range(nb + 1)]) for j in range(ns + 1)])
    drift = np.max(np.abs(np.diff(dx)) / spec.heights) / DRIFT_LIMIT
    return Pr, Mr, Vr, drift


def broad_validation(spec, x, pm, n_random=600, n_pool=200_000, n_tail=250, u_tail=0.85,
                     seed=2024, member_E_cov=0.03, log=None):
    """Surrogate vs full P-Delta re-analysis for one design.

    * Random set: ``n_random`` samples of scenario S2 at the nominal load model.
      Compares U and drift and the drift-SLS failure classification at the actual
      SLS reliability level of the design.
    * Tail set: from a pool of ``n_pool`` samples, the surrogate selects the
      stratum T = {U_sur >= u_tail}; ``n_tail`` samples of T are drawn at random
      and both scenarios are re-analysed (a scenario only when its surrogate
      range) in two strata of the pool: A = {U_sur >= 1} (surrogate failures) and
      B = {u_tail <= U_sur < 1} (near misses). With the stratum probabilities
      known from the pool, the system ULS failure probability of the full model is
      estimated as P(A) f_A + P(B) f_B, f = fraction of full-model failures in the
      stratum sample; failures with U_sur < u_tail are excluded by the observed
      ratio U_full / U_sur.
    """
    from .reliability import evaluate_samples, Samples
    from .limit_states import capacities
    r = analyse(spec, x)
    t = MemberTable(r)
    nm = len(t.A)
    rng = np.random.default_rng([seed, 7])
    s = draw(pm, n_pool, nm, seed)
    s_E = _member_E_factors(rng, n_pool, nm, member_E_cov)
    s_bar = Samples(s.d, s.l50, s.lapt, s.w50, np.mean(s_E, axis=1), s.fy, s.fab, s.th)
    ind = evaluate_samples(r, s_bar, t)   # surrogate, frame-wide modulus = member mean
    out = {'weight': r.weight, 'n_pool': n_pool, 'u_tail': u_tail,
           'pf_sur_pool': float(np.mean(ind['U'] >= 1)), 'nfail_sur_pool': int(np.sum(ind['U'] >= 1))}

    def utilisation_full(idx, dem):
        E_mem = E_STEEL * s_E[idx]
        U, _ = utilisation(t, dem, FY_NOM * s.fy[idx], E_mem, s.fab[idx])
        return np.max(U, axis=1)

    # ---- random set (scenario S2)
    idx_r = np.arange(n_random)
    res = [_full_sample(spec, x, t, s, s_E, k, 'S2', 1.0) for k in idx_r]
    dem = {'Pr': np.array([q[0] for q in res]), 'Mr': np.array([q[1] for q in res]),
           'Vr': np.array([q[2] for q in res])}
    out['random'] = {'U_full': utilisation_full(idx_r, dem), 'U_sur': ind['U2'][idx_r],
                     'D_full': np.array([q[3] for q in res]), 'D_sur': ind['drift'][idx_r]}
    if log:
        log(f'random set done ({n_random})')
    # ---- tail strata: A = surrogate failures, B = near misses u_tail <= U_sur < 1
    out['tail'] = {}
    for name, sel in (('A', ind['U'] >= 1.0), ('B', (ind['U'] >= u_tail) & (ind['U'] < 1.0))):
        T = np.flatnonzero(sel)
        pick = np.sort(rng.choice(T, size=min(n_tail, len(T)), replace=False))
        U_full = np.zeros(len(pick))
        n_an = 0
        for q, k in enumerate(pick):
            best = 0.0
            gov = 'S1' if ind['U1'][k] >= ind['U2'][k] else 'S2'
            for sc, us in (('S1', ind['U1'][k]), ('S2', ind['U2'][k])):
                if us >= u_tail or sc == gov:
                    Pr, Mr, Vr, _ = _full_sample(spec, x, t, s, s_E, k, sc, 1.0)
                    dem = {'Pr': Pr[None], 'Mr': Mr[None], 'Vr': Vr[None]}
                    best = max(best, float(utilisation_full([k], dem)[0]))
                    n_an += 1
            U_full[q] = best
            if log and q % 50 == 0:
                log(f'stratum {name} sample {q}/{len(pick)}')
        out['tail'][name] = {'n_T': int(len(T)), 'p_T': len(T) / n_pool, 'idx': pick, 'U_full': U_full,
                             'U_sur': ind['U'][pick], 'gov_S1': ind['U1'][pick] >= ind['U2'][pick],
                             'n_analyses': n_an}
    return out


def excluded_stratum(spec, x, pm, n_pool=200_000, u_low=0.70, u_tail=0.85, n_C=300, seed=2024,
                     member_E_cov=0.03, log=None):
    """Check of the stratum that ``broad_validation`` excludes: C = {u_low <= U_sur < u_tail}.

    The pool, the member modulus factors and the surrogate utilisations are regenerated
    exactly as in ``broad_validation`` (same seeds); ``n_C`` samples of C are drawn with a
    separate stream and re-analysed with full P-Delta. Samples with U_sur < u_low (stratum
    D) are not analysed; their contribution is bounded separately from the observed
    distribution of U_full / U_sur (see make_figures.val_rows)."""
    from .reliability import evaluate_samples, Samples
    r = analyse(spec, x)
    t = MemberTable(r)
    nm = len(t.A)
    rng = np.random.default_rng([seed, 7])
    s = draw(pm, n_pool, nm, seed)
    s_E = _member_E_factors(rng, n_pool, nm, member_E_cov)
    s_bar = Samples(s.d, s.l50, s.lapt, s.w50, np.mean(s_E, axis=1), s.fy, s.fab, s.th)
    ind = evaluate_samples(r, s_bar, t)
    rng_c = np.random.default_rng([seed, 8])
    T = np.flatnonzero((ind['U'] >= u_low) & (ind['U'] < u_tail))
    pick = np.sort(rng_c.choice(T, size=min(n_C, len(T)), replace=False))
    U_full = np.zeros(len(pick))
    n_an = 0
    for q, k in enumerate(pick):
        best = 0.0
        gov = 'S1' if ind['U1'][k] >= ind['U2'][k] else 'S2'
        for sc, us in (('S1', ind['U1'][k]), ('S2', ind['U2'][k])):
            if us >= u_low or sc == gov:
                Pr, Mr, Vr, _ = _full_sample(spec, x, t, s, s_E, k, sc, 1.0)
                Um, _ = utilisation(t, {'Pr': Pr[None], 'Mr': Mr[None], 'Vr': Vr[None]},
                                    FY_NOM * s.fy[[k]], E_STEEL * s_E[[k]], s.fab[[k]])
                best = max(best, float(np.max(Um)))
                n_an += 1
        U_full[q] = best
        if log and q % 50 == 0:
            log(f'stratum C sample {q}/{len(pick)}')
    # pool histogram of U_sur below u_tail (for the residual-risk bound of the unanalysed samples)
    edges = np.round(np.arange(0.0, u_tail + 1e-9, 0.005), 3)
    counts, _ = np.histogram(ind['U'][ind['U'] < u_tail], bins=edges)
    return {'hist_edges': edges, 'hist_counts': counts,
            'u_low': u_low, 'u_tail': u_tail, 'n_pool': n_pool,
            'n_T': int(len(T)), 'p_T': len(T) / n_pool, 'n_D': int(np.sum(ind['U'] < u_low)),
            'idx': pick, 'U_full': U_full, 'U_sur': ind['U'][pick], 'n_analyses': n_an}
