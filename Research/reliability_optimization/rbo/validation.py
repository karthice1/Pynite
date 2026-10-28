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
