"""Parameterised planar steel moment frames analysed with PyNite.

A frame is defined by a :class:`FrameSpec` (topology, geometry and nominal loads)
and a design vector of integer indices into the W-shape catalogues. For each
design, PyNite solves five unit load cases once (first-order, linear elastic):

* ``D``  - nominal dead load (superimposed dead load plus member self-weight)
* ``L``  - nominal live load
* ``W``  - nominal (50-year) wind load, applied as storey forces
* ``ND`` - lateral floor forces equal to the dead-load gravity of each floor
* ``NL`` - lateral floor forces equal to the live-load gravity of each floor

``ND`` and ``NL`` scaled by an out-of-plumb angle reproduce the equivalent
lateral loads of an initial sway imperfection (and the AISC notional loads).
Because the analysis is linear and member stiffnesses scale uniformly with the
elastic modulus, any load combination, any random load intensity and any global
modulus factor can be evaluated exactly by superposition of these unit
responses; second-order effects are added with the AISC 360 Appendix 8 B1-B2
amplifiers (verified against PyNite's own P-Delta solver and OpenSees).
"""
from dataclasses import dataclass, field
import numpy as np
from Pynite import FEModel3D

from .sections import COLUMN_SHAPES, BEAM_SHAPES, RHO_STEEL

E_STEEL = 200e6      # kN/m2
G_STEEL = 77.2e6     # kN/m2
FY_NOM = 345e3       # kN/m2 (ASTM A992, Fy = 50 ksi)
GAMMA_STEEL = RHO_STEEL * 9.81e-3  # kN/m3
CASES = ('D', 'L', 'W', 'ND', 'NL')
N_BEAM_PTS = 11


@dataclass
class FrameSpec:
    """Geometry and nominal loading of a regular planar moment frame (kN, m)."""
    n_storeys: int = 6
    n_bays: int = 3
    bay_width: float = 8.0
    storey_height: float = 3.8
    ground_storey_height: float = 4.5
    trib_width: float = 7.5          # out-of-plane spacing of frames
    q_dead: float = 4.0              # kPa, floors (slab, finishes, services)
    q_live: float = 2.4              # kPa, floors (office)
    q_dead_roof: float = 3.0         # kPa
    q_live_roof: float = 1.0         # kPa
    p_wind: float = 1.2              # kPa, net (windward + leeward) nominal pressure
    group_size: int = 2              # storeys sharing the same sections
    beam_unbraced_frac: float = 1/3  # beam Lb / span (secondary beams at third points)
    name: str = 'frame'

    @property
    def heights(self):
        return np.array([self.ground_storey_height] + [self.storey_height] * (self.n_storeys - 1))

    @property
    def levels(self):
        return np.concatenate([[0.0], np.cumsum(self.heights)])

    @property
    def total_height(self):
        return float(self.levels[-1])

    @property
    def n_groups(self):
        return int(np.ceil(self.n_storeys / self.group_size))

    @property
    def n_vars(self):
        # exterior column, interior column, beam per storey group
        return 3 * self.n_groups

    def var_bounds(self):
        """Upper index bound (exclusive) of each design variable."""
        ub = []
        for _ in range(self.n_groups):
            ub += [len(COLUMN_SHAPES), len(COLUMN_SHAPES), len(BEAM_SHAPES)]
        return np.array(ub)

    def group_of_storey(self, s):
        return s // self.group_size

    def describe_design(self, x):
        out = []
        for g in range(self.n_groups):
            s0 = g * self.group_size + 1
            s1 = min((g + 1) * self.group_size, self.n_storeys)
            out.append({'storeys': f'{s0}-{s1}' if s1 > s0 else f'{s0}',
                        'ext_col': COLUMN_SHAPES[x[3 * g]].name,
                        'int_col': COLUMN_SHAPES[x[3 * g + 1]].name,
                        'beam': BEAM_SHAPES[x[3 * g + 2]].name})
        return out


@dataclass
class Member:
    name: str
    kind: str            # 'col' or 'beam'
    storey: int          # columns: storey index (0 = ground); beams: floor index (0 = level 1)
    L: float
    shape: object
    exterior: bool = False
    i_node: str = ''
    j_node: str = ''


@dataclass
class UnitResponse:
    """Unit-case responses of one design (all arrays ordered as CASES)."""
    spec: FrameSpec
    x: tuple
    members: list
    weight: float                       # kg
    col_idx: np.ndarray                 # indices of columns in members
    beam_idx: np.ndarray
    # columns
    col_M: np.ndarray = None            # (5, ncol, 2) end moments (i, j)
    col_P: np.ndarray = None            # (5, ncol) axial (compression +)
    col_V: np.ndarray = None            # (5, ncol)
    # beams
    beam_M: np.ndarray = None           # (5, nbeam, N_BEAM_PTS)
    beam_P: np.ndarray = None           # (5, nbeam)
    beam_V: np.ndarray = None           # (5, nbeam, 2)
    beam_defl_L: np.ndarray = None      # (nbeam,) mid-span deflection rel. to chord, L case
    # storeys
    drift: np.ndarray = None            # (5, nst) inter-storey drift (floor-average DX)
    storey_P: np.ndarray = None         # (5, nst) total column axial load in storey
    storey_K: np.ndarray = None         # (nst,) first-order lateral stiffness from W case
    floor_gravity: dict = field(default_factory=dict)


def build_model(spec, x, E=E_STEEL, member_E=None, member_prop_factor=None):
    """Create the PyNite model of design ``x`` with the five unit load cases.

    ``member_E`` / ``member_prop_factor`` (dicts keyed by member name) allow
    member-by-member randomisation of stiffness and section properties, used
    when validating the superposition surrogate by full re-analysis.
    """
    m = FEModel3D()
    nb, ns = spec.n_bays, spec.n_storeys
    lv = spec.levels
    for j in range(ns + 1):
        for i in range(nb + 1):
            n = f'N{i}_{j}'
            m.add_node(n, i * spec.bay_width, lv[j], 0.0)
            if j == 0:
                m.def_support(n, True, True, True, True, True, True)
            else:
                m.def_support(n, False, False, True, True, True, False)

    members = []
    used = {}

    def add(name, kind, storey, i_node, j_node, shape, exterior=False):
        Ef = E * (member_E.get(name, 1.0) if member_E else 1.0)
        pf = member_prop_factor.get(name, 1.0) if member_prop_factor else 1.0
        key = (shape.name, round(Ef, 3), round(pf, 6))
        if key not in used:
            mat = f'S{len(used)}'
            m.add_material(mat, Ef, G_STEEL * Ef / E, 0.3, GAMMA_STEEL)
            m.add_section(mat, shape.A * pf, shape.Iy * pf, shape.Ix * pf, shape.J * pf)
            used[key] = mat
        mat = used[key]
        m.add_member(name, i_node, j_node, mat, mat)
        L = float(np.hypot(m.nodes[j_node].X - m.nodes[i_node].X, m.nodes[j_node].Y - m.nodes[i_node].Y))
        members.append(Member(name, kind, storey, L, shape, exterior, i_node, j_node))

    for s in range(ns):
        g = spec.group_of_storey(s)
        ext, inn, bm = COLUMN_SHAPES[x[3 * g]], COLUMN_SHAPES[x[3 * g + 1]], BEAM_SHAPES[x[3 * g + 2]]
        for i in range(nb + 1):
            exterior = i in (0, nb)
            add(f'C{i}_{s}', 'col', s, f'N{i}_{s}', f'N{i}_{s + 1}', ext if exterior else inn, exterior)
        for i in range(nb):
            add(f'B{i}_{s}', 'beam', s, f'N{i}_{s + 1}', f'N{i + 1}_{s + 1}', bm)

    # --- loads -------------------------------------------------------------
    t = spec.trib_width
    span = nb * spec.bay_width
    floor_D, floor_L = np.zeros(ns), np.zeros(ns)
    for mem in members:
        if mem.kind == 'beam':
            roof = mem.storey == ns - 1
            qd = (spec.q_dead_roof if roof else spec.q_dead) * t
            ql = (spec.q_live_roof if roof else spec.q_live) * t
            m.add_member_dist_load(mem.name, 'FY', -qd, -qd, case='D')
            m.add_member_dist_load(mem.name, 'FY', -ql, -ql, case='L')
            floor_D[mem.storey] += qd * mem.L + GAMMA_STEEL * mem.shape.A * mem.L
            floor_L[mem.storey] += ql * mem.L
        else:
            w = GAMMA_STEEL * mem.shape.A * mem.L
            floor_D[mem.storey] += w / 2           # top half to the floor above
            if mem.storey > 0:
                floor_D[mem.storey - 1] += w / 2   # bottom half to the floor below
    m.add_member_self_weight('FY', -1.0, case='D')

    h = spec.heights
    wind = np.array([spec.p_wind * t * (h[j] / 2 + (h[j + 1] / 2 if j + 1 < ns else 0.0))
                     for j in range(ns)])
    for j in range(ns):
        for i in range(nb + 1):
            n = f'N{i}_{j + 1}'
            m.add_node_load(n, 'FX', wind[j] / (nb + 1), case='W')
            m.add_node_load(n, 'FX', floor_D[j] / (nb + 1), case='ND')
            m.add_node_load(n, 'FX', floor_L[j] / (nb + 1), case='NL')
    for c in CASES:
        m.add_load_combo(c, {c: 1.0})
    info = {'floor_D': floor_D, 'floor_L': floor_L, 'wind': wind, 'span': span}
    return m, members, info


def analyse(spec, x, model_out=False, **kw):
    """Solve the five unit cases for design ``x`` and extract the responses."""
    x = tuple(int(v) for v in x)
    m, members, info = build_model(spec, x, **kw)
    m.analyze_linear(check_stability=False)
    ns, nb = spec.n_storeys, spec.n_bays
    col_idx = np.array([k for k, mm in enumerate(members) if mm.kind == 'col'])
    beam_idx = np.array([k for k, mm in enumerate(members) if mm.kind == 'beam'])
    weight = sum(mm.shape.mass_per_m * mm.L for mm in members)
    r = UnitResponse(spec, x, members, weight, col_idx, beam_idx, floor_gravity=info)

    nc, nbm = len(col_idx), len(beam_idx)
    r.col_M = np.zeros((5, nc, 2)); r.col_P = np.zeros((5, nc)); r.col_V = np.zeros((5, nc))
    r.beam_M = np.zeros((5, nbm, N_BEAM_PTS)); r.beam_P = np.zeros((5, nbm)); r.beam_V = np.zeros((5, nbm, 2))
    r.beam_defl_L = np.zeros(nbm)
    r.drift = np.zeros((5, ns)); r.storey_P = np.zeros((5, ns))
    for c, case in enumerate(CASES):
        for k, ci in enumerate(col_idx):
            mem = m.members[members[ci].name]
            f = mem.f(case).ravel()
            # local end forces: [Fx, Fy, Fz, Mx, My, Mz]_i, [...]_j ; moments as internal
            r.col_M[c, k] = [f[5], -f[11]]
            r.col_P[c, k] = f[0]
            r.col_V[c, k] = f[1]
            r.storey_P[c, members[ci].storey] += f[0]
        for k, bi in enumerate(beam_idx):
            mem = m.members[members[bi].name]
            L = members[bi].L
            xs = np.linspace(0, L, N_BEAM_PTS)
            r.beam_M[c, k] = mem.moment_array('Mz', N_BEAM_PTS, case, x_array=xs)[1]
            r.beam_P[c, k] = mem.axial(L / 2, case)
            f = mem.f(case).ravel()
            r.beam_V[c, k] = [f[1], -f[7]]
            if case == 'L':
                r.beam_defl_L[k] = mem.rel_deflection('dy', L / 2, case)
        dx = np.array([np.mean([m.nodes[f'N{i}_{j}'].DX[case] for i in range(nb + 1)])
                       for j in range(ns + 1)])
        r.drift[c] = np.diff(dx)
    shear_W = np.cumsum(info['wind'][::-1])[::-1]
    r.storey_K = shear_W / r.drift[CASES.index('W')]
    if model_out:
        return r, m
    return r
