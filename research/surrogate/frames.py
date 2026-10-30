"""Parametric planar moment frames, PyNite analysis and frame statics.

Units: kN, m. Every frame lies in the global X-Y plane (Y vertical). Each
frame is described by a plain dict so that it can be pickled, regenerated and
fed to both the surrogate and the finite element (FE) solver.

Member force convention (used everywhere in this package)
--------------------------------------------------------
Each member e (i -> j) carries three *basic forces* q_e = [N, M_i, M_j]:
N is the axial force (tension positive) and M_i, M_j are the end moments
acting on the member, counter-clockwise positive (global Z). Together with a
statically admissible particular solution for the member load (simply
supported beam reactions) they define all six end forces, so element
equilibrium holds identically. Nodal equilibrium is then the linear
constraint  B q + p0 = P  on the free degrees of freedom.
"""

from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

# Sampling ranges shared by all scenarios (kN, m)
E_RANGE = (25e6, 210e6)           # concrete-like to steel-like Young's modulus (kN/m^2)
I_COL_RANGE = (5e-5, 2e-3)        # column second moment of area (m^4)
I_BEAM_RANGE = (5e-5, 2e-3)       # beam second moment of area (m^4)
R_GYR_RANGE = (0.08, 0.30)        # radius of gyration used to derive area A = I / r^2 (m)
BAY_RANGE = (3.0, 9.0)            # bay width (m)
STOREY_RANGE = (2.8, 4.5)         # storey height (m)
W_RANGE = (5.0, 40.0)             # beam gravity line load (kN/m)
F_RANGE = (5.0, 80.0)             # lateral force at the roof level (kN)


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def sample_frame(rng: np.random.Generator, scenario: str = 'train') -> dict:
    """Sample one frame definition.

    scenario:
      'train'  - in-distribution: 1-5 storeys, 1-4 bays, uniform gravity per
                 storey, uniform or inverted-triangular lateral profile.
      'oodG'   - unseen geometries: 6-8 storeys and/or 5-6 bays, first storey
                 up to 6 m tall (outside the training height range).
      'oodL'   - unseen loading: training geometries but bay-by-bay pattern
                 gravity loading (some bays unloaded) and parabolic or
                 roof-concentrated lateral profiles.
    """
    if scenario == 'oodG':
        if rng.random() < 0.5:
            n_st, n_bay = int(rng.integers(6, 9)), int(rng.integers(1, 7))
        else:
            n_st, n_bay = int(rng.integers(1, 9)), int(rng.integers(5, 7))
    else:
        n_st, n_bay = int(rng.integers(1, 6)), int(rng.integers(1, 5))

    bays = rng.uniform(*BAY_RANGE, n_bay)
    heights = rng.uniform(*STOREY_RANGE, n_st)
    if scenario == 'oodG' and rng.random() < 0.5:
        heights[0] = rng.uniform(4.5, 6.0)

    E = _logu(rng, *E_RANGE)
    # Section properties per storey; columns tend to get lighter going up
    taper = rng.uniform(0.5, 1.0)
    Ic0 = _logu(rng, *I_COL_RANGE)
    Ic = np.array([Ic0 * taper ** (s / max(n_st - 1, 1)) for s in range(n_st)])
    Ib = np.array([_logu(rng, *I_BEAM_RANGE) for _ in range(n_st)])
    rc = rng.uniform(*R_GYR_RANGE)
    rb = rng.uniform(*R_GYR_RANGE)
    Ac, Ab = Ic / rc ** 2, Ib / rb ** 2

    fixed_base = bool(rng.random() < 0.7)

    # Gravity loads: w[s, b] on the beam of bay b at floor s+1
    w_floor = rng.uniform(*W_RANGE, n_st)
    w = np.repeat(w_floor[:, None], n_bay, axis=1)
    if scenario == 'oodL':
        # Pattern (checkerboard-like random) live load: each bay keeps a dead
        # part and randomly receives or loses the live part
        dead = rng.uniform(0.3, 0.6, n_st)[:, None] * w
        mask = rng.random((n_st, n_bay)) < 0.5
        w = np.where(mask, w, dead)

    # Lateral loads at each floor (positive +X)
    elev = np.cumsum(heights)
    Froof = rng.uniform(*F_RANGE)
    if scenario == 'oodL':
        kind = rng.choice(['parabolic', 'roof'])
    else:
        kind = rng.choice(['uniform', 'triangular'])
    if kind == 'uniform':
        prof = np.ones(n_st)
    elif kind == 'triangular':
        prof = elev / elev[-1]
    elif kind == 'parabolic':
        prof = (elev / elev[-1]) ** 2
    else:  # roof
        prof = np.zeros(n_st)
        prof[-1] = 1.0
    F = Froof * prof
    if rng.random() < 0.5:
        F = -F

    return dict(n_st=n_st, n_bay=n_bay, bays=bays, heights=heights, E=E,
                Ic=Ic, Ib=Ib, Ac=Ac, Ab=Ab, fixed_base=fixed_base, w=w, F=F,
                profile=str(kind), scenario=scenario)


def sample_design_frame(rng: np.random.Generator) -> dict:
    """Design-driven steel moment frame with correlated sections and loads.

    Used only as an independent test set ('test_real'). Geometry follows
    common office-building proportions; gravity loads come from area loads on
    a tributary width (1.2 D + 1.0 L); wind forces come from a uniform pressure
    on the facade; beams are sized for an L/360 live-load deflection and for
    a simple-span moment demand, columns for an h/400 storey-drift target
    under wind (shear-building estimate with a 0.5 beam-flexibility factor)
    and change section every two storeys. Sections are clipped to the
    training ranges so that only the parameter *correlations* are new.
    """
    E = 200e6
    n_st, n_bay = int(rng.integers(1, 6)), int(rng.integers(1, 5))
    bays = rng.choice([6.0, 7.5, 9.0], n_bay)
    heights = np.full(n_st, rng.uniform(3.2, 3.8))
    heights[0] = rng.uniform(4.0, 4.5)
    trib = rng.uniform(4.0, 6.0)                     # tributary width of the frame (m)
    D, Lq = rng.uniform(3.5, 5.0), rng.uniform(2.0, 3.0)   # kPa
    w_floor = np.full(n_st, (1.2 * D + 1.0 * Lq) * trib)
    w_floor[-1] = (1.2 * (D - 1.0) + 1.0 * 1.0) * trib    # lighter roof
    w = np.repeat(w_floor[:, None], n_bay, axis=1)
    qw = rng.uniform(0.6, 1.2)                       # wind pressure (kPa)
    F = np.array([qw * trib * ((heights[s] + (heights[s + 1] if s + 1 < n_st else 0)) / 2) for s in range(n_st)])
    if rng.random() < 0.5:
        F = -F
    Lmax = bays.max()
    I_defl = 5 * Lq * trib * Lmax ** 4 / (384 * E * Lmax / 360)
    I_str = (w_floor.max() * Lmax ** 2 / 8) / 250e3 * 0.15            # M / fy * (d/2) with d ~ 0.3 m
    Ib = np.full(n_st, max(I_defl, I_str))
    ncol = n_bay + 1
    V = np.array([np.abs(F[s:]).sum() for s in range(n_st)])
    Ic = np.array([V[s] * heights[s] ** 3 / (0.5 * 12 * E * ncol * heights[s] / 400) for s in range(n_st)])
    for s in range(0, n_st, 2):                      # one section per two storeys
        Ic[s:s + 2] = Ic[s:s + 2].max()
    Ic = np.clip(np.maximum(Ic, 0.3 * Ib), I_COL_RANGE[0], I_COL_RANGE[1])
    Ib = np.clip(Ib, I_BEAM_RANGE[0], I_BEAM_RANGE[1])
    rc, rb = 0.16, 0.18
    return dict(n_st=n_st, n_bay=n_bay, bays=bays, heights=heights, E=E, Ic=Ic, Ib=Ib,
                Ac=Ic / rc ** 2, Ab=Ib / rb ** 2, fixed_base=bool(rng.random() < 0.8), w=w, F=F,
                profile='wind', scenario='real')


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def topology(p: dict) -> dict:
    """Node coordinates, member connectivity and loads in array form."""
    n_st, n_bay = p['n_st'], p['n_bay']
    xs = np.concatenate([[0.0], np.cumsum(p['bays'])])
    ys = np.concatenate([[0.0], np.cumsum(p['heights'])])
    ncol = n_bay + 1

    def nid(level, line):
        return level * ncol + line

    coords = np.array([[xs[c], ys[l]] for l in range(n_st + 1) for c in range(ncol)])
    level = np.array([l for l in range(n_st + 1) for c in range(ncol)])
    line = np.array([c for l in range(n_st + 1) for c in range(ncol)])

    conn, mtype, mstorey, E, A, I, wloc = [], [], [], [], [], [], []
    for s in range(n_st):           # columns (bottom -> top)
        for c in range(ncol):
            conn.append((nid(s, c), nid(s + 1, c)))
            mtype.append(0); mstorey.append(s)
            A.append(p['Ac'][s]); I.append(p['Ic'][s]); wloc.append(0.0)
    for s in range(n_st):           # beams (left -> right)
        for b in range(n_bay):
            conn.append((nid(s + 1, b), nid(s + 1, b + 1)))
            mtype.append(1); mstorey.append(s)
            A.append(p['Ab'][s]); I.append(p['Ib'][s]); wloc.append(-p['w'][s, b])
    conn = np.array(conn)
    n = len(coords)
    m = len(conn)
    E = np.full(m, p['E'])

    # Nodal loads [FX, FY, MZ]; lateral load applied at the left column line
    P = np.zeros((n, 3))
    for s in range(n_st):
        P[nid(s + 1, 0), 0] = p['F'][s]

    # Restraints: base nodes; True = restrained
    restr = np.zeros((n, 3), dtype=bool)
    restr[:ncol, 0] = True
    restr[:ncol, 1] = True
    restr[:ncol, 2] = p['fixed_base']

    d = coords[conn[:, 1]] - coords[conn[:, 0]]
    L = np.linalg.norm(d, axis=1)
    return dict(coords=coords, level=level, line=line, conn=conn,
                mtype=np.array(mtype), mstorey=np.array(mstorey), E=E,
                A=np.array(A), I=np.array(I), wg=np.array(wloc), L=L,
                c=d[:, 0] / L, s=d[:, 1] / L, P=P, restr=restr)


# ---------------------------------------------------------------------------
# Statics: equilibrium matrix
# ---------------------------------------------------------------------------

def member_end_forces_global(t: dict, q: np.ndarray) -> np.ndarray:
    """Global end forces acting on each member, shape (m, 6) = [Fx,Fy,Mz]_i,[...]_j."""
    N, Mi, Mj = q[:, 0], q[:, 1], q[:, 2]
    L, c, s = t['L'], t['c'], t['s']
    wy = t['wg'] * c          # local transverse component of the global-Y load
    Fxi, Fxj = -N, N
    Fyi = (Mi + Mj) / L - wy * L / 2
    Fyj = -(Mi + Mj) / L - wy * L / 2
    out = np.empty((len(L), 6))
    out[:, 0] = c * Fxi - s * Fyi
    out[:, 1] = s * Fxi + c * Fyi
    out[:, 2] = Mi
    out[:, 3] = c * Fxj - s * Fyj
    out[:, 4] = s * Fxj + c * Fyj
    out[:, 5] = Mj
    return out


def equilibrium_system(t: dict):
    """Return (B, p0) with sum of member end forces at node DOFs = B q + p0.

    B has shape (3n, 3m); q is flattened member-wise [N, Mi, Mj] per member.
    The map is affine in q, so it is built column by column from unit vectors.
    """
    n, m = len(t['coords']), len(t['conn'])
    p0 = _assemble(t, member_end_forces_global(t, np.zeros((m, 3))))
    B = np.zeros((3 * n, 3 * m))
    for k in range(3):
        q = np.zeros((m, 3)); q[:, k] = 1.0
        fe = member_end_forces_global(t, q) - member_end_forces_global(t, np.zeros((m, 3)))
        for e in range(m):
            i, j = t['conn'][e]
            B[3 * i:3 * i + 3, 3 * e + k] += fe[e, :3]
            B[3 * j:3 * j + 3, 3 * e + k] += fe[e, 3:]
    return B, p0


def _assemble(t, fe):
    n = len(t['coords'])
    out = np.zeros(3 * n)
    for e, (i, j) in enumerate(t['conn']):
        out[3 * i:3 * i + 3] += fe[e, :3]
        out[3 * j:3 * j + 3] += fe[e, 3:]
    return out


def equilibrium_residual(t: dict, q: np.ndarray, B=None, p0=None) -> np.ndarray:
    """Out-of-balance force vector at the free DOFs for basic forces q (m, 3)."""
    if B is None:
        B, p0 = equilibrium_system(t)
    free = ~t['restr'].ravel()
    return (B @ q.ravel() + p0 - t['P'].ravel())[free]


# ---------------------------------------------------------------------------
# PyNite analysis
# ---------------------------------------------------------------------------

def case_topology(t: dict, case: str) -> dict:
    """Topology restricted to one load case: 'W' (nodal loads) or 'G' (member loads)."""
    tc = dict(t)
    if case == 'W':
        tc['wg'] = np.zeros_like(t['wg'])
    elif case == 'G':
        tc['P'] = np.zeros_like(t['P'])
    return tc


def build_pynite(p: dict, t: dict | None = None):
    from Pynite import FEModel3D
    t = topology(p) if t is None else t
    mdl = FEModel3D()
    for k, (x, y) in enumerate(t['coords']):
        mdl.add_node(f'N{k}', float(x), float(y), 0.0)
    G = p['E'] / (2 * (1 + 0.3))
    mdl.add_material('Mat', p['E'], G, 0.3, 0.0)
    for e, (i, j) in enumerate(t['conn']):
        sec = f'S{e}'
        mdl.add_section(sec, float(t['A'][e]), float(t['I'][e]), float(t['I'][e]), float(t['I'][e]))
        mdl.add_member(f'M{e}', f'N{i}', f'N{j}', 'Mat', sec)
        if t['wg'][e] != 0.0:
            mdl.add_member_dist_load(f'M{e}', 'FY', float(t['wg'][e]), float(t['wg'][e]), case='G')
    for k in range(len(t['coords'])):
        r = t['restr'][k]
        # Planar frame: out-of-plane DOFs (DZ, RX, RY) are restrained everywhere
        mdl.def_support(f'N{k}', bool(r[0]), bool(r[1]), True, True, True, bool(r[2]))
        if t['P'][k, 0] != 0.0:
            mdl.add_node_load(f'N{k}', 'FX', float(t['P'][k, 0]), case='W')
        if t['P'][k, 1] != 0.0:
            mdl.add_node_load(f'N{k}', 'FY', float(t['P'][k, 1]), case='W')
    # Lateral (W) and gravity (G) load cases are solved as separate combinations
    mdl.add_load_combo('W', {'W': 1.0})
    mdl.add_load_combo('G', {'G': 1.0})
    return mdl


def run_fe(p: dict) -> dict:
    """Run a first-order linear PyNite analysis and extract the response.

    Returns, for the lateral (W) and gravity (G) load cases and their sum,
    nodal displacements u (n, 3) = [DX, DY, RZ] and member basic forces
    q (m, 3) = [N, Mi, Mj], plus the wall-clock time of the PyNite call chain
    (model build + analysis + result extraction).
    """
    t0 = time.perf_counter()
    t = topology(p)
    mdl = build_pynite(p, t)
    mdl.analyze_linear(check_stability=False, check_statics=False)
    n, m = len(t['coords']), len(t['conn'])
    out = {}
    for case in ('W', 'G'):
        u = np.zeros((n, 3))
        for k in range(n):
            nd = mdl.nodes[f'N{k}']
            u[k] = nd.DX[case], nd.DY[case], nd.RZ[case]
        fg = np.zeros((m, 6))
        for e in range(m):
            mem = mdl.members[f'M{e}']
            f_glob = (mem.T().T @ mem.f(case)).ravel()
            fg[e] = f_glob[[0, 1, 5, 6, 7, 11]]
        out['u_' + case] = u
        out['q_' + case] = basic_from_global(t, fg)
    out['t_fe'] = time.perf_counter() - t0
    out['u'] = out['u_W'] + out['u_G']
    out['q'] = out['q_W'] + out['q_G']
    return out


def basic_from_global(t: dict, fg: np.ndarray) -> np.ndarray:
    """Recover [N, Mi, Mj] from global end forces on each member."""
    c, s = t['c'], t['s']
    Fxj_loc = c * fg[:, 3] + s * fg[:, 4]
    return np.stack([Fxj_loc, fg[:, 2], fg[:, 5]], axis=1)


# ---------------------------------------------------------------------------
# Mechanics-informed reference scales and statics features
# ---------------------------------------------------------------------------

def scales(p: dict, t: dict) -> dict:
    """Reference quantities used to non-dimensionalise inputs and outputs.

    lam: load scale (kN) = total applied lateral + gravity load magnitude
    H:   reference length = mean storey height
    EI:  reference rigidity = E * mean column I
    Displacement scale  u0 = lam H^3 / EI, rotation scale u0 / H,
    force scale lam, moment scale lam H. For linear elasticity the
    dimensionless response is then exactly independent of lam and E.
    """
    lam = float(np.sum(np.abs(p['F'])) + np.sum(np.abs(t['wg']) * t['L']))
    H = float(np.mean(p['heights']))
    EI = float(p['E'] * np.mean(p['Ic']))
    return dict(lam=lam, H=H, EI=EI, u0=lam * H ** 3 / EI)
