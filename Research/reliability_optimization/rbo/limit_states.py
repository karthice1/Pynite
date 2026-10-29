"""Member resistances (AISC 360-16), second-order demands and limit-state checks.

The code basis is fixed (CODE_BASIS): designs are checked against these editions only, not
against later ones such as AISC 360-22 or ASCE/SEI 7-22.

All functions are vectorised over an arbitrary number of "realisations" (rows):
a realisation is either a factored LRFD load combination (deterministic design)
or one Monte Carlo sample of the random variables (reliability analysis).
"""
import numpy as np

CODE_BASIS = {'resistance': 'AISC 360-16', 'loads': 'ASCE/SEI 7-05'}

from .frames import E_STEEL, FY_NOM, CASES

iD, iL, iW, iND, iNL = (CASES.index(c) for c in CASES)
R_M = 0.85            # AISC App. 8, all columns part of moment frames
DRIFT_LIMIT = 1 / 400
DEFL_LIMIT = 1 / 360


class MemberTable:
    """Per-member geometric data stacked into arrays (columns first, then beams)."""

    def __init__(self, r):
        spec = r.spec
        mem = [r.members[i] for i in r.col_idx] + [r.members[i] for i in r.beam_idx]
        self.ncol = len(r.col_idx)
        self.nbeam = len(r.beam_idx)
        g = lambda f: np.array([f(m) for m in mem], dtype=float)
        self.A = g(lambda m: m.shape.A)
        self.Ix = g(lambda m: m.shape.Ix)
        self.Zx = g(lambda m: m.shape.Zx)
        self.Sx = g(lambda m: m.shape.Sx)
        self.rx = g(lambda m: m.shape.rx)
        self.ry = g(lambda m: m.shape.ry)
        self.rts = g(lambda m: m.shape.rts)
        self.ho = g(lambda m: m.shape.ho)
        self.J = g(lambda m: m.shape.J)
        self.lam_f = g(lambda m: m.shape.lam_f)
        self.Aw = g(lambda m: m.shape.Aw)
        self.h_tw = g(lambda m: (m.shape.d - 2 * m.shape.tf) / m.shape.tw)
        self.L = g(lambda m: m.L)
        # unbraced length for flexure/weak-axis buckling: columns braced at floors,
        # beams braced by secondary beams
        self.Lb = g(lambda m: m.L if m.kind == 'col' else m.L * spec.beam_unbraced_frac)
        self.names = [m.name for m in mem]
        self.storey_col = np.array([m.storey for m in mem[:self.ncol]])
        self.floor_beam = np.array([m.storey for m in mem[self.ncol:]])


# ---------------------------------------------------------------- resistances
def axial_capacity(t, Fy, E, fab):
    """Nominal compressive (AISC E3, K = 1) and tensile yield strengths."""
    slender = np.maximum(t.L / t.rx, t.Lb / t.ry)
    Fe = np.pi ** 2 * E / slender ** 2
    ratio = Fy / Fe
    Fcr = np.where(ratio <= 2.25, 0.658 ** ratio * Fy, 0.877 * Fe)
    return Fcr * t.A * fab, Fy * t.A * fab


def flexural_capacity(t, Fy, E, fab, Cb=1.0):
    """Nominal strong-axis flexural strength: AISC F2 (LTB) and F3 (flange local buckling)."""
    Mp = Fy * t.Zx * fab
    Sx = t.Sx * fab
    Lp = 1.76 * t.ry * np.sqrt(E / Fy)
    jc = t.J / (t.Sx * t.ho)
    Lr = 1.95 * t.rts * E / (0.7 * Fy) * np.sqrt(jc + np.sqrt(jc ** 2 + 6.76 * (0.7 * Fy / E) ** 2))
    Lb = t.Lb
    inel = Cb * (Mp - (Mp - 0.7 * Fy * Sx) * (Lb - Lp) / (Lr - Lp))
    Fcr = Cb * np.pi ** 2 * E / (Lb / t.rts) ** 2 * np.sqrt(1 + 0.078 * jc * (Lb / t.rts) ** 2)
    M_ltb = np.where(Lb <= Lp, Mp, np.where(Lb <= Lr, inel, Fcr * Sx))
    lam_p = 0.38 * np.sqrt(E / Fy)
    lam_r = 1.0 * np.sqrt(E / Fy)
    M_flb = np.where(t.lam_f <= lam_p, Mp,
                     Mp - (Mp - 0.7 * Fy * Sx) * (t.lam_f - lam_p) / (lam_r - lam_p))
    return np.minimum(np.minimum(M_ltb, M_flb), Mp)


def shear_capacity(t, Fy, E, fab):
    """Nominal shear strength (AISC G2.1) and the associated resistance factor."""
    lim1 = 2.24 * np.sqrt(E / Fy)
    lim2 = 1.10 * np.sqrt(5.34 * E / Fy)
    Cv1 = np.where(t.h_tw <= lim2, 1.0, lim2 / t.h_tw)
    phi = np.where(t.h_tw <= lim1, 1.0, 0.9)
    return 0.6 * Fy * t.Aw * Cv1 * fab, phi


def interaction(Pr, Pc, Mr, Mc):
    """AISC H1-1 interaction ratio."""
    pr = Pr / Pc
    return np.where(pr >= 0.2, pr + 8 / 9 * Mr / Mc, pr / 2 + Mr / Mc)


# ------------------------------------------------------------------ demands
def second_order_demands(r, t, d, l, w, th, e, tau=1.0):
    """B1-B2 amplified member demands for realisations with load intensities
    d, l, w (multiples of nominal D, L, W), sway angle th (rad) and modulus factor
    e. ``tau`` is the stiffness reduction (0.8 for the Direct Analysis Method).

    Returns a dict with Pr, Mr, Vr arrays of shape (n, nmembers) (compression
    positive), B2 (n, nstoreys), first-order-amplified drifts (n, nstoreys).
    """
    d, l, w, th, e = (np.asarray(v, float)[:, None] for v in (d, l, w, th, e))
    nst = r.spec.n_storeys
    h = r.spec.heights

    # storey sway amplifiers
    Pstory = d * r.storey_P[iD] + l * r.storey_P[iL]
    Pe_story = R_M * r.storey_K[None, :] * e * tau * h[None, :]
    rat = Pstory / Pe_story
    with np.errstate(divide='ignore'):
        B2 = np.where(rat < 1, 1 / (1 - rat), np.inf)

    z = np.zeros_like(d)
    C_nt = np.hstack([d, l, z, z, z]).astype(np.float32)
    C_lt = np.hstack([z, z, w, th * d, th * l]).astype(np.float32)

    def lin(arr):
        """Combine unit-case array (5, m[, k]) into non-sway and sway parts (n, m[, k]).

        Single precision is ample for the demand combination (relative error ~1e-7)."""
        flat = arr.reshape(5, -1).astype(np.float32)
        shp = (-1,) + arr.shape[1:]
        return (C_nt @ flat).reshape(shp), (C_lt @ flat).reshape(shp)

    # ---- columns
    sc = t.storey_col
    B2c = B2[:, sc]
    Mnt, Mlt = lin(r.col_M)                         # (n, ncol, 2)
    Pnt, Plt = lin(r.col_P)
    Vnt, Vlt = lin(r.col_V)
    Pr_c = Pnt + B2c * Plt
    a, b = Mnt[..., 0], Mnt[..., 1]
    big = np.where(np.abs(a) >= np.abs(b), a, b)
    small = np.where(np.abs(a) >= np.abs(b), b, a)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(big != 0, -small / big, 0.0)   # + for reverse curvature
    Cm = 0.6 - 0.4 * ratio
    EIc = E_STEEL * e * tau * t.Ix[None, :t.ncol]
    Pe1 = np.pi ** 2 * EIc / t.L[None, :t.ncol] ** 2
    with np.errstate(divide='ignore'):
        B1 = np.where(Pr_c < Pe1, Cm / (1 - np.maximum(Pr_c, 0) / Pe1), np.inf)
    B1 = np.maximum(B1, 1.0)
    Mr_c = np.max(np.abs(B1[..., None] * Mnt + B2c[..., None] * Mlt), axis=-1)
    Vr_c = np.abs(Vnt + B2c * Vlt)

    # ---- beams (B2 of the storeys below/above the floor)
    fb = t.floor_beam
    B2b = np.maximum(B2[:, fb], B2[:, np.minimum(fb + 1, nst - 1)]).astype(np.float32)
    Mnt, Mlt = lin(r.beam_M)
    Mr_b = np.max(np.abs(Mnt + B2b[..., None] * Mlt), axis=-1)
    Pnt, Plt = lin(r.beam_P)
    Pr_b = Pnt + B2b * Plt
    Vnt, Vlt = lin(r.beam_V)
    Vr_b = np.max(np.abs(Vnt + B2b[..., None] * Vlt), axis=-1)

    # ---- drifts (actual stiffness, no tau)
    dnt, dlt = lin(r.drift)
    drift = np.abs(dnt + B2 * dlt) / e / h[None, :]

    return {'Pr': np.concatenate([Pr_c, Pr_b], 1), 'Mr': np.concatenate([Mr_c, Mr_b], 1),
            'Vr': np.concatenate([Vr_c, Vr_b], 1), 'B2': B2, 'drift_ratio': drift,
            'B1': B1}


def capacities(t, Fy, E, fab):
    """Nominal (Pn_c, Pn_t, Mn, Vn, phi_v) for each realisation and member."""
    Pn_c, Pn_t = axial_capacity(t, Fy, E, fab)
    Mn = flexural_capacity(t, Fy, E, fab)
    Vn, phi_v = shear_capacity(t, Fy, E, fab)
    return Pn_c, Pn_t, Mn, Vn, phi_v


def utilisation(t, dem, Fy, E, fab, phi_c=1.0, phi_b=1.0, phi_t=1.0, use_phi_v=False, cap=None):
    """Member utilisation = max(H1-1 interaction, shear ratio), shape (n, nm).

    ``cap`` may pass precomputed :func:`capacities` (reused across load scenarios)."""
    Pn_c, Pn_t, Mn, Vn, phi_v = cap if cap is not None else capacities(t, Fy, E, fab)
    if not use_phi_v:
        phi_v = 1.0
    Pr, Mr, Vr = dem['Pr'], dem['Mr'], dem['Vr']
    Pc = np.where(Pr >= 0, phi_c * Pn_c, phi_t * Pn_t)
    U = interaction(np.abs(Pr), Pc, Mr, phi_b * Mn)
    Us = Vr / (phi_v * Vn)
    U = np.maximum(U, Us)
    # dynamic instability of a storey -> collapse
    U = np.where(np.isfinite(U), U, np.inf)
    return U, {'P_ratio': np.abs(Pr) / Pc, 'shear': Us}


# ------------------------------------------------------- deterministic design
def lrfd_combos(stability_notional=0.002, tau_notional=0.001):
    """ASCE 7-05 strength combinations (wind in both directions) and DAM notional
    loads (AISC 360-16 C2.2b, C2.3(c) with tau_b = 1).

    Returns arrays d, l, w, th (theta = notional-load ratio)."""
    rows = []
    for sgn in (1, -1):
        Nf = sgn * (stability_notional + tau_notional)
        Nw = sgn * tau_notional
        rows += [(1.4, 0.0, 0.0, Nf),
                 (1.2, 1.6, 0.0, Nf),
                 (1.2, 0.5, 1.6 * sgn, Nw),
                 (0.9, 0.0, 1.6 * sgn, Nw)]
    return np.array(rows).T


def deterministic_check(r, t=None, drift_limit=DRIFT_LIMIT):
    """LRFD strength (DAM, B1-B2) and serviceability checks of a design.

    Returns a dict with the governing strength ratio, drift ratio / limit and
    beam live-load deflection / limit (all must be <= 1)."""
    t = t or MemberTable(r)
    d, l, w, th = lrfd_combos()
    n = len(d)
    dem = second_order_demands(r, t, d, l, w, th, np.ones(n), tau=0.8)
    Fy = np.full((n, len(t.A)), FY_NOM)
    U, extra = utilisation(t, dem, Fy, E_STEEL, 1.0, phi_c=0.9, phi_b=0.9, phi_t=0.9, use_phi_v=True)
    strength = float(np.max(U))
    # serviceability: drift under D + 0.5L + W (nominal) with actual stiffness
    sd = second_order_demands(r, t, [1.0, 1.0], [0.5, 0.5], [1.0, -1.0], [0.0, 0.0], [1.0, 1.0], tau=1.0)
    drift = float(np.max(sd['drift_ratio'])) / drift_limit
    span = np.array([r.members[i].L for i in r.beam_idx])
    defl = float(np.max(np.abs(r.beam_defl_L) / span)) / DEFL_LIMIT
    return {'strength': strength, 'drift': drift, 'deflection': defl,
            'max_B2': float(np.max(dem['B2'])),
            'beam_P_ratio': float(np.max(extra['P_ratio'][:, t.ncol:])),
            'U_members': np.max(U, axis=0)}
