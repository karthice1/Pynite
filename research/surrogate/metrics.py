"""Engineering error metrics, computed per frame in physical units.

Definitions (hat = surrogate, no hat = PyNite reference; ||.|| = Euclidean
norm over all entries of the frame):

* e_dx, e_dy, e_rz: relative L2 errors of horizontal and vertical nodal
  displacements and nodal rotations, ||a_hat - a|| / ||a||.
* e_drift: |max_s d_s_hat - max_s d_s| / max_s d_s, with d_s the largest
  inter-storey drift ratio of storey s.
* e_defl: the same relative error for the largest beam mid-span deflection
  (relative to the beam chord, Hermite interpolation of the end rotations
  plus the fixed-end particular solution w L^4 / 384 EI).
* e_M, e_N, e_V: relative L2 errors of all member end moments, axial forces
  and end shears.
* e_R: relative L2 error of the support reactions.
* e_Mmax, e_Nmax: relative error of the largest absolute end moment / axial
  force in the frame.
* gov_top1 / gov_top3: the member with the largest end moment in the
  reference is the predicted largest / among the predicted three largest.
* eq_res: ||r|| / ||P - p0|| on the free DOFs, with r = B q + p0 - P and
  moment rows divided by the mean storey height H.
* compat / compat_M: ||q - q_c(u_hat)|| / ||q_c(u_hat)||, where q_c(u_hat)
  are the basic forces implied by the predicted displacement field through
  the element stiffness relations; moments are divided by H, compat_M uses
  the end moments only.
"""

from __future__ import annotations

import numpy as np

import solvers as S
from features import physical, total_p0


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def _row_scale(g):
    t = g['t']
    return np.tile([1.0, 1.0, g['H']], len(t['coords']))


def equilibrium_residual_phys(g, q):
    t = g['t']
    free = ~t['restr'].ravel()
    rs = _row_scale(g)[free]
    P = t['P'].ravel()
    p0 = total_p0(g)
    r = (g['B'] @ q.ravel() + p0 - P)[free] / rs
    ref = (P - p0)[free] / rs
    return float(np.linalg.norm(r) / (np.linalg.norm(ref) + 1e-300))


def equilibrium_residual(g, q_dimless):
    """Relative out-of-balance of the total predicted member forces."""
    q = sum(q_dimless[:, 3 * k:3 * k + 3] * s for k, s in enumerate(g['qsc']))
    return equilibrium_residual_phys(g, q)


def drift_ratios(g, u):
    """Maximum inter-storey drift ratio per storey (absolute), from DX."""
    t = g['t']
    lev = t['level']
    y = np.unique(t['coords'][:, 1])
    out = []
    for s in range(1, lev.max() + 1):
        dx_top = u[lev == s, 0]; dx_bot = u[lev == s - 1, 0]
        out.append(np.max(np.abs(dx_top - dx_bot)) / (y[s] - y[s - 1]))
    return np.array(out)


def beam_midspan_deflection(g, u):
    """Mid-span deflection of each beam relative to its chord (local y)."""
    t = g['t']
    b = t['mtype'] == 1
    i, j = t['conn'][b, 0], t['conn'][b, 1]
    L = t['L'][b]
    psi = (u[j, 1] - u[i, 1]) / L
    wy = t['wg'][b] * t['c'][b]
    return L / 8 * ((u[i, 2] - psi) - (u[j, 2] - psi)) + wy * L ** 4 / (384 * t['E'][b] * t['I'][b])


def end_shears(g, q):
    t = g['t']
    wy = t['wg'] * t['c']
    s = (q[:, 1] + q[:, 2]) / t['L']
    return np.concatenate([s - wy * t['L'] / 2, -s - wy * t['L'] / 2])


def reactions(g, q):
    t = g['t']
    R = g['B'] @ q.ravel() + total_p0(g) - t['P'].ravel()
    return R[t['restr'].ravel()]


def mats(g):
    if '_mats' not in g:
        g['_mats'] = S.element_matrices(g['t'])
    return g['_mats']


def forces_from_displacements(g, u):
    """Basic forces implied by a (total) displacement field."""
    return S.basic_from_local(S.member_local_forces(g['t'], u, mats(g)))


def compat_residual(g, u_hat, q_hat):
    qc = forces_from_displacements(g, u_hat)
    sc = np.array([1.0, 1 / g['H'], 1 / g['H']])
    full = _rel(q_hat * sc, qc * sc)
    mom = _rel(q_hat[:, 1:], qc[:, 1:])
    return full, mom


def frame_metrics_phys(g, u_hat, q_hat, compat=True):
    """Errors of one frame for physical total predictions u_hat (n, 3), q_hat (m, 3)."""
    u, q = g['u_true'], g['q_true']
    dr_hat, dr = drift_ratios(g, u_hat), drift_ratios(g, u)
    Mhat, M = q_hat[:, 1:].ravel(), q[:, 1:].ravel()
    dm_hat, dm = np.abs(q_hat[:, 1:]).max(1), np.abs(q[:, 1:]).max(1)
    out = dict(
        e_dx=_rel(u_hat[:, 0], u[:, 0]),
        e_dy=_rel(u_hat[:, 1], u[:, 1]),
        e_rz=_rel(u_hat[:, 2], u[:, 2]),
        e_drift=abs(dr_hat.max() - dr.max()) / dr.max(),
        e_M=_rel(Mhat, M),
        e_N=_rel(q_hat[:, 0], q[:, 0]),
        e_V=_rel(end_shears(g, q_hat), end_shears(g, q)),
        e_R=_rel(reactions(g, q_hat), reactions(g, q)),
        e_Mmax=abs(dm_hat.max() - dm.max()) / dm.max(),
        e_Nmax=abs(np.abs(q_hat[:, 0]).max() - np.abs(q[:, 0]).max()) / np.abs(q[:, 0]).max(),
        gov_top1=float(np.argmax(dm_hat) == np.argmax(dm)),
        gov_top3=float(np.argmax(dm) in np.argsort(-dm_hat)[:3]),
        eq_res=equilibrium_residual_phys(g, q_hat),
    )
    if (g['t']['mtype'] == 1).any():
        df_hat, df = beam_midspan_deflection(g, u_hat), beam_midspan_deflection(g, u)
        out['e_defl'] = abs(np.abs(df_hat).max() - np.abs(df).max()) / np.abs(df).max()
    if compat:
        out['compat'], out['compat_M'] = compat_residual(g, u_hat, q_hat)
    return out


def frame_metrics(g, pred_u, pred_q, compat=True):
    """Errors of one frame. Inputs are dimensionless predictions."""
    return frame_metrics_phys(g, *physical(g, pred_u, pred_q), compat=compat)


def summarise(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k + '_mean'] = float(v.mean())
        out[k + '_med'] = float(np.median(v))
        out[k + '_p95'] = float(np.percentile(v, 95))
    return out
