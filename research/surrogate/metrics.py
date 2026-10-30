"""Engineering error metrics, computed per frame in physical units."""

from __future__ import annotations

import numpy as np

from features import physical


def equilibrium_residual(g, q_dimless):
    """Relative out-of-balance of the total predicted member forces."""
    r, ref = 0.0, 0.0
    for k, d in enumerate(g['d']):
        lam = g['qsc'][k][0]
        r = r + lam * (g['C'] @ q_dimless[:, 3 * k:3 * k + 3].ravel() - d)
        ref = ref + lam * d
    return float(np.linalg.norm(r) / (np.linalg.norm(ref) + 1e-12))


def _rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


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


def frame_metrics(g, pred_u, pred_q):
    """Errors of one frame. Inputs are dimensionless predictions."""
    u_hat, q_hat = physical(g, pred_u, pred_q)
    u, q = g['u_true'], g['q_true']
    dr_hat, dr = drift_ratios(g, u_hat), drift_ratios(g, u)
    Mhat, M = q_hat[:, 1:].ravel(), q[:, 1:].ravel()
    return dict(
        e_dx=_rel(u_hat[:, 0], u[:, 0]),
        e_dy=_rel(u_hat[:, 1], u[:, 1]),
        e_rz=_rel(u_hat[:, 2], u[:, 2]),
        e_drift=abs(dr_hat.max() - dr.max()) / dr.max(),
        e_M=_rel(Mhat, M),
        e_N=_rel(q_hat[:, 0], q[:, 0]),
        e_Mmax=abs(np.abs(Mhat).max() - np.abs(M).max()) / np.abs(M).max(),
        eq_res=equilibrium_residual(g, pred_q),
    )


def summarise(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows])
        out[k + '_mean'] = float(v.mean())
        out[k + '_med'] = float(np.median(v))
        out[k + '_p95'] = float(np.percentile(v, 95))
    return out
