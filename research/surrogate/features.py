"""Dimensionless, mechanics-informed graph features for frame records.

All inputs are scaled by the reference load lam, length H and rigidity EI
(see frames.scales) so that the learned map is exactly homogeneous of degree
one in the loads and of degree minus one in E, as linear elasticity requires.
Exact statics quantities (storey shear, overturning moment, gravity load
carried by each column line) and a shear-building drift estimate are added as
node features; these cost O(n) to compute and need no FE solve.
"""

from __future__ import annotations

import numpy as np

import frames as F

NODE_FEATS = ['lev', 'line', 'is_base', 'is_ext', 'x', 'y', 'n_st', 'n_bay', 'fixed',
              'Px', 'V_above', 'OTM_above', 'G_above', 'Ic_below', 'Ic_above',
              'Ib_left', 'Ib_right', 'Lb_left', 'Lb_right', 'h_below', 'h_above',
              'rho_level', 'drift_sb', 'disp_sb', 'w_trib']
EDGE_FEATS = ['is_col', 'is_beam', 'storey', 'L', 'I', 'A', 'wL', 'rho_storey',
              'V_storey', 'drift_sb']
# Conventional graph-learning inputs (no statics features, no load scaling):
# physical coordinates, supports and loads; log section properties.
NODE_FEATS_RAW = ['x', 'y', 'is_base', 'fixed', 'Px']
EDGE_FEATS_RAW = ['is_col', 'is_beam', 'L', 'log10E', 'log10I', 'log10A', 'w']


def graph(rec: dict, mode: str = 'split') -> dict:
    """Build the feature graph and dimensionless targets for one record.

    mode='split': exact load-case superposition. The lateral (W) and gravity
    (G) cases are scaled by their own intensities lamW (signed base shear) and
    lamG (total gravity load) and predicted as separate unit responses.
    mode='total': a single combined case scaled by lamW_abs + lamG (ablation).
    mode='raw': conventional GNN inputs and targets in physical units (single
    combined case, no statics features, no dimensional scaling); the network
    normaliser then standardises inputs and targets over the training set.
    """
    p = rec['p']
    t = F.topology(p)
    H = float(np.mean(p['heights']))
    EI = float(p['E'] * np.mean(p['Ic']))
    Iref = EI / p['E']
    n_st, n_bay = p['n_st'], p['n_bay']
    ncol = n_bay + 1
    n, m = len(t['coords']), len(t['conn'])
    lev, line = t['level'], t['line']
    lamW = float(np.sum(p['F']))
    lamG = float(np.sum(np.abs(t['wg']) * t['L']))
    if mode == 'split':
        cases, lams = ['W', 'G'], [lamW, lamG]
        lat_scale, grav_scale = lamW, lamG
    elif mode == 'raw':
        return _graph_raw(rec, p, t)
    else:
        lam = abs(lamW) + lamG
        cases, lams = ['T'], [lam]
        lat_scale, grav_scale = lam, lam

    # Storey-level statics (exact) -------------------------------------------------
    F_fl = np.asarray(p['F'])                         # lateral force at floor s+1
    elev = np.concatenate([[0.0], np.cumsum(p['heights'])])
    V = np.array([F_fl[s:].sum() for s in range(n_st)])          # shear in storey s
    # Overturning moment about the bottom of storey s from loads above
    OTM = np.array([np.sum(F_fl[s:] * (elev[s + 1:] - elev[s])) for s in range(n_st)])
    # Gravity load carried by each column line in storey s (tributary, exact for
    # simply supported beams)
    trib = np.zeros((n_st, ncol))
    for s in range(n_st):
        for b in range(n_bay):
            half = p['w'][s, b] * p['bays'][b] / 2
            trib[s, b] += half
            trib[s, b + 1] += half
    G_above = np.array([trib[s:].sum(axis=0) for s in range(n_st)])   # (n_st, ncol)

    # Shear-building (rigid beam) lateral stiffness and drift estimate
    h = np.asarray(p['heights'])
    k_col = 12 * p['E'] * p['Ic'] / h ** 3
    if not p['fixed_base']:
        k_col = k_col.copy()
        k_col[0] = 3 * p['E'] * p['Ic'][0] / h[0] ** 3
    k_st = ncol * k_col
    drift = V / k_st
    disp = np.cumsum(drift)
    # Beam-to-column stiffness ratio per storey (classical frame parameter)
    rho = (n_bay * p['Ib'] / np.mean(p['bays'])) / (ncol * p['Ic'] / h)

    u0_lat = lat_scale * H ** 3 / EI
    X = np.zeros((n, len(NODE_FEATS)))
    for k in range(n):
        l, c = lev[k], line[k]
        sb, sa = l - 1, l          # storey below / above the node
        f = dict(lev=l / n_st, line=c / max(n_bay, 1), is_base=float(l == 0),
                 is_ext=float(c in (0, n_bay)), x=t['coords'][k, 0] / H,
                 y=t['coords'][k, 1] / H, n_st=n_st / 5, n_bay=n_bay / 4,
                 fixed=float(p['fixed_base']), Px=t['P'][k, 0] / lat_scale)
        s_ref = sa if sa < n_st else sb
        f['V_above'] = V[s_ref] / lat_scale if l < n_st else 0.0
        f['OTM_above'] = OTM[s_ref] / (lat_scale * H) if l < n_st else 0.0
        f['G_above'] = G_above[s_ref, c] / grav_scale if l < n_st else 0.0
        f['Ic_below'] = p['Ic'][sb] / Iref if sb >= 0 else 0.0
        f['Ic_above'] = p['Ic'][sa] / Iref if sa < n_st else 0.0
        f['Ib_left'] = p['Ib'][sb] / Iref if (l > 0 and c > 0) else 0.0
        f['Ib_right'] = p['Ib'][sb] / Iref if (l > 0 and c < n_bay) else 0.0
        f['Lb_left'] = p['bays'][c - 1] / H if c > 0 else 0.0
        f['Lb_right'] = p['bays'][c] / H if c < n_bay else 0.0
        f['h_below'] = h[sb] / H if sb >= 0 else 0.0
        f['h_above'] = h[sa] / H if sa < n_st else 0.0
        f['rho_level'] = rho[s_ref]
        f['drift_sb'] = drift[s_ref] / u0_lat
        f['disp_sb'] = disp[sb] / u0_lat if sb >= 0 else 0.0
        f['w_trib'] = trib[sb, c] / grav_scale if sb >= 0 else 0.0
        X[k] = [f[name] for name in NODE_FEATS]

    Ef = np.zeros((m, len(EDGE_FEATS)))
    for e in range(m):
        s = t['mstorey'][e]
        is_col = t['mtype'][e] == 0
        Ef[e] = [float(is_col), float(not is_col), s / n_st, t['L'][e] / H, t['I'][e] / Iref,
                 t['A'][e] * H ** 2 / Iref, t['wg'][e] * t['L'][e] / grav_scale, rho[s],
                 V[s] / lat_scale, drift[s] / u0_lat]

    usc = [np.array([lm * H ** 3 / EI] * 2 + [lm * H ** 2 / EI]) for lm in lams]
    qsc = [np.array([lm, lm * H, lm * H]) for lm in lams]
    out = dict(x=X, e=Ef, conn=t['conn'], restr=t['restr'], t=t, cases=cases,
               usc=usc, qsc=qsc, H=H, mode=mode)
    if 'u_W' in rec:
        if mode == 'split':
            uc, qc = [rec['u_W'], rec['u_G']], [rec['q_W'], rec['q_G']]
        else:
            uc, qc = [rec['u_W'] + rec['u_G']], [rec['q_W'] + rec['q_G']]
        out['y_u'] = np.hstack([u / s for u, s in zip(uc, usc)])
        out['y_q'] = np.hstack([q / s for q, s in zip(qc, qsc)])
        out['u_true'] = rec['u_W'] + rec['u_G']
        out['q_true'] = rec['q_W'] + rec['q_G']

    # Dimensionless equilibrium system C qhat = d_c for each case, with
    # q = qhat * [lam, lam H, lam H] and residuals scaled by lam (forces) and
    # lam H (moments). C is identical for every case; only d changes.
    B, p0 = F.equilibrium_system(F.case_topology(t, 'W'))
    free = ~t['restr'].ravel()
    D = np.tile([1.0, H, H], m)
    row_scale = np.tile([1.0, 1.0, H], n)[free]
    C = (B[free] * D[None, :]) / row_scale[:, None]
    ds = []
    for case, lm in zip(cases, lams):
        tc = F.case_topology(t, case) if case != 'T' else t
        Bc, p0c = F.equilibrium_system(tc)
        ds.append((tc['P'].ravel() - p0c)[free] / row_scale / lm)
    out.update(C=C, d=ds, B=B, p0=None)
    return out


def _graph_raw(rec, p, t):
    n = len(t['coords'])
    X = np.zeros((n, len(NODE_FEATS_RAW)))
    X[:, 0], X[:, 1] = t['coords'][:, 0], t['coords'][:, 1]
    X[:, 2] = t['level'] == 0
    X[:, 3] = float(p['fixed_base'])
    X[:, 4] = t['P'][:, 0]
    Ef = np.stack([t['mtype'] == 0, t['mtype'] == 1, t['L'], np.log10(t['E']), np.log10(t['I']),
                   np.log10(t['A']), t['wg']], axis=1).astype(float)
    one = np.ones(3)
    out = dict(x=X, e=Ef, conn=t['conn'], restr=t['restr'], t=t, cases=['T'], usc=[one], qsc=[one],
               H=float(np.mean(p['heights'])), mode='raw')
    if 'u_W' in rec:
        out['u_true'] = rec['u_W'] + rec['u_G']
        out['q_true'] = rec['q_W'] + rec['q_G']
        out['y_u'], out['y_q'] = out['u_true'].copy(), out['q_true'].copy()
    B, p0 = F.equilibrium_system(t)
    free = ~t['restr'].ravel()
    out.update(C=B[free], d=[(t['P'].ravel() - p0)[free]], B=B, p0=p0)
    return out


def total_p0(g):
    """Constant term of the equilibrium map for the combined (W + G) loading."""
    if g.get('p0') is None:
        g['p0'] = F.equilibrium_system(g['t'])[1]
    return g['p0']


def physical(g, y_u, y_q):
    """Sum the dimensionless case responses back to physical total u, q."""
    u = sum(y_u[:, 3 * k:3 * k + 3] * s for k, s in enumerate(g['usc']))
    q = sum(y_q[:, 3 * k:3 * k + 3] * s for k, s in enumerate(g['qsc']))
    return u, q


def tabular(graphs: list[dict]):
    """Flatten graphs into per-node and per-member tables for the baselines.

    Member rows concatenate the member features with both end-node features,
    which gives the tabular models the same local information the first layer
    of the graph network sees.
    """
    Xn, Yn, Mn, Xe, Ye = [], [], [], [], []
    for g in graphs:
        Xn.append(g['x'])
        Mn.append(np.tile(~g['restr'], len(g['cases'])))
        i, j = g['conn'][:, 0], g['conn'][:, 1]
        Xe.append(np.hstack([g['e'], g['x'][i], g['x'][j]]))
        if 'y_u' in g:
            Yn.append(g['y_u'])
            Ye.append(g['y_q'])
    Xn, Mn, Xe = np.vstack(Xn), np.vstack(Mn), np.vstack(Xe)
    Yn = np.vstack(Yn) if Yn else None
    Ye = np.vstack(Ye) if Ye else None
    return Xn, Yn, Mn, Xe, Ye
