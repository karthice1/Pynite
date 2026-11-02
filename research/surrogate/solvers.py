"""Independent reference solvers for cross-validating the PyNite dataset.

* ``dsm_solve``: a vectorised 2D direct-stiffness solver written for this
  study (Euler-Bernoulli frame elements, uniform member loads, sparse solve).
  It shares no code with PyNite and is also used for the compatibility
  analysis (member forces implied by a displacement field).
* ``opensees_solve``: the same frame in OpenSeesPy (elasticBeamColumn,
  Linear transformation, beamUniform loads, linear static analysis).

Both return nodal displacements u (n, 3), member basic forces q (m, 3) =
[N, Mi, Mj] in the convention of frames.py, and support reactions (n, 3).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import frames as F


def element_matrices(t):
    """Local stiffness (m, 6, 6), rotation (m, 6, 6) and local fixed-end forces (m, 6)."""
    E, A, I, L, c, s = t['E'], t['A'], t['I'], t['L'], t['c'], t['s']
    m = len(L)
    k = np.zeros((m, 6, 6))
    ea, ei = E * A / L, E * I
    k[:, 0, 0] = k[:, 3, 3] = ea; k[:, 0, 3] = k[:, 3, 0] = -ea
    a, b, d, e = 12 * ei / L ** 3, 6 * ei / L ** 2, 4 * ei / L, 2 * ei / L
    for (i, j, v) in [(1, 1, a), (1, 2, b), (1, 4, -a), (1, 5, b), (2, 2, d), (2, 4, -b), (2, 5, e),
                      (4, 4, a), (4, 5, -b), (5, 5, d)]:
        k[:, i, j] = v; k[:, j, i] = v
    T = np.zeros((m, 6, 6))
    for o in (0, 3):
        T[:, o, o] = c; T[:, o, o + 1] = s; T[:, o + 1, o] = -s; T[:, o + 1, o + 1] = c; T[:, o + 2, o + 2] = 1
    wy = t['wg'] * c                     # local transverse component of the global-Y line load
    fef = np.zeros((m, 6))               # forces on the member from the fixed supports
    fef[:, 1] = fef[:, 4] = -wy * L / 2
    fef[:, 2] = -wy * L ** 2 / 12
    fef[:, 5] = wy * L ** 2 / 12
    return k, T, fef


def member_local_forces(t, u, mats=None):
    """Local end forces f = k T u_e + fef for a nodal displacement field u (n, 3)."""
    k, T, fef = mats if mats is not None else element_matrices(t)
    i, j = t['conn'][:, 0], t['conn'][:, 1]
    ue = np.concatenate([u[i], u[j]], axis=1)
    return np.einsum('mab,mbc,mc->ma', k, T, ue) + fef


def basic_from_local(f):
    return np.stack([f[:, 3], f[:, 2], f[:, 5]], axis=1)


def dsm_solve(p, case='both'):
    t = F.topology(p)
    if case in ('W', 'G'):
        t = F.case_topology(t, case)
    n = len(t['coords'])
    k, T, fef = mats = element_matrices(t)
    kg = np.einsum('mba,mbc,mcd->mad', T, k, T)          # T^T k T
    dofs = np.concatenate([3 * t['conn'][:, :1] + np.arange(3), 3 * t['conn'][:, 1:] + np.arange(3)], axis=1)
    rows = np.repeat(dofs, 6, axis=1).ravel(); cols = np.tile(dofs, (1, 6)).ravel()
    K = sp.csr_matrix((kg.ravel(), (rows, cols)), shape=(3 * n, 3 * n))
    Pf = t['P'].ravel().copy()
    np.add.at(Pf, dofs.ravel(), -np.einsum('mba,mb->ma', T, fef).ravel())
    free = ~t['restr'].ravel()
    u = np.zeros(3 * n)
    u[free] = spla.spsolve(K[free][:, free].tocsc(), Pf[free])
    u = u.reshape(n, 3)
    f = member_local_forces(t, u, mats)
    R = (K @ u.ravel() - Pf).reshape(n, 3) * t['restr']
    return dict(u=u, q=basic_from_local(f), f_local=f, R=R)


def opensees_solve(p, case='both'):
    import openseespy.opensees as ops
    t = F.topology(p)
    if case in ('W', 'G'):
        t = F.case_topology(t, case)
    ops.wipe(); ops.model('basic', '-ndm', 2, '-ndf', 3)
    for k_, (x, y) in enumerate(t['coords']):
        ops.node(k_ + 1, float(x), float(y))
        r = t['restr'][k_]
        if r.any():
            ops.fix(k_ + 1, int(r[0]), int(r[1]), int(r[2]))
    ops.geomTransf('Linear', 1)
    for e, (i, j) in enumerate(t['conn']):
        ops.element('elasticBeamColumn', e + 1, int(i) + 1, int(j) + 1, float(t['A'][e]), float(t['E'][e]),
                    float(t['I'][e]), 1)
    ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)
    for k_ in range(len(t['coords'])):
        if np.any(t['P'][k_] != 0):
            ops.load(k_ + 1, *[float(v) for v in t['P'][k_]])
    for e in range(len(t['conn'])):
        wy = t['wg'][e] * t['c'][e]
        if wy != 0:
            ops.eleLoad('-ele', e + 1, '-type', '-beamUniform', float(wy))
    ops.system('UmfPack'); ops.numberer('RCM'); ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0); ops.algorithm('Linear'); ops.analysis('Static')
    ops.analyze(1); ops.reactions()
    n, m = len(t['coords']), len(t['conn'])
    u = np.array([ops.nodeDisp(k_ + 1) for k_ in range(n)])
    f = np.array([ops.eleResponse(e + 1, 'localForce') for e in range(m)])
    R = np.array([ops.nodeReaction(k_ + 1) for k_ in range(n)])
    ops.wipe()
    return dict(u=u, q=basic_from_local(f), f_local=f, R=R)
