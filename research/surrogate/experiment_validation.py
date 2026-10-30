"""Independent validation of the PyNite reference data (reviewer Experiment A).

    python experiment_validation.py     # writes results/validation.json, data/realistic.pkl

1. Cross-solver check: every test frame (1,800) and the design-driven
   benchmark (300) is re-analysed with an independent direct-stiffness solver
   written for this study and with OpenSeesPy; displacements, rotations,
   reactions, column/beam end moments, shears and axial forces are compared.
2. Closed-form benchmarks analysed with PyNite.
3. Generation of the design-driven benchmark set 'test_real' with PyNite.
"""

import json
import pickle
from multiprocessing import Pool

import numpy as np

import frames as F
import solvers as S

TESTS = ['test_id', 'test_oodG', 'test_oodL']


def _real(seed):
    rng = np.random.default_rng(seed)
    p = F.sample_design_frame(rng)
    r = F.run_fe(p)
    t = F.topology(p)
    for case in ('W', 'G'):
        assert np.abs(F.equilibrium_residual(F.case_topology(t, case), r['q_' + case])).max() < 1e-6 * max(1, np.abs(t['P']).max(), np.abs(t['wg'] * t['L']).max())
    return {k: r[k] for k in ('u_W', 'q_W', 'u_G', 'q_G', 't_fe')} | dict(p=p)


def quantities(t, u, q, R):
    """Response quantities compared between solvers."""
    col, beam = t['mtype'] == 0, t['mtype'] == 1
    V = (q[:, 1] + q[:, 2]) / t['L']                 # end shear from end moments (member-load part identical)
    free = ~t['restr']
    return dict(ux=u[:, 0], uy=u[:, 1], rz=u[:, 2], reactions=R[t['restr']],
                M_col=q[col, 1:].ravel(), M_beam=q[beam, 1:].ravel(), V=V, N=q[:, 0],
                _free=free)


def compare(p, r):
    t = F.topology(p)
    uP, qP = r['u_W'] + r['u_G'], r['q_W'] + r['q_G']
    B, p0 = F.equilibrium_system(t)
    RP = (B @ qP.ravel() + p0 - t['P'].ravel()).reshape(-1, 3)
    out = {}
    ref = quantities(t, uP, qP, RP)
    for name, fn in (('DSM', S.dsm_solve), ('OpenSeesPy', S.opensees_solve)):
        o = fn(p)
        R = o['R']
        Q = quantities(t, o['u'], o['q'], R)
        out[name] = {k: float(np.linalg.norm(Q[k] - ref[k]) / (np.linalg.norm(ref[k]) + 1e-300))
                     for k in ref if not k.startswith('_') and np.linalg.norm(ref[k]) > 0}
    return out


def closed_form():
    """Closed-form checks run through PyNite via the same model builder."""
    res = []
    E, h, L, Ic, P = 200e6, 3.5, 6.0, 1e-4, 50.0
    for fixed in (True, False):
        p = dict(n_st=1, n_bay=1, bays=np.array([L]), heights=np.array([h]), E=E, Ic=np.array([Ic]),
                 Ib=np.array([Ic * 1e5]), Ac=np.array([1e3]), Ab=np.array([1e3]), fixed_base=fixed,
                 w=np.zeros((1, 1)), F=np.array([P]))
        r = F.run_fe(p)
        u, q = r['u_W'], r['q_W']
        k = 12 if fixed else 3
        exact_sway = P * h ** 3 / (2 * k * E * Ic)
        exact_base = P * h / 4 if fixed else 0.0
        exact_top = P * h / 4 if fixed else P * h / 2
        res.append(dict(case=f'portal, rigid beam, {"fixed" if fixed else "pinned"} base, lateral load',
                        sway_pynite=float(u[2, 0]), sway_exact=exact_sway,
                        base_moment_pynite=float(abs(q[0, 1])), base_moment_exact=exact_base,
                        top_moment_pynite=float(abs(q[0, 2])), top_moment_exact=exact_top))
    # Fixed-fixed beam: a one-bay frame with rigid, axially stiff columns and gravity load
    w = 20.0
    p = dict(n_st=1, n_bay=1, bays=np.array([L]), heights=np.array([h]), E=E, Ic=np.array([1e2]),
             Ib=np.array([Ic]), Ac=np.array([1e6]), Ab=np.array([1e3]), fixed_base=True,
             w=np.array([[w]]), F=np.array([1e-9]))
    r = F.run_fe(p)
    q = r['q_G']
    res.append(dict(case='fixed-fixed beam (rigid columns), uniform load',
                    end_moment_pynite=float(abs(q[2, 1])), end_moment_exact=w * L ** 2 / 12))
    return res


def main():
    D = pickle.load(open('data/frames.pkl', 'rb'))
    with Pool(4) as pool:
        real = pool.map(_real, [9_000_000 + k for k in range(300)], chunksize=10)
    pickle.dump(real, open('data/realistic.pkl', 'wb'))
    out = dict(closed_form=closed_form())
    sets = {s: D[s] for s in TESTS} | {'test_real': real}
    for s, recs in sets.items():
        with Pool(4) as pool:
            rows = pool.starmap(compare, [(r['p'], r) for r in recs], chunksize=10)
        out[s] = {solver: {k: dict(max=float(np.max([r[solver][k] for r in rows if k in r[solver]])),
                                   median=float(np.median([r[solver][k] for r in rows if k in r[solver]])))
                           for k in rows[0][solver]} for solver in rows[0]}
        print(s, {sv: {k: f"{v['max']:.1e}" for k, v in d.items()} for sv, d in out[s].items()}, flush=True)
    json.dump(out, open('results/validation.json', 'w'), indent=1)
    print(json.dumps(out['closed_form'], indent=1))


if __name__ == '__main__':
    main()
