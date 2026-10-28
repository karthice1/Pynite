"""Fast regression tests of the reliability-optimisation framework.

Run from Research/reliability_optimization:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbo.sections import check_catalogue  # noqa: E402
from rbo.frames import FrameSpec, analyse, build_model, CASES  # noqa: E402
from rbo.limit_states import MemberTable, second_order_demands, deterministic_check  # noqa: E402
from rbo.reliability import ProbModel, draw, evaluate_samples, estimate  # noqa: E402
from rbo import validation  # noqa: E402

SPEC = FrameSpec(n_storeys=3, name='test')
X = [5, 7, 10, 3, 5, 8]


def test_catalogue_consistency():
    for name, a, ix, zx, iy in check_catalogue():
        for ratio in (a, ix, zx, iy):
            assert 0.97 < ratio < 1.05, name


def test_closed_form_benchmarks():
    for row in validation.run_closed_form():
        assert row['rel_err'] < 1e-4, row['case']


def test_pdelta_cantilever_converges():
    for row in validation.bench_pdelta_cantilever(P_ratios=(0.4,), n_els=(1, 4)):
        assert abs(row['pynite_4'] - row['exact']) / row['exact'] < 1e-3


def test_superposition_matches_direct_combination():
    """A first-order combination from unit cases equals a direct PyNite solve."""
    r = analyse(SPEC, X)
    t = MemberTable(r)
    combo = {'D': 1.2, 'L': 0.5, 'W': 1.6, 'ND': 1.2 * 0.003, 'NL': 0.5 * 0.003}
    m, members, _ = build_model(SPEC, X)
    m.add_load_combo('C', combo)
    m.analyze_linear(check_stability=False)
    cols = [mm for mm in members if mm.kind == 'col']
    direct = np.array([[m.members[c.name].f('C').ravel()[5], -m.members[c.name].f('C').ravel()[11]] for c in cols])
    sup = sum(combo[c] * r.col_M[i] for i, c in enumerate(CASES))
    assert np.allclose(direct, sup, rtol=1e-8, atol=1e-6)


def test_b2_close_to_pdelta():
    r = analyse(SPEC, X)
    t = MemberTable(r)
    dem = second_order_demands(r, t, [1.2], [0.5], [1.6], [0.003], [1.0])
    m, _, _ = build_model(SPEC, X)
    m.add_load_combo('PD', {'D': 1.2, 'L': 0.5, 'W': 1.6, 'ND': 1.2 * 0.003, 'NL': 0.5 * 0.003})
    m.analyze_PDelta(check_stability=False)
    dx = [np.mean([m.nodes[f'N{i}_{j}'].DX['PD'] for i in range(SPEC.n_bays + 1)]) for j in range(4)]
    pd = np.abs(np.diff(dx))
    b2 = dem['drift_ratio'][0] * SPEC.heights
    assert np.max(np.abs(b2 - pd) / pd) < 0.03


def test_reliability_monotonic_in_size():
    pm = ProbModel()
    light, heavy = analyse(SPEC, [2, 3, 6, 1, 2, 5]), analyse(SPEC, [9, 11, 16, 7, 9, 14])
    tl, th = MemberTable(light), MemberTable(heavy)
    s = draw(pm, 20000, len(tl.A), 3)
    bl = estimate(evaluate_samples(light, s, tl))
    bh = estimate(evaluate_samples(heavy, s, th))
    assert bh['beta_uls'] >= bl['beta_uls']
    assert bh['beta_drift'] > bl['beta_drift']
    assert deterministic_check(heavy)['strength'] < deterministic_check(light)['strength']
