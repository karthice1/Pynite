"""Figures and tables of the revised manuscript.

    python make_figures_rev.py    # needs results/revision/*.json, results/rev/*.pkl, results/timing.json

Writes ../../manuscript/surrogate/figures/*.pdf and tables/*.tex, and
results/revision/uq.json, results/revision/adaptive_summary.json,
results/revision/dataset_stats.json.
"""

import glob
import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import analysis_revision as AR
import features as FT
import frames as F
import metrics as MT

OUT = os.path.join('..', '..', 'manuscript', 'surrogate')
FIG, TAB = os.path.join(OUT, 'figures'), os.path.join(OUT, 'tables')
RES = 'results/revision'
TESTS = ['test_id', 'test_oodG', 'test_oodL', 'test_real']
LAB = {'test_id': 'In-distribution (ID)', 'test_oodG': 'Unseen geometries (OOD-G)',
       'test_oodL': 'Unseen loading (OOD-L)', 'test_real': 'Design-driven frames (DES)'}
ABBR = {'test_id': 'ID', 'test_oodG': 'OOD-G', 'test_oodL': 'OOD-L', 'test_real': 'DES'}
NFR = {'test_id': 600, 'test_oodG': 600, 'test_oodL': 600, 'test_real': 300}
BASE = ['Linear ridge', 'Quadratic ridge', 'Random forest', 'Gradient boosting', 'MLP']
PROP = 'Proposed'
COL = {'Linear ridge': '#9e9e9e', 'Quadratic ridge': '#6d6d6d', 'Random forest': '#8c6d31',
       'Gradient boosting': '#d08c2c', 'MLP': '#7b5ea7', 'GNN ensemble (raw)': '#5b9bd5', PROP: '#1f4e9c'}

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})


def pct(x, d=1):
    return f'{100 * x:.{d}f}'


def pm(v, d=1):
    return f"{100 * v['mean']:.{d}f} $\\pm$ {100 * v['std']:.{d}f}"


def J(name):
    p = f'{RES}/{name}.json'
    return json.load(open(p)) if os.path.exists(p) else None


def write(name, lines):
    open(os.path.join(TAB, name), 'w').write('\n'.join(lines) + '\n')


def sci(x):
    if x < 1e-12:
        return '$<10^{-12}$'
    ex = int(np.floor(np.log10(x)))
    return f'${x / 10 ** ex:.1f}\\times10^{{{ex}}}$'


# ---------------------------------------------------------------------------
def load():
    D = AR.data()
    G = AR.graphs('split')
    P0 = AR.ensemble_preds(0)
    BP = pickle.load(open('results/rev/baseline_preds.pkl', 'rb')) if os.path.exists('results/rev/baseline_preds.pkl') else {}
    return D, G, P0, BP


def prop_phys(g, p):
    return AR.proposed(g, p, 'variance-weighted')


def rows_of(G, P, s, kind):
    key = ('rows', kind, s)
    if key not in AR._cache:
        if kind == PROP:
            AR._cache[key] = [MT.frame_metrics_phys(g, *prop_phys(g, p), compat=False) for g, p in zip(G[s], P[s])]
        elif kind == 'GNN ensemble (raw)':
            AR._cache[key] = [MT.frame_metrics(g, p['u'], p['q'], compat=False) for g, p in zip(G[s], P[s])]
        else:
            AR._cache[key] = [MT.frame_metrics(g, p['u'], p['q'], compat=False) for g, p in zip(G[s], P[kind][s])]
    return AR._cache[key]


# ---------------------------------------------------------------------------
def fig_frames(D):
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
    picks = {'test_id': 7, 'test_oodG': 3, 'test_oodL': 11, 'test_real': 5}
    for ax, (s, k) in zip(axs.ravel(), picks.items()):
        p = D[s][k]['p']; t = F.topology(p)
        for e, (i, j) in enumerate(t['conn']):
            xy = t['coords'][[i, j]]
            ax.plot(xy[:, 0], xy[:, 1], color='#1f4e9c' if t['mtype'][e] == 0 else '#333333', lw=1.3)
            if t['wg'][e] != 0:
                ax.fill_between(xy[:, 0], xy[0, 1], xy[0, 1] + 0.012 * abs(t['wg'][e]), color='#d08c2c', alpha=0.35, lw=0)
        for kk in np.nonzero(t['P'][:, 0])[0]:
            x, y = t['coords'][kk]
            sgn = np.sign(t['P'][kk, 0])
            ax.annotate('', xy=(x, y), xytext=(x - sgn * (0.6 + 0.03 * abs(t['P'][kk, 0])), y),
                        arrowprops=dict(arrowstyle='->', color='#b2182b', lw=1.0))
        base = t['coords'][t['level'] == 0]
        ax.plot(base[:, 0], base[:, 1], 's' if p['fixed_base'] else '^', color='k', ms=4)
        ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
        ax.set_title(f"{LAB[s]}\n{p['n_st']} storeys, {p['n_bay']} bays")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_frames.pdf')); plt.close(fig)


def fig_cdf(G, P0, BP):
    names = [n for n in ['Linear ridge', 'Random forest', 'Gradient boosting', 'MLP'] if n in BP] + ['GNN ensemble (raw)', PROP]
    fig, axs = plt.subplots(4, 2, figsize=(7.0, 9.0), sharey=True)
    for r, s in enumerate(TESTS):
        for name in names:
            rows = rows_of(G, BP if name in BP else P0, s, name)
            for c, key in enumerate(['e_dx', 'e_M']):
                v = np.sort([x[key] for x in rows])
                axs[r, c].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[name],
                               lw=1.8 if name == PROP else 1.0, label=name)
        for c, lab in enumerate([r'relative $L_2$ error, horizontal displacements $e_{u_x}$',
                                 r'relative $L_2$ error, end moments $e_M$']):
            axs[r, c].set_xscale('log'); axs[r, c].set_xlim(3e-3, 3); axs[r, c].grid(alpha=0.25)
            axs[r, c].set_title(LAB[s])
            if r == 3:
                axs[r, c].set_xlabel(lab)
        axs[r, 0].set_ylabel('Fraction of frames')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, 'fig_error_cdf.pdf')); plt.close(fig)


def fig_parity(G, P0, BP):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.9))
    for c, s in enumerate(TESTS):
        for name, mk in [('Gradient boosting', 'x'), (PROP, '.')]:
            if name not in BP and name != PROP:
                continue
            rt, rp, Mt, Mp = [], [], [], []
            for i, g in enumerate(G[s]):
                if name == PROP:
                    u, q = prop_phys(g, P0[s][i])
                else:
                    u, q = FT.physical(g, BP[name][s][i]['u'], BP[name][s][i]['q'])
                top = g['t']['level'] == g['t']['level'].max()
                rt.append(np.mean(g['u_true'][top, 0]) * 1e3); rp.append(np.mean(u[top, 0]) * 1e3)
                Mt.append(np.abs(g['q_true'][:, 1:]).max()); Mp.append(np.abs(q[:, 1:]).max())
            axs[0, c].plot(rt, rp, mk, ms=2.5, alpha=0.6, color=COL[name], label=name)
            axs[1, c].plot(Mt, Mp, mk, ms=2.5, alpha=0.6, color=COL[name], label=name)
        for r in range(2):
            lim = np.array(axs[r, c].get_xlim()); axs[r, c].plot(lim, lim, 'k-', lw=0.6)
        axs[0, c].set_title(ABBR[s])
        axs[0, c].set_xlabel('PyNite roof $u_x$ (mm)'); axs[1, c].set_xlabel(r'PyNite max $|M|$ (kN m)')
    axs[0, 0].set_ylabel('Predicted (mm)'); axs[1, 0].set_ylabel('Predicted (kN m)')
    axs[0, 0].legend(frameon=False, loc='upper left', fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_parity.pdf')); plt.close(fig)


def fig_example(G, P0, BP):
    s = 'test_oodG'
    sizes = [len(g['conn']) for g in G[s]]
    k = int(np.argsort(sizes)[len(sizes) // 2])
    g = G[s][k]
    _, q = prop_phys(g, P0[s][k])
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    x = np.arange(2 * len(q))
    ax.plot(x, g['q_true'][:, 1:].ravel(), 'k-', lw=1.6, label='PyNite')
    ax.plot(x, q[:, 1:].ravel(), '.', color=COL[PROP], ms=4, label='Proposed')
    if 'Gradient boosting' in BP:
        _, qb = FT.physical(g, BP['Gradient boosting'][s][k]['u'], BP['Gradient boosting'][s][k]['q'])
        ax.plot(x, qb[:, 1:].ravel(), 'x', color=COL['Gradient boosting'], ms=3, label='Gradient boosting')
    ax.set_xlabel('Member end index (columns first, then beams)'); ax.set_ylabel('End moment (kN m)')
    ax.set_title(f"Unseen-geometry frame: {int(g['t']['level'].max())} storeys, {int(g['t']['line'].max())} bays")
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3))
    fig.savefig(os.path.join(FIG, 'fig_example.pdf')); plt.close(fig)


def fig_equilibrium(G, P0, BP):
    names = [n for n in ['Random forest', 'Gradient boosting', 'MLP'] if n in BP] + ['GNN ensemble (raw)', PROP]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    data = [np.maximum([x['eq_res'] for x in rows_of(G, BP if n in BP else P0, 'test_id', n)], 1e-16) for n in names]
    ax.boxplot(data, orientation='horizontal', whis=(5, 95), showfliers=False)
    ax.set_yticks(range(1, len(names) + 1), names)
    ax.set_xscale('log'); ax.set_xlabel('Relative nodal out-of-balance force (ID)')
    fig.savefig(os.path.join(FIG, 'fig_equilibrium.pdf')); plt.close(fig)


# ---------------------------------------------------------------------------
def uq(G, P0, ood):
    out, curves_all = {}, {}
    for s in TESTS:
        err, sc, eq, cover = [], [], [], []
        for g, p in zip(G[s], P0[s]):
            u, q = prop_phys(g, p)
            sdM = np.sqrt(sum(p['q_var'][:, 3 * k + 1:3 * k + 3] * g['qsc'][k][1] ** 2 for k in range(len(g['cases']))))
            sc.append(np.linalg.norm(sdM) / (np.linalg.norm(q[:, 1:]) + 1e-12))
            err.append(MT._rel(q[:, 1:], g['q_true'][:, 1:]))
            eq.append(MT.equilibrium_residual(g, p['q']))
            cover.append(np.mean(np.abs(q[:, 1:] - g['q_true'][:, 1:]) <= 2 * sdM))
        err, sc, eq = map(np.array, (err, sc, eq))
        fr = np.linspace(0, 0.9, 46)
        rng = np.random.default_rng(0)
        curves = {}
        for lab, v in [('Ensemble spread', sc), ('Equilibrium residual', eq), ('Random', rng.random(len(err))), ('Oracle', err)]:
            o = np.argsort(-v)
            curves[lab] = [float(np.percentile(err[o[int(round(f * len(err))):]], 95)) for f in fr]
        out[s] = dict(spearman_spread=float(spearmanr(sc, err)[0]), spearman_eq=float(spearmanr(eq, err)[0]),
                      coverage_2sigma=float(np.mean(cover)), frac_eM_gt_10=float(np.mean(err > 0.10)),
                      p95_all=float(np.percentile(err, 95)))
        for f in (0.1, 0.2):
            k = int(round(f * len(err))); o = np.argsort(-sc)
            out[s][f'p95_after_{int(100 * f)}'] = float(np.percentile(err[o[k:]], 95))
            out[s][f'recall_bad_at_{int(100 * f)}'] = float(np.sum(err[o[:k]] > 0.1) / max(np.sum(err > 0.1), 1))
        curves_all[s] = dict(fr=fr, curves=curves, sc=sc, err=err)
    json.dump(out, open(f'{RES}/uq.json', 'w'), indent=1)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
    cmap = {'Ensemble spread': ('#1f4e9c', '-'), 'Equilibrium residual': ('#d08c2c', '--'),
            'Random': ('#777777', ':'), 'Oracle': ('#2a9d55', '-.')}
    for ax, s in zip(axs.ravel(), TESTS):
        c = curves_all[s]
        for lab, (col, st) in cmap.items():
            ax.plot(100 * c['fr'], 100 * np.array(c['curves'][lab]), st, color=col, lw=1.3, label=lab)
        if ood is not None:
            for tau, mk in (('0.05', 'o'), ('0.1', 's')):
                pol = ood['policy'][tau][s]
                if pol['accepted_p95'] is not None:
                    ax.plot(100 * pol['referred_total'], 100 * pol['accepted_p95'], mk, color='k', ms=5, mfc='none',
                            label=f'Two-stage rule, $\\tau$={int(100 * float(tau))}%')
        ax.set_title(LAB[s]); ax.set_xlabel('Frames referred to FE (%)'); ax.grid(alpha=0.25)
        ax.set_ylabel('p95 moment error of accepted (%)')
        ax.set_xlim(0, 100)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, 'fig_triage.pdf')); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.8))
    for ax, s in zip(axs.ravel(), TESTS):
        c = curves_all[s]
        ax.loglog(c['sc'], c['err'], '.', ms=3, alpha=0.5, color=COL[PROP])
        if ood is not None:
            x = np.logspace(np.log10(c['sc'].min()), np.log10(c['sc'].max()), 50)
            ax.plot(x, ood['conformal_qhat'] * x, 'r--', lw=1, label=f"conformal bound $\\hat a s$ ($\\hat a$={ood['conformal_qhat']:.2f})")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, 'k-', lw=0.5)
        ax.set_title(f"{LAB[s]} ($\\rho_s$={out[s]['spearman_spread']:.2f})")
        ax.set_xlabel('Ensemble relative spread $s$ (moments)'); ax.set_ylabel('True moment error $e_M$')
    axs[0, 0].legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_uq_scatter.pdf')); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
def adaptive_runs():
    """{strategy: {seed: history}} from the per-strategy files and the original two seed files."""
    runs = {}
    for f in sorted(glob.glob('results/adaptive_seed*.json')):
        a = json.load(open(f)); seed = int(f.split('seed')[1].split('.')[0])
        for st, v in a['results'].items():
            runs.setdefault(st, {})[seed] = v['history']
    for f in sorted(glob.glob('results/adaptive/seed*_*.json')):
        a = json.load(open(f))
        for st, v in a['results'].items():
            runs.setdefault(st, {})[a['seed']] = v['history']
    return runs


def fig_adaptive_and_table():
    runs = adaptive_runs()
    if not runs:
        return None
    TS = ['test_id', 'test_oodG', 'test_oodL']
    sty = {'random': ('#777777', 'o', 'Random'), 'ensemble variance': ('#1f4e9c', 's', 'Ensemble variance'),
           'equilibrium residual': ('#d08c2c', '^', 'Equilibrium residual'),
           'diverse equilibrium residual': ('#2a9d55', 'D', 'Equilibrium residual + diversity')}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.4))
    summ = {}
    for st, (c, mk, lab) in sty.items():
        if st not in runs:
            continue
        H = runs[st]
        n = np.array([h['n_fe'] for h in next(iter(H.values()))])
        for j, s in enumerate(TS):
            for r, key in enumerate(['e_M_med', 'e_M_p95']):
                v = 100 * np.array([[h[s][key] for h in hist] for hist in H.values()])
                axs[r, j].plot(n, v.mean(0), marker=mk, ms=3.5, color=c, label=lab, lw=1.2)
                if len(v) > 1:
                    sd = v.std(0, ddof=1)
                    axs[r, j].fill_between(n, v.mean(0) - sd, v.mean(0) + sd, color=c, alpha=0.15, lw=0)
            axs[0, j].set_title(LAB[s].split(' (')[0]); axs[1, j].set_xlabel('PyNite analyses in training set')
    axs[0, 0].set_ylabel('Median moment error (%)'); axs[1, 0].set_ylabel('p95 moment error (%)')
    for a in axs.ravel():
        a.grid(alpha=0.25)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.legend(h, l, loc='lower center', ncol=2, frameon=False)
    fig.savefig(os.path.join(FIG, 'fig_adaptive.pdf')); plt.close(fig)

    # final-round statistics and FE analyses needed to reach random sampling's final median error
    lines = []
    rnd_final = {s: np.mean([hist[-1][s]['e_M_med'] for hist in runs['random'].values()]) for s in TS}
    for st, (c, mk, lab) in sty.items():
        if st not in runs:
            continue
        H = runs[st]
        summ[st] = {'n_seeds': len(H)}
        cells = []
        for s in TS:
            fm = np.array([hist[-1][s]['e_M_med'] for hist in H.values()])
            fp = np.array([hist[-1][s]['e_M_p95'] for hist in H.values()])
            curve = np.mean([[h[s]['e_M_med'] for h in hist] for hist in H.values()], 0)
            n = [h['n_fe'] for h in next(iter(H.values()))]
            hit = np.nonzero(curve <= rnd_final[s])[0]
            n_hit = int(n[hit[0]]) if len(hit) else None
            # paired difference to random over common seeds
            common = sorted(set(H) & set(runs['random']))
            d = np.array([H[k][-1][s]['e_M_med'] - runs['random'][k][-1][s]['e_M_med'] for k in common])
            summ[st][s] = dict(final_med_mean=float(fm.mean()), final_med_std=float(fm.std(ddof=1)) if len(fm) > 1 else 0.0,
                               final_p95_mean=float(fp.mean()), final_p95_std=float(fp.std(ddof=1)) if len(fp) > 1 else 0.0,
                               n_to_random_final=n_hit, diff_to_random=AR.t_ci(d) if len(d) > 1 and st != 'random' else None)
            cells += [f"{100 * fm.mean():.1f} $\\pm$ {100 * summ[st][s]['final_med_std']:.1f}",
                      f"{100 * fp.mean():.1f} $\\pm$ {100 * summ[st][s]['final_p95_std']:.1f}",
                      str(n_hit) if n_hit else '--']
        lines.append(f"{lab} ({len(H)}) & " + ' & '.join(cells) + r'\\')
    write('tab_adaptive.tex', lines)
    json.dump(summ, open(f'{RES}/adaptive_summary.json', 'w'), indent=1)
    return summ


# ---------------------------------------------------------------------------
def tab_validation():
    v = json.load(open('results/validation.json'))
    keys = [('ux', '$u_x$'), ('uy', '$u_y$'), ('rz', r'$\theta_z$'), ('reactions', 'Reactions'),
            ('M_col', 'Column $M$'), ('M_beam', 'Beam $M$'), ('V', 'Shear $V$'), ('N', 'Axial $N$')]
    lines = [r'\begin{tabular}{ll' + 'c' * len(keys) + '}', r'\toprule',
             'Test set & Solver & ' + ' & '.join(k[1] for k in keys) + r'\\', r'\midrule']
    for s in TESTS:
        for sv in ('DSM', 'OpenSeesPy'):
            lines.append((ABBR[s] if sv == 'DSM' else '') + f' & {"Own DSM" if sv == "DSM" else sv} & ' +
                         ' & '.join(f"{v[s][sv][k]['max']:.1e}" for k, _ in keys) + r'\\')
    lines += [r'\midrule', r'\multicolumn{10}{l}{\textit{Closed-form benchmarks (PyNite / exact)}}\\']
    cf = v['closed_form']
    lines.append(r'\multicolumn{10}{l}{Portal, rigid beam, fixed bases: sway ' +
                 f"{1e3 * cf[0]['sway_pynite']:.4f} / {1e3 * cf[0]['sway_exact']:.4f} mm; base moment "
                 f"{cf[0]['base_moment_pynite']:.3f} / {cf[0]['base_moment_exact']:.3f} kN\\,m" + r'}\\')
    lines.append(r'\multicolumn{10}{l}{Portal, rigid beam, pinned bases: sway ' +
                 f"{1e3 * cf[1]['sway_pynite']:.4f} / {1e3 * cf[1]['sway_exact']:.4f} mm; top moment "
                 f"{cf[1]['top_moment_pynite']:.3f} / {cf[1]['top_moment_exact']:.3f} kN\\,m" + r'}\\')
    lines.append(r'\multicolumn{10}{l}{Beam with fixed ends, uniform load: end moment ' +
                 f"{cf[2]['end_moment_pynite']:.4f} / {cf[2]['end_moment_exact']:.4f} kN\\,m" + r'}\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('tab_validation.tex', lines)


def tab_main(base, runs, G, P0):
    """Two compact tables: ID + DES, and OOD-G + OOD-L."""
    for fname, sets in (('tab_main_a.tex', ['test_id', 'test_real']), ('tab_main_b.tex', ['test_oodG', 'test_oodL'])):
        lines = []
        for s in sets:
            lines.append(r'\multicolumn{7}{l}{\textit{' + LAB[s] + f' ({NFR[s]} frames)' + r'}}\\')
            entries = []
            for n in BASE:
                if base and n in base:
                    b = base[n][s]
                    entries.append((n, b['e_dx']['med'], b['e_drift']['med'], b['e_M']['med'], b['e_M']['p95'],
                                    b['e_N']['med'], b['eq_res']['med']))
            raw = AR.summ(rows_of(G, P0, s, 'GNN ensemble (raw)'))
            entries.append(('GNN ensemble, raw forces', raw['e_dx']['med'], raw['e_drift']['med'], raw['e_M']['med'],
                            raw['e_M']['p95'], raw['e_N']['med'], raw['eq_res']['med']))
            pr = AR.summ(rows_of(G, P0, s, PROP))
            entries.append(('Proposed (ensemble + var.-weighted proj.)', pr['e_dx']['med'], pr['e_drift']['med'],
                            pr['e_M']['med'], pr['e_M']['p95'], pr['e_N']['med'], pr['eq_res']['med']))
            best = [min(e[i] for e in entries) for i in range(1, 6)]
            for e in entries:
                cells = [(r'\textbf{' + pct(x) + '}') if np.isclose(x, b) else pct(x) for x, b in zip(e[1:6], best)]
                lines.append(e[0] + ' & ' + ' & '.join(cells) + ' & ' + sci(e[6]) + r'\\')
            if runs:
                A = runs['across_runs'][s]
                lines.append(r'\quad Proposed, 5 independent runs (mean $\pm$ s.d.) & ' + ' & '.join(
                    [pm(A['e_dx']['med']), pm(A['e_drift']['med']), pm(A['e_M']['med']), pm(A['e_M']['p95']),
                     pm(A['e_N']['med'])]) + r' & --\\')
            lines.append(r'\midrule')
        write(fname, lines[:-1])


def tab_extended(runs):
    keys = [('e_dx', '$e_{u_x}$'), ('e_dy', '$e_{u_y}$'), ('e_rz', r'$e_\theta$'), ('e_drift', r'$e_\mathrm{drift}$'),
            ('e_defl', r'$e_\mathrm{defl}$'), ('e_M', '$e_M$'), ('e_N', '$e_N$'), ('e_V', '$e_V$'), ('e_R', '$e_R$'),
            ('e_Mmax', r'$e_{M,\max}$'), ('e_Nmax', r'$e_{N,\max}$')]
    lines = []
    for k, lab in keys:
        cells = []
        for s in TESTS:
            A = runs['across_runs'][s][k]
            cells += [pm(A['med']), pm(A['p95'])]
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    lines.append(r'\midrule')
    for k, lab in (('gov_top1', 'Governing member, top-1 (\\%)'), ('gov_top3', 'Governing member, top-3 (\\%)')):
        cells = []
        for s in TESTS:
            cells += [r'\multicolumn{2}{c}{' + f"{100 * runs['gov'][s][k]['mean']:.1f} $\\pm$ {100 * runs['gov'][s][k]['std']:.1f}" + '}']
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    write('tab_extended.tex', lines)


def tab_stats(runs):
    lines = []
    for s in TESTS:
        b = runs['bootstrap_ens0'][s]
        A = runs['across_runs'][s]
        d = runs['raw_minus_projected_median_eM']['across_runs'][s]
        bd = b['eM_raw_minus_proj']
        lines.append(f"{ABBR[s]} & {pct(b['e_M']['med'], 2)} [{pct(b['e_M']['med_ci'][0], 2)}, {pct(b['e_M']['med_ci'][1], 2)}]"
                     f" & {pct(b['e_M']['p95'], 1)} [{pct(b['e_M']['p95_ci'][0], 1)}, {pct(b['e_M']['p95_ci'][1], 1)}]"
                     f" & {pct(A['e_M']['med']['mean'], 2)} [{pct(A['e_M']['med']['ci95'][0], 2)}, {pct(A['e_M']['med']['ci95'][1], 2)}]"
                     f" & {pct(bd['est'], 2)} [{pct(bd['ci'][0], 2)}, {pct(bd['ci'][1], 2)}]"
                     f" & {pct(d['mean'], 2)} [{pct(d['ci95'][0], 2)}, {pct(d['ci95'][1], 2)}]" + r'\\')
    write('tab_stats.tex', lines)


GRAPH_ORDER = [
    ('Proposed, single network (depth 8)', 'Proposed network, single (8 steps)'),
    ('Conventional GNN (physical inputs)', 'Conventional GNN (physical inputs/targets)'),
    ('Global pooling', 'Proposed + global pooling'),
    ('StructGNN-style (storey pseudo nodes)', 'StructGNN-style'),
    ('Displacement-only + force recovery', 'Displacement-only + force recovery'),
    ('Displacement-only + stiffness residual (Leng-style)', 'Leng-style, residual weight $10^{-2}$'),
    ('Leng-style, residual weight 1e-4', 'Leng-style, residual weight $10^{-4}$'),
    ('Soft equilibrium penalty', 'Proposed + soft equilibrium penalty'),
    ('No load-case superposition', 'Proposed, no load-case split'),
    ('Depth 4', 'Depth 4'), ('Depth 12', 'Depth 12'), ('Depth 16', 'Depth 16'),
]


def tab_graph(gr):
    lines = []
    for key, lab in GRAPH_ORDER:
        if key not in gr:
            continue
        if key == 'Depth 4':
            lines.append(r'\midrule')
            lines.append(r'\multicolumn{9}{l}{\textit{Message-passing depth (proposed network, single)}}\\')
        R = gr[key]['raw']
        cells = []
        for s in TESTS:
            cells += [f"{100 * R[s]['e_dx']['med_mean']:.1f} $\\pm$ {100 * R[s]['e_dx']['med_std']:.1f}",
                      f"{100 * R[s]['e_M']['med_mean']:.1f} $\\pm$ {100 * R[s]['e_M']['med_std']:.1f}"]
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    write('tab_graph.tex', lines)
    # equilibrium residuals and projected moment errors of the graph baselines
    lines = []
    for key, lab in GRAPH_ORDER[:9]:
        if key not in gr:
            continue
        R, Pj = gr[key]['raw'], gr[key]['projected']
        cells = []
        for s in ('test_id', 'test_oodG'):
            cells += [sci(R[s]['eq_res']['med_mean']), f"{100 * Pj[s]['e_M']['med_mean']:.1f}",
                      f"{100 * R[s]['compat_M']['med_mean']:.1f}"]
        t = gr[key].get('train_s')
        if key.startswith('Proposed, single') or key in ('Soft equilibrium penalty', 'No load-case superposition'):
            tag = {'Soft equilibrium penalty': 'soft_s', 'No load-case superposition': 'nosup_s'}.get(key)
            files = (glob.glob(f'results/rev/{tag}*.pkl') if tag else glob.glob('results/rev/ens[1-4]_m*.pkl'))
            t = [pickle.load(open(f, 'rb'))['train_s'] for f in files]
        lines.append(lab + ' & ' + ' & '.join(cells) + (f" & {np.mean(t) / 60:.0f}" if t else ' & --') + r'\\')
    write('tab_graph_eq.tex', lines)


def tab_compat(var):
    names = [('raw', 'Raw ensemble forces'), ('euclidean', 'Euclidean projection'),
             ('variance-weighted', 'Variance-weighted projection'), ('recovered', 'Recovered from $\\hat{\\mathbf{u}}$')]
    lines = []
    for s in TESTS:
        lines.append(r'\multicolumn{7}{l}{\textit{' + LAB[s] + r'}}\\')
        for k, lab in names:
            v = var[k][s]
            lines.append(f"{lab} & {pct(v['e_M']['med'])} & {pct(v['e_N']['med'])} & {pct(v['e_V']['med'])} & "
                         f"{sci(v['eq_res']['med'])} & {pct(v['compat']['med'])} & {pct(v['compat_M']['med'])}" + r'\\')
        lines.append(r'\midrule')
    write('tab_compat.tex', lines[:-1])


def tab_sensitivity(sen):
    order = [('euclidean', 'Euclidean ($\\boldsymbol{\\Sigma}=\\mathbf{I}$)'), ('1.0', '$\\max(\\sigma^2, 1\\,\\bar\\sigma^2)$'),
             ('0.01', '$\\max(\\sigma^2, 10^{-2}\\bar\\sigma^2)$'), ('0.0001', '$\\max(\\sigma^2, 10^{-4}\\bar\\sigma^2)$'),
             ('1e-06', '$\\max(\\sigma^2, 10^{-6}\\bar\\sigma^2)$'), ('1e-08', '$\\max(\\sigma^2, 10^{-8}\\bar\\sigma^2)$'),
             ('additive 1e-6', '$\\sigma^2 + 10^{-6}\\bar\\sigma^2$ (used)')]
    lines = []
    for k, lab in order:
        cells = []
        for s in TESTS:
            v = sen[k][s]
            cells += [pct(v['e_M_med'], 2), f"{v['cond_max']:.0e}".replace('e+0', 'e').replace('e+', 'e')]
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    write('tab_sensitivity.tex', lines)


def tab_ood(ood):
    lines = []
    for grp, lab in (('geometry', 'Geometry descriptors'), ('load', 'Load-pattern descriptors'), ('combined', 'Combined (used)')):
        lines.append(lab + ' & ' + ' & '.join(pct(ood[grp]['flagged'][s]) for s in ['val'] + TESTS) + r'\\')
    write('tab_ood_domain.tex', lines)
    lines = []
    for tau in ('0.05', '0.1'):
        for s in TESTS:
            p = ood['policy'][tau][s]
            f = lambda x, d=1: '--' if x is None else pct(x, d)
            lines.append((f"{int(100 * float(tau))}\\%" if s == 'test_id' else '') + f" & {ABBR[s]} & {f(p['referred_domain'])} & "
                         f"{f(p['referred_spread'])} & {f(p['referred_total'])} & {f(p['coverage_in_domain'])} & "
                         f"{f(p['accepted_violation'])} & {f(p['accepted_p95'])} & {f(p['all_p95'])}" + r'\\')
        lines.append(r'\midrule')
    write('tab_ood_policy.tex', lines[:-1])


def tab_baseline_proj(base):
    lines = []
    for n in ['Linear ridge', 'Random forest', 'Gradient boosting', 'MLP']:
        if n not in base:
            continue
        cells = []
        for s in TESTS:
            a, b = base[n][s], base[n][s]['projected']
            cells += [f"{pct(a['e_M']['med'])} $\\to$ {pct(b['e_M']['med'])}", f"{pct(a['e_N']['med'])} $\\to$ {pct(b['e_N']['med'])}"]
        lines.append(n + ' & ' + ' & '.join(cells) + r'\\')
    write('tab_proj_baselines.tex', lines)


def tab_uq(u, ood):
    lines = []
    for s in TESTS:
        x = u[s]
        lines.append(f"{ABBR[s]} & {x['spearman_spread']:.2f} & {x['spearman_eq']:.2f} & {pct(x['coverage_2sigma'])} & "
                     f"{pct(x['frac_eM_gt_10'])} & {pct(x['recall_bad_at_20'], 0)} & {pct(x['p95_all'])} & {pct(x['p95_after_20'])}" + r'\\')
    write('tab_uq.tex', lines)


def tab_timing(u):
    tm = json.load(open('results/timing.json'))
    if 'dsm_per_frame_s' not in tm.get('test_id', {}):
        return None
    lines = []
    out = {}
    for s in ['test_id', 'test_oodG', 'test_oodL']:
        T = tm[s]
        f = lambda k: f"{1e3 * T[k]['mean']:.2f} $\\pm$ {1e3 * T[k]['std']:.2f}"
        bat = T['batched_per_frame_s']['mean']
        out[s] = dict(speedup_pynite=T['fe_per_frame_s']['mean'] / bat, speedup_dsm=T['dsm_per_frame_s']['mean'] / bat,
                      speedup_opensees=T['opensees_per_frame_s']['mean'] / bat)
        lines.append(f"{ABBR[s]} & {f('fe_per_frame_s')} & {f('opensees_per_frame_s')} & {f('dsm_per_frame_s')} & "
                     f"{f('batched_per_frame_s')} & {f('single_per_frame_s')} & {out[s]['speedup_pynite']:.1f} & "
                     f"{out[s]['speedup_dsm']:.2f}" + r'\\')
    write('tab_timing.tex', lines)
    json.dump(out, open(f'{RES}/speedups.json', 'w'), indent=1)
    return out


def dataset_stats(D):
    out = {}
    for s in ['pool', 'val'] + TESTS:
        recs = D[s]
        nm = np.array([len(F.topology(r['p'])['conn']) for r in recs])
        out[s] = dict(n=len(recs), storeys=[int(min(r['p']['n_st'] for r in recs)), int(max(r['p']['n_st'] for r in recs))],
                      bays=[int(min(r['p']['n_bay'] for r in recs)), int(max(r['p']['n_bay'] for r in recs))],
                      members=[int(nm.min()), int(nm.max()), float(nm.mean())],
                      t_fe_ms_mean=1e3 * float(np.mean([r['t_fe'] for r in recs])))
    json.dump(out, open(f'{RES}/dataset_stats.json', 'w'), indent=1)


def gov_stats(G):
    """Mean / s.d. over the five runs of the governing-member agreement rates."""
    out = {s: {'gov_top1': [], 'gov_top3': []} for s in TESTS}
    for r in range(5):
        P = AR.ensemble_preds(r)
        for s in TESTS:
            rows = [MT.frame_metrics_phys(g, *prop_phys(g, p), compat=False) for g, p in zip(G[s], P[s])]
            for k in out[s]:
                out[s][k].append(np.mean([x[k] for x in rows]))
    return {s: {k: dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1))) for k, v in d.items()} for s, d in out.items()}


def fig_depth(gr):
    keys = [('Depth 4', 4), ('Proposed, single network (depth 8)', 8), ('Depth 12', 12), ('Depth 16', 16)]
    if not all(k in gr for k, _ in keys):
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, met, lab in zip(axs, ['e_dx', 'e_M'], ['Median $e_{u_x}$ (%)', 'Median $e_M$ (%)']):
        for s, c in zip(TESTS, ['#1f4e9c', '#d08c2c', '#2a9d55', '#7b5ea7']):
            m = [100 * gr[k]['raw'][s][met]['med_mean'] for k, _ in keys]
            sd = [100 * gr[k]['raw'][s][met]['med_std'] for k, _ in keys]
            ax.errorbar([d for _, d in keys], m, yerr=sd, marker='o', ms=4, capsize=3, color=c, label=ABBR[s])
        ax.set_xticks([4, 8, 12, 16]); ax.set_xlabel('Message-passing steps'); ax.set_ylabel(lab); ax.grid(alpha=0.25)
    axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_depth.pdf')); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    D, G, P0, BP = load()
    var, runs, gr, base, sen, ood = J('variants'), J('runs'), J('graphs'), J('baselines'), J('sensitivity'), J('ood')
    dataset_stats(D)
    tab_validation()
    fig_frames(D)
    fig_cdf(G, P0, BP)
    fig_parity(G, P0, BP)
    fig_example(G, P0, BP)
    fig_equilibrium(G, P0, BP)
    u = uq(G, P0, ood)
    tab_uq(u, ood)
    fig_adaptive_and_table()
    if runs:
        if 'gov' not in runs:
            runs['gov'] = gov_stats(G)
            json.dump(runs, open(f'{RES}/runs.json', 'w'), indent=1)
        tab_extended(runs); tab_stats(runs)
    tab_main(base, runs, G, P0)
    if var:
        tab_compat(var)
    if sen:
        tab_sensitivity(sen)
    if ood:
        tab_ood(ood)
    if base:
        tab_baseline_proj(base)
    if gr:
        tab_graph(gr); fig_depth(gr)
    if os.path.exists('results/timing.json'):
        tab_timing(u)


if __name__ == '__main__':
    main()
