"""Post-process experiment outputs into the manuscript figures and tables.

    python make_figures.py     # writes ../../manuscript/surrogate/figures/*.pdf, tables/*.tex, results/uq.json
"""

import json
import os
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import features as FT
import frames as F
import metrics as MT

OUT = os.path.join('..', '..', 'manuscript', 'surrogate')
FIG, TAB = os.path.join(OUT, 'figures'), os.path.join(OUT, 'tables')
TESTS = ['test_id', 'test_oodG', 'test_oodL']
TEST_LABEL = {'test_id': 'In-distribution', 'test_oodG': 'Unseen geometries', 'test_oodL': 'Unseen loading'}
PROPOSED = 'GNN ensemble + variance-weighted projection'
SHORT = {
    'Linear ridge': 'Linear ridge', 'Quadratic ridge': 'Quadratic ridge',
    'Random forest': 'Random forest', 'Gradient boosting': 'Gradient boosting', 'MLP': 'MLP',
    'GNN (single, raw)': 'GNN, single (raw)', 'GNN ensemble (raw)': 'GNN ensemble (raw)',
    'GNN ensemble + Euclidean projection': 'GNN ensemble + Eucl. proj.',
    PROPOSED: 'Proposed: GNN ens. + var.-weighted proj.',
}
COL = {'Linear ridge': '#9e9e9e', 'Quadratic ridge': '#6d6d6d', 'Random forest': '#8c6d31',
       'Gradient boosting': '#d08c2c', 'MLP': '#7b5ea7', 'GNN ensemble (raw)': '#5b9bd5',
       PROPOSED: '#1f4e9c'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight'})


def load():
    D = pickle.load(open('data/frames.pkl', 'rb'))
    G = {s: [FT.graph(r) for r in D[s]] for s in TESTS + ['val']}
    main = json.load(open('results/main.json'))
    preds = pickle.load(open('results/main_preds.pkl', 'rb'))
    import glob
    ad = [json.load(open(f)) for f in sorted(glob.glob('results/adaptive_seed*.json'))] or None
    return D, G, main, preds, ad


def pct(x):
    return f'{100 * x:.1f}'


# ---------------------------------------------------------------------------
def fig_frames(D):
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.6))
    picks = {'test_id': 7, 'test_oodG': 3, 'test_oodL': 11}
    for ax, (s, k) in zip(axs, picks.items()):
        p = D[s][k]['p']; t = F.topology(p)
        for e, (i, j) in enumerate(t['conn']):
            xy = t['coords'][[i, j]]
            ax.plot(xy[:, 0], xy[:, 1], color='#1f4e9c' if t['mtype'][e] == 0 else '#333333', lw=1.2)
            if t['wg'][e] != 0:
                ax.fill_between(xy[:, 0], xy[0, 1], xy[0, 1] + 0.012 * abs(t['wg'][e]),
                                color='#d08c2c', alpha=0.35, lw=0)
        for kk in np.nonzero(t['P'][:, 0])[0]:
            x, y = t['coords'][kk]
            sgn = np.sign(t['P'][kk, 0])
            ax.annotate('', xy=(x, y), xytext=(x - sgn * (0.6 + 0.03 * abs(t['P'][kk, 0])), y),
                        arrowprops=dict(arrowstyle='->', color='#b2182b', lw=0.9))
        base = t['coords'][t['level'] == 0]
        ax.plot(base[:, 0], base[:, 1], 's' if p['fixed_base'] else '^', color='k', ms=4)
        ax.set_aspect('equal'); ax.set_xlabel('x (m)')
        ax.set_title(f"{TEST_LABEL[s]}: {p['n_st']} storeys, {p['n_bay']} bays")
    axs[0].set_ylabel('y (m)')
    fig.savefig(os.path.join(FIG, 'fig_frames.pdf')); plt.close(fig)


def fig_parity(G, preds):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.4))
    for c, s in enumerate(TESTS):
        for name, mk in [('Gradient boosting', 'x'), (PROPOSED, '.')]:
            roof_t, roof_p, M_t, M_p = [], [], [], []
            for g, pr in zip(G[s], preds[name][s]):
                u, q = FT.physical(g, pr['u'], pr['q'])
                top = g['t']['level'] == g['t']['level'].max()
                roof_t.append(np.mean(g['u_true'][top, 0]) * 1e3); roof_p.append(np.mean(u[top, 0]) * 1e3)
                np.argmax(np.abs(g['q_true'][:, 1:]))
                M_t.append(np.abs(g['q_true'][:, 1:]).max()); M_p.append(np.abs(q[:, 1:]).max())
            axs[0, c].plot(roof_t, roof_p, mk, ms=2.5, alpha=0.6, color=COL[name], label=SHORT[name])
            axs[1, c].plot(M_t, M_p, mk, ms=2.5, alpha=0.6, color=COL[name], label=SHORT[name])
        for r in range(2):
            lim = np.array(axs[r, c].get_xlim())
            axs[r, c].plot(lim, lim, 'k-', lw=0.6)
        axs[0, c].set_title(TEST_LABEL[s])
        axs[0, c].set_xlabel('PyNite roof displacement (mm)'); axs[1, c].set_xlabel(r'PyNite max $|M|$ (kN m)')
    axs[0, 0].set_ylabel('Predicted (mm)'); axs[1, 0].set_ylabel('Predicted (kN m)')
    axs[0, 0].legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_parity.pdf')); plt.close(fig)


def fig_cdf(G, preds):
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.2), sharey=True)
    names = ['Linear ridge', 'Random forest', 'Gradient boosting', 'MLP', 'GNN ensemble (raw)', PROPOSED]
    for c, s in enumerate(TESTS):
        for name in names:
            rows = [MT.frame_metrics(g, p['u'], p['q']) for g, p in zip(G[s], preds[name][s])]
            for r, key in enumerate(['e_dx', 'e_M']):
                v = np.sort([x[key] for x in rows])
                axs[r, c].plot(v, np.arange(1, len(v) + 1) / len(v), color=COL[name],
                               lw=1.6 if name == PROPOSED else 1.0, label=SHORT[name])
        axs[0, c].set_title(TEST_LABEL[s])
        for r, lab in enumerate([r'relative $L_2$ error, lateral displacements', r'relative $L_2$ error, end moments']):
            axs[r, c].set_xscale('log'); axs[r, c].set_xlim(3e-3, 3); axs[r, c].set_xlabel(lab)
            axs[r, c].grid(alpha=0.2)
    axs[0, 0].set_ylabel('Fraction of frames'); axs[1, 0].set_ylabel('Fraction of frames')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, 'fig_error_cdf.pdf')); plt.close(fig)


def fig_equilibrium(G, preds):
    names = ['Random forest', 'Gradient boosting', 'MLP', 'GNN ensemble (raw)', PROPOSED]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    data = []
    for name in names:
        v = [MT.equilibrium_residual(g, p['q']) for g, p in zip(G['test_id'], preds[name]['test_id'])]
        data.append(np.maximum(v, 1e-16))
    ax.boxplot(data, vert=False, whis=(5, 95), showfliers=False)
    ax.set_yticks(range(1, len(names) + 1), [SHORT[n] for n in names])
    ax.set_xscale('log'); ax.set_xlabel('Relative nodal out-of-balance force')
    fig.savefig(os.path.join(FIG, 'fig_equilibrium.pdf')); plt.close(fig)


def uq_analysis(G, preds, main):
    """Uncertainty quality and FE-referral (triage) analysis."""
    out = {}
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for c, s in enumerate(TESTS):
        P = preds[PROPOSED][s]; V = preds['_ens_var'][s]; R = preds['GNN ensemble (raw)'][s]
        err = np.array([MT.frame_metrics(g, p['u'], p['q'])['e_M'] for g, p in zip(G[s], P)])
        errdx = np.array([MT.frame_metrics(g, p['u'], p['q'])['e_dx'] for g, p in zip(G[s], P)])
        score_var, score_eq, cover = [], [], []
        for g, p, v, r in zip(G[s], P, V, R):
            # predicted relative std of the end moments (physical units)
            _, q = FT.physical(g, p['u'], p['q'])
            sdM = np.sqrt(sum(v['q_var'][:, 3 * k + 1:3 * k + 3] * g['qsc'][k][1] ** 2 for k in range(len(g['cases']))))
            score_var.append(np.linalg.norm(sdM) / (np.linalg.norm(q[:, 1:]) + 1e-12))
            score_eq.append(MT.equilibrium_residual(g, r['q']))
            cover.append(np.mean(np.abs(q[:, 1:] - g['q_true'][:, 1:]) <= 2 * sdM))
        score_var, score_eq = np.array(score_var), np.array(score_eq)
        rng = np.random.default_rng(0)
        fr = np.linspace(0, 0.5, 26)
        curves = {}
        for lab, sc in [('Ensemble spread', score_var), ('Equilibrium residual', score_eq),
                        ('Random', rng.random(len(err))), ('Oracle', err)]:
            order = np.argsort(-sc)
            curves[lab] = [float(np.percentile(err[order[int(round(f * len(err))):]], 95)) for f in fr]
        tau = 0.10
        out[s] = dict(spearman_var=float(spearmanr(score_var, err)[0]), spearman_eq=float(spearmanr(score_eq, err)[0]),
                      spearman_var_dx=float(spearmanr(score_var, errdx)[0]),
                      coverage_2sigma=float(np.mean(cover)),
                      frac_eM_gt_10=float(np.mean(err > tau)),
                      curves=curves, fractions=fr.tolist())
        # Recall of large-error frames when 10 % / 20 % are referred
        for f in (0.1, 0.2):
            k = int(round(f * len(err)))
            sel = np.argsort(-score_var)[:k]
            out[s][f'recall_bad_at_{int(100 * f)}'] = float(np.sum(err[sel] > tau) / max(np.sum(err > tau), 1))
            out[s][f'p95_after_{int(100 * f)}'] = float(np.percentile(err[np.argsort(-score_var)[k:]], 95))
            out[s][f'max_after_{int(100 * f)}'] = float(np.max(err[np.argsort(-score_var)[k:]]))
        for lab, st in [('Ensemble spread', '-'), ('Equilibrium residual', '--'), ('Random', ':'), ('Oracle', '-.')]:
            axs[c].plot(100 * fr, 100 * np.array(curves[lab]), st, lw=1.1, label=lab,
                        color={'Ensemble spread': '#1f4e9c', 'Equilibrium residual': '#d08c2c',
                               'Random': '#777777', 'Oracle': '#2a9d55'}[lab])
        axs[c].set_title(TEST_LABEL[s]); axs[c].set_xlabel('Frames referred to FE (%)')
        axs[c].grid(alpha=0.2)
    axs[0].set_ylabel('95th pct. moment error of\naccepted predictions (%)')
    axs[0].legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_triage.pdf')); plt.close(fig)

    # scatter for the in-distribution and unseen-geometry sets
    fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.3))
    for ax, s in zip(axs, ['test_id', 'test_oodG']):
        P = preds[PROPOSED][s]; V = preds['_ens_var'][s]
        e, sc = [], []
        for g, p, v in zip(G[s], P, V):
            _, q = FT.physical(g, p['u'], p['q'])
            sdM = np.sqrt(sum(v['q_var'][:, 3 * k + 1:3 * k + 3] * g['qsc'][k][1] ** 2 for k in range(len(g['cases']))))
            sc.append(np.linalg.norm(sdM) / np.linalg.norm(q[:, 1:])); e.append(MT.frame_metrics(g, p['u'], p['q'])['e_M'])
        ax.loglog(sc, e, '.', ms=2.5, alpha=0.5, color='#1f4e9c')
        ax.set_xlabel('Ensemble relative spread (moments)'); ax.set_title(f"{TEST_LABEL[s]} ($\\rho_s$={out[s]['spearman_var']:.2f})")
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, 'k-', lw=0.5)
    axs[0].set_ylabel('True relative moment error')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_uq_scatter.pdf')); plt.close(fig)

    # Workflow cost: refer the top-f fraction to PyNite, accept the rest
    if os.path.exists('results/timing.json'):
        tm = json.load(open('results/timing.json'))
        overhead = tm['train']['ensemble_s_est'] + tm['train']['fe_training_set_s']
        for s in TESTS:
            fe, sur = tm[s]['fe_mean_s'], tm[s]['batched_per_frame_s']
            out[s]['t_fe_mean'] = fe; out[s]['t_sur_mean'] = sur
            out[s]['speedup_0'] = fe / sur
            for f in (0.1, 0.2):
                cost = sur + f * fe
                out[s][f'speedup_at_{int(100 * f)}'] = fe / cost
                out[s][f'breakeven_at_{int(100 * f)}'] = overhead / (fe - cost) if fe > cost else float('inf')
    json.dump(out, open('results/uq.json', 'w'), indent=1)
    return out


def adaptive_curves(ad, strat, s, key):
    """Mean over seeds (and per-seed values) of a test metric along the rounds."""
    runs = [a['results'][strat]['history'] for a in ad if strat in a['results']]
    n_rounds = min(len(h) for h in runs)
    n = [runs[0][r]['n_fe'] for r in range(n_rounds)]
    vals = np.array([[h[r][s][key] for r in range(n_rounds)] for h in runs])
    return np.array(n), vals


def fig_adaptive(ad):
    if ad is None:
        return
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.0))
    sty = {'random': ('#777777', 'o', 'Random sampling'), 'ensemble variance': ('#1f4e9c', 's', 'Ensemble-variance acquisition'),
           'equilibrium residual': ('#d08c2c', '^', 'Equilibrium-residual acquisition')}
    for strat in ad[0]['results']:
        c, mk, lab = sty[strat]
        for j, s in enumerate(TESTS):
            for r, key in enumerate(['e_M_med', 'e_M_p95']):
                n, v = adaptive_curves(ad, strat, s, key)
                axs[r, j].plot(n, 100 * v.mean(0), marker=mk, ms=3, color=c, label=lab, lw=1)
                if len(v) > 1:
                    axs[r, j].fill_between(n, 100 * v.min(0), 100 * v.max(0), color=c, alpha=0.15, lw=0)
            axs[0, j].set_title(TEST_LABEL[s])
            axs[1, j].set_xlabel('Number of PyNite analyses')
    axs[0, 0].set_ylabel('Median moment error (%)'); axs[1, 0].set_ylabel('95th pct. moment error (%)')
    for a in axs.ravel():
        a.grid(alpha=0.2)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
    fig.savefig(os.path.join(FIG, 'fig_adaptive.pdf')); plt.close(fig)


def fig_example(G, preds):
    s = 'test_oodG'
    sizes = [len(g['conn']) for g in G[s]]
    k = int(np.argsort(sizes)[len(sizes) // 2])
    g, p = G[s][k], preds[PROPOSED][s][k]
    _, q = FT.physical(g, p['u'], p['q'])
    _, qb = FT.physical(g, preds['Gradient boosting'][s][k]['u'], preds['Gradient boosting'][s][k]['q'])
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    x = np.arange(2 * len(q))
    ax.plot(x, g['q_true'][:, 1:].ravel(), 'k-', lw=1.6, label='PyNite')
    ax.plot(x, q[:, 1:].ravel(), '.', color='#1f4e9c', ms=4, label=SHORT[PROPOSED])
    ax.plot(x, qb[:, 1:].ravel(), 'x', color='#d08c2c', ms=3, label='Gradient boosting')
    ax.set_xlabel('Member end index (columns first, then beams)'); ax.set_ylabel('End moment (kN m)')
    pp = g['t']
    ax.set_title(f"Unseen-geometry frame: {int(pp['level'].max())} storeys, {int(pp['line'].max())} bays")
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.28))
    fig.savefig(os.path.join(FIG, 'fig_example.pdf')); plt.close(fig)


# ---------------------------------------------------------------------------
def tables(main, uq, ad):
    R = main['results']
    rows = ['Linear ridge', 'Quadratic ridge', 'Random forest', 'Gradient boosting', 'MLP',
            'GNN (single, raw)', 'GNN ensemble (raw)', 'GNN ensemble + Euclidean projection', PROPOSED]
    lines = []
    for s in TESTS:
        lines.append(r'\multicolumn{8}{l}{\textit{' + TEST_LABEL[s] + r' (600 frames)}}\\')
        best = {k: min(R[n][s][k] for n in rows) for k in ['e_dx_med', 'e_drift_med', 'e_M_med', 'e_M_p95', 'e_N_med', 'e_Mmax_med']}
        for n in rows:
            r = R[n][s]
            cell = lambda k: (r'\textbf{' + pct(r[k]) + '}') if np.isclose(r[k], best[k]) else pct(r[k])
            eq = r['eq_res_med']
            if eq < 1e-12:
                eqs = '$<10^{-12}$'
            else:
                ex = int(np.floor(np.log10(eq)))
                eqs = f'${eq / 10 ** ex:.1f}\\times10^{{{ex}}}$'
            lines.append(f"{SHORT[n]} & {cell('e_dx_med')} & {cell('e_drift_med')} & {cell('e_M_med')} & "
                         f"{cell('e_M_p95')} & {cell('e_N_med')} & {cell('e_Mmax_med')} & {eqs}" + r'\\')
        lines.append(r'\midrule')
    open(os.path.join(TAB, 'tab_main.tex'), 'w').write('\n'.join(lines[:-1]) + '\n')

    # projection applied to baselines
    lines = []
    for n in ['Linear ridge', 'Random forest', 'Gradient boosting', 'MLP']:
        cells = []
        for s in TESTS:
            a, b = R[n][s], R[n + ' + Euclidean projection'][s]
            cells += [pct(a['e_M_med']), pct(b['e_M_med']), pct(a['e_N_med']), pct(b['e_N_med'])]
        lines.append(SHORT[n] + ' & ' + ' & '.join(cells) + r'\\')
    open(os.path.join(TAB, 'tab_proj_baselines.tex'), 'w').write('\n'.join(lines) + '\n')

    # ablation
    lines = []
    abl = [('GNN (single, raw)', 'Superposition, data loss only'),
           ('Ablation: soft equilibrium penalty', 'Superposition + soft equilibrium penalty'),
           ('Ablation: no load-case superposition', 'Single combined load scale'),
           ('Ablation: no load-case superposition + Euclidean projection', 'Single combined load scale + Eucl. proj.')]
    for n, lab in abl:
        cells = []
        for s in TESTS:
            r = R[n][s]
            cells += [pct(r['e_dx_med']), pct(r['e_M_med']), f"{r['eq_res_med']:.3f}" if r['eq_res_med'] > 1e-9 else '0']
        lines.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    open(os.path.join(TAB, 'tab_ablation.tex'), 'w').write('\n'.join(lines) + '\n')

    # UQ / workflow
    lines = []
    for s in TESTS:
        u = uq[s]
        lines.append(f"{TEST_LABEL[s]} & {u['spearman_var']:.2f} & {u['spearman_eq']:.2f} & {100 * u['coverage_2sigma']:.1f} & "
                     f"{100 * u['frac_eM_gt_10']:.1f} & {100 * u['recall_bad_at_20']:.0f} & {pct(u['p95_after_20'])} & "
                     f"{u.get('speedup_at_20', float('nan')):.1f}" + r'\\')
    open(os.path.join(TAB, 'tab_uq.tex'), 'w').write('\n'.join(lines) + '\n')

    # timing
    if os.path.exists('results/timing.json'):
        tm = json.load(open('results/timing.json'))
        lines = []
        for s in TESTS:
            T = tm[s]
            lines.append(f"{TEST_LABEL[s]} & {1e3 * T['fe_mean_s']:.1f} & {1e3 * T['features_per_frame_s']:.2f} & "
                         f"{1e3 * T['inference_per_frame_s']:.2f} & {1e3 * T['projection_per_frame_s']:.2f} & "
                         f"{1e3 * T['batched_per_frame_s']:.2f} & {1e3 * T['single_frame_end_to_end_s']:.1f} & "
                         f"{uq[s]['speedup_0']:.1f}" + r'\\')
        open(os.path.join(TAB, 'tab_timing.tex'), 'w').write('\n'.join(lines) + '\n')

    if ad is not None:
        lines = []
        for strat in ad[0]['results']:
            cells = []
            for s in TESTS:
                for key in ('e_M_med', 'e_M_p95', 'e_dx_med'):
                    _, v = adaptive_curves(ad, strat, s, key)
                    cells.append(pct(v[:, -1].mean()))
            lines.append(strat.capitalize() + ' & ' + ' & '.join(cells) + r'\\')
        open(os.path.join(TAB, 'tab_adaptive.tex'), 'w').write('\n'.join(lines) + '\n')
        # FE analyses needed by each strategy to reach the final random-sampling error
        summ = {}
        for s in TESTS:
            n, vr = adaptive_curves(ad, 'random', s, 'e_M_med')
            target = vr[:, -1].mean()
            summ[s] = {}
            for strat in ad[0]['results']:
                n, v = adaptive_curves(ad, strat, s, 'e_M_med')
                m = v.mean(0)
                hit = np.nonzero(m <= target)[0]
                summ[s][strat] = dict(final_med=float(m[-1]), n_to_random_final=int(n[hit[0]]) if len(hit) else None,
                                      final_p95=float(adaptive_curves(ad, strat, s, 'e_M_p95')[1][:, -1].mean()),
                                      n_seeds=len(v))
        json.dump(summ, open('results/adaptive_summary.json', 'w'), indent=1)


def dataset_stats(D):
    out = {}
    for s in ['pool', 'val'] + TESTS:
        recs = D[s]
        nst = np.array([r['p']['n_st'] for r in recs]); nb = np.array([r['p']['n_bay'] for r in recs])
        nm = np.array([len(F.topology(r['p'])['conn']) for r in recs])
        drift = np.array([MT.drift_ratios(g, g['u_true']).max() for g in (FT.graph(r) for r in recs)])
        out[s] = dict(drift_median=float(np.median(drift)), drift_frac_gt_2pct=float(np.mean(drift > 0.02)),
                      drift_frac_gt_1_400=float(np.mean(drift > 1 / 400)),n=len(recs), storeys=[int(nst.min()), int(nst.max())], bays=[int(nb.min()), int(nb.max())],
                      members=[int(nm.min()), int(nm.max()), float(nm.mean())],
                      t_fe_ms=[1e3 * float(np.mean([r['t_fe'] for r in recs])), 1e3 * float(np.max([r['t_fe'] for r in recs]))],
                      fixed_frac=float(np.mean([r['p']['fixed_base'] for r in recs])))
    json.dump(out, open('results/dataset_stats.json', 'w'), indent=1)


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    D, G, main_r, preds, ad = load()
    dataset_stats(D)
    fig_frames(D)
    fig_parity(G, preds)
    fig_cdf(G, preds)
    fig_equilibrium(G, preds)
    uq = uq_analysis(G, preds, main_r)
    fig_adaptive(ad)
    fig_example(G, preds)
    tables(main_r, uq, ad)


if __name__ == '__main__':
    main()
