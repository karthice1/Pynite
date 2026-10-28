"""Generate the figures (PDF) and LaTeX tables of the paper from results/*.json."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from run_experiments import FRAMES, BASE, PM, BETA_ULS, BETA_SLS, N_MC_OPT
from rbo.sections import COLUMN_SHAPES, BEAM_SHAPES

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'paper', 'figures')
TAB = os.path.join(HERE, 'paper', 'tables')
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

C_DDO, C_RBDO, C_3, C_4 = '#eb6834', '#2a78d6', '#1baf7a', '#6250d6'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.5,
    'savefig.bbox': 'tight', 'savefig.dpi': 300, 'legend.frameon': False,
})
W1, W2 = 3.5, 7.0   # single / double column width (in)
FRAME_LABEL = {'base': 'Base', 'S3': '3 storeys', 'S9': '9 storeys', 'B6': 'Bay 6 m',
               'B10': 'Bay 10 m', 'W08': '$p_w$ = 0.8 kPa', 'W16': '$p_w$ = 1.6 kPa'}
FRAME_ORDER = ['S3', 'base', 'S9', 'B6', 'B10', 'W08', 'W16']


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def best(frame, mode):
    d = load(f'opt_{frame}_{mode}.json')
    feas = [r for r in d['runs'] if r['feasible']] or d['runs']
    return min(feas, key=lambda r: r['weight'])


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    fig.savefig(os.path.join(FIG, name + '.png'))
    plt.close(fig)


def write_table(name, text):
    with open(os.path.join(TAB, name + '.tex'), 'w') as f:
        f.write(text)


# ------------------------------------------------------------------ figures
def fig_frame():
    sp = BASE
    fig, ax = plt.subplots(figsize=(W1, 3.2))
    lv = sp.levels
    xs = np.arange(sp.n_bays + 1) * sp.bay_width
    colors = [C_RBDO, C_DDO, C_3]
    for s in range(sp.n_storeys):
        g = sp.group_of_storey(s)
        for i, x in enumerate(xs):
            ext = i in (0, sp.n_bays)
            ax.plot([x, x], [lv[s], lv[s + 1]], color=INK if ext else INK2,
                    lw=2.2 if ext else 1.4, solid_capstyle='butt')
        ax.plot([xs[0], xs[-1]], [lv[s + 1], lv[s + 1]], color=colors[g % 3], lw=2.0)
        for k in range(sp.n_bays):
            for q in np.linspace(xs[k] + 0.4, xs[k + 1] - 0.4, 7):
                ax.annotate('', (q, lv[s + 1] + 0.05), (q, lv[s + 1] + 0.55),
                            arrowprops=dict(arrowstyle='-|>', lw=0.4, color=INK2, mutation_scale=4))
        ax.annotate('', (xs[0] - 0.1, lv[s + 1]), (xs[0] - 2.2, lv[s + 1]),
                    arrowprops=dict(arrowstyle='-|>', lw=0.8, color=INK, mutation_scale=6))
    for x in xs:
        ax.plot([x - 0.5, x + 0.5], [0, 0], color=INK, lw=1.5)
        for q in np.linspace(x - 0.5, x + 0.4, 5):
            ax.plot([q, q - 0.25], [0, -0.3], color=INK2, lw=0.5)
    for g in range(sp.n_groups):
        y = lv[min((g + 1) * sp.group_size, sp.n_storeys)] - 1.9
        ax.text(xs[-1] + 0.6, y, f'group {g + 1}', color=colors[g % 3], fontsize=7, va='center')
    ax.text(xs[0] - 2.4, lv[-1] + 0.7, '$W_i$', fontsize=7)
    ax.text(xs[1] / 2, lv[-1] + 0.9, '$D$, $L$', fontsize=7, ha='center')
    ax.set_xlim(-3, xs[-1] + 4.2)
    ax.set_ylim(-0.6, lv[-1] + 1.6)
    ax.set_aspect('equal')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{x:.0f}' for x in xs])
    ax.set_yticks(lv)
    ax.set_yticklabels([f'{y:.1f}' for y in lv])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('Elevation (m)')
    ax.grid(False)
    savefig(fig, 'frame')


def fig_ga_history():
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3), sharey=False)
    for ax, mode, c in zip(axs, ('DDO', 'RBDO'), (C_DDO, C_RBDO)):
        d = load(f'opt_base_{mode}.json')
        for k, r in enumerate(d['runs']):
            h = r['history']
            ev = [x['n_eval'] for x in h]
            w = [x['best_weight'] / 1000 if x['feasible'] else np.nan for x in h]
            ax.plot(ev, w, color=c, alpha=1.0 if k == 0 else 0.55, lw=1.5 if k == 0 else 1.0,
                    label=f'seed {r["seed"]}')
            ax.plot([ev[-1]], [r['weight'] / 1000], 'o', color=c, ms=4, mec='white', mew=0.8)
        ax.set_title(f'{mode}', loc='left', color=INK)
        ax.set_xlabel('Distinct designs evaluated')
        ax.set_ylabel('Best feasible mass (t)')
        ax.legend()
    savefig(fig, 'ga_history')


def fig_pareto():
    p = load('pareto_base.json')
    v = load('verify.json')
    front = v.get('pareto') or [{'weight': f['weight'], 'est': f['info']} for f in p['front']]
    w = np.array([f['weight'] for f in front]) / 1000
    bu = np.array([f['est']['beta_uls'] for f in front])
    bd = np.array([f['est']['beta_drift'] for f in front])
    nf = np.array([f['est']['nfail_uls'] for f in front])
    arch = p['archive']
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    ax = axs[0]
    arch = [a for a in arch if a['weight'] / 1000 <= 21.5]
    ax.scatter([a['weight'] / 1000 for a in arch], [a['beta_uls'] for a in arch], s=6, color=GRID,
               label='feasible designs visited', zorder=1)
    o = np.argsort(w)
    ax.plot(w[o], bu[o], '-', color=C_RBDO, lw=1.0, zorder=2)
    ax.scatter(w[nf > 0], bu[nf > 0], s=18, color=C_RBDO, edgecolor='white', lw=0.8, zorder=3,
               label='Pareto designs (independent MC)')
    if np.any(nf == 0):
        ax.scatter(w[nf == 0], bu[nf == 0], s=18, facecolor='white', edgecolor=C_RBDO, lw=1.0, zorder=3,
                   label='no failures in sample (lower bound)')
    for mode, c, mk in (('DDO', C_DDO, 's'), ('RBDO', C_RBDO, 'D')):
        e = v[f'base_{mode}']
        ax.scatter([e['weight'] / 1000], [e['est']['beta_uls']], marker=mk, s=30, color=c,
                   edgecolor=INK, lw=0.6, zorder=4, label=f'{mode} optimum')
    ax.axhline(BETA_ULS, color=INK2, lw=0.8, ls='--')
    ax.text(ax.get_xlim()[1], BETA_ULS, r' $\beta_T$', va='center', ha='right', color=INK2, fontsize=7)
    ax.set_xlim(18.0, 21.5)
    ax.set_xlabel('Steel mass (t)')
    ax.set_ylabel(r'System ULS reliability index $\beta_{ULS}$')
    ax.legend(loc='lower right')
    ax = axs[1]
    ax.plot(w[o], bd[o], '-', color=C_3, lw=1.0)
    ax.scatter(w, bd, s=18, color=C_3, edgecolor='white', lw=0.8, label='Pareto designs')
    for mode, c, mk in (('DDO', C_DDO, 's'), ('RBDO', C_RBDO, 'D')):
        e = v[f'base_{mode}']
        ax.scatter([e['weight'] / 1000], [e['est']['beta_drift']], marker=mk, s=30, color=c,
                   edgecolor=INK, lw=0.6, zorder=4, label=f'{mode} optimum')
    ax.axhline(BETA_SLS, color=INK2, lw=0.8, ls='--')
    ax.set_xlabel('Steel mass (t)')
    ax.set_ylabel(r'Drift SLS reliability index $\beta_{drift}$')
    ax.legend(loc='lower right')
    axs[0].set_title('(a) cost vs. ultimate reliability', loc='left')
    axs[1].set_title('(b) drift reliability of the same designs', loc='left')
    savefig(fig, 'pareto')


def fig_parametric():
    v = load('verify.json')
    frames = [f for f in FRAME_ORDER if f'{f}_DDO' in v and f'{f}_RBDO' in v]
    x = np.arange(len(frames))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
    wd = np.array([v[f'{f}_DDO']['weight'] for f in frames]) / 1000
    wr = np.array([v[f'{f}_RBDO']['weight'] for f in frames]) / 1000
    ax = axs[0]
    ax.bar(x - 0.19, wd, 0.36, color=C_DDO, label='DDO')
    ax.bar(x + 0.19, wr, 0.36, color=C_RBDO, label='RBDO')
    ax.set_xticks(x); ax.set_xticklabels([FRAME_LABEL[f] for f in frames], rotation=45, ha='right')
    ax.set_ylabel('Steel mass (t)'); ax.legend(); ax.set_title('(a) optimum mass', loc='left')
    ax = axs[1]
    for mode, c, off, mk in (('DDO', C_DDO, -0.12, 's'), ('RBDO', C_RBDO, 0.12, 'D')):
        b = [v[f'{f}_{mode}']['est']['beta_uls'] for f in frames]
        lb = [v[f'{f}_{mode}']['est']['nfail_uls'] == 0 for f in frames]
        for xi, bi, l in zip(x + off, b, lb):
            ax.scatter([xi], [bi], marker=mk, s=22, color='white' if l else c, edgecolor=c, lw=1.0)
        ax.scatter([], [], marker=mk, s=22, color=c, label=mode)
    ax.axhline(BETA_ULS, color=INK2, ls='--', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([FRAME_LABEL[f] for f in frames], rotation=45, ha='right')
    ax.set_ylabel(r'$\beta_{ULS}$'); ax.legend(); ax.set_title('(b) ultimate reliability', loc='left')
    ax = axs[2]
    for mode, c, off, mk in (('DDO', C_DDO, -0.12, 's'), ('RBDO', C_RBDO, 0.12, 'D')):
        b = [v[f'{f}_{mode}']['est']['beta_drift'] for f in frames]
        ax.scatter(x + off, b, marker=mk, s=22, color=c, edgecolor=c, label=mode)
    ax.axhline(BETA_SLS, color=INK2, ls='--', lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([FRAME_LABEL[f] for f in frames], rotation=45, ha='right')
    ax.set_ylabel(r'$\beta_{drift}$'); ax.legend(); ax.set_title('(c) drift reliability', loc='left')
    fig.tight_layout()
    savefig(fig, 'parametric')


def fig_distributions():
    v = load('verify.json')
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3))
    edges_u = np.linspace(0, 1.2, 121)
    edges_d = np.linspace(0, 2.4, 121)
    for mode, c in (('DDO', C_DDO), ('RBDO', C_RBDO)):
        e = v[f'base_{mode}']
        for ax, key, edges in ((axs[0], 'hist_U', edges_u), (axs[1], 'hist_drift', edges_d)):
            h = np.array(e[key], float)
            n = sum(h)
            # complementary CDF (exceedance probability), computed from the histogram
            exc = 1 - np.cumsum(h) / n
            exc = np.where(exc > 0, exc, np.nan)
            ax.semilogy(edges[1:], exc, color=c, label=mode)
    for ax, lim, lab in ((axs[0], 1.0, 'System utilisation $U_{max}$'),
                         (axs[1], 1.0, r'Max. drift ratio / ($h$/400)')):
        ax.axvline(lim, color=INK2, ls='--', lw=0.8)
        ax.set_xlabel(lab); ax.set_ylabel('Exceedance probability'); ax.legend()
    axs[0].set_title('(a) ultimate limit state', loc='left')
    axs[1].set_title('(b) drift serviceability', loc='left')
    savefig(fig, 'exceedance')


def fig_member_util():
    v = load('verify.json')
    fig, axs = plt.subplots(2, 1, figsize=(W2, 3.6), sharex=True)
    for ax, mode, c in zip(axs, ('DDO', 'RBDO'), (C_DDO, C_RBDO)):
        e = v[f'base_{mode}']
        names = e['member_names']
        x = np.arange(len(names))
        ax.bar(x, e['det_U_members'], 0.7, color=GRID, label='LRFD ratio (factored, $\\phi$)')
        ax.plot(x, e['U_member_median'], 'o', ms=3, color=c, label='MC median $U$')
        ax.plot(x, e['U_member_p999'], '^', ms=3.5, color=INK, label='MC 99.9% quantile $U$')
        ax.axhline(1.0, color=INK2, ls='--', lw=0.8)
        ax.set_ylabel(f'{mode}\nutilisation')
        ax.set_ylim(0, 1.2)
    axs[0].legend(ncol=3, loc='upper right')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names, rotation=90, fontsize=5.5)
    savefig(fig, 'member_util')


def fig_pdelta():
    o = load('opensees.json')
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6))
    for ax, mode in zip(axs, ('DDO', 'RBDO')):
        pd = o[mode]['pdelta']
        ns = len(pd['opensees_drift'])
        st = np.arange(1, ns + 1)
        h = BASE.heights
        for key, lab, c, mk, ls in (('first_order_drift', 'first order (PyNite)', INK2, None, ':'),
                                    ('b2_drift', 'first order $\\times B_2$ (surrogate)', C_RBDO, 'o', '-'),
                                    ('pynite_drift', 'PyNite P-$\\Delta$', C_DDO, 's', '-'),
                                    ('opensees_drift', 'OpenSees P-$\\Delta$', C_3, '^', '--')):
            ax.plot(np.array(pd[key]) / h * 1000, st, ls=ls, marker=mk, ms=3.5, color=c, label=lab, lw=1.2)
        ax.set_xlabel('Inter-storey drift ratio (‰)')
        ax.set_ylabel('Storey')
        ax.set_yticks(st)
        ax.set_title(f'({"ab"[mode == "RBDO"]}) {mode} optimum, 1.2D+0.5L+1.6W', loc='left')
    axs[0].legend(loc='lower left')
    savefig(fig, 'pdelta')


def fig_fullmc():
    f = load('fullmc.json')
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    for key, c, lab in (('ws1.0', C_RBDO, '$W_{50}$'), ('ws2.0', C_DDO, r'$2\,W_{50}$ (stress test)')):
        d = f[key]
        axs[0].scatter(d['U_full'], d['U_sur'], s=5, color=c, alpha=0.6, lw=0, label=lab)
        axs[1].scatter(d['D_full'], d['D_sur'], s=5, color=c, alpha=0.6, lw=0, label=lab)
    for ax, lab in ((axs[0], 'system utilisation $U_{max}$'), (axs[1], 'drift ratio / ($h$/400)')):
        lo, hi = 0, max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Full P-$\\Delta$ re-analysis: {lab}')
        ax.set_ylabel(f'Surrogate: {lab}')
        ax.legend(loc='upper left')
        ax.set_aspect('equal')
    axs[0].set_title('(a) ultimate', loc='left'); axs[1].set_title('(b) drift', loc='left')
    savefig(fig, 'fullmc')


def fig_mc_convergence():
    d = load('convergence.json')
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3))
    for ax, key, lab in ((axs[0], 'beta_uls', r'$\beta_{ULS}$'), (axs[1], 'beta_drift', r'$\beta_{drift}$')):
        for mode, c in (('DDO', C_DDO), ('RBDO', C_RBDO)):
            rows = d[mode]
            Ns = sorted(set(r['n'] for r in rows))
            reps = sorted(set(r['rep'] for r in rows))
            for rep in reps:
                y = [next(r[key] for r in rows if r['n'] == n and r['rep'] == rep) for n in Ns]
                ok = [next(r['nfail_uls'] for r in rows if r['n'] == n and r['rep'] == rep) > 0 or key != 'beta_uls'
                      for n in Ns]
                y = [yy if o else np.nan for yy, o in zip(y, ok)]
                ax.semilogx(Ns, y, color=c, lw=0.9, alpha=0.8, label=mode if rep == 0 else None)
        ax.axvline(N_MC_OPT, color=INK2, ls=':', lw=0.8)
        ax.set_xlabel('Number of samples $N$')
        ax.set_ylabel(lab)
        ax.legend()
    axs[0].set_title('(a) ultimate (5 independent streams)', loc='left')
    axs[1].set_title('(b) drift', loc='left')
    savefig(fig, 'mc_convergence')


def fig_sensitivity():
    s = load('sensitivity.json')
    keys = [k for k in s['DDO'] if k != 'baseline']
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), sharey=True)
    y = np.arange(len(keys))
    for ax, b, lab in ((axs[0], 'beta_uls', r'$\Delta\beta_{ULS}$'), (axs[1], 'beta_drift', r'$\Delta\beta_{drift}$')):
        for mode, c, off, mk in (('DDO', C_DDO, 0.15, 's'), ('RBDO', C_RBDO, -0.15, 'D')):
            base = s[mode]['baseline'][b]
            dv = [s[mode][k][b] - base for k in keys]
            ax.hlines(y + off, 0, dv, color=c, lw=1.5)
            ax.scatter(dv, y + off, marker=mk, s=16, color=c, label=f'{mode} (baseline {base:.2f})', zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlabel(lab)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    axs[0].set_yticks(y); axs[0].set_yticklabels(keys)
    axs[0].invert_yaxis()
    savefig(fig, 'sensitivity')


# ------------------------------------------------------------------- tables
def tab_random_variables():
    rows = PM.table()
    lines = [r'\begin{tabular}{llcc}', r'\toprule',
             r'Variable & Distribution & Mean/nominal & CoV \\', r'\midrule']
    for name, dist, bias, cov in rows:
        if 'theta' in name:
            lines.append(f'{name} & {dist} & 0 & s.d. 1/{1 / cov:.0f} \\\\')
        else:
            lines.append(f'{name} & {dist} & {bias:.2f} & {cov:.2f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('random_variables', '\n'.join(lines))


def tab_closed_form():
    d = load('closed_form.json')
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Quantity & Closed form & PyNite & Rel. error \\', r'\midrule']
    for r in d['static']:
        lines.append(f"{r['case']} ({r['unit']}) & {r['analytical']:.5g} & {r['pynite']:.5g} & "
                     f"{r['rel_err']:.1e} \\\\")
    lines.append(r'\midrule')
    lines.append(r'\multicolumn{4}{l}{Second-order cantilever, tip sway (mm): exact vs PyNite P-$\Delta$ with $n$ elements} \\')
    lines.append(r'$P/P_{cr}$ & exact & $n=1$ / $n=2$ / $n=8$ & err. ($n=1$, $n=8$) \\')
    for r in d['pdelta_cantilever']:
        e1 = abs(r['pynite_1'] - r['exact']) / r['exact']
        e8 = abs(r['pynite_8'] - r['exact']) / r['exact']
        lines.append(f"{r['P/Pcr']:.1f} & {r['exact'] * 1000:.3f} & {r['pynite_1'] * 1000:.3f} / "
                     f"{r['pynite_2'] * 1000:.3f} / {r['pynite_8'] * 1000:.3f} & {e1:.1e}, {e8:.1e} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('closed_form', '\n'.join(lines))


def tab_opensees():
    o = load('opensees.json')
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Design & Case & Drift & Column $M$ & Column $N$ & Beam end $M$ \\', r'\midrule']
    for key in [k for k in ('DDO', 'RBDO', 'S3_DDO', 'S9_DDO') if k in o]:
        lab = {'DDO': 'Base DDO', 'RBDO': 'Base RBDO', 'S3_DDO': '3-storey DDO', 'S9_DDO': '9-storey DDO'}[key]
        for r in o[key]['unit']:
            if r['case'] in ('D', 'L') and r['drift'] > 1e-3:
                dr = '--'
            else:
                dr = f"{r['drift']:.1e}"
            lines.append(f"{lab} & {r['case']} & {dr} & {r['col_M']:.1e} & {r['col_P']:.1e} & {r['beam_M']:.1e} \\\\")
            lab = ''
        pd = o[key]['pdelta']
        e_pn = np.max(np.abs(np.array(pd['pynite_drift']) - pd['opensees_drift']) / np.array(pd['opensees_drift']))
        e_b2 = np.max(np.abs(np.array(pd['b2_drift']) - pd['opensees_drift']) / np.array(pd['opensees_drift']))
        m_pn = np.max(np.abs(np.array(pd['pynite_base_M']) - pd['opensees_base_M']) / np.array(pd['opensees_base_M']))
        m_b2 = np.max(np.abs(np.array(pd['b2_base_M']) - pd['opensees_base_M']) / np.array(pd['opensees_base_M']))
        lines.append(f" & P-$\\Delta$: PyNite & {e_pn:.1e} & {m_pn:.1e} & & \\\\")
        lines.append(f" & P-$\\Delta$: $B_1$--$B_2$ & {e_b2:.1e} & {m_b2:.1e} & & \\\\")
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines += [r'\end{tabular}']
    write_table('opensees', '\n'.join(lines))


def tab_designs():
    v = load('verify.json')
    d, r = v['base_DDO'], v['base_RBDO']
    lines = [r'\begin{tabular}{lllllll}', r'\toprule',
             r' & \multicolumn{3}{c}{DDO} & \multicolumn{3}{c}{RBDO} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Storeys & Ext. col. & Int. col. & Beam & Ext. col. & Int. col. & Beam \\', r'\midrule']
    for a, b in zip(d['design'], r['design']):
        lines.append(f"{a['storeys']} & {a['ext_col']} & {a['int_col']} & {a['beam']} & "
                     f"{b['ext_col']} & {b['int_col']} & {b['beam']} \\\\")
    lines.append(r'\midrule')

    def fmt_beta(e, k):
        return (r'$>$' if e[f'nfail_{k}'] == 0 else '') + f"{e[f'beta_{k}']:.2f}"
    rows = [('Steel mass (t)', f"{d['weight'] / 1000:.2f}", f"{r['weight'] / 1000:.2f}"),
            ('LRFD strength ratio', f"{d['det']['strength']:.3f}", f"{r['det']['strength']:.3f}"),
            ('Drift ratio / limit (D+0.5L+W)', f"{d['det']['drift']:.3f}", f"{r['det']['drift']:.3f}"),
            ('Deflection / limit (L)', f"{d['det']['deflection']:.3f}", f"{r['det']['deflection']:.3f}"),
            ('Max. $B_2$ (DAM)', f"{d['det']['max_B2']:.3f}", f"{r['det']['max_B2']:.3f}"),
            (r'$\beta_{ULS}$ ($N=10^6$)', fmt_beta(d['est'], 'uls'), fmt_beta(r['est'], 'uls')),
            (r'$P_{f,ULS}$', f"{d['est']['pf_uls']:.2e}", f"{r['est']['pf_uls']:.2e}"),
            (r'$\beta_{drift}$', fmt_beta(d['est'], 'drift'), fmt_beta(r['est'], 'drift')),
            (r'$\beta_{defl}$', fmt_beta(d['est'], 'defl'), fmt_beta(r['est'], 'defl'))]
    for name, a, b in rows:
        lines.append(f"{name} & \\multicolumn{{3}}{{l}}{{{a}}} & \\multicolumn{{3}}{{l}}{{{b}}} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('designs', '\n'.join(lines))


def tab_parametric():
    v = load('verify.json')
    lines = [r'\begin{tabular}{lrrrrrrrrrr}', r'\toprule',
             r' & \multicolumn{3}{c}{DDO ($h/400$)} & \multicolumn{3}{c}{RBDO} & & \multicolumn{3}{c}{DDO ($h/500$)} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){9-11}',
             r'Frame & Mass (t) & $\beta_{ULS}$ & $\beta_{drift}$ & Mass (t) & $\beta_{ULS}$ & $\beta_{drift}$ & $\Delta$mass & Mass (t) & $\beta_{ULS}$ & $\beta_{drift}$ \\',
             r'\midrule']

    def fb(e, k):
        return (r'$>$' if e[f'nfail_{k}'] == 0 else '') + f"{e[f'beta_{k}']:.2f}"
    for f in FRAME_ORDER:
        if f'{f}_DDO' not in v or f'{f}_RBDO' not in v:
            continue
        a, b = v[f'{f}_DDO'], v[f'{f}_RBDO']
        dm = (b['weight'] - a['weight']) / a['weight'] * 100
        c = v.get(f'{f}_DDO-h500')
        h5 = (f"{c['weight'] / 1000:.2f} & {fb(c['est'], 'uls')} & {fb(c['est'], 'drift')}" if c else '-- & -- & --')
        lines.append(f"{FRAME_LABEL[f]} & {a['weight'] / 1000:.2f} & {fb(a['est'], 'uls')} & {fb(a['est'], 'drift')} & "
                     f"{b['weight'] / 1000:.2f} & {fb(b['est'], 'uls')} & {fb(b['est'], 'drift')} & "
                     f"{dm:+.1f}\\% & {h5} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('parametric', '\n'.join(lines))


def tab_targets():
    """Base frame: optimum mass for different reliability targets and drift limits."""
    v = load('verify.json')

    def fb(e, k):
        return (r'$>$' if e[f'nfail_{k}'] == 0 else '') + f"{e[f'beta_{k}']:.2f}"
    rows = [('DDO, drift $h/400$', '--', '--', v['base_DDO']),
            ('DDO, drift $h/500$', '--', '--', v.get('base_DDO-h500'))]
    for k in sorted(k for k in v if k.startswith('base_RBDO')):
        e = v[k]
        bu = e.get('beta_uls_T', BETA_ULS)
        bs = e.get('beta_sls_T', BETA_SLS)
        rows.append(('RBDO', f'{bu:.1f}', f'{bs:.1f}', e))
    rows = [r for r in rows if r[3] is not None]
    rows[2:] = sorted(rows[2:], key=lambda r: (r[1], r[2]))
    m0 = v['base_DDO']['weight']
    lines = [r'\begin{tabular}{lccrrrrr}', r'\toprule',
             r'Formulation & $\beta_T$ & $\beta_{SLS}$ & Mass (t) & $\Delta$mass & $\beta_{ULS}$ & $\beta_{drift}$ & LRFD ratio \\',
             r'\midrule']
    for name, bu, bs, e in rows:
        lines.append(f"{name} & {bu} & {bs} & {e['weight'] / 1000:.2f} & {(e['weight'] - m0) / m0 * 100:+.1f}\\% & "
                     f"{fb(e['est'], 'uls')} & {fb(e['est'], 'drift')} & {e['det']['strength']:.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('targets', '\n'.join(lines))


def tab_opt_runs():
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Frame & Mode & Seeds & Best mass (t) & Spread (\%) & Designs evaluated & Wall time (min) \\',
             r'\midrule']
    for f in FRAME_ORDER:
        for mode in ('DDO', 'DDO-h500', 'RBDO'):
            try:
                d = load(f'opt_{f}_{mode}.json')
            except FileNotFoundError:
                continue
            ws = [r['weight'] for r in d['runs'] if r['feasible']]
            spread = (max(ws) - min(ws)) / min(ws) * 100 if len(ws) > 1 else 0.0
            ne = np.mean([r['n_eval'] for r in d['runs']])
            tm = np.mean([r['time_s'] for r in d['runs']]) / 60
            lines.append(f"{FRAME_LABEL[f]} & {mode} & {len(d['runs'])} & {min(ws) / 1000:.2f} & {spread:.1f} & {ne:.0f} & {tm:.0f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('opt_runs', '\n'.join(lines))


def tab_frames():
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Frame & Storeys & Bays & Bay (m) & $p_w$ (kPa) & Groups & Variables \\', r'\midrule']
    for f in FRAME_ORDER:
        s = FRAMES[f]
        lines.append(f"{FRAME_LABEL[f]} & {s.n_storeys} & {s.n_bays} & {s.bay_width:.0f} & {s.p_wind:.1f} & "
                     f"{s.n_groups} & {s.n_vars} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('frames', '\n'.join(lines))


def summary():
    """Print the numbers quoted in the manuscript text."""
    out = {}
    v = load('verify.json')
    for k, e in v.items():
        if k == 'pareto':
            continue
        out[k] = {'mass_t': round(e['weight'] / 1000, 3), 'beta_uls': round(e['est']['beta_uls'], 3),
                  'nfail_uls': e['est']['nfail_uls'], 'beta_drift': round(e['est']['beta_drift'], 3),
                  'beta_defl': round(e['est']['beta_defl'], 3), 'lrfd': round(e['det']['strength'], 3),
                  'drift_det': round(e['det']['drift'], 3), 'B2': round(e['det']['max_B2'], 3)}
        if 'governing_member_counts' in e:
            out[k]['governing'] = e['governing_member_counts']
    with open(os.path.join(RES, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == '__main__':
    import sys
    todo = sys.argv[1:] or None
    for fn in [fig_frame, fig_ga_history, fig_pareto, fig_parametric, fig_distributions, fig_member_util,
               fig_pdelta, fig_fullmc, fig_mc_convergence, fig_sensitivity, tab_random_variables, tab_closed_form,
               tab_opensees, tab_designs, tab_parametric, tab_targets, tab_opt_runs, tab_frames, summary]:
        if todo and fn.__name__ not in todo:
            continue
        try:
            fn()
            print('ok', fn.__name__)
        except FileNotFoundError as e:
            print('skip', fn.__name__, e)
