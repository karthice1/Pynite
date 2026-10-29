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
    for ax, mode, c in zip(axs, ('DDO', 'RBDO-C'), (C_DDO, C_RBDO)):
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




def fig_distributions():
    v = load('verify.json')
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.3))
    edges_u = np.linspace(0, 1.2, 121)
    edges_d = np.linspace(0, 2.4, 121)
    for mode, c in (('DDO', C_DDO), ('RBDO-C', C_RBDO)):
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
    for ax, mode, c in zip(axs, ('DDO', 'RBDO-C'), (C_DDO, C_RBDO)):
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
    for ax, mode in zip(axs, ('DDO', 'RBDO-C')):
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
        ax.set_title(f'({"ab"[mode != "DDO"]}) {mode} optimum, 1.2D+0.5L+1.6W', loc='left')
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
        for mode, c in (('DDO', C_DDO), ('RBDO-C', C_RBDO)):
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
        for mode, c, off, mk in (('DDO', C_DDO, 0.15, 's'), ('RBDO-C', C_RBDO, -0.15, 'D')):
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
    for key in [k for k in ('DDO', 'RBDO-C', 'S3_DDO', 'S9_DDO') if k in o]:
        lab = {'DDO': 'Base DDO', 'RBDO-C': 'Base RBDO-C', 'S3_DDO': '3-storey DDO', 'S9_DDO': '9-storey DDO'}[key]
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






def tab_frames():
    lines = [r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'Frame & Storeys & Bays & Bay (m) & $p_w$ (kPa) & Groups & Variables \\', r'\midrule']
    for f in FRAME_ORDER:
        s = FRAMES[f]
        lines.append(f"{FRAME_LABEL[f]} & {s.n_storeys} & {s.n_bays} & {s.bay_width:.0f} & {s.p_wind:.1f} & "
                     f"{s.n_groups} & {s.n_vars} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('frames', '\n'.join(lines))



# ------------------------------------------------------- revision: helpers
MODES3 = (('DDO', 'DDO', C_DDO, 's'), ('RBDO', 'RBDO (reliability only)', C_4, 'o'),
          ('RBDO-C', 'RBDO-C (code-compliant)', C_RBDO, 'D'))
TARGET = {'uls': BETA_ULS, 'drift': BETA_SLS, 'defl': BETA_SLS}


def fb(e, k, d=2):
    """Point estimate of beta, or the one-sided 95 % lower confidence bound when no failure."""
    if e[f'nfail_{k}'] == 0:
        return f"$\\geq${e[f'beta_lcb_{k}']:.{d}f}"
    return f"{e[f'beta_{k}']:.{d}f}"


def fci(e, k):
    lo, hi = e[f'ci_beta_{k}']
    if e[f'nfail_{k}'] == 0:
        return f"[{e[f'beta_lcb_{k}']:.2f}, $\\infty$)"
    return f"[{lo:.2f}, {hi:.2f}]"


def verdict(e, k, target=None):
    """Position of the target relative to the exact 95 % interval of beta."""
    t = TARGET[k] if target is None else target
    lo, hi = e[f'ci_beta_{k}']
    if e[f'nfail_{k}'] == 0:
        lo = e[f'beta_lcb_{k}']
    if lo >= t:
        return 'above'
    if hi < t:
        return 'below'
    return 'within'


def n_types(design):
    return len({d['ext_col'] for d in design} | {d['int_col'] for d in design} | {d['beam'] for d in design})


def mass_split(frame, x):
    """Steel mass (kg) of exterior columns, interior columns and beams."""
    from rbo.sections import COLUMN_SHAPES as CS, BEAM_SHAPES as BS
    sp = FRAMES[frame]
    out = np.zeros(3)
    for st in range(sp.n_storeys):
        g = sp.group_of_storey(st)
        h = sp.heights[st]
        out[0] += 2 * h * CS[x[3 * g]].mass_per_m
        out[1] += (sp.n_bays - 1) * h * CS[x[3 * g + 1]].mass_per_m
        out[2] += sp.n_bays * sp.bay_width * BS[x[3 * g + 2]].mass_per_m
    return out


def ver():
    return load('verify.json')


# ------------------------------------------------------- revision: figures
def fig_parametric():
    v = ver()
    frames = [f for f in FRAME_ORDER if all(f'{f}_{m}' in v for m, *_ in MODES3)]
    x = np.arange(len(frames))
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.6))
    off = {'DDO': -0.26, 'RBDO': 0.0, 'RBDO-C': 0.26}
    ax = axs[0]
    for m, lab, c, mk in MODES3:
        ax.bar(x + off[m], [v[f'{f}_{m}']['weight'] / 1000 for f in frames], 0.25, color=c, label=lab)
    ax.set_ylabel('Steel mass (t)'); ax.set_title('(a) optimum mass', loc='left')
    for ax, k, lab, t in ((axs[1], 'uls', r'$\beta_{ULS}$', '(b) system ULS reliability'),
                          (axs[2], 'drift', r'$\beta_{drift}$', '(c) drift SLS reliability')):
        for m, l, c, mk in MODES3:
            for xi, f in zip(x + off[m], frames):
                e = v[f'{f}_{m}']['est']
                lo, hi = e[f'ci_beta_{k}']
                ax.errorbar([xi], [e[f'beta_{k}']], yerr=[[e[f'beta_{k}'] - lo], [hi - e[f'beta_{k}']]],
                            fmt=mk, ms=3.5, color=c, mec=c, elinewidth=0.8, capsize=1.5)
        ax.axhline(TARGET[k], color=INK2, ls='--', lw=0.8)
        ax.set_ylabel(lab); ax.set_title(t, loc='left')
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels([FRAME_LABEL[f] for f in frames], rotation=45, ha='right')
    handles = [plt.Line2D([], [], marker=mk, color=c, ls='none', ms=5, label=lab) for m, lab, c, mk in MODES3]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, 'parametric')


def fig_pareto():
    p = load('pareto_base.json')
    v = ver()
    front = v['pareto']
    w = np.array([f['weight'] for f in front]) / 1000
    o = np.argsort(w)
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.7))
    arch = [a for a in p['archive'] if a['weight'] / 1000 <= 21.5]
    axs[0].scatter([a['weight'] / 1000 for a in arch], [a['beta_uls'] for a in arch], s=6, color=GRID,
                   label='feasible designs visited (SAA, $N=5\\times10^4$)', zorder=1)
    for ax, k, c in ((axs[0], 'uls', C_RBDO), (axs[1], 'drift', C_3)):
        b = np.array([f['est'][f'beta_{k}'] for f in front])
        ci = np.array([f['est'][f'ci_beta_{k}'] for f in front])
        lrfd = np.array([f.get('lrfd', 0.0) for f in front])
        ax.plot(w[o], b[o], '-', color=c, lw=0.8, zorder=2)
        ok = lrfd <= 1.0
        for sel, fc, lab in ((ok, c, 'non-dominated, LRFD-compliant'), (~ok, 'white', 'non-dominated, LRFD ratio $>1$')):
            if np.any(sel):
                ax.errorbar(w[sel], b[sel], yerr=[b[sel] - ci[sel, 0], ci[sel, 1] - b[sel]], fmt='o', ms=4,
                            mfc=fc, mec=c, color=c, elinewidth=0.8, capsize=1.5, zorder=3,
                            label=lab + ' (independent $N=4\\times10^5$, 95% CI)')
        for mode, cc, mk in (('DDO', C_DDO, 's'), ('RBDO-C', C_RBDO, 'D')):
            e = v[f'base_{mode}']
            ax.scatter([e['weight'] / 1000], [e['est'][f'beta_{k}']], marker=mk, s=30, color=cc,
                       edgecolor=INK, lw=0.6, zorder=4, label=f'{mode} optimum ($N=10^6$)')
        ax.axhline(TARGET[k], color=INK2, lw=0.8, ls='--')
        ax.set_xlabel('Steel mass (t)')
    axs[0].set_xlim(18.0, 21.5)
    axs[1].set_xlim(18.0, 21.5)
    axs[0].set_ylabel(r'System ULS reliability index $\beta_{ULS}$')
    axs[1].set_ylabel(r'Drift SLS reliability index $\beta_{drift}$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6)
    axs[0].set_title('(a) mass vs. ultimate reliability', loc='left')
    axs[1].set_title('(b) drift reliability of the same designs', loc='left')
    savefig(fig, 'pareto')


def fig_validation():
    keys = [k for k in VAL_KEYS if os.path.exists(os.path.join(RES, f'validate_{k}.json'))]
    cols = plt.get_cmap('tab10')
    fig, axs = plt.subplots(1, 3, figsize=(W2, 2.5))
    for i, k in enumerate(keys):
        d = load(f'validate_{k}.json')
        c = cols(i % 10)
        lab = val_label(k)
        for st, mk in (('A', 'o'), ('B', '.')):
            t = d['tail'][st]
            axs[0].scatter(t['U_full'], t['U_sur'], s=5 if st == 'A' else 3, color=c, lw=0, alpha=0.7,
                           label=lab if st == 'A' else None)
        r = d['random']
        axs[1].scatter(r['D_full'], r['D_sur'], s=3, color=c, lw=0, alpha=0.6)
    for ax, lo, hi, lab in ((axs[0], 0.8, 1.35, 'system utilisation $U$ (tail strata)'),
                            (axs[1], 0.0, 2.0, 'drift / ($h$/400) (random set, $S_2$)')):
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8)
        ax.axvline(1, color=GRID, lw=0.8); ax.axhline(1, color=GRID, lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel(f'Full P-$\\Delta$: {lab}', fontsize=6.5)
        ax.set_ylabel(f'Surrogate', fontsize=6.5)
    axs[0].set_title('(a) ULS tail', loc='left'); axs[1].set_title('(b) drift', loc='left')
    ax = axs[2]
    rows = val_rows()
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        ax.errorbar([r['dbeta']], [yi], xerr=[[r['dbeta'] - r['dbeta_ci'][0]], [r['dbeta_ci'][1] - r['dbeta']]],
                    fmt='o', ms=3.5, color=cols(list(keys).index(r['key']) % 10), capsize=1.5, elinewidth=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([val_label(r['key']) for r in rows], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel(r'$\beta_{ULS}$(full) $-$ $\beta_{ULS}$(surrogate)', fontsize=6.5)
    ax.set_title('(c) effect on $\\beta_{ULS}$ (95% CI)', loc='left')
    fig.tight_layout()
    savefig(fig, 'validation')


def fig_driftlimits():
    v = ver()
    lims = [400, 450, 500, 600]
    fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
    cols = plt.get_cmap('tab10')
    for i, f in enumerate(FRAME_ORDER):
        ks = [f'{f}_DDO' if L == 400 else f'{f}_DDO-h{L}' for L in lims]
        if not all(k in v for k in ks) or f'{f}_RBDO-C' not in v:
            continue
        m0 = v[f'{f}_DDO']['weight']
        bd = [v[k]['est']['beta_drift'] for k in ks]
        dm = [(v[k]['weight'] - m0) / m0 * 100 for k in ks]
        axs[0].plot(lims, bd, '-o', ms=3, color=cols(i), label=FRAME_LABEL[f])
        axs[1].plot(lims, dm, '-o', ms=3, color=cols(i), label=FRAME_LABEL[f])
        r = v[f'{f}_RBDO-C']
        axs[1].scatter([660], [(r['weight'] - m0) / m0 * 100], marker='D', s=16, color=cols(i))
    axs[0].axhline(BETA_SLS, color=INK2, ls='--', lw=0.8)
    axs[0].set_ylabel(r'$\beta_{drift}$ of DDO optimum'); axs[1].set_ylabel(r'$\Delta$mass vs. DDO $h/400$ (%)')
    for ax in axs:
        ax.set_xticks(lims + ([660] if ax is axs[1] else []))
        ax.set_xticklabels([f'h/{L}' for L in lims] + (['RBDO-C'] if ax is axs[1] else []))
        ax.set_xlabel('Deterministic drift limit')
    axs[0].legend(fontsize=5.5, ncol=2)
    axs[0].set_title('(a) drift reliability', loc='left'); axs[1].set_title('(b) extra steel', loc='left')
    fig.tight_layout()
    savefig(fig, 'driftlimits')


# ------------------------------------------------------- revision: tables
def tab_random_variables():
    src = {
        'Dead load $D$': ('Ellingwood et al.\\ (1980)', 'sourced'),
        '50-yr max. live load $L_{50}$': ('Ellingwood et al.\\ (1980)', 'sourced'),
        'Arbitrary-point-in-time live load $L_{apt}$': ('Ellingwood et al.\\ (1980)', 'sourced'),
        '50-yr max. wind load $W_{50}$': ('Ellingwood et al.\\ (1980)', 'sourced'),
        'Elastic modulus $E$': ('JCSS (2001): c.o.v., lognormal', 'sourced'),
        'Yield stress $F_y$ (per member)': ('JCSS (2001): c.o.v., lognormal; mean ratio and $\\rho$', 'assumed (mean, $\\rho$)'),
        'Fabrication factor $F$ (per member)': ('Ellingwood et al.\\ (1980)', 'sourced'),
        'Out-of-plumbness $\\theta$ (rad)': ('1/500 tolerance taken as 2 s.d.', 'assumed'),
    }
    lines = [r'\begin{tabular}{llccll}', r'\toprule',
             r'Variable & Distribution & Mean/nominal & c.o.v. & Basis & Status \\', r'\midrule']
    for name, dist, bias, cov in PM.table():
        s, st = src.get(name, ('', ''))
        if 'theta' in name:
            lines.append(f'{name} & {dist} & 0 & s.d. 1/{1 / cov:.0f} & {s} & {st} \\\\')
        else:
            lines.append(f'{name} & {dist} & {bias:.2f} & {cov:.2f} & {s} & {st} \\\\')
    lines += [r'\midrule',
              r'Model factors $\theta_R,\theta_E,\theta_\delta$ (frame-wide) & Lognormal & 1.00 & 0 (0.05, 0.10) & '
              r'baseline: 0; sensitivity study: 0.05, 0.10; RBDO-C-MU (2 frames): 0.05 & assumed \\',
              r'\bottomrule', r'\end{tabular}']
    write_table('random_variables', '\n'.join(lines))


def tab_designs():
    v = ver()
    d, r = v['base_DDO'], v['base_RBDO-C']
    lines = [r'\begin{tabular}{lllllll}', r'\toprule',
             r' & \multicolumn{3}{c}{DDO} & \multicolumn{3}{c}{RBDO-C} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Storeys & Ext. col. & Int. col. & Beam & Ext. col. & Int. col. & Beam \\', r'\midrule']
    for a, b in zip(d['design'], r['design']):
        lines.append(f"{a['storeys']} & {a['ext_col']} & {a['int_col']} & {a['beam']} & "
                     f"{b['ext_col']} & {b['int_col']} & {b['beam']} \\\\")
    lines.append(r'\midrule')

    def pf(e, k):
        lo, hi = e[f'ci_pf_{k}']
        return f"{e[f'nfail_{k}'] / e[f'n_{k}']:.3e} [{lo:.3e}, {hi:.3e}]"
    rows = [('Steel mass (kg)', f"{d['weight']:.0f}", f"{r['weight']:.0f}"),
            ('Distinct sections', f"{n_types(d['design'])}", f"{n_types(r['design'])}"),
            ('LRFD strength ratio', f"{d['det']['strength']:.3f}", f"{r['det']['strength']:.3f}"),
            ('Drift ratio / limit (D+0.5L+W)', f"{d['det']['drift']:.3f}", f"{r['det']['drift']:.3f}"),
            ('Deflection / limit (L)', f"{d['det']['deflection']:.3f}", f"{r['det']['deflection']:.3f}"),
            ('Max. $B_2$ (DAM)', f"{d['det']['max_B2']:.3f}", f"{r['det']['max_B2']:.3f}"),
            (r'ULS failures $k$ in $N=10^6$', f"{d['est']['nfail_uls']}", f"{r['est']['nfail_uls']}"),
            (r'$\hat P_{f,ULS}$ [95\% CI]', pf(d['est'], 'uls'), pf(r['est'], 'uls')),
            (r'$\hat\beta_{ULS}$ [95\% CI]', f"{d['est']['beta_uls']:.3f} {fci(d['est'], 'uls')}",
             f"{r['est']['beta_uls']:.3f} {fci(r['est'], 'uls')}"),
            (r'Drift failures $k$', f"{d['est']['nfail_drift']}", f"{r['est']['nfail_drift']}"),
            (r'$\hat\beta_{drift}$ [95\% CI]', f"{d['est']['beta_drift']:.3f} {fci(d['est'], 'drift')}",
             f"{r['est']['beta_drift']:.3f} {fci(r['est'], 'drift')}"),
            (r'Deflection failures $k$; $\beta_{defl}$', f"{d['est']['nfail_defl']}; {fb(d['est'], 'defl')}",
             f"{r['est']['nfail_defl']}; {fb(r['est'], 'defl')}")]
    for name, a, b in rows:
        lines.append(f"{name} & \\multicolumn{{3}}{{l}}{{{a}}} & \\multicolumn{{3}}{{l}}{{{b}}} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('designs', '\n'.join(lines))


def tab_verification():
    """All final designs: mass, code checks, failure counts, CIs and verdict vs. the targets."""
    v = ver()
    lines = [r'\begin{tabular}{llrrrrlrll}', r'\toprule',
             r'Frame & Formulation & Mass (kg) & $\Delta m$ & LRFD & $k_{ULS}$ & $\hat\beta_{ULS}$ [95\% CI] & '
             r'$k_{drift}$ & $\hat\beta_{drift}$ [95\% CI] & vs. targets \\', r'\midrule']
    for f in FRAME_ORDER:
        m0 = v[f'{f}_DDO']['weight']
        first = True
        for m, lab, *_ in MODES3:
            e = v.get(f'{f}_{m}')
            if e is None:
                continue
            s = e['est']
            vd = f"{verdict(s, 'uls')[0]}/{verdict(s, 'drift')[0]}"
            dm = '--' if m == 'DDO' else f"{(e['weight'] - m0) / m0 * 100:+.2f}\\%"
            lines.append(f"{FRAME_LABEL[f] if first else ''} & {m} & {e['weight']:.0f} & {dm} & "
                         f"{e['det']['strength']:.3f} & {s['nfail_uls']} & {s['beta_uls']:.3f} {fci(s, 'uls')} & "
                         f"{s['nfail_drift']} & {s['beta_drift']:.3f} {fci(s, 'drift')} & {vd} \\\\")
            first = False
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    write_table('verification', '\n'.join(lines))


def claim(e, k, target=None):
    """Strongest of the three claims of Section 2.5: 'L' (one-sided 95 % lower bound >= target),
    'P, I' (point estimate >= target, interval contains it), 'I' (point estimate below the target,
    interval contains it) or 'below'."""
    t = TARGET[k] if target is None else target
    if e[f'beta_lcb_{k}'] >= t:
        return 'L'
    if verdict(e, k, t) == 'below':
        return 'below'
    return 'P, I' if e[f'beta_{k}'] >= t else 'I'


def tab_strict():
    """Point-estimate (RBDO-C) vs confidence-bounded (RBDO-C95) designs, drift bias and model-
    uncertainty-aware designs."""
    v = ver()
    db = load('driftbias.json')
    lines = [r'\begin{tabular}{llrrllllll}', r'\toprule',
             r'Frame & Formulation & Mass (kg) & $\Delta m$ vs.\ RBDO-C & $\hat\beta_{ULS}$ ($\beta^{lcb}$) & claim & '
             r'$\hat\beta_{drift}$ ($\beta^{lcb}$) & claim & $\hat\beta_{drift}$, drift $+1\%$ ($\beta^{lcb}$) & claim \\',
             r'\midrule']
    for f in FRAME_ORDER:
        m0 = v[f'{f}_RBDO-C']['weight']
        first = True
        for key, lab in ((f'{f}_RBDO-C', 'RBDO-C'), (f'{f}_RBDO-C_lcb', 'RBDO-C95'),
                         (f'{f}_RBDO-C_bu3.07_bs1.5', r'RBDO-C, $\beta_T=3.07$')):
            if key not in v:
                continue
            e = v[key]
            s = e['est']
            dm = '--' if lab == 'RBDO-C' else f"{(e['weight'] - m0) / m0 * 100:+.2f}\\%"
            dkey = key if key in db else next((k for k in db if k.startswith(f + '_') and v[k]['x'] == e['x']), None)
            if dkey:
                d1 = db[dkey]['1.01']
                c1 = 'L' if d1['beta_lcb'] >= BETA_SLS else ('below' if d1['ci_beta'][1] < BETA_SLS else
                                                            ('P, I' if d1['beta'] >= BETA_SLS else 'I'))
                bias = f"{d1['beta']:.3f} ({d1['beta_lcb']:.3f}) & {c1}"
            else:
                bias = '-- & --'
            lines.append(f"{FRAME_LABEL[f] if first else ''} & {lab} & {e['weight']:.0f} & {dm} & "
                         f"{s['beta_uls']:.3f} ({s['beta_lcb_uls']:.3f}) & {claim(s, 'uls')} & "
                         f"{s['beta_drift']:.3f} ({s['beta_lcb_drift']:.3f}) & {claim(s, 'drift')} & {bias} \\\\")
            first = False
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    write_table('strict', '\n'.join(lines))
    # model-uncertainty-aware designs
    lines = [r'\begin{tabular}{llrrlll}', r'\toprule',
             r'Frame & Formulation & Mass (kg) & $\Delta m$ vs.\ RBDO-C & '
             r'$\hat\beta_{ULS}$, c.o.v.\ 0.05 ($\beta^{lcb}$) & $\hat\beta_{drift}$, c.o.v.\ 0.05 ($\beta^{lcb}$) & '
             r'$\hat\beta_{ULS}$ / $\hat\beta_{drift}$ without model factors \\', r'\midrule']
    mu = load('modelunc.json')
    for f in ('S3', 'base'):
        e0 = v[f'{f}_RBDO-C']
        m0 = e0['weight']
        u = mu[f'{f}_RBDO-C']['MU-a']
        lines.append(f"{FRAME_LABEL[f]} & RBDO-C & {m0:.0f} & -- & {u['beta_uls']:.3f} ({u['beta_lcb_uls']:.3f}) & "
                     f"{u['beta_drift']:.3f} ({u['beta_lcb_drift']:.3f}) & "
                     f"{e0['est']['beta_uls']:.3f} / {e0['est']['beta_drift']:.3f} \\\\")
        e = v[f'{f}_RBDO-C_MU-a']
        s, b = e['est'], e['est_baseline']
        lines.append(f" & RBDO-C-MU & {e['weight']:.0f} & {(e['weight'] - m0) / m0 * 100:+.2f}\\% & "
                     f"{s['beta_uls']:.3f} ({s['beta_lcb_uls']:.3f}) & {s['beta_drift']:.3f} ({s['beta_lcb_drift']:.3f}) & "
                     f"{b['beta_uls']:.3f} / {b['beta_drift']:.3f} \\\\")
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    write_table('mudesign', '\n'.join(lines))


def tab_targets():
    """Base frame: optimum mass for different reliability targets, drift limits and SAA settings."""
    v = ver()
    m0 = v['base_DDO']['weight']
    rows = []
    for L in (400, 450, 500, 600):
        k = 'base_DDO' if L == 400 else f'base_DDO-h{L}'
        if k in v:
            rows.append((f'DDO, drift $h/{L}$', '--', '--', v[k]))
    for k in sorted(k for k in v if k.startswith('base_RBDO') and k != 'pareto'):
        e = v[k]
        if (e.get('n_mc', N_MC_OPT) != N_MC_OPT or e.get('accept', 'point') != 'point' or '_saa' in k
                or e.get('pm', 'none') != 'none'):
            continue
        rows.append((e['mode'], f"{e['beta_uls_T']:.1f}", f"{e['beta_sls_T']:.1f}", e))
    lines = [r'\begin{tabular}{lccrrrlrl}', r'\toprule',
             r'Formulation & $\beta_T$ & $\beta_{SLS}$ & Mass (kg) & $\Delta m$ & $k_{ULS}$ & $\hat\beta_{ULS}$ [95\% CI] & '
             r'$\hat\beta_{drift}$ & LRFD \\', r'\midrule']
    for name, bu, bs, e in rows:
        s = e['est']
        lines.append(f"{name} & {bu} & {bs} & {e['weight']:.0f} & {(e['weight'] - m0) / m0 * 100:+.2f}\\% & "
                     f"{s['nfail_uls']} & {s['beta_uls']:.3f} {fci(s, 'uls')} & {s['beta_drift']:.3f} & "
                     f"{e['det']['strength']:.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('targets', '\n'.join(lines))


def tab_saa():
    """Base frame RBDO-C: effect of SAA sample size, SAA sample stream and acceptance rule."""
    v = ver()
    rows = []
    for k in sorted(k for k in v if k.startswith('base_RBDO-C')):
        e = v[k]
        if e['beta_uls_T'] != BETA_ULS or e['beta_sls_T'] != BETA_SLS:
            continue
        rows.append(e | {'key': k})
    rows.sort(key=lambda e: (e['accept'] != 'point', e['n_mc'], e['key']))
    lines = [r'\begin{tabular}{lrlrrlrl}', r'\toprule',
             r'SAA setting & Mass (kg) & SAA $\hat\beta_{ULS}$ / $\hat\beta_{drift}$ & $k_{ULS}$ & $\hat\beta_{ULS}$ [95\% CI] & '
             r'& $\hat\beta_{drift}$ [95\% CI] & \\', r'\midrule']
    lines[2] = (r'SAA setting & Mass (kg) & SAA $\hat\beta_{ULS}$, $\hat\beta_{drift}$ & $k_{ULS}$ & '
                r'Verified $\hat\beta_{ULS}$ [95\% CI] & $k_{drift}$ & Verified $\hat\beta_{drift}$ [95\% CI] & vs. targets \\')
    for e in rows:
        stream = 'A' if '_saa' not in e['key'] else {'_saa24680': 'B', '_saa13579': 'C'}.get(
            '_saa' + e['key'].split('_saa')[1].split('_')[0], '?')
        name = f"$N={e['n_mc']:,}$".replace(',', '\\,') + f", stream {stream}" + (
            ', 95\\% LCB' if e['accept'] == 'lcb' else '')
        s, a = e['est'], e['saa']
        lines.append(f"{name} & {e['weight']:.0f} & {a['beta_uls']:.2f}, {a['beta_drift']:.2f} & {s['nfail_uls']} & "
                     f"{s['beta_uls']:.3f} {fci(s, 'uls')} & {s['nfail_drift']} & {s['beta_drift']:.3f} {fci(s, 'drift')} & "
                     f"{verdict(s, 'uls')[0]}/{verdict(s, 'drift')[0]} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('saa', '\n'.join(lines))


def tab_robustness():
    v = ver()
    lines = [r'\begin{tabular}{llrrrrrrr}', r'\toprule',
             r'Problem & Formulation & Runs & Best (kg) & Median (kg) & Worst (kg) & Distinct finals & Runs at best & Designs eval. \\',
             r'\midrule']
    for f in FRAME_ORDER:
        for m in ('DDO', 'RBDO', 'RBDO-C'):
            try:
                d = load(f'opt_{f}_{m}.json')
            except FileNotFoundError:
                continue
            ws = np.array([r['weight'] for r in d['runs'] if r['feasible']])
            if len(ws) < 2:
                continue
            xs = {tuple(r['x']) for r in d['runs'] if r['feasible']}
            lines.append(f"{FRAME_LABEL[f]} (9 var.) & {m} & {len(d['runs'])} & {ws.min():.0f} & {np.median(ws):.0f} & "
                         f"{ws.max():.0f} & {len(xs)} & {int(np.sum(ws <= ws.min() + 1e-6))} & "
                         f"{np.mean([r['n_eval'] for r in d['runs']]):.0f} \\\\")
    for m in ('DDO', 'RBDO', 'RBDO-C'):
        try:
            d = load(f'enum_R1_{m}.json')
        except FileNotFoundError:
            continue
        ws = np.array([r['weight'] for r in d['ga_runs']])
        xs = {tuple(r['x']) for r in d['ga_runs']}
        opt = d['optimum']['weight']
        lines.append(f"Reduced (3 var.) & {m} & {len(ws)} & {ws.min():.0f} & {np.median(ws):.0f} & {ws.max():.0f} & "
                     f"{len(xs)} & {int(np.sum(np.abs(ws - opt) < 1e-6))} & {np.mean([r['n_eval'] for r in d['ga_runs']]):.0f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('robustness', '\n'.join(lines))


def tab_enum():
    lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Formulation & Designs & Exact optimum (kg) & Lighter designs (infeasible) & GA runs reaching it & GA without local search \\',
             r'\midrule']
    for m in ('DDO', 'RBDO', 'RBDO-C'):
        try:
            d = load(f'enum_R1_{m}.json')
        except FileNotFoundError:
            continue
        o = d['optimum']
        ws = np.array([r['weight'] for r in d['ga_runs']])
        wg = np.array([r['ga_weight'] for r in d['ga_runs']])
        lines.append(f"{m} & {d['n_designs']} & {o['weight']:.0f} & {o['rank_in_mass_order']} & "
                     f"{int(np.sum(np.abs(ws - o['weight']) < 1e-6))}/{len(ws)} & "
                     f"{int(np.sum(np.abs(wg - o['weight']) < 1e-6))}/{len(ws)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('enumeration', '\n'.join(lines))


def tab_driftlimits():
    """Deterministic optima with tightened drift limits vs. RBDO-C and RBDO-C95. Bold: both targets
    met with 95 % confidence (claim L); italic: neither target below its interval (claims P/I)."""
    v = ver()
    lims = (400, 450, 500, 600)
    lines = [r'\begin{tabular}{l' + 'r' * (len(lims) + 4) + '}', r'\toprule',
             'Frame & ' + ' & '.join(f'DDO $h/{L}$' for L in lims) +
             r' & RBDO-C & RBDO-C95 & $\Delta m_1$ & $\Delta m_2$ \\',
             r'\midrule']
    for f in FRAME_ORDER:
        cells = []
        best_ok, best_l = None, None
        for L in lims:
            k = f'{f}_DDO' if L == 400 else f'{f}_DDO-h{L}'
            e = v.get(k)
            if e is None:
                cells.append('--')
                continue
            s = e['est']
            ok = verdict(s, 'uls') != 'below' and verdict(s, 'drift') != 'below'
            okl = claim(s, 'uls') == 'L' and claim(s, 'drift') == 'L'
            w = (lambda t: '\\textbf{' + t + '}') if okl else (lambda t: '\\textit{' + t + '}') if ok else (lambda t: t)
            cells.append('\\begin{tabular}[c]{@{}r@{}}' + w(f"{e['weight']:.0f}") + '\\\\' + w(f"({s['beta_uls']:.2f}/{s['beta_drift']:.2f})") + '\\end{tabular}')
            if ok and (best_ok is None or e['weight'] < best_ok):
                best_ok = e['weight']
            if okl and (best_l is None or e['weight'] < best_l):
                best_l = e['weight']
        r, r95 = v.get(f'{f}_RBDO-C'), v.get(f'{f}_RBDO-C_lcb')
        if r is None or r95 is None:
            continue
        txt = lambda e: ('\\begin{tabular}[c]{@{}r@{}}' + f"{e['weight']:.0f}" + '\\\\' +
                         f"({e['est']['beta_uls']:.2f}/{e['est']['beta_drift']:.2f})" + '\\end{tabular}')
        pct = lambda m, ref: '--' if ref is None else f"{(m - ref) / ref * 100:+.2f}\\%"
        lines.append(f"{FRAME_LABEL[f]} & " + ' & '.join(cells) + f" & {txt(r)} & {txt(r95)} & "
                     f"{pct(r['weight'], best_ok)} & {pct(r95['weight'], best_l)} \\\\ \\addlinespace")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('driftlimits', '\n'.join(lines))


COST_CASES = ((0.25, 0.0), (0.5, 0.0), (1.0, 0.0), (0.5, 0.25), (1.0, 0.25))


def cost_index(e, frame, tau, kappa):
    """Illustrative cost in tonnes of steel equivalent: mass + tau * distinct sections
    + kappa * beam mass (connection/fabrication effort assumed to scale with beam size)."""
    ms = mass_split(frame, e['x'])
    return e['weight'] / 1000 + tau * n_types(e['design']) + kappa * ms[2] / 1000


def tab_cost():
    v = ver()
    head = ' & '.join(f'$\\tau={t}$, $\\kappa={k}$' for t, k in COST_CASES)
    lines = [r'\begin{tabular}{lrrrrr' + 'r' * len(COST_CASES) + '}', r'\toprule',
             r' & \multicolumn{2}{c}{Distinct sections} & \multicolumn{2}{c}{$\Delta m$ RBDO-C vs. DDO (kg)} & & '
             + r'\multicolumn{' + str(len(COST_CASES)) + r'}{c}{Cost premium of RBDO-C over DDO} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){7-' + str(6 + len(COST_CASES)) + '}',
             r'Frame & DDO & RBDO-C & Columns & Beams & Mass & ' + head + r' \\', r'\midrule']
    for f in FRAME_ORDER:
        d, r = v.get(f'{f}_DDO'), v.get(f'{f}_RBDO-C')
        if d is None or r is None:
            continue
        sd, sr = mass_split(f, d['x']), mass_split(f, r['x'])
        prem = [(cost_index(r, f, t, k) - cost_index(d, f, t, k)) / cost_index(d, f, t, k) * 100 for t, k in COST_CASES]
        lines.append(f"{FRAME_LABEL[f]} & {n_types(d['design'])} & {n_types(r['design'])} & "
                     f"{sr[0] + sr[1] - sd[0] - sd[1]:+.0f} & {sr[2] - sd[2]:+.0f} & "
                     f"{(r['weight'] - d['weight']) / d['weight'] * 100:+.2f}\\% & " + ' & '.join(f'{p:+.2f}\\%' for p in prem) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('cost', '\n'.join(lines))


def tab_members():
    """Where failures occur (member type, scenario) for the DDO and RBDO-C designs of every frame."""
    v = ver()
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Frame & Design & $k_{ULS}$ & Beams (\%) & Ext. col. (\%) & Int. col. (\%) & $S_1$ only / $S_2$ only / both (\%) \\',
             r'\midrule']
    for f in FRAME_ORDER:
        first = True
        for m in ('DDO', 'RBDO-C'):
            e = v.get(f'{f}_{m}')
            if e is None or 'governing_member_counts' not in e:
                continue
            g = e['governing_member_counts']
            n = max(sum(g.values()), 1)
            ext = {f'C{i}_' for i in (0, FRAMES[f].n_bays)}
            beams = sum(c for k, c in g.items() if k.startswith('B'))
            exc = sum(c for k, c in g.items() if k.startswith('C') and k[:k.index('_') + 1] in ext)
            inc = n - beams - exc if sum(g.values()) else 0
            k = e['est']['nfail_uls']
            sc = [e.get('fail_S1_only', 0), e.get('fail_S2_only', 0), e.get('fail_both', 0)]
            lines.append(f"{FRAME_LABEL[f] if first else ''} & {m} & {k} & {beams / n * 100:.0f} & {exc / n * 100:.0f} & "
                         f"{inc / n * 100:.0f} & " + ' / '.join(f'{c / max(k, 1) * 100:.0f}' for c in sc) + r' \\')
            first = False
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('members', '\n'.join(lines))


def tab_modelunc():
    d = load('modelunc.json')
    cases = ('none', 'MU-a', 'MU-b')
    lines = [r'\begin{tabular}{ll' + 'r' * (2 * len(cases)) + '}', r'\toprule',
             r' & & \multicolumn{3}{c}{$\hat\beta_{ULS}$} & \multicolumn{3}{c}{$\hat\beta_{drift}$} \\',
             r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
             r'Frame & Design & none & 0.05 & 0.10 & none & 0.05 & 0.10 \\', r'\midrule']
    for f in FRAME_ORDER:
        first = True
        for m in ('DDO', 'RBDO-C'):
            e = d.get(f'{f}_{m}')
            if e is None:
                continue
            lines.append(f"{FRAME_LABEL[f] if first else ''} & {m} & " +
                         ' & '.join(fb(e[c], 'uls') for c in cases) + ' & ' +
                         ' & '.join(fb(e[c], 'drift') for c in cases) + r' \\')
            first = False
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('modelunc', '\n'.join(lines))


def tab_opt_runs():
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Frame & Formulation & Runs & Best mass (kg) & Designs evaluated & Wall time (min) \\', r'\midrule']
    for f in FRAME_ORDER:
        for mode in ('DDO', 'DDO-h450', 'DDO-h500', 'DDO-h600', 'RBDO', 'RBDO-C'):
            try:
                d = load(f'opt_{f}_{mode}.json')
            except FileNotFoundError:
                continue
            ws = [r['weight'] for r in d['runs'] if r['feasible']]
            ne = np.mean([r['n_eval'] for r in d['runs']])
            tm = np.mean([r['time_s'] for r in d['runs']]) / 60
            lines.append(f"{FRAME_LABEL[f]} & {mode} & {len(d['runs'])} & {min(ws):.0f} & {ne:.0f} & {tm:.0f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('opt_runs', '\n'.join(lines))


# ------------------------------------------------- broad surrogate validation
VAL_KEYS = ['base_DDO', 'base_RBDO-C', 'S3_RBDO-C', 'S9_RBDO-C', 'B6_RBDO-C', 'B10_RBDO-C', 'W08_RBDO-C', 'W16_RBDO-C',
            'S3_RBDO-C_lcb', 'S3_RBDO-C_bu3.21_bs1.5_lcb']


def val_label(k):
    f, m = k.split('_', 1)
    return f"{FRAME_LABEL[f]} {m.replace('RBDO-C_bu3.21_bs1.5_lcb', 'RBDO-C95, $\\beta_T=3.21$').replace('RBDO-C_lcb', 'RBDO-C95')}"


def val_rows(n_sim=200_000, seed=1):
    """Stratified estimate of the full-model P_f and its 95 % interval for each validated design.

    Jeffreys posteriors of the stratum fractions and of the stratum probabilities
    (pool counts) are propagated by simulation; the surrogate P_f of the same pool is
    propagated jointly, so the interval of the difference accounts for both."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in VAL_KEYS:
        fn = os.path.join(RES, f'validate_{k}.json')
        if not os.path.exists(fn):
            continue
        d = load(f'validate_{k}.json')
        N = d['n_pool']
        A, B = d['tail']['A'], d['tail']['B']
        nA, nB = len(A['U_full']), len(B['U_full'])
        kA = int(np.sum(np.array(A['U_full']) >= 1))   # surrogate failures that also fail in the full model
        kB = int(np.sum(np.array(B['U_full']) >= 1))   # near misses that fail in the full model
        NA, NB = A['n_T'], B['n_T']
        # posterior draws: stratum counts in the pool (multinomial-Dirichlet) and fractions
        pr = rng.dirichlet([NA + 0.5, NB + 0.5, N - NA - NB + 0.5], n_sim)
        fA = rng.beta(kA + 0.5, nA - kA + 0.5, n_sim)
        fB = rng.beta(kB + 0.5, nB - kB + 0.5, n_sim)
        pf_full = pr[:, 0] * fA + pr[:, 1] * fB
        pf_sur = pr[:, 0]
        from scipy import stats as st
        b_full = -st.norm.ppf(pf_full)
        b_sur = -st.norm.ppf(pf_sur)
        db = b_full - b_sur
        r = d['random']
        Ur = np.array(r['U_sur']) / np.array(r['U_full']) - 1
        Dr = np.array(r['D_sur']) / np.array(r['D_full']) - 1
        Df, Ds = np.array(r['D_full']) >= 1, np.array(r['D_sur']) >= 1
        # utilisation ratio over everything analysed (supports the u_tail cut-off)
        ratio_all = np.concatenate([np.array(A['U_full']) / np.array(A['U_sur']),
                                    np.array(B['U_full']) / np.array(B['U_sur']),
                                    np.array(r['U_full']) / np.array(r['U_sur'])])
        pf_hat = (NA / N) * (kA / nA) + (NB / N) * (kB / max(nB, 1))
        rows.append({'key': k, 'nA': nA, 'kA': kA, 'nB': nB, 'kB': kB, 'NA': NA, 'NB': NB, 'N': N,
                     'pf_sur': NA / N, 'pf_full': pf_hat, 'pf_full_ci': np.quantile(pf_full, [0.025, 0.975]).tolist(),
                     'ratio': pf_hat / (NA / N), 'ratio_ci': np.quantile(pf_full / pf_sur, [0.025, 0.975]).tolist(),
                     'dbeta': float(-stats.norm.ppf(pf_hat) + stats.norm.ppf(NA / N)),
                     'dbeta_ci': np.quantile(db, [0.025, 0.975]).tolist(),
                     'U_bias': float(np.mean(Ur)), 'U_sd': float(np.std(Ur)), 'D_bias': float(np.mean(Dr)),
                     'D_sd': float(np.std(Dr)), 'n_random': len(Ur), 'kD_full': int(Df.sum()), 'kD_sur': int(Ds.sum()),
                     'D_discord': int(np.sum(Df != Ds)), 'max_ratio_full_sur': float(ratio_all.max()),
                     'u_tail': d['u_tail']})
    return rows


def tab_validation():
    rows = val_rows()
    lines = [r'\begin{tabular}{lrrrrrrrrl}', r'\toprule',
             r' & \multicolumn{4}{c}{Random set, scenario $S_2$} & \multicolumn{2}{c}{Tail strata: full failures} & & & \\',
             r'\cmidrule(lr){2-5}\cmidrule(lr){6-7}',
             r'Design & $U$ bias (\%) & drift bias (\%) & $k_{drift}$ full/sur. & discordant & A: $U_s\geq1$ & B: $0.85\leq U_s<1$ & '
             r'$\hat P_{f}^{sur}$ & $\hat P_{f}^{full}$ [95\% CI] & $\Delta\beta_{ULS}$ [95\% CI] \\', r'\midrule']
    for r in rows:
        lo, hi = r['pf_full_ci']
        lines.append(f"{val_label(r['key'])} & {r['U_bias'] * 100:+.1f} $\\pm$ {r['U_sd'] * 100:.1f} & "
                     f"{r['D_bias'] * 100:+.1f} $\\pm$ {r['D_sd'] * 100:.1f} & {r['kD_full']}/{r['kD_sur']} & {r['D_discord']} & "
                     f"{r['kA']}/{r['nA']} & {r['kB']}/{r['nB']} & {r['pf_sur']:.2e} & {r['pf_full']:.2e} [{lo:.2e}, {hi:.2e}] & "
                     f"{r['dbeta']:+.2f} [{r['dbeta_ci'][0]:+.2f}, {r['dbeta_ci'][1]:+.2f}] \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('validation', '\n'.join(lines))
    with open(os.path.join(RES, 'validation_summary.json'), 'w') as f:
        json.dump(rows, f, indent=1)


def sci(x):
    """LaTeX scientific notation with two significant digits."""
    if x <= 0:
        return '0'
    m, e = f'{x:.1e}'.split('e')
    return f'${m}\\times10^{{{int(e)}}}$'


def tail_rows(n_sim=200_000, seed=2):
    """Excluded-tail check: full re-analysis of C = {0.70 <= U_sur < 0.85} and a residual-risk
    estimate for D = {U_sur < 0.70}, which is not analysed.

    The residual of D is (1/N) sum_k P(R >= 1/U_k) over the pool histogram (bin upper edges,
    conservative), with R = U_full / U_sur. log R is fitted to all re-analysed samples with
    U_sur >= 0.70 (strata B and C) by a normal distribution and, as a heavy-tailed
    alternative, by a Student t with 3 degrees of freedom and the same scale.
    The stratified estimate with the lower threshold is P(A) f_A + P(B) f_B + P(C) f_C (+ R_D)."""
    from scipy import stats as st
    rng = np.random.default_rng(seed)
    out = []
    for k in VAL_KEYS:
        try:
            d = load(f'validate_{k}.json')
            c = load(f'validateC_{k}.json')
        except FileNotFoundError:
            continue
        N = d['n_pool']
        A, B = d['tail']['A'], d['tail']['B']
        nA, nB, nC = len(A['U_full']), len(B['U_full']), len(c['U_full'])
        kA = int(np.sum(np.array(A['U_full']) >= 1))
        kB = int(np.sum(np.array(B['U_full']) >= 1))
        kC = int(np.sum(np.array(c['U_full']) >= 1))
        NA, NB, NC = A['n_T'], B['n_T'], c['n_T']
        NDD = N - NA - NB - NC
        lr = np.log(np.concatenate([np.array(B['U_full']) / np.array(B['U_sur']),
                                    np.array(c['U_full']) / np.array(c['U_sur'])]))
        mu, sd = float(np.mean(lr)), float(np.std(lr, ddof=1))
        rC = np.array(c['U_full']) / np.array(c['U_sur'])
        edges = np.array(c['hist_edges'])
        cnt = np.array(c['hist_counts'])
        sel = edges[1:] <= c['u_low'] + 1e-9            # bins of D
        need = np.log(1 / edges[1:][sel])                  # log ratio needed to fail (bin upper edge)
        RD_norm = float(np.sum(cnt[sel] * st.norm.sf(need, mu, sd)) / N)
        RD_t3 = float(np.sum(cnt[sel] * st.t.sf((need - mu) / sd, 3)) / N)
        # model-based residual of C as well (for comparison with the sampled estimate)
        selC = ~sel
        needC = np.log(1 / edges[1:][selC])
        RC_norm = float(np.sum(cnt[selC] * st.norm.sf(needC, mu, sd)) / N)
        RC_t3 = float(np.sum(cnt[selC] * st.t.sf((needC - mu) / sd, 3)) / N)
        pf85 = (NA / N) * (kA / nA) + (NB / N) * (kB / nB)
        pf70 = pf85 + (NC / N) * (kC / nC)
        pr = rng.dirichlet([NA + 0.5, NB + 0.5, NC + 0.5, NDD + 0.5], n_sim)
        fA = rng.beta(kA + 0.5, nA - kA + 0.5, n_sim)
        fB = rng.beta(kB + 0.5, nB - kB + 0.5, n_sim)
        fC = rng.beta(kC + 0.5, nC - kC + 0.5, n_sim)
        p85 = pr[:, 0] * fA + pr[:, 1] * fB
        p70 = p85 + pr[:, 2] * fC
        sur = pr[:, 0]
        b = lambda p: -st.norm.ppf(p)
        # combined estimate: sampled strata (Jeffreys posterior) + t3 residual of all samples with
        # U_sur < 0.85, with the uncertainty of the fitted (mu, sd) propagated by a nonparametric
        # bootstrap of the log-ratios. Conditional on the t3 family; not distribution-free.
        need85 = np.log(1 / edges[1:])
        n_boot = 4000
        R_boot = np.empty(n_boot)
        for i in range(n_boot):
            lb = rng.choice(lr, size=len(lr), replace=True)
            m_b, s_b = lb.mean(), lb.std(ddof=1)
            R_boot[i] = np.sum(cnt * st.t.sf((need85 - m_b) / s_b, 3)) / N
        comb = rng.choice(p85, n_boot) + R_boot
        b_comb = b(comb)
        R85_t2 = float(np.sum(cnt * st.t.sf((need85 - mu) / sd, 2)) / N)
        # observed tail of the standardised log-ratios vs. normal and t3 predictions
        z = (lr - mu) / sd
        tailcmp = {f'gt{q}': [int(np.sum(z > q)), float(len(z) * st.norm.sf(q)), float(len(z) * st.t.sf(q, 3))]
                   for q in (2, 3, 4)}
        out.append({'key': k, 'NC': NC, 'pC': NC / N, 'nC': nC, 'kC': kC, 'ND': NDD, 'pD': NDD / N,
                    'max_ratio_C': float(rC.max()), 'mu_logR': mu, 'sd_logR': sd,
                    'pf_sur': NA / N, 'pf85': pf85, 'pf70': pf70,
                    'pf70_ci': np.quantile(p70, [0.025, 0.975]).tolist(),
                    'pf85_ci': np.quantile(p85, [0.025, 0.975]).tolist(),
                    'contribC_ci': np.quantile(pr[:, 2] * fC, [0.025, 0.975]).tolist(),
                    'RC_norm': RC_norm, 'RC_t3': RC_t3, 'RD_norm': RD_norm, 'RD_t3': RD_t3,
                    'R85_norm': RC_norm + RD_norm, 'R85_t3': RC_t3 + RD_t3,
                    'beta85_t3': float(b(pf85 + RC_t3 + RD_t3)),
                    'beta_comb_ci': [float(x) for x in np.quantile(b_comb, [0.025, 0.975])],
                    'beta_comb_lcb': float(np.quantile(b_comb, 0.05)),
                    'R85_t3_ci': [float(x) for x in np.quantile(R_boot, [0.025, 0.975])],
                    'beta85_t2': float(b(pf85 + R85_t2)), 'R85_t2': R85_t2,
                    'n_lr': int(len(lr)), 'z_max': float(z.max()), 'tailcmp': tailcmp,
                    'beta85': float(b(pf85)), 'beta70': float(b(pf70)),
                    'beta70_ci': [float(x) for x in b(np.quantile(p70, [0.975, 0.025]))],
                    'beta70_t3': float(b(pf70 + RD_t3)),
                    'dbeta70': float(b(pf70) - b(NA / N)),
                    'dbeta70_ci': np.quantile(b(p70) - b(sur), [0.025, 0.975]).tolist()})
    return out


def tab_tail():
    rows = tail_rows()
    lines = [r'\begin{tabular}{lrrrllrll}', r'\toprule',
             r' & \multicolumn{3}{c}{Stratum $C$: $0.70\leq U_s<0.85$} & '
             r'\multicolumn{2}{c}{Sampled strata only (Jeffreys 95\% CI)} & '
             r'\multicolumn{3}{c}{Sampled strata + $t$-model residual of $U_s<0.85$} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}',
             r'Design & $P(C)$ & $k_C/n_C$ & max $U_f/U_s$ & $A,B$ & $A,B,C$ & '
             r'$R_{t_3}$ & $\hat\beta$ [95\% CI] ($\beta^{lcb}$), $t_3$ & $\hat\beta$, $t_2$ \\', r'\midrule']
    from scipy import stats as st
    for r in rows:
        lo, hi = r['beta70_ci']
        l85, h85 = [float(-st.norm.ppf(x)) for x in r['pf85_ci'][::-1]]
        cl, ch = r['beta_comb_ci']
        lines.append(f"{val_label(r['key'])} & {r['pC']:.3f} & {r['kC']}/{r['nC']} & {r['max_ratio_C']:.3f} & "
                     f"{r['beta85']:.3f} [{l85:.2f}, {h85:.2f}] & {r['beta70']:.3f} [{lo:.2f}, {hi:.2f}] & "
                     f"{sci(r['R85_t3'])} & {r['beta85_t3']:.3f} [{cl:.2f}, {ch:.2f}] ({r['beta_comb_lcb']:.2f}) & "
                     f"{r['beta85_t2']:.3f} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('tail', '\n'.join(lines))
    with open(os.path.join(RES, 'tail_summary.json'), 'w') as f:
        json.dump(rows, f, indent=1)


def summary():
    """Numbers quoted in the manuscript text."""
    out = {}
    v = ver()
    for k, e in v.items():
        if k == 'pareto':
            continue
        s = e['est']
        out[k] = {'mass_kg': round(e['weight'], 1), 'n_types': n_types(e['design']),
                  'beta_uls': round(s['beta_uls'], 3), 'ci_beta_uls': [round(c, 3) for c in s['ci_beta_uls']],
                  'nfail_uls': s['nfail_uls'], 'beta_drift': round(s['beta_drift'], 3),
                  'ci_beta_drift': [round(c, 3) for c in s['ci_beta_drift']], 'nfail_drift': s['nfail_drift'],
                  'beta_lcb_defl': round(s['beta_lcb_defl'], 3), 'lrfd': round(e['det']['strength'], 4),
                  'drift_det': round(e['det']['drift'], 4), 'B2': round(e['det']['max_B2'], 3),
                  'verdict': [verdict(s, 'uls'), verdict(s, 'drift')],
                  'runs': [(r['seed'], round(r['weight'], 1)) for r in e.get('runs', [])]}
        if 'saa' in e and 'beta_uls' in e['saa']:
            out[k]['saa'] = {kk: round(vv, 3) for kk, vv in e['saa'].items() if kk.startswith('beta_') and 'lcb' not in kk}
        if 'governing_member_counts' in e:
            out[k]['governing'] = e['governing_member_counts']
            out[k]['scenario'] = [e.get('fail_S1_only'), e.get('fail_S2_only'), e.get('fail_both')]
    with open(os.path.join(RES, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out

if __name__ == '__main__':
    import sys
    todo = sys.argv[1:] or None
    for fn in [fig_frame, fig_ga_history, fig_pareto, fig_parametric, fig_distributions, fig_member_util,
               fig_pdelta, fig_fullmc, fig_mc_convergence, fig_sensitivity, fig_validation, fig_driftlimits,
               tab_random_variables, tab_closed_form, tab_opensees, tab_designs, tab_verification, tab_targets,
               tab_saa, tab_strict, tab_robustness, tab_enum, tab_driftlimits, tab_cost, tab_members, tab_modelunc,
               tab_validation, tab_tail, tab_opt_runs, tab_frames, summary]:
        if todo and fn.__name__ not in todo:
            continue
        try:
            fn()
            print('ok', fn.__name__)
        except (FileNotFoundError, KeyError) as e:
            print('skip', fn.__name__, repr(e))
