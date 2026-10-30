"""Mechanics-constrained graph surrogate for planar frames.

Components
----------
* FrameGNN: encode-process-decode message passing network on the frame graph
  (joints = nodes, members = edges). Node head -> dimensionless nodal
  displacements; edge head -> dimensionless member basic forces [N, Mi, Mj].
* Hard boundary conditions: restrained DOFs are multiplied by zero.
* Optional soft equilibrium penalty during training (||C q - d||^2).
* Equilibrium projection at inference: the minimum-(weighted)-norm correction
  that makes the predicted member forces satisfy nodal equilibrium exactly.
* Deep ensemble for predictive uncertainty.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


def mlp(i, h, o, layers=2, norm=True):
    mods, d = [], i
    for _ in range(layers - 1):
        mods += [nn.Linear(d, h), nn.SiLU()]
        d = h
    mods.append(nn.Linear(d, o))
    if norm:
        mods.append(nn.LayerNorm(o))
    return nn.Sequential(*mods)


class FrameGNN(nn.Module):
    """Encode-process-decode network; pool=True adds a graph-level mean of the
    node states to every node update (global context in one step)."""
    kind = 'frame'

    def __init__(self, nx, ne, hidden=96, steps=8, n_out=3, pool=False):
        super().__init__()
        self.pool = pool
        k = 4 if pool else 3
        self.enc_n = mlp(nx, hidden, hidden)
        self.enc_e = mlp(ne, hidden, hidden)
        self.edge_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(steps)])
        self.node_up = nn.ModuleList([mlp(k * hidden, hidden, hidden) for _ in range(steps)])
        self.dec_n = mlp(hidden, hidden, n_out, norm=False)
        self.dec_e = mlp(3 * hidden, hidden, n_out, norm=False)

    def forward(self, x, e, src, dst, gid=None, ng=None):
        h, z = self.enc_n(x), self.enc_e(e)
        n = h.shape[0]
        pool = getattr(self, 'pool', False)
        if pool:
            cnt = torch.zeros(ng).index_add_(0, gid, torch.ones(n)).clamp(min=1)[:, None]
        for eu, nu in zip(self.edge_up, self.node_up):
            z = z + eu(torch.cat([z, h[src], h[dst]], 1))
            # Separate aggregation of messages arriving at member start / end nodes
            agg_i = torch.zeros(n, z.shape[1]).index_add_(0, src, z)
            agg_j = torch.zeros(n, z.shape[1]).index_add_(0, dst, z)
            parts = [h, agg_i, agg_j]
            if pool:
                g_mean = torch.zeros(ng, h.shape[1]).index_add_(0, gid, h) / cnt
                parts.append(g_mean[gid])
            h = h + nu(torch.cat(parts, 1))
        return self.dec_n(h), self.dec_e(torch.cat([z, h[src], h[dst]], 1))


class StoryGNN(nn.Module):
    """Re-implementation of the StructGNN idea (Chou et al., 2024) for planar frames.

    The frame graph is augmented with one pseudo node per storey (rigid-floor
    proxy) linked to every joint of that floor and to the pseudo node below
    (see story_graph). Linear encoders; message passing with linear messages
    and mean aggregation in both edge directions; the number of propagation
    steps equals n_storeys + 1 so that information can cross the building
    height. Weights are shared across steps so that taller test frames can use
    more steps than seen in training; a residual connection is added for
    stable training. Node / member decoders have one ReLU hidden layer.
    """
    kind = 'story'

    def __init__(self, nx, ne, hidden=256, n_out=3):
        super().__init__()
        self.enc_n = nn.Linear(nx, hidden)
        self.enc_e = nn.Linear(ne, hidden)
        self.msg_f = nn.Linear(3 * hidden, hidden)
        self.msg_b = nn.Linear(3 * hidden, hidden)
        self.upd = nn.Linear(3 * hidden, hidden)
        self.dec_n = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.dec_e = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, x, e, src, dst, steps):
        h, z = self.enc_n(x), self.enc_e(e)
        n = h.shape[0]
        ones = torch.ones(len(src))
        deg_in = torch.zeros(n).index_add_(0, dst, ones).clamp(min=1)[:, None]
        deg_out = torch.zeros(n).index_add_(0, src, ones).clamp(min=1)[:, None]
        for _ in range(steps):
            mf = self.msg_f(torch.cat([h[src], h[dst], z], 1))
            mb = self.msg_b(torch.cat([h[dst], h[src], z], 1))
            a_in = torch.zeros(n, h.shape[1]).index_add_(0, dst, mf) / deg_in
            a_out = torch.zeros(n, h.shape[1]).index_add_(0, src, mb) / deg_out
            h = h + torch.relu(self.upd(torch.cat([h, a_in, a_out], 1)))
        return self.dec_n(h), self.dec_e(torch.cat([z, h[src], h[dst]], 1))


def story_graph(g):
    """Augment a frame graph with storey pseudo nodes (for StoryGNN).

    Pseudo node s stands for floor s+1; it is linked to every joint of that
    floor (joint -> pseudo) and to the pseudo node of the floor below
    (pseudo s -> pseudo s-1). Pseudo-node features are -1 except the floor's
    total lateral load, the floor level and an indicator; pseudo edges carry
    -1 member features and an edge-kind indicator. Targets of pseudo nodes and
    pseudo edges are zero and masked out of the loss and the outputs.
    """
    t = g['t']
    n, m = len(g['x']), len(g['conn'])
    lev = t['level']
    n_st = int(lev.max())
    nf, ef = g['x'].shape[1], g['e'].shape[1]
    ipx = 9 if g['mode'] != 'raw' else 4          # 'Px' column
    xp = -np.ones((n_st, nf))
    for s in range(n_st):
        xp[s, ipx] = g['x'][lev == s + 1, ipx].sum()
        xp[s, 0] = (s + 1) / n_st if g['mode'] != 'raw' else t['coords'][lev == s + 1, 1][0]
    X = np.vstack([np.hstack([g['x'], np.zeros((n, 1))]), np.hstack([xp, np.ones((n_st, 1))])])
    jn = np.nonzero(lev > 0)[0]
    e_np = np.stack([jn, n + lev[jn] - 1], 1)
    e_pp = np.stack([n + np.arange(1, n_st), n + np.arange(0, n_st - 1)], 1).reshape(-1, 2)
    conn = np.vstack([g['conn'], e_np, e_pp])
    k = len(conn) - m
    E = np.vstack([np.hstack([g['e'], np.tile([1, 0, 0], (m, 1))]),
                   np.hstack([-np.ones((len(e_np), ef)), np.tile([0, 1, 0], (len(e_np), 1))]),
                   np.hstack([-np.ones((len(e_pp), ef)), np.tile([0, 0, 1], (len(e_pp), 1))])])
    restr = np.vstack([g['restr'], np.ones((n_st, 3), bool)])
    out = dict(g, x=X, e=E, conn=conn, restr=restr, n_real=n, m_real=m, steps=n_st + 1,
               emask=np.r_[np.ones(m), np.zeros(k)])
    if 'y_u' in g:
        out['y_u'] = np.vstack([g['y_u'], np.zeros((n_st, g['y_u'].shape[1]))])
        out['y_q'] = np.vstack([g['y_q'], np.zeros((k, g['y_q'].shape[1]))])
    return out


def stiffness_system(g):
    """Dimensionless free-DOF stiffness system per load case (for the
    displacement-only, residual-trained model): K_d y = f_c with y the
    dimensionless displacements; rows are divided by lam (forces) or lam H
    (moments), so K_d is the same for both cases."""
    import scipy.sparse as sp
    import frames as F_
    import solvers as S_
    if '_K' in g:
        return g['_K']
    t = g['t']
    n = len(t['coords'])
    k, T, fef = S_.element_matrices(t)
    kg = np.einsum('mba,mbc,mcd->mad', T, k, T)
    dofs = np.concatenate([3 * t['conn'][:, :1] + np.arange(3), 3 * t['conn'][:, 1:] + np.arange(3)], axis=1)
    K = sp.csr_matrix((kg.ravel(), (np.repeat(dofs, 6, axis=1).ravel(), np.tile(dofs, (1, 6)).ravel())),
                      shape=(3 * n, 3 * n))
    free = ~t['restr'].ravel()
    H = g['H']
    rs = np.tile([1.0, 1.0, H], n)[free]
    fs = []
    for kc, case in enumerate(g['cases']):
        tc = F_.case_topology(t, case) if case != 'T' else t
        _, _, fc = S_.element_matrices(tc)
        Pf = tc['P'].ravel().copy()
        np.add.at(Pf, dofs.ravel(), -np.einsum('mba,mb->ma', T, fc).ravel())
        lam = g['qsc'][kc][0]
        fs.append(Pf[free] / rs / lam)
    cs = np.tile(g['usc'][0] / g['qsc'][0][0], n)[free]          # u = y * usc
    Kd = sp.diags(1 / rs) @ K[free][:, free] @ sp.diags(cs)
    g['_K'] = (Kd.tocoo(), fs, np.nonzero(free)[0])
    return g['_K']


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class Normaliser:
    def __init__(self, graphs):
        X = np.vstack([g['x'] for g in graphs]); E = np.vstack([g['e'] for g in graphs])
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-8
        self.em, self.es = E.mean(0), E.std(0) + 1e-8
        mask = np.vstack([np.tile(~g['restr'], len(g['cases'])) for g in graphs])
        Yu = np.vstack([g['y_u'] for g in graphs])
        self.us = np.sqrt((Yu ** 2 * mask).sum(0) / mask.sum(0)) + 1e-12
        Yq = np.vstack([g['y_q'] for g in graphs])
        em = np.concatenate([g.get('emask', np.ones(len(g['conn']))) for g in graphs])[:, None]
        self.qs = np.sqrt((Yq ** 2 * em).sum(0) / em.sum()) + 1e-12


def collate(graphs, nrm: Normaliser, with_targets=True, with_eq=False, with_K=False):
    xs, es, src, dst, masks, yu, yq = [], [], [], [], [], [], []
    gid, emask = [], []
    K_parts, f_parts, krow = [], [[] for _ in graphs[0]['cases']], []
    koff = 0
    off = 0
    nc = len(graphs[0]['cases'])
    C_parts = [[] for _ in range(nc)]
    d_parts = [[] for _ in range(nc)]
    roff, coff = 0, 0
    for g in graphs:
        n, m = len(g['x']), len(g['conn'])
        xs.append((g['x'] - nrm.xm) / nrm.xs)
        es.append((g['e'] - nrm.em) / nrm.es)
        src.append(g['conn'][:, 0] + off); dst.append(g['conn'][:, 1] + off)
        masks.append(np.tile(~g['restr'], nc))
        gid.append(np.full(n, len(gid)))
        emask.append(g.get('emask', np.ones(m)))
        if with_K:
            Kd, fs, fdof = stiffness_system(g)
            # network outputs y / us per DOF type; fold us into the columns
            us_c = [nrm.us[3 * c:3 * c + 3][fdof % 3] for c in range(nc)]
            K_parts.append((Kd.row + koff, Kd.col + koff, Kd.data, us_c, fdof + 3 * off))
            for c in range(nc):
                f_parts[c].append(fs[c])
            krow.append(np.full(len(fdof), len(krow)))
            koff += len(fdof)
        if with_targets:
            yu.append(g['y_u'] / nrm.us); yq.append(g['y_q'] / nrm.qs)
        if with_eq:
            # C acts on dimensionless q; the network outputs q / qs per case
            for k in range(nc):
                Cs = g['C'] * np.tile(nrm.qs[3 * k:3 * k + 3], m)[None, :]
                r, c = np.nonzero(Cs)
                C_parts[k].append((r + roff, c + coff, Cs[r, c]))
                d_parts[k].append(g['d'][k])
            roff += g['C'].shape[0]; coff += 3 * m
        off += n
    T = lambda a, dt=torch.float32: torch.as_tensor(np.concatenate(a), dtype=dt)
    b = dict(x=T(xs), e=T(es), src=T(src, torch.long), dst=T(dst, torch.long),
             mask=T(masks, torch.float32), gid=T(gid, torch.long), ng=len(graphs),
             emask=T(emask)[:, None])
    if 'steps' in graphs[0]:
        b['steps'] = graphs[0]['steps']
    if with_K:
        b['K'], b['f'] = [], []
        r = np.concatenate([p[0] for p in K_parts]); c = np.concatenate([p[1] for p in K_parts])
        v = np.concatenate([p[2] for p in K_parts])
        b['kdof'] = T([p[4] for p in K_parts], torch.long)
        b['krow'] = T(krow, torch.long)
        for k in range(nc):
            us_c = np.concatenate([p[3][k] for p in K_parts])
            idx = torch.as_tensor(np.vstack([r, c]))
            b['K'].append(torch.sparse_coo_tensor(idx, torch.as_tensor(v * us_c[c], dtype=torch.float32),
                                                  (koff, koff)).coalesce())
            b['f'].append(T(f_parts[k]))
    if with_targets:
        b.update(yu=T(yu), yq=T(yq))
    if with_eq:
        b['C'], b['d'] = [], []
        for k in range(nc):
            r = np.concatenate([p[0] for p in C_parts[k]]); c = np.concatenate([p[1] for p in C_parts[k]])
            v = np.concatenate([p[2] for p in C_parts[k]])
            idx = torch.as_tensor(np.vstack([r, c]))
            b['C'].append(torch.sparse_coo_tensor(idx, torch.as_tensor(v, dtype=torch.float32), (roff, coff)).coalesce())
            b['d'].append(T(d_parts[k]))
    return b


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run(model, b):
    if getattr(model, 'kind', 'frame') == 'story':
        return model(b['x'], b['e'], b['src'], b['dst'], b['steps'])
    return model(b['x'], b['e'], b['src'], b['dst'], b['gid'], b['ng'])


def _batches(graphs, batch, rng):
    """Random mini-batches; graphs with a 'steps' key are grouped by it."""
    if 'steps' not in graphs[0]:
        perm = rng.permutation(len(graphs))
        return [perm[i:i + batch] for i in range(0, len(perm), batch)]
    out = []
    steps = np.array([g['steps'] for g in graphs])
    for s in np.unique(steps):
        idx = rng.permutation(np.nonzero(steps == s)[0])
        out += [idx[i:i + batch] for i in range(0, len(idx), batch)]
    return [out[i] for i in rng.permutation(len(out))]


def train_model(train_g, val_g=None, seed=0, epochs=150, batch=32, lr=2e-3, hidden=96,
                steps=8, eq_weight=0.0, verbose=False, nrm=None, arch='frame', loss_type='mse',
                k_weight=0.0, disp_only=False):
    """arch: 'frame' (FrameGNN), 'pool' (FrameGNN + global pooling) or 'story'
    (StoryGNN on story_graph inputs). disp_only trains the displacement head
    only (member forces are then recovered from displacements); k_weight > 0
    adds the normalised free-DOF stiffness residual ||K u - f||^2 / ||f||^2."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nrm = nrm or Normaliser(train_g)
    nx, ne, no = train_g[0]['x'].shape[1], train_g[0]['e'].shape[1], 3 * len(train_g[0]['cases'])
    if arch == 'story':
        model = StoryGNN(nx, ne, hidden, no)
    else:
        model = FrameGNN(nx, ne, hidden, steps, no, pool=arch == 'pool')
    model.disp_only = disp_only
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    nb = len(_batches(train_g, batch, np.random.default_rng(0)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb + 10)
    err = (lambda a: a ** 2) if loss_type == 'mse' else torch.abs
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for idx in _batches(train_g, batch, rng):
            b = collate([train_g[i] for i in idx], nrm, with_eq=eq_weight > 0, with_K=k_weight > 0)
            pu, pq = run(model, b)
            pu = pu * b['mask']
            loss = (err(pu - b['yu']) * b['mask']).sum() / b['mask'].sum()
            if not disp_only:
                loss = loss + (err(pq - b['yq']) * b['emask']).sum() / (b['emask'].sum() * pq.shape[1])
            if k_weight > 0:
                for k in range(len(b['K'])):
                    y = pu[:, 3 * k:3 * k + 3].reshape(-1)[b['kdof']]
                    r = torch.sparse.mm(b['K'][k], y[:, None]).squeeze(1) - b['f'][k]
                    num = torch.zeros(b['ng']).index_add_(0, b['krow'], r ** 2)
                    den = torch.zeros(b['ng']).index_add_(0, b['krow'], b['f'][k] ** 2) + 1e-12
                    loss = loss + k_weight * (num / den).mean()
            if eq_weight > 0:
                for k in range(len(b['C'])):
                    r = torch.sparse.mm(b['C'][k], pq[:, 3 * k:3 * k + 3].reshape(-1, 1)).squeeze(1) - b['d'][k]
                    loss = loss + eq_weight * (r ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if verbose and (ep % 25 == 0 or ep == epochs - 1):
            msg = f'ep {ep:4d} loss {tot / nb:.4e}'
            if val_g is not None:
                pu, pq = predict(model, nrm, val_g)
                msg += f'  val relL2 u {rel_err_list(pu, [g["y_u"] for g in val_g]):.4f}'
                msg += f' q {rel_err_list(pq, [g["y_q"] for g in val_g]):.4f}'
            print(msg, f'({time.time() - t0:.0f}s)', flush=True)
    return model, nrm


@torch.no_grad()
def predict(model, nrm, graphs, chunk=256):
    """Dimensionless predictions (u, q) per graph; pseudo nodes/edges removed."""
    model.eval()
    U, Q = [None] * len(graphs), [None] * len(graphs)
    if 'steps' in graphs[0]:
        steps = np.array([g['steps'] for g in graphs])
        groups = [np.nonzero(steps == s)[0] for s in np.unique(steps)]
    else:
        groups = [np.arange(len(graphs))]
    for grp in groups:
        for s in range(0, len(grp), chunk):
            ids = grp[s:s + chunk]
            gs = [graphs[i] for i in ids]
            b = collate(gs, nrm, with_targets=False)
            pu, pq = run(model, b)
            pu = (pu * b['mask']).numpy() * nrm.us
            pq = pq.numpy() * nrm.qs
            on, oe = 0, 0
            for i, g in zip(ids, gs):
                n, m = len(g['x']), len(g['conn'])
                U[i] = pu[on:on + g.get('n_real', n)]; Q[i] = pq[oe:oe + g.get('m_real', m)]
                on += n; oe += m
    return U, Q


def rel_err_list(pred, true):
    return float(np.mean([np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-12) for p, t in zip(pred, true)]))


# ---------------------------------------------------------------------------
# Ensemble and equilibrium projection
# ---------------------------------------------------------------------------

class Ensemble:
    def __init__(self, members):
        self.members = members          # list of (model, nrm)

    @classmethod
    def train(cls, train_g, n_models=5, seed0=0, **kw):
        nrm = Normaliser(train_g)
        return cls([train_model(train_g, seed=seed0 + k, nrm=nrm, **kw) for k in range(n_models)])

    def predict(self, graphs):
        outs = [predict(m, n, graphs) for m, n in self.members]
        res = []
        for i in range(len(graphs)):
            Us = np.stack([o[0][i] for o in outs]); Qs = np.stack([o[1][i] for o in outs])
            res.append(dict(u=Us.mean(0), u_var=Us.var(0), q=Qs.mean(0), q_var=Qs.var(0)))
        return res


def project(g, q, weights=None):
    """Minimum weighted-norm correction of q so that equilibrium holds exactly.

    q: (m, 3 * n_cases) dimensionless basic forces; weights: same shape,
    positive variances (metric W = diag(1/weights)); None means the Euclidean
    metric on the dimensionless forces. For each load case c solves
        q* = q + W^-1 C^T (C W^-1 C^T)^-1 (d_c - C q).
    """
    C = g['C']
    out = np.empty_like(q)
    for k, d in enumerate(g['d']):
        qv = q[:, 3 * k:3 * k + 3].ravel()
        Winv = np.ones_like(qv) if weights is None else weights[:, 3 * k:3 * k + 3].ravel()
        CW = C * Winv[None, :]
        lamb = np.linalg.solve(CW @ C.T, d - C @ qv)
        out[:, 3 * k:3 * k + 3] = (qv + CW.T @ lamb).reshape(-1, 3)
    return out
