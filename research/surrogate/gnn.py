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
    def __init__(self, nx, ne, hidden=96, steps=8, n_out=3):
        super().__init__()
        self.enc_n = mlp(nx, hidden, hidden)
        self.enc_e = mlp(ne, hidden, hidden)
        self.edge_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(steps)])
        self.node_up = nn.ModuleList([mlp(3 * hidden, hidden, hidden) for _ in range(steps)])
        self.dec_n = mlp(hidden, hidden, n_out, norm=False)
        self.dec_e = mlp(3 * hidden, hidden, n_out, norm=False)

    def forward(self, x, e, src, dst):
        h, z = self.enc_n(x), self.enc_e(e)
        n = h.shape[0]
        for eu, nu in zip(self.edge_up, self.node_up):
            z = z + eu(torch.cat([z, h[src], h[dst]], 1))
            # Separate aggregation of messages arriving at member start / end nodes
            agg_i = torch.zeros(n, z.shape[1]).index_add_(0, src, z)
            agg_j = torch.zeros(n, z.shape[1]).index_add_(0, dst, z)
            h = h + nu(torch.cat([h, agg_i, agg_j], 1))
        return self.dec_n(h), self.dec_e(torch.cat([z, h[src], h[dst]], 1))


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
        self.qs = np.sqrt((Yq ** 2).mean(0)) + 1e-12


def collate(graphs, nrm: Normaliser, with_targets=True, with_eq=False):
    xs, es, src, dst, masks, yu, yq = [], [], [], [], [], [], []
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
             mask=T(masks, torch.float32))
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

def train_model(train_g, val_g=None, seed=0, epochs=150, batch=32, lr=2e-3, hidden=96,
                steps=8, eq_weight=0.0, verbose=False, nrm=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nrm = nrm or Normaliser(train_g)
    model = FrameGNN(train_g[0]['x'].shape[1], train_g[0]['e'].shape[1], hidden, steps,
                     3 * len(train_g[0]['cases']))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    nb = int(np.ceil(len(train_g) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(train_g))
        tot = 0.0
        for bi in range(nb):
            idx = perm[bi * batch:(bi + 1) * batch]
            b = collate([train_g[i] for i in idx], nrm, with_eq=eq_weight > 0)
            pu, pq = model(b['x'], b['e'], b['src'], b['dst'])
            pu = pu * b['mask']
            loss = ((pu - b['yu']) ** 2 * b['mask']).sum() / b['mask'].sum() + ((pq - b['yq']) ** 2).mean()
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
    model.eval()
    U, Q = [], []
    for s in range(0, len(graphs), chunk):
        gs = graphs[s:s + chunk]
        b = collate(gs, nrm, with_targets=False)
        pu, pq = model(b['x'], b['e'], b['src'], b['dst'])
        pu = (pu * b['mask']).numpy() * nrm.us
        pq = pq.numpy() * nrm.qs
        on, oe = 0, 0
        for g in gs:
            n, m = len(g['x']), len(g['conn'])
            U.append(pu[on:on + n]); Q.append(pq[oe:oe + m])
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
