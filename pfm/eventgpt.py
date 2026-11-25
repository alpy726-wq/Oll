"""Reference model for the scaling test: a causal transformer on the same real events.
Input  = value embedding (linear for continuous streams, table for discrete) + stream embedding + log(time gap) features
Output = per stream: 5-component Gaussian mixture (continuous) or softmax (discrete). Same NLL definition as OneMachine."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class EventGPT(nn.Module):
    def __init__(self, streams, d=128, L=4, H=4, T=512, K=5):
        super().__init__()
        self.streams, self.K = streams, K
        self.n_st = len(streams)
        self.val = nn.Linear(1, d)
        self.sym = nn.Embedding(256, d)
        self.st = nn.Embedding(self.n_st, d)
        self.gap = nn.Linear(2, d)
        self.pos = nn.Embedding(T, d)
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L)
        self.ln = nn.LayerNorm(d)
        self.q_st = nn.Embedding(self.n_st, d)   # which stream is asked
        self.q_gap = nn.Linear(2, d)             # and when
        self.mdn = nn.Linear(d, 3 * K)
        self.cls = nn.Linear(d, 256)
        self.discrete = torch.tensor([st.kind == "discrete" for st in streams])

    def forward(self, x, s, dt, q):
        disc = self.discrete.to(x.device)[s]
        xv = torch.where(disc, torch.zeros_like(x), x).float()[..., None]
        sym = torch.where(disc, x, torch.zeros_like(x)).long().clamp(0, 255)
        g = torch.stack([torch.log1p(dt.float()), (dt > 0).float()], -1)
        h = torch.where(disc[..., None], self.sym(sym), self.val(xv)) + self.st(s) + self.gap(g) + self.pos(torch.arange(x.shape[1], device=x.device))
        T = x.shape[1]
        h = self.enc(h, mask=nn.Transformer.generate_square_subsequent_mask(T, device=x.device), is_causal=True)
        return self.ln(h), q

    def nll(self, hq, s_out, y):
        h, q = hq
        h = h + self.q_st(s_out) + self.q_gap(torch.stack([torch.log1p(q.float()), (q > 0).float()], -1))
        disc = self.discrete.to(y.device)[s_out]
        lw, mu, ls = self.mdn(h).chunk(3, -1)
        ls = ls.clamp(-7, 3)
        lp = -0.5 * ((y.float()[..., None] - mu) / ls.exp()) ** 2 - ls - 0.5 * math.log(2 * math.pi)
        cont = -torch.logsumexp(lp + torch.log_softmax(lw, -1), -1)
        lg = self.cls(h)
        dis = Fn.cross_entropy(lg.reshape(-1, 256), y.long().clamp(0, 255).reshape(-1), reduction="none").reshape(y.shape)
        return torch.where(disc, dis, cont)
