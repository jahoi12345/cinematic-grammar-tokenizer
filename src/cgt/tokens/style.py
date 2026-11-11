"""Style (RQ-VAE) feasibility pieces, CPU only, torch with a thread cap. Inputs: L2-normalised SigLIP pooled embeddings; the residual after a ridge
regression on attribute tokens; a 3-level x 256-code residual-quantised autoencoder. Fitted on ShotQA film-split train stills only. Nothing here sees our films during fitting."""
import numpy as np
import torch
from torch import nn

LEVELS = 3
CODES = 256


def ridge_fit(t: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge with intercept: x ~ t @ w + b. Returns (w (d_t, d_x), b (d_x,))."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    tm, xm = t.mean(0), x.mean(0)
    tc = t - tm
    w = np.linalg.solve(tc.T @ tc + alpha * np.eye(t.shape[1]), tc.T @ (x - xm))
    return w, xm - tm @ w


def residual(t: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) - (np.asarray(t, dtype=np.float64) @ w + b)).astype(np.float32)


def one_hot(idx: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(idx), k), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


class ResidualQuantizer(nn.Module):
    """EMA-codebook residual quantiser: each level quantises what the previous levels left over. Dead codes are re-seeded from the current batch."""

    def __init__(self, dim: int, levels: int = LEVELS, codes: int = CODES, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.levels, self.codes, self.decay, self.eps = levels, codes, decay, eps
        self.register_buffer("emb", torch.randn(levels, codes, dim) * 0.1)
        self.register_buffer("cluster", torch.ones(levels, codes))
        self.register_buffer("sums", self.emb.clone())
        self.register_buffer("initialised", torch.zeros(1))

    def forward(self, z: torch.Tensor):
        """z (N, d) -> (quantised (N, d), indices (N, levels) long, commitment loss)."""
        if self.training and self.initialised.item() == 0:  # seed every level's codebook from encoder outputs
            r = z.detach()
            for lv in range(self.levels):
                pick = r[torch.randperm(len(r))[: self.codes]]
                if len(pick) < self.codes:
                    pick = r[torch.randint(0, len(r), (self.codes,))]
                self.emb[lv] = pick
                self.sums[lv] = pick
                idx = torch.cdist(r, self.emb[lv]).argmin(1)
                r = r - self.emb[lv][idx]
            self.initialised.fill_(1)
        res, out, idxs, loss = z, torch.zeros_like(z), [], z.new_zeros(())
        for lv in range(self.levels):
            e = self.emb[lv]
            idx = torch.cdist(res.detach(), e).argmin(1)
            q = e[idx]
            if self.training:
                with torch.no_grad():
                    oh = torch.zeros(len(idx), self.codes, device=z.device)
                    oh.scatter_(1, idx[:, None], 1.0)
                    self.cluster[lv].mul_(self.decay).add_(oh.sum(0), alpha=1 - self.decay)
                    self.sums[lv].mul_(self.decay).add_(oh.T @ res.detach(), alpha=1 - self.decay)
                    n = self.cluster[lv].sum()
                    c = (self.cluster[lv] + self.eps) / (n + self.codes * self.eps) * n
                    self.emb[lv] = self.sums[lv] / c[:, None]
                    dead = self.cluster[lv] < 1e-2
                    if dead.any():  # re-seed unused codes with random residuals of this batch
                        fresh = res.detach()[torch.randint(0, len(res), (int(dead.sum()),))]
                        self.emb[lv][dead] = fresh
                        self.sums[lv][dead] = fresh
                        self.cluster[lv][dead] = 1.0
            loss = loss + ((res - q.detach()) ** 2).mean()  # commitment (encoder side)
            out = out + q
            res = res - q.detach()
            idxs.append(idx)
        return z + (out - z).detach(), torch.stack(idxs, 1), loss / self.levels


class RQVAE(nn.Module):
    def __init__(self, d_in: int = 768, hidden: int = 512, d_lat: int = 64, levels: int = LEVELS, codes: int = CODES):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_lat))
        self.rq = ResidualQuantizer(d_lat, levels, codes)
        self.dec = nn.Sequential(nn.Linear(d_lat, hidden), nn.GELU(), nn.Linear(hidden, d_in))

    def forward(self, x):
        z = self.enc(x)
        q, idx, commit = self.rq(z)
        return self.dec(q), idx, commit

    @torch.no_grad()
    def encode(self, x: np.ndarray, batch: int = 4096) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(x), batch):
            out.append(self.rq(self.enc(torch.from_numpy(x[i:i + batch])))[1].numpy())
        return np.concatenate(out)


def train_rqvae(x: np.ndarray, epochs: int = 30, batch: int = 512, lr: float = 1e-3, seed: int = 0, max_seconds: float = 1500.0, log=print,
                hidden: int = 512, d_lat: int = 64, levels: int = LEVELS, codes: int = CODES) -> tuple[RQVAE, dict]:
    """Bounded training: stops after `epochs` or `max_seconds`, whichever comes first. Loss = reconstruction MSE + commitment (EMA codebooks)."""
    import time
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = RQVAE(x.shape[1], hidden, d_lat, levels, codes)
    opt = torch.optim.Adam(list(m.enc.parameters()) + list(m.dec.parameters()), lr=lr)
    xt = torch.from_numpy(x)
    hist, t0 = [], time.time()
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(xt), generator=g)
        tot = 0.0
        for i in range(0, len(xt), batch):
            xb = xt[perm[i:i + batch]]
            rec, _, commit = m(xb)
            loss = ((rec - xb) ** 2).mean() + 0.25 * commit
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(xb)
        hist.append(tot / len(xt))
        log(f"epoch {ep + 1}/{epochs} loss {hist[-1]:.5f} ({time.time() - t0:.0f}s)")
        if time.time() - t0 > max_seconds:
            log("time bound reached")
            break
    return m, {"loss": hist, "epochs_run": len(hist), "seconds": time.time() - t0}


def usage_stats(codes: np.ndarray, k: int = CODES) -> dict:
    """Per level: number of used codes, perplexity exp(H) of the code distribution, largest code share."""
    out = []
    for lv in range(codes.shape[1]):
        p = np.bincount(codes[:, lv], minlength=k) / len(codes)
        nz = p[p > 0]
        out.append({"level": lv + 1, "used": int((p > 0).sum()), "perplexity": float(np.exp(-(nz * np.log(nz)).sum())), "max_share": float(p.max())})
    return {"levels": out}


def same_shot_agreement(codes_by_kf: dict[str, np.ndarray], level: int = 0) -> dict:
    """Share of shots whose three keyframes receive the same code at `level` (all three equal), plus the pairwise rates, and the chance level
    from pairing keyframes of different (shuffled) shots (seeded)."""
    s, m, e = (codes_by_kf[k][:, level] for k in ("kf_start", "kf_mid", "kf_end"))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(m))
    return {"all_three": float(((s == m) & (m == e)).mean()), "start_mid": float((s == m).mean()), "mid_end": float((m == e).mean()),
            "chance_all_three_shuffled": float(((s[perm] == m) & (m == e[rng.permutation(len(m))])).mean())}
