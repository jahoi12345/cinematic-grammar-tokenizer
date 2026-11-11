"""Classification heads on frozen features: multi-hot logistic/MLP, softmax, CORAL ordinal (docs/phase3_interface.md s.4)."""
import numpy as np
import torch
from torch import nn


def coral_loss(cum_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CORAL ordinal loss. cum_logits: (N, K-1) logits of P(y > k); y: (N,) long in [0, K-1]. Mean binary cross-entropy over
    the K-1 cumulative levels (target[k] = 1 iff y > k). Misses are penalised by how many levels they cross."""
    k1 = cum_logits.shape[1]
    levels = (y[:, None] > torch.arange(k1, device=y.device)[None, :]).float()
    return nn.functional.binary_cross_entropy_with_logits(cum_logits, levels)


def coral_class_probs(cum_logits: torch.Tensor) -> torch.Tensor:
    """(N, K-1) cumulative logits -> (N, K) class probabilities (rows sum to 1). P(y>k) is made monotone with a running
    minimum, then p_0 = 1 - s_0, p_k = s_{k-1} - s_k, p_{K-1} = s_{K-2}; clamped at 1e-8 and renormalised."""
    s = torch.cummin(torch.sigmoid(cum_logits), dim=1).values
    p = torch.cat([1 - s[:, :1], s[:, :-1] - s[:, 1:], s[:, -1:]], 1).clamp_min(1e-8)
    return p / p.sum(1, keepdim=True)


def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


class Head(nn.Module):
    def __init__(self, d_in: int, k: int, kind: str, hidden: int = 0, dropout: float = 0.2):
        super().__init__()
        self.kind, self.k = kind, k
        out = {"multihot": k, "softmax": k, "coral": 1}[kind]
        self.net = nn.Linear(d_in, out) if hidden == 0 else nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))
        # CORAL: shared score + K-1 level biases (initialised descending so P(y>k) decreases with k)
        self.bias = nn.Parameter(torch.linspace(1.0, -1.0, k - 1)) if kind == "coral" else None

    def forward(self, x):
        z = self.net(x)
        return z if self.kind != "coral" else z + self.bias  # (N, K-1) cumulative logits

    def loss(self, x, y):
        z = self.forward(x)
        if self.kind == "multihot":
            return nn.functional.binary_cross_entropy_with_logits(z, y)
        if self.kind == "softmax":
            return nn.functional.cross_entropy(z, y)
        return coral_loss(z, y)

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        """multihot: logits (N,K); softmax: probs (N,K); coral: class probs (N,K) from cumulative P(y>k)."""
        z = self.forward(x)
        if self.kind == "multihot":
            return z.cpu().numpy()
        if self.kind == "softmax":
            return torch.softmax(z, 1).cpu().numpy()
        return coral_class_probs(z).cpu().numpy()


def fit_head(kind: str, xtr, ytr, xva, yva, seed: int = 0, hidden: int = 0, epochs: int = 60, lr: float = 1e-3,
             wd: float = 1e-4, batch: int = 512, patience: int = 6) -> tuple[Head, dict]:
    """Train on (xtr, ytr); early-stop on validation loss (ShotQA val only). Returns best head and history."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    k = ytr.shape[1] if kind == "multihot" else int(max(ytr.max(), yva.max()) + 1)
    h = Head(xtr.shape[1], k, kind, hidden)
    opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=wd)
    cast = torch.float32 if kind == "multihot" else torch.long
    xt, yt = torch.from_numpy(xtr), torch.from_numpy(ytr).to(cast)
    xv, yv = torch.from_numpy(xva), torch.from_numpy(yva).to(cast)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        h.train()
        perm = torch.randperm(len(xt), generator=g)
        for i in range(0, len(xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            h.loss(xt[idx], yt[idx]).backward()
            opt.step()
        h.eval()
        with torch.no_grad():
            vl = float(h.loss(xv, yv))
        hist.append(vl)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k_: v.clone() for k_, v in h.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    h.load_state_dict(best_state)
    h.eval()
    return h, {"best_val_loss": best, "epochs": len(hist), "seed": seed}
