"""Held-out split of CineScale2 for Step B.

CineScale2 ships no film titles, so a true film-held-out split is impossible. The split is held out by NEAR-DUPLICATE GROUP:
images are joined (union-find) when they are a pixel-verified same-frame pair (dHash candidate within `group_distance`,
32x32 thumbnail MAD < 0.05; flat frames excluded from edges, see cgt.cinescale2.contamination); whole groups go to one side. This keeps identical / re-encoded / letterboxed copies of a frame on one
side, but NOT different shots of the same film (stated as a caveat wherever the split is used).
Assignment: groups are sorted by their smallest relpath, permuted with numpy.random.default_rng(seed), and the first
round(n_groups * val_frac) groups are validation. Excluded ids (contamination, synthetic) are removed from both sides first;
a group that contains any excluded id is excluded in full (conservative)."""
from collections.abc import Iterable, Sequence

import numpy as np


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def groups(ids: Sequence[str], pairs: Iterable[tuple[int, int]]) -> dict[str, str]:
    """{id: group id}; the group id is the smallest member id (stable, order independent)."""
    uf = UnionFind(len(ids))
    for i, j in pairs:
        uf.union(i, j)
    members: dict[int, list[str]] = {}
    for k, i in enumerate(ids):
        members.setdefault(uf.find(k), []).append(i)
    out = {}
    for m in members.values():
        g = min(m)
        for i in m:
            out[i] = g
    return out


def group_split(group_of: dict[str, str], val_frac: float = 0.15, seed: int = 0,
                exclude: Iterable[str] = ()) -> tuple[list[str], list[str], dict]:
    """(train_ids, val_ids, info). Groups touching an excluded id are dropped in full. Both lists sorted, no group on both sides."""
    ex = set(exclude)
    bad_groups = {group_of[i] for i in ex if i in group_of}
    kept = {i: g for i, g in group_of.items() if g not in bad_groups}
    gs = sorted(set(kept.values()))
    perm = np.random.default_rng(seed).permutation(len(gs))
    val_groups = {gs[k] for k in perm[: round(len(gs) * val_frac)]}
    val = sorted(i for i, g in kept.items() if g in val_groups)
    train = sorted(i for i, g in kept.items() if g not in val_groups)
    info = {"n_groups": len(gs), "n_val_groups": len(val_groups), "train": len(train), "val": len(val),
            "excluded_ids": len(ex & set(group_of)), "dropped_with_group": len(group_of) - len(kept),
            "seed": seed, "val_frac": val_frac}
    return train, val, info
