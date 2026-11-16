"""Read-only canonical per-link Gaussians, uploaded to the GPU once.

All links are concatenated into one contiguous set of buffers (link slices are
contiguous, ``offsets[i]:offsets[i+1]``). SH coefficients are zero-padded to the
largest basis count: evaluating a degree-d link at a higher degree with zero
higher-band coefficients is exactly its degree-d value, so one SH call serves
links of mixed degree.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..assets.gaussian_ply import CanonicalGaussians, load_gaussian_ply


@dataclass
class GaussianScene:
    link_names: tuple[str, ...]
    T_link_ply: "object"     # torch [L,4,4] float32
    means: "object"          # torch [N,3]   canonical (PLY frame), never written
    quats: "object"          # torch [N,4]   wxyz unit
    scales: "object"         # torch [N,3]
    opacities: "object"      # torch [N]
    sh: "object"             # torch [N,K,3]
    link_ids: "object"       # torch [N] int64
    offsets: tuple[int, ...]  # len L+1
    degrees: tuple[int, ...]  # per link
    canonical: tuple[CanonicalGaussians, ...]

    @property
    def count(self) -> int:
        return int(self.offsets[-1])

    @property
    def max_degree(self) -> int:
        return max(self.degrees)

    def link_slice(self, i: int) -> slice:
        return slice(self.offsets[i], self.offsets[i + 1])

    def checksum(self) -> str:
        """Hash of the device canonical buffers (immutability regression check)."""
        h = hashlib.sha256()
        for t in (self.means, self.quats, self.scales, self.opacities, self.sh):
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()


def load_scene(bindings, device: str = "cuda") -> GaussianScene:
    """bindings: iterable of objects with ``.link``, ``.ply``, ``.T_link_ply``."""
    import torch

    bindings = list(bindings)
    if not bindings:
        raise ValueError("no Gaussian link bindings")
    names = [b.link for b in bindings]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate link bindings {names}")
    canon = [load_gaussian_ply(b.ply) for b in bindings]
    kmax = max(c.sh.shape[1] for c in canon)
    sh = []
    for c in canon:
        pad = np.zeros((c.count, kmax, 3), np.float32)
        pad[:, : c.sh.shape[1]] = c.sh
        sh.append(pad)
    counts = [c.count for c in canon]
    offsets = tuple(int(x) for x in np.concatenate([[0], np.cumsum(counts)]))

    def t(a, dtype=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)

    return GaussianScene(
        link_names=tuple(names),
        T_link_ply=t(np.stack([b.T_link_ply for b in bindings])),
        means=t(np.concatenate([c.means for c in canon])),
        quats=t(np.concatenate([c.quats for c in canon])),
        scales=t(np.concatenate([c.scales for c in canon])),
        opacities=t(np.concatenate([c.opacities for c in canon])),
        sh=t(np.concatenate(sh)),
        link_ids=t(np.repeat(np.arange(len(canon)), counts), torch.int64),
        offsets=offsets,
        degrees=tuple(c.sh_degree for c in canon),
        canonical=tuple(canon),
    )
