"""Graphdeco (INRIA 3DGS) PLY codec — canonical, immutable, CPU NumPy.

Stored fields (per vertex)::

    x y z                       mean (PLY frame P)
    opacity                     logit  -> sigmoid on decode
    scale_0..2                  log    -> exp on decode
    rot_0..3                    quaternion (w, x, y, z), scalar first, unnormalized
    f_dc_0..2                   SH degree-0 coefficients (R, G, B)
    f_rest_0..3*(K-1)-1         channel-major: [R basis 1..K-1, G ..., B ...]

Decoded SH is [N, K, 3] (basis-major, RGB last), the layout gsplat's
``spherical_harmonics`` expects. All Gaussians are preserved (no subsampling).
Plain point clouds (x,y,z,red,green,blue) are rejected, never silently splatted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814
REQUIRED = ("x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")


class PLYFormatError(ValueError):
    pass


@dataclass(frozen=True)
class CanonicalGaussians:
    """Decoded canonical Gaussians of one link, in that link's PLY frame.

    Arrays are read-only (``flags.writeable = False``).
    """

    path: Path
    means: np.ndarray      # [N,3] float32
    quats: np.ndarray      # [N,4] float32, wxyz, unit
    scales: np.ndarray     # [N,3] float32, linear (exp of log-scale)
    opacities: np.ndarray  # [N]   float32, in (0,1)
    sh: np.ndarray         # [N,K,3] float32
    sh_degree: int

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        return self.means.min(0), self.means.max(0)


def ordered_columns(names, prefix: str) -> list[str]:
    matched = [v for v in names if v.startswith(prefix)]
    try:
        vals = sorted((int(v[len(prefix):]), v) for v in matched)
    except ValueError as exc:
        raise PLYFormatError(f"invalid numeric suffix in {prefix}* columns") from exc
    if vals and [x for x, _ in vals] != list(range(len(vals))):
        raise PLYFormatError(f"{prefix}*: columns must be numbered continuously from zero")
    return [v for _, v in vals]


def unpack_graphdeco_sh(dc, rest) -> np.ndarray:
    """dc [N,3] + channel-major rest [N,3*(K-1)] -> [N,K,3]."""
    dc = np.asarray(dc, np.float32)
    rest = np.asarray(rest, np.float32)
    if dc.ndim != 2 or dc.shape[1] != 3 or rest.ndim != 2 or rest.shape[0] != dc.shape[0] or rest.shape[1] % 3:
        raise PLYFormatError(f"SH fields inconsistent: dc {dc.shape}, rest {rest.shape}")
    n = dc.shape[0]
    return np.ascontiguousarray(
        np.concatenate((dc[:, None, :], rest.reshape(n, 3, -1).transpose(0, 2, 1)), axis=1))


def pack_graphdeco_sh(sh) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`unpack_graphdeco_sh`."""
    sh = np.asarray(sh, np.float32)
    n = sh.shape[0]
    return sh[:, 0, :], sh[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)


def _readonly(a):
    a = np.ascontiguousarray(a)
    a.flags.writeable = False
    return a


def load_gaussian_ply(path: str | Path) -> CanonicalGaussians:
    from plyfile import PlyData

    path = Path(path)
    try:
        ply = PlyData.read(str(path))
    except Exception as exc:  # plyfile raises several types
        raise PLYFormatError(f"{path}: cannot read PLY ({exc})") from exc
    if "vertex" not in ply:
        raise PLYFormatError(f"{path}: no 'vertex' element")
    v = ply["vertex"]
    names = v.data.dtype.names
    missing = [f for f in REQUIRED if f not in names]
    if missing:
        hint = ""
        if {"x", "y", "z"} <= set(names) and {"red", "green", "blue"} & set(names):
            hint = " (looks like a plain colored point cloud, not 3D Gaussians)"
        raise PLYFormatError(f"{path}: not a Graphdeco 3DGS PLY; missing {missing}{hint}")
    n = len(v.data)
    if n == 0:
        raise PLYFormatError(f"{path}: empty Gaussian PLY")

    def cols(fields):
        return np.stack([np.asarray(v[f], dtype=np.float32) for f in fields], axis=-1)

    means = cols(("x", "y", "z"))
    log_scales = cols(("scale_0", "scale_1", "scale_2"))
    quats = cols(("rot_0", "rot_1", "rot_2", "rot_3"))
    opacity_logit = np.asarray(v["opacity"], dtype=np.float32)
    rest_names = ordered_columns(names, "f_rest_")
    rest = cols(rest_names) if rest_names else np.zeros((n, 0), np.float32)
    sh = unpack_graphdeco_sh(cols(("f_dc_0", "f_dc_1", "f_dc_2")), rest)
    k = sh.shape[1]
    degree = math.isqrt(k) - 1
    if degree not in (0, 1, 2, 3) or (degree + 1) ** 2 != k:
        raise PLYFormatError(f"{path}: {k} SH bases per channel; expected 1, 4, 9 or 16")
    for nm, a in (("means", means), ("scale", log_scales), ("rot", quats),
                  ("opacity", opacity_logit), ("SH", sh)):
        if not np.isfinite(a).all():
            raise PLYFormatError(f"{path}: non-finite {nm} values")
    qn = np.linalg.norm(quats, axis=-1, keepdims=True)
    if (qn < 1e-8).any():
        raise PLYFormatError(f"{path}: {(qn < 1e-8).sum()} zero-norm Gaussian quaternion(s)")
    if (np.abs(log_scales) > 50).any():
        raise PLYFormatError(f"{path}: implausible log-scale outside [-50, 50]")
    return CanonicalGaussians(
        path=path,
        means=_readonly(means),
        quats=_readonly((quats / qn).astype(np.float32)),
        scales=_readonly(np.exp(log_scales).astype(np.float32)),
        opacities=_readonly((1.0 / (1.0 + np.exp(-opacity_logit))).astype(np.float32)),
        sh=_readonly(sh),
        sh_degree=degree,
    )


def write_gaussian_ply(path: str | Path, means, quats_wxyz, scales, opacities, sh, binary: bool = True) -> Path:
    """Write activated parameters back to a Graphdeco PLY (inverse activations)."""
    from plyfile import PlyData, PlyElement

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    means = np.asarray(means, np.float32)
    n = len(means)
    sh = np.asarray(sh, np.float32).reshape(n, -1, 3)
    dc, rest = pack_graphdeco_sh(sh)
    op = np.clip(np.asarray(opacities, np.float64), 1e-7, 1 - 1e-7)
    fields = {
        "x": means[:, 0], "y": means[:, 1], "z": means[:, 2],
        "opacity": np.log(op / (1 - op)).astype(np.float32),
    }
    for i in range(3):
        fields[f"f_dc_{i}"] = dc[:, i]
    for i in range(rest.shape[1]):
        fields[f"f_rest_{i}"] = rest[:, i]
    ls = np.log(np.asarray(scales, np.float64)).astype(np.float32)
    for i in range(3):
        fields[f"scale_{i}"] = ls[:, i]
    q = np.asarray(quats_wxyz, np.float32)
    for i in range(4):
        fields[f"rot_{i}"] = q[:, i]
    arr = np.empty(n, dtype=[(k, "f4") for k in fields])
    for k, val in fields.items():
        arr[k] = val
    PlyData([PlyElement.describe(arr, "vertex")], text=not binary).write(str(path))
    return path
