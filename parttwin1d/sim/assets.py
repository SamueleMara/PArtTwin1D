"""Configuration and raw Graphdeco-format 3DGS PLYs; no implicit frame changes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
from .geometry import se3


@dataclass(frozen=True)
class GaussianLink:
    link: str
    path: Path
    T_link_ply: np.ndarray


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    K: np.ndarray
    T_world_camera: np.ndarray  # OpenCV optical camera-to-world


@dataclass(frozen=True)
class AssetConfig:
    source: Path
    urdf: Path
    links: tuple[GaussianLink, ...]
    cameras: tuple[CameraSpec, ...]
    root_fixed: bool
    initial_root_pose: np.ndarray
    render_background: tuple[float, float, float]
    sh_degree: int  # -1 for all available coefficients


def _path(cfg_path: Path, value: str) -> Path:
    p = Path(value).expanduser()
    return (cfg_path.parent / p).resolve() if not p.is_absolute() else p.resolve()


def load_config(path: str | Path) -> AssetConfig:
    cfg_path = Path(path).expanduser().resolve()
    cfg = json.loads(cfg_path.read_text())
    urdf = _path(cfg_path, cfg["urdf"])
    if not urdf.is_file():
        raise FileNotFoundError(f"URDF: {urdf}")
    specs = []
    for name, value in cfg["links"].items():
        if not isinstance(value, dict) or "ply" not in value:
            raise ValueError(f"links.{name}: must have a PLY path")
        ply = _path(cfg_path, value["ply"])
        if not ply.is_file():
            raise FileNotFoundError(f"links.{name}: {ply}")
        specs.append(GaussianLink(name, ply, se3(value.get("T_link_ply", np.eye(4)), f"links.{name}.T_link_ply")))
    if not specs:
        raise ValueError("Configure at least one Gaussian link")
    cams = []
    for idx, c in enumerate(cfg.get("cameras", [])):
        name = c.get("name", f"camera_{idx}")
        w, h = int(c["width"]), int(c["height"])
        K = np.asarray(c["K"], dtype=np.float64)
        if w <= 0 or h <= 0 or K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError(f"Camera {name}: invalid dimensions or K")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError(f"Camera {name}: K must have positive fx/fy and homogeneous bottom row")
        cams.append(CameraSpec(name, w, h, K, se3(c["T_world_camera"], f"cameras.{name}.T_world_camera")))
    if not cams:
        raise ValueError("At least one explicit calibrated OpenCV camera is required")
    bg = tuple(float(v) for v in cfg.get("render", {}).get("background_rgb", [1, 1, 1]))
    if len(bg) != 3 or any(not 0 <= v <= 1 for v in bg):
        raise ValueError("render.background_rgb must have three values in [0,1]")
    degree = int(cfg.get("render", {}).get("sh_degree", -1))
    if degree < -1 or degree > 3:
        raise ValueError("SH degree must be -1 or 0..3")
    return AssetConfig(cfg_path, urdf, tuple(specs), tuple(cams),
                       bool(cfg.get("fixed_base", True)),
                       se3(cfg.get("T_world_root", np.eye(4)), "T_world_root"),
                       bg, degree)


def ordered_columns(names, prefix):
    matched = [v for v in names if v.startswith(prefix)]
    try:
        vals = sorted(((int(v[len(prefix):]), v) for v in matched))
    except ValueError as exc:
        raise ValueError(f"Invalid numeric {prefix} suffix") from exc
    if vals and [x for x, _ in vals] != list(range(len(vals))):
        raise ValueError(f"{prefix}: columns must be numbered continuously from zero")
    return [v for _, v in vals]


def unpack_graphdeco_sh(dc, rest):
    """Graphdeco f_rest stored channel-major: [R basis, G basis, B basis]."""
    dc, rest = np.asarray(dc, np.float32), np.asarray(rest, np.float32)
    if dc.ndim != 2 or dc.shape[-1] != 3 or rest.ndim != 2 or rest.shape[0] != dc.shape[0] or rest.shape[-1] % 3:
        raise ValueError("SH fields inconsistent")
    return np.concatenate((dc[:, None, :], rest.reshape(len(dc), 3, -1).transpose(0, 2, 1)), axis=1).copy()


def load_graphdeco_ply(path, device="cuda", limit=0):
    """Return canonical tensors; opacity logits and log scales are decoded once.

    Accepts original Graphdeco-style fields. Does NOT accept an XYZ/RGB PCD as a
    Gaussian scene. No automatic scaling, centering, or global pose guessing.
    """
    from plyfile import PlyData
    import torch
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    required = ["x", "y", "z", "opacity", *[f"f_dc_{i}" for i in range(3)],
                *[f"scale_{i}" for i in range(3)], *[f"rot_{i}" for i in range(4)]]
    missing = sorted(set(required) - set(names))
    if missing:
        raise ValueError(f"{path}: not a 3DGS PLY; missing: {missing}")
    if len(v) == 0:
        raise ValueError(f"{path}: empty Gaussian PLY")
    inds = np.arange(len(v), dtype=np.int64)
    if limit and len(inds) > limit:
        inds = np.linspace(0, len(v)-1, int(limit), dtype=np.int64)
    def cols(fields):
        return np.stack([np.asarray(v[f])[inds] for f in fields], axis=-1).astype(np.float32)
    means = cols(("x", "y", "z"))
    logs = cols([f"scale_{i}" for i in range(3)])
    quats = cols([f"rot_{i}" for i in range(4)])
    opacity = np.asarray(v["opacity"])[inds].astype(np.float32)
    dc = cols([f"f_dc_{i}" for i in range(3)])
    rest_names = ordered_columns(names, "f_rest_")
    rest = cols(rest_names) if rest_names else np.zeros((len(inds), 0), dtype=np.float32)
    sh = unpack_graphdeco_sh(dc, rest)
    basis = sh.shape[1]
    degree = math.isqrt(basis)-1
    if degree not in (0, 1, 2, 3) or (degree+1)**2 != basis:
        raise ValueError(f"{path}: invalid SH basis count {basis}; expected 1,4,9,16")
    for name, a in (("means", means), ("logscales", logs), ("quats", quats), ("opacities", opacity), ("sh", sh)):
        if not np.isfinite(a).all():
            raise ValueError(f"{path}: non-finite {name}")
    if (np.linalg.norm(quats, axis=-1) < 1e-8).any():
        raise ValueError(f"{path}: zero-norm Gaussian quaternion")
    # Very large log-scales overflow exp; fail rather than silently altering geometry.
    if (logs > 50).any() or (logs < -50).any():
        raise ValueError(f"{path}: implausible Gaussian log-scale outside [-50,50]")
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    quats_t = t(quats)
    return {
        "means": t(means),
        "quats": torch.nn.functional.normalize(quats_t, dim=-1),
        "scales": t(np.exp(logs)),
        "opacities": torch.sigmoid(t(opacity)),
        "sh": t(sh),
        "degree": degree,
        "count": len(inds),
    }
