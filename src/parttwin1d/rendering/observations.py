"""Explicit GPU -> CPU export boundary: PNG / NPZ / JSON (and GIF for sweeps).

Never called from the render hot path. Metadata records exactly what was
rendered: K, resolution, OpenCV c2w and w2c, q, T_world_link, snapshot
revision, timestamps, depth semantics and units.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

DEPTH_SEMANTICS = ("expected optical-axis Z depth in metres from gsplat 'RGB+ED' "
                   "(sum_i w_i z_i / sum_i w_i); NaN where alpha < alpha_threshold; "
                   "not a hard surface or collision depth")

LINK_PALETTE = np.array([[230, 25, 75], [60, 180, 75], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                         [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 212], [0, 128, 128]],
                        dtype=np.uint8)


def to_numpy(result) -> dict:
    d = {"rgb": result.rgb.detach().cpu().numpy(), "alpha": result.alpha.detach().cpu().numpy(),
         "depth": result.depth.detach().cpu().numpy(), "valid": result.valid.detach().cpu().numpy()}
    if result.link_id is not None:
        d["link_id"] = result.link_id.detach().cpu().numpy()
    return d


def rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def link_id_image(link_id: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
    img = np.empty(link_id.shape + (3,), np.uint8)
    img[:] = background
    m = link_id >= 0
    img[m] = LINK_PALETTE[link_id[m] % len(LINK_PALETTE)]
    return img


def depth_colormap(depth: np.ndarray, dmin: float, dmax: float, cmap: str = "turbo") -> np.ndarray:
    import matplotlib

    x = (depth - dmin) / max(dmax - dmin, 1e-9)
    rgba = matplotlib.colormaps[cmap](np.clip(np.nan_to_num(x, nan=0.0), 0, 1))
    img = (rgba[..., :3] * 255).astype(np.uint8)
    img[~np.isfinite(depth)] = 0
    return img


def metadata(result, extra: dict | None = None) -> dict:
    md = {
        "camera": result.camera.to_json(),
        "q": result.q, "revision": result.revision, "sim_time_s": result.sim_time,
        "T_world_link": {k: np.asarray(v).tolist() for k, v in result.T_world_link.items()},
        "link_names": list(result.link_names), "gaussians": result.gaussians,
        "render_ms": result.render_ms, "alpha_threshold": result.alpha_threshold,
        "depth_semantics": DEPTH_SEMANTICS, "units": {"length": "m", "revolute": "rad", "prismatic": "m"},
        "frames": "T_A_B maps frame-B points to frame A; camera is OpenCV optical",
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        md.update(extra)
    return md


def export_result(result, out_dir, prefix: str = "frame", extra_meta: dict | None = None) -> dict:
    """Write <prefix>_rgb.png, _alpha.png, _depth.npy, _arrays.npz, _meta.json (+ _linkid.png)."""
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a = to_numpy(result)
    paths = {
        "rgb": out / f"{prefix}_rgb.png", "alpha": out / f"{prefix}_alpha.png",
        "depth": out / f"{prefix}_depth.npy", "npz": out / f"{prefix}_arrays.npz",
        "meta": out / f"{prefix}_meta.json",
    }
    Image.fromarray(rgb_uint8(a["rgb"]), "RGB").save(paths["rgb"])
    Image.fromarray((np.clip(a["alpha"], 0, 1) * 255 + 0.5).astype(np.uint8), "L").save(paths["alpha"])
    np.save(paths["depth"], a["depth"].astype(np.float32))
    np.savez_compressed(paths["npz"], **{k: v for k, v in a.items()})
    if "link_id" in a:
        paths["link_id"] = out / f"{prefix}_linkid.png"
        Image.fromarray(link_id_image(a["link_id"]), "RGB").save(paths["link_id"])
    md = metadata(result, extra_meta)
    md["files"] = {k: p.name for k, p in paths.items()}
    paths["meta"].write_text(json.dumps(md, indent=2))
    return paths


def load_export(meta_path) -> dict:
    """Reload an export written by :func:`export_result` (for round-trip tests)."""
    from PIL import Image

    meta_path = Path(meta_path)
    md = json.loads(meta_path.read_text())
    base = meta_path.parent
    f = md["files"]
    arrays = dict(np.load(base / f["npz"]))
    return {"meta": md, "rgb_png": np.asarray(Image.open(base / f["rgb"])),
            "depth": np.load(base / f["depth"]), **arrays}


def save_gif(frames, path, duration_ms: int = 100) -> Path:
    from PIL import Image

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
    return Path(path)
