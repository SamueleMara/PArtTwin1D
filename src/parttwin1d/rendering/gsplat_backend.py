"""ArticulatedGaussianRenderer — the single rendering backend (gsplat 1.5.3 CUDA).

    renderer = ArticulatedGaussianRenderer.from_config(cfg, device="cuda")
    renderer.bind_articulation(articulation)       # exact link-name validation
    renderer.update_from_snapshot(snapshot)        # T_W_L for every bound link
    result = renderer.render(camera)               # GPU RGB/alpha/depth (+ link id)

* All links are composited in ONE ``gsplat.rendering.rasterization`` call per
  camera batch, so occlusion is resolved globally by depth sorting, independent
  of link / PLY order. No per-link RGBA pasting.
* Canonical buffers are never written; world buffers are recomputed from them
  per snapshot revision (no drift).
* No new rasterizer, no fallback: if gsplat's CUDA extension is unavailable
  the constructor raises.
* Thread-safe: one internal lock serialises update+render (the GUI renders from
  nerfview worker threads; exports run from GUI callbacks).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .cameras import RenderCamera
from .gaussians import GaussianScene, load_scene
from .link_adapter import LinkAdapter


class RendererError(RuntimeError):
    pass


@dataclass
class RenderResult:
    """One camera's render. Image tensors live on the render device."""

    camera: RenderCamera
    rgb: "object"           # [H,W,3] float32 in [0,1]
    alpha: "object"         # [H,W]   float32 in [0,1]
    depth: "object"         # [H,W]   float32 metres (expected optical Z), NaN where invalid
    valid: "object"         # [H,W]   bool: alpha >= alpha_threshold
    link_id: "object" = None  # [H,W] int16 index into link_names, -1 background
    link_names: tuple = ()
    q: dict = field(default_factory=dict)
    revision: int = -1
    sim_time: float = 0.0
    T_world_link: dict = field(default_factory=dict)  # link -> np 4x4
    render_ms: float = float("nan")
    gaussians: int = 0
    alpha_threshold: float = 0.01

    def validate(self) -> "RenderResult":
        import torch

        H, W = self.camera.height, self.camera.width
        checks = [("rgb", self.rgb, (H, W, 3), torch.float32), ("alpha", self.alpha, (H, W), torch.float32),
                  ("depth", self.depth, (H, W), torch.float32), ("valid", self.valid, (H, W), torch.bool)]
        if self.link_id is not None:
            checks.append(("link_id", self.link_id, (H, W), torch.int16))
        dev = self.rgb.device
        for name, t, shape, dtype in checks:
            if tuple(t.shape) != shape or t.dtype != dtype or t.device != dev:
                raise RendererError(f"RenderResult.{name}: got {tuple(t.shape)} {t.dtype} {t.device}, "
                                    f"expected {shape} {dtype} {dev}")
        if not torch.isfinite(self.rgb).all():
            raise RendererError("RenderResult.rgb contains non-finite values")
        return self


def gsplat_cuda_available() -> tuple[bool, str]:
    try:
        from gsplat.cuda._backend import _C
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"import failed: {exc}"
    if _C is None:
        return False, "gsplat CUDA extension (_C) is None (no CUDA toolkit or build failed)"
    return True, getattr(_C, "__file__", "?")


class ArticulatedGaussianRenderer:
    def __init__(self, scene: GaussianScene, *, background=(1.0, 1.0, 1.0), sh_degree: int = -1,
                 near_plane: float = 0.01, far_plane: float = 100.0, alpha_threshold: float = 0.01):
        import torch

        if scene.means.device.type != "cuda":
            raise RendererError("gsplat rasterization requires the scene on a CUDA device")
        ok, info = gsplat_cuda_available()
        if not ok:
            raise RendererError(f"gsplat CUDA kernels unavailable: {info}")
        self.extension_path = info
        self.scene = scene
        self.device = scene.means.device
        self.adapter = LinkAdapter(scene)
        self.background = torch.tensor(background, dtype=torch.float32, device=self.device)
        self.sh_degree = scene.max_degree if sh_degree < 0 else min(sh_degree, scene.max_degree)
        self.near_plane, self.far_plane = float(near_plane), float(far_plane)
        self.alpha_threshold = float(alpha_threshold)
        self._lock = threading.RLock()
        self._bound = False
        self._updated = False
        self._revision = None
        self._snapshot = None
        self._onehot = None

    # ---------------------------------------------------------------- set-up
    @classmethod
    def from_config(cls, cfg, device: str = "cuda") -> "ArticulatedGaussianRenderer":
        r = cfg.render
        return cls(load_scene(cfg.links, device), background=r.background_rgb, sh_degree=r.sh_degree,
                   near_plane=r.near_plane, far_plane=r.far_plane, alpha_threshold=r.alpha_threshold)

    @property
    def link_names(self) -> tuple[str, ...]:
        return self.scene.link_names

    @property
    def gaussian_count(self) -> int:
        return self.scene.count

    def bind_articulation(self, articulation_or_links) -> None:
        """Validate that every Gaussian-bound link exists in the articulation."""
        names = articulation_or_links.links_map.keys() if hasattr(articulation_or_links, "links_map") \
            else articulation_or_links
        names = set(names)
        absent = [n for n in self.scene.link_names if n not in names]
        if absent:
            raise RendererError(f"Gaussian-bound links {absent} not in articulation links {sorted(names)}")
        self._bound = True

    def set_background(self, rgb) -> None:
        import torch

        rgb = [float(v) for v in rgb]
        if len(rgb) != 3 or any(not 0.0 <= v <= 1.0 for v in rgb):
            raise ValueError(f"background must be 3 values in [0,1], got {rgb}")
        with self._lock:
            self.background = torch.tensor(rgb, dtype=torch.float32, device=self.device)

    # ---------------------------------------------------------------- state
    def update_from_link_poses(self, T_world_link: dict, *, revision=None, snapshot=None) -> None:
        import torch

        missing = [n for n in self.scene.link_names if n not in T_world_link]
        if missing:
            raise RendererError(f"pose missing for bound links {missing}")
        with self._lock:
            if revision is not None and revision == self._revision:
                return
            T = torch.as_tensor(np.stack([np.asarray(T_world_link[n], np.float64) for n in self.scene.link_names]),
                                dtype=torch.float32, device=self.device)
            self._revision = None
            self.adapter.update(T)
            self._snapshot = snapshot
            self._updated = True
            self._revision = revision  # set last: an interrupted update is recomputed

    def update_from_snapshot(self, snapshot) -> None:
        if not self._bound:
            self.bind_articulation(snapshot.T_world_link.keys())
        self.update_from_link_poses(snapshot.T_world_link, revision=snapshot.revision, snapshot=snapshot)

    # ---------------------------------------------------------------- render
    def _link_onehot(self):
        import torch

        if self._onehot is None:
            L = len(self.scene.link_names)
            self._onehot = torch.nn.functional.one_hot(self.scene.link_ids, L).to(torch.float32)
        return self._onehot

    def render_batch(self, cameras, *, segmentation: bool = False, sh_degree: int | None = None, background=None):
        """Render C cameras sharing one resolution in a single rasterization call.

        Returns dict of GPU tensors: rgb [C,H,W,3], alpha [C,H,W], depth [C,H,W] (NaN invalid),
        valid [C,H,W], optionally link_id [C,H,W] int16.
        """
        import torch
        from gsplat.rendering import rasterization

        cameras = list(cameras)
        if not cameras:
            raise ValueError("no cameras")
        W, H = cameras[0].width, cameras[0].height
        if any((c.width, c.height) != (W, H) for c in cameras):
            raise ValueError("render_batch needs equal resolutions; call per resolution group")
        deg = self.sh_degree if sh_degree is None else max(0, min(int(sh_degree), self.scene.max_degree))
        with self._lock, torch.no_grad():
            if not self._updated:
                raise RendererError("render() before update_from_snapshot()/update_from_link_poses()")
            viewmats = torch.as_tensor(np.stack([c.viewmat for c in cameras]), dtype=torch.float32, device=self.device)
            Ks = torch.as_tensor(np.stack([c.K for c in cameras]), dtype=torch.float32, device=self.device)
            cam_pos = torch.as_tensor(np.stack([c.position for c in cameras]), dtype=torch.float32, device=self.device)
            colors = self.adapter.colors(cam_pos, deg)  # [C,N,3]
            C = len(cameras)
            bgv = self.background if background is None else torch.as_tensor(background, dtype=torch.float32, device=self.device)
            bg = bgv[None].expand(C, 3)
            if segmentation:
                oh = self._link_onehot()[None].expand(C, -1, -1)
                colors = torch.cat([colors, oh], dim=-1)
                bg = torch.cat([bg, torch.zeros((C, oh.shape[-1]), device=self.device)], dim=-1)
            out, alphas, _meta = rasterization(
                means=self.adapter.means_w, quats=self.adapter.quats_w, scales=self.scene.scales,
                opacities=self.scene.opacities, colors=colors.contiguous(), viewmats=viewmats, Ks=Ks,
                width=W, height=H, near_plane=self.near_plane, far_plane=self.far_plane,
                sh_degree=None, backgrounds=bg.contiguous(), render_mode="RGB+ED",
                # packed=True + backgrounds trips an assertion in gsplat 1.5.3
                # (rasterize_to_pixels takes image_dims from packed means2d [nnz,2]
                # -> (), so a [C,D] background never matches). gsplat's own
                # examples/simple_viewer.py renders with packed=False.
                packed=False)
            alpha = alphas[..., 0]
            valid = alpha >= self.alpha_threshold
            depth = out[..., -1].masked_fill(~valid, float("nan"))
            res = {"rgb": out[..., :3].clamp(0.0, 1.0), "alpha": alpha, "depth": depth, "valid": valid}
            if segmentation:
                w = out[..., 3:-1]
                lid = w.argmax(-1).to(torch.int16)
                res["link_id"] = lid.masked_fill(~valid, -1)
            return res

    def render(self, camera: RenderCamera, *, segmentation: bool = False, sh_degree: int | None = None,
               background=None) -> RenderResult:
        """background=(0,0,0) yields premultiplied colour (used for compositing)."""
        import torch

        t0 = time.perf_counter()
        with self._lock:
            out = self.render_batch([camera], segmentation=segmentation, sh_degree=sh_degree, background=background)
            snap = self._snapshot
            torch.cuda.synchronize(self.device)
        ms = (time.perf_counter() - t0) * 1e3
        return RenderResult(
            camera=camera, rgb=out["rgb"][0], alpha=out["alpha"][0], depth=out["depth"][0],
            valid=out["valid"][0], link_id=out["link_id"][0] if segmentation else None,
            link_names=self.scene.link_names,
            q=snap.q_dict() if snap is not None else {}, revision=snap.revision if snap is not None else -1,
            sim_time=snap.sim_time if snap is not None else 0.0,
            T_world_link={k: np.array(v) for k, v in snap.T_world_link.items()} if snap is not None else {},
            render_ms=ms, gaussians=self.scene.count, alpha_threshold=self.alpha_threshold,
        )

    def render_observation(self, snapshot, cameras, *, segmentation: bool = False) -> dict:
        """Future visual-PPO interface: snapshot + cameras -> batched GPU tensors.

        Single environment today; the camera batch is rendered in one call. A
        vectorised multi-env path would feed ``sim.state.link_poses_tensor``
        [B,L,4,4] directly (no host copy) — not implemented/benchmarked yet.
        """
        self.update_from_snapshot(snapshot)
        return self.render_batch(cameras, segmentation=segmentation)

    def world_means(self):
        """Current world-frame Gaussian means (GPU, read-only view for tests/overlays)."""
        return self.adapter.means_w
