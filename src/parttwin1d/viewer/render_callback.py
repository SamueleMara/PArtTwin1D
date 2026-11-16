"""nerfview ``render_fn``: browser camera state -> live gsplat render.

Called by nerfview's per-client ``Renderer`` thread (pinned commit 4538024)
as ``render_fn(camera_state, render_tab_state)`` under nerfview's global
``VIEWER_LOCK``. It must return ``uint8 [H,W,3]`` or ``(uint8 image, float32
depth [H,W])``; the depth lets viser composite scene overlays against the
splats.

nerfview behaviour this module defends against (see _renderer.py @ 4538024):

* any exception escaping render_fn -> ``traceback; os._exit(1)`` (kills the
  server). We therefore catch every exception, show it in the GUI and return
  an error image. The one exception that MUST propagate is nerfview's
  ``InterruptRenderException`` (raised via ``sys.settrace`` to abort a stale
  render when the camera moves).
* a ``TypeError`` would switch nerfview permanently to its legacy
  ``render_fn(camera_state, img_wh)`` API — also prevented by the catch-all.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..rendering.cameras import RenderCamera
from ..rendering.observations import depth_colormap, link_id_image, rgb_uint8

try:  # resolved at import so the isinstance check is cheap
    from nerfview._renderer import InterruptRenderException
except Exception:  # pragma: no cover
    class InterruptRenderException(Exception):  # type: ignore[no-redef]
        pass

RENDER_MODES = ("rgb", "alpha", "depth", "link_id")
VISER_MAX_DEPTH = 167.0  # viser fixed-point depth range (2^24-1)/1e5 m


@dataclass
class GuiRenderOptions:
    mode: str = "rgb"
    depth_min: float = 0.2
    depth_max: float = 4.0
    sh_degree: int | None = None
    occlude_overlays: bool = True


@dataclass(frozen=True)
class RenderEvent:
    t: float
    client_hint: str
    revision: int
    width: int
    height: int
    K: tuple
    c2w: tuple
    ms: float
    visible_px: int
    mode: str


def _error_image(w: int, h: int, msg: str) -> np.ndarray:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (max(w, 16), max(h, 16)), (60, 20, 20))
    d = ImageDraw.Draw(img)
    y = 10
    for line in ("RENDER ERROR", *[msg[i:i + 70] for i in range(0, min(len(msg), 700), 70)]):
        d.text((10, y), line, fill=(255, 220, 220))
        y += 14
    return np.asarray(img)


def _with_colorbar(img: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    h, w = img.shape[:2]
    bw = max(10, w // 40)
    grad = np.linspace(dmax, dmin, h)[:, None].repeat(bw, 1)
    bar = depth_colormap(grad, dmin, dmax)
    out = img.copy()
    out[:, w - bw:] = bar
    pim = Image.fromarray(out)
    d = ImageDraw.Draw(pim)
    d.text((w - bw - 64, 4), f"{dmax:.2f} m", fill=(0, 0, 0))
    d.text((w - bw - 64, h - 16), f"{dmin:.2f} m", fill=(0, 0, 0))
    return np.asarray(pim)


class ViewerRenderCallback:
    def __init__(self, renderer, coordinator, options: GuiRenderOptions | None = None,
                 on_error=None, on_frame=None, keep_last: bool = True):
        self.renderer = renderer
        self.coordinator = coordinator
        self.options = options or GuiRenderOptions()
        self.on_error = on_error
        self.on_frame = on_frame
        self.keep_last = keep_last
        self.log: collections.deque = collections.deque(maxlen=2000)
        self.count = 0
        self.last = None  # (RenderCamera, rgb uint8, snapshot revision) for parity checks
        self._lock = threading.Lock()

    def __call__(self, camera_state, render_tab_state):
        W = int(getattr(render_tab_state, "viewer_width", 640))
        H = int(getattr(render_tab_state, "viewer_height", 480))
        try:
            snap = self.coordinator.snapshot()
            if snap is None:
                raise RuntimeError("no simulation snapshot published yet")
            cam = RenderCamera.virtual(np.asarray(camera_state.c2w, np.float64), float(camera_state.fov), W, H)
            self.renderer.update_from_snapshot(snap)
            opt = self.options
            res = self.renderer.render(cam, segmentation=(opt.mode == "link_id"), sh_degree=opt.sh_degree)
            rgb = res.rgb.detach().cpu().numpy()
            depth = res.depth.detach().cpu().numpy()
            if opt.mode == "rgb":
                img = rgb_uint8(rgb)
            elif opt.mode == "alpha":
                a = (np.clip(res.alpha.detach().cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                img = np.repeat(a[..., None], 3, -1)
            elif opt.mode == "depth":
                img = _with_colorbar(depth_colormap(depth, opt.depth_min, opt.depth_max), opt.depth_min, opt.depth_max)
            elif opt.mode == "link_id":
                img = link_id_image(res.link_id.detach().cpu().numpy())
            else:
                raise ValueError(f"unknown render mode {opt.mode}")
            visible = int(np.isfinite(depth).sum())
            # nerfview renders each browser client from its own Renderer thread (has .client)
            client = getattr(threading.current_thread(), "client", None)
            cid = str(getattr(client, "client_id", "")) if client is not None else ""
            ev = RenderEvent(time.time(), cid, res.revision, W, H, tuple(cam.K.ravel()),
                             tuple(cam.T_world_camera.ravel()), res.render_ms, visible, opt.mode)
            with self._lock:
                self.count += 1
                self.log.append(ev)
                if self.keep_last:
                    self.last = (cam, rgb_uint8(rgb), res.revision)
            if self.on_frame is not None:
                self.on_frame(ev, res)
            if not opt.occlude_overlays:
                return img
            d = np.nan_to_num(depth, nan=VISER_MAX_DEPTH).astype(np.float32)
            return img, np.minimum(d, VISER_MAX_DEPTH)
        except InterruptRenderException:
            raise
        except Exception as exc:  # surface, never let nerfview os._exit()
            msg = f"{type(exc).__name__}: {exc}"
            try:
                import torch

                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    msg += " — CUDA OOM: lower 'Max resolution' in the Render panel"
            except Exception:
                pass
            if self.on_error is not None:
                try:
                    self.on_error("render", exc, msg)
                except Exception:
                    pass
            return _error_image(W, H, msg)

    def fps(self, window_s: float = 2.0) -> float:
        now = time.time()
        with self._lock:
            n = sum(1 for e in self.log if now - e.t <= window_s)
        return n / window_s

    def latest(self):
        with self._lock:
            return self.log[-1] if self.log else None
