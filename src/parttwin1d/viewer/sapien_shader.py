"""Gaussian-compositing shader pack for the SAPIEN / ManiSkill viewer.

Generated at runtime from the *installed* SAPIEN's own ``vulkan_shader/default``
pack (so it always matches the SAPIEN binary; nothing Apache-licensed is
vendored). Only the final composite pass ``composite0.frag`` is patched:

* two extra samplers ``samplerGaussianColor`` (premultiplied RGB + alpha, from a
  gsplat render with black background) and ``samplerGaussianDepth`` (expected
  optical-Z depth in metres, huge where empty). SAPIEN binds them by name from
  ``RenderWindow.set_camera_texture("GaussianColor"|"GaussianDepth", tex)``.
* per pixel: if a Gaussian layer exists and is nearer than the rasterized SAPIEN
  surface (``-position.z`` is the OpenGL view-space depth of the gbuffer), the
  colour is composited ``g.rgb + (1 - g.a) * scene.rgb``; SAPIEN lines/points
  (joint axes, frames) are still drawn on top afterwards, as in the original.

Verified on SAPIEN 3.0.3: custom samplers bind by name; texture row 0 is the
top image row; depth test against the gbuffer is correct (see
tests/integration/test_maniskill_viewer.py).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

PACK_NAME = "parttwin_gaussian_composite"

_SAMPLER_ANCHOR = "layout(set = 0, binding = 7) uniform sampler2D samplerGbufferDepth;"
_SAMPLER_ADD = """
// --- PArtTwin1D: gsplat layer (bound by name via RenderWindow.set_camera_texture) ---
layout(set = 0, binding = 8) uniform sampler2D samplerGaussianColor;  // premultiplied rgb, alpha
layout(set = 0, binding = 9) uniform sampler2D samplerGaussianDepth;  // optical Z [m]
"""
_COMPOSITE_ANCHOR = "  vec4 lineColor = texture(samplerLine, inUV);"
_COMPOSITE_ADD = """  // --- PArtTwin1D: depth-composite the Gaussian layer with the SAPIEN scene ---
  {
    vec4 g = texture(samplerGaussianColor, inUV);
    float gz = texture(samplerGaussianDepth, inUV).x;
    bool sceneEmpty = texture(samplerGbufferDepth, inUV).x >= 1.0;
    if (g.a > 0.0 && (sceneEmpty || gz < -position.z)) {
      outColor = vec4(g.rgb + (1.0 - g.a) * outColor.rgb, 1.0);
    }
  }
"""


class ShaderPatchError(RuntimeError):
    pass


def _sapien_shader_root() -> Path:
    import sapien

    return Path(sapien.__file__).resolve().parent / "vulkan_shader"


def patch_composite(src: str) -> str:
    for anchor in (_SAMPLER_ANCHOR, _COMPOSITE_ANCHOR):
        if src.count(anchor) != 1:
            raise ShaderPatchError(f"composite0.frag anchor not found exactly once: {anchor!r}; "
                                   "SAPIEN's default shader changed — update parttwin1d.viewer.sapien_shader")
    src = src.replace(_SAMPLER_ANCHOR, _SAMPLER_ANCHOR + "\n" + _SAMPLER_ADD)
    return src.replace(_COMPOSITE_ANCHOR, _COMPOSITE_ADD + _COMPOSITE_ANCHOR)


def build_shader_pack(cache_root: str | Path | None = None) -> Path:
    """Create (idempotently) and return the absolute shader-pack directory."""
    import sapien

    root = _sapien_shader_root()
    default, common = root / "default", root / "common"
    src = (default / "composite0.frag").read_text()
    patched = patch_composite(src)
    key = hashlib.sha256((sapien.__version__ + src).encode()).hexdigest()[:12]
    base = Path(cache_root or Path.home() / ".cache" / "parttwin1d" / "sapien_shaders") / f"sapien{sapien.__version__}-{key}"
    pack = base / PACK_NAME
    stamp = pack / ".parttwin_ok"
    if stamp.exists():
        return pack
    tmp = base.with_name(base.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(default, tmp / PACK_NAME)
    shutil.copytree(common, tmp / "common")  # the default pack #includes ../common/*.glsl
    (tmp / PACK_NAME / "composite0.frag").write_text(patched)
    (tmp / PACK_NAME / ".parttwin_ok").write_text(f"generated from {default} (SAPIEN {sapien.__version__})\n")
    shutil.rmtree(base, ignore_errors=True)
    tmp.rename(base)
    return pack
