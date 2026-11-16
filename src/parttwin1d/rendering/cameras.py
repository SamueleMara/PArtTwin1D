"""Render-camera objects shared by calibrated sensors and GUI orbit cameras.

Both camera sources go through the *same* renderer; they differ only in where
K and T_world_camera come from:

``calibrated``  exact K (fx, fy, cx, cy) and T_world_camera from the config.
``virtual``     GUI orbit camera: c2w from the browser (viser/nerfview, already
                OpenCV: +Z forward, +Y down) and a *chosen* pinhole K from the
                viewport FOV (fx = fy, principal point at the image centre).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..geometry.cameras import intrinsics_from_vertical_fov, scale_intrinsics, validate_K
from ..geometry.se3 import invert, validate_se3


@dataclass(frozen=True)
class RenderCamera:
    name: str
    width: int
    height: int
    K: np.ndarray               # [3,3] float64
    T_world_camera: np.ndarray  # [4,4] OpenCV optical c2w
    source: str = "calibrated"  # "calibrated" | "virtual"

    def __post_init__(self):
        object.__setattr__(self, "K", validate_K(self.K, f"{self.name}.K"))
        object.__setattr__(self, "T_world_camera", validate_se3(self.T_world_camera, f"{self.name}.T_world_camera"))
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"{self.name}: invalid resolution {self.width}x{self.height}")

    @property
    def viewmat(self) -> np.ndarray:
        """World-to-camera (gsplat ``viewmats``)."""
        return invert(self.T_world_camera)

    @property
    def position(self) -> np.ndarray:
        return self.T_world_camera[:3, 3].copy()

    @classmethod
    def from_spec(cls, spec) -> "RenderCamera":
        return cls(spec.name, spec.width, spec.height, spec.K, spec.T_world_camera, "calibrated")

    @classmethod
    def virtual(cls, c2w, fov_y: float, width: int, height: int, name: str = "orbit") -> "RenderCamera":
        return cls(name, int(width), int(height), intrinsics_from_vertical_fov(fov_y, width, height),
                   np.asarray(c2w, np.float64), "virtual")

    def resized(self, width: int, height: int) -> "RenderCamera":
        """Same sensor at another resolution with exactly rescaled K (fx,fy,cx,cy)."""
        sx, sy = width / self.width, height / self.height
        return replace(self, width=int(width), height=int(height), K=scale_intrinsics(self.K, sx, sy))

    def to_json(self) -> dict:
        return {"name": self.name, "source": self.source, "width": self.width, "height": self.height,
                "K": self.K.tolist(), "T_world_camera_c2w_opencv": self.T_world_camera.tolist(),
                "viewmat_w2c": self.viewmat.tolist()}
