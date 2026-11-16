"""Typed, immutable configuration objects (schema_version 1).

Every geometric field is explicit: ``T_world_root`` and every ``T_link_ply``
are required — identity is never assumed silently (docs/urdf_and_gaussians.md).
All paths are absolute after loading (resolved relative to the config file).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LinkBinding:
    """One canonical Gaussian PLY rigidly attached to one URDF link.

    T_link_ply = T_{L<-P}: maps PLY coordinates into the URDF link frame.
    """

    link: str
    ply: Path
    T_link_ply: np.ndarray


@dataclass(frozen=True)
class CameraSpec:
    """A calibrated real camera: exact K, OpenCV optical T_world_camera (c2w)."""

    name: str
    width: int
    height: int
    K: np.ndarray
    T_world_camera: np.ndarray
    source: str = "calibrated"


@dataclass(frozen=True)
class RenderSettings:
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    sh_degree: int = -1  # -1: all degrees present in the PLYs
    near_plane: float = 0.01
    far_plane: float = 100.0
    alpha_threshold: float = 0.01  # depth is NaN where alpha < threshold


@dataclass(frozen=True)
class ViewerSettings:
    host: str = "127.0.0.1"
    port: int = 8080
    preview_width: int = 640
    preview_height: int = 480
    max_resolution: int = 960
    jpeg_quality: int = 85
    playback_hz: float = 30.0


@dataclass(frozen=True)
class SimSettings:
    backend: str = "physx_cpu"  # physx_cpu | physx_cuda
    sim_freq: int = 500
    control_freq: int = 50
    # Default PD joint drive used by the explicit "dynamics preview" mode only.
    drive_stiffness: float = 50.0
    drive_damping: float = 5.0
    drive_force_limit: float = 100.0


@dataclass(frozen=True)
class ObjectConfig:
    source: Path
    schema_version: int
    object_name: str
    urdf: Path
    fixed_base: bool
    T_world_root: np.ndarray
    links: tuple[LinkBinding, ...]
    cameras: tuple[CameraSpec, ...]
    initial_q: dict = field(default_factory=dict)  # joint name -> value (rad|m)
    render: RenderSettings = RenderSettings()
    viewer: ViewerSettings = ViewerSettings()
    sim: SimSettings = SimSettings()

    def link_binding(self, link: str) -> LinkBinding:
        for b in self.links:
            if b.link == link:
                return b
        raise KeyError(link)

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(b.link for b in self.links)

    # Backwards-compatible aliases used by the pre-refactor scripts.
    @property
    def root_fixed(self) -> bool:
        return self.fixed_base

    @property
    def initial_root_pose(self) -> np.ndarray:
        return self.T_world_root


# Allowed keys (unknown keys are rejected to catch typos such as "T_link_ply ").
TOP_LEVEL_KEYS = {
    "schema_version", "object_name", "urdf", "fixed_base", "T_world_root",
    "links", "cameras", "initial_q", "render", "viewer", "sim", "notes",
}
LINK_KEYS = {"ply", "T_link_ply", "notes"}
CAMERA_KEYS = {"name", "width", "height", "K", "T_world_camera", "distortion", "notes"}
RENDER_KEYS = {"background_rgb", "sh_degree", "near_plane", "far_plane", "alpha_threshold"}
VIEWER_KEYS = {"host", "port", "preview_width", "preview_height", "max_resolution", "jpeg_quality", "playback_hz"}
SIM_KEYS = {"backend", "sim_freq", "control_freq", "drive_stiffness", "drive_damping", "drive_force_limit"}
