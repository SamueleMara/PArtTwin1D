"""Load an object configuration (YAML or JSON) into a validated ObjectConfig.

Relative paths resolve against the configuration file's directory, never the
current working directory. Unknown keys are errors.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import (
    CAMERA_KEYS, LINK_KEYS, RENDER_KEYS, SCHEMA_VERSION, SIM_KEYS, TOP_LEVEL_KEYS, VIEWER_KEYS,
    CameraSpec, LinkBinding, ObjectConfig, RenderSettings, SimSettings, ViewerSettings,
)
from .validate import ConfigError, K_field, require_keys, resolve_path, rgb_field, se3_field


def read_mapping(path: Path) -> dict:
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ConfigError(f"{path}: unsupported config extension (use .yaml/.yml/.json)")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def _int(v, where, lo=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise ConfigError(f"{where}: expected an integer, got {v!r}")
    v = int(v)
    if lo is not None and v < lo:
        raise ConfigError(f"{where}: must be >= {lo}")
    return v


def _parse_render(d: dict) -> RenderSettings:
    require_keys(d, RENDER_KEYS, "render")
    rs = RenderSettings()
    deg = _int(d.get("sh_degree", rs.sh_degree), "render.sh_degree")
    if deg < -1 or deg > 3:
        raise ConfigError("render.sh_degree: must be -1 (all) or 0..3")
    near = float(d.get("near_plane", rs.near_plane))
    far = float(d.get("far_plane", rs.far_plane))
    if not 0 < near < far:
        raise ConfigError("render: require 0 < near_plane < far_plane")
    a = float(d.get("alpha_threshold", rs.alpha_threshold))
    if not 0 < a < 1:
        raise ConfigError("render.alpha_threshold: must be in (0, 1)")
    return RenderSettings(rgb_field(d.get("background_rgb", rs.background_rgb), "render.background_rgb"),
                          deg, near, far, a)


def _parse_viewer(d: dict) -> ViewerSettings:
    require_keys(d, VIEWER_KEYS, "viewer")
    vs = ViewerSettings()
    out = ViewerSettings(
        host=str(d.get("host", vs.host)),
        port=_int(d.get("port", vs.port), "viewer.port", 1),
        preview_width=_int(d.get("preview_width", vs.preview_width), "viewer.preview_width", 16),
        preview_height=_int(d.get("preview_height", vs.preview_height), "viewer.preview_height", 16),
        max_resolution=_int(d.get("max_resolution", vs.max_resolution), "viewer.max_resolution", 64),
        jpeg_quality=_int(d.get("jpeg_quality", vs.jpeg_quality), "viewer.jpeg_quality", 1),
        playback_hz=float(d.get("playback_hz", vs.playback_hz)),
    )
    if out.port > 65535 or out.jpeg_quality > 100 or not 1 <= out.playback_hz <= 240:
        raise ConfigError("viewer: port<=65535, jpeg_quality<=100, playback_hz in [1,240]")
    return out


def _parse_sim(d: dict) -> SimSettings:
    require_keys(d, SIM_KEYS, "sim")
    ss = SimSettings()
    backend = str(d.get("backend", ss.backend))
    if backend not in ("physx_cpu", "physx_cuda"):
        raise ConfigError("sim.backend: 'physx_cpu' or 'physx_cuda'")
    out = SimSettings(
        backend=backend,
        sim_freq=_int(d.get("sim_freq", ss.sim_freq), "sim.sim_freq", 1),
        control_freq=_int(d.get("control_freq", ss.control_freq), "sim.control_freq", 1),
        drive_stiffness=float(d.get("drive_stiffness", ss.drive_stiffness)),
        drive_damping=float(d.get("drive_damping", ss.drive_damping)),
        drive_force_limit=float(d.get("drive_force_limit", ss.drive_force_limit)),
    )
    if out.sim_freq % out.control_freq:
        raise ConfigError("sim: sim_freq must be an integer multiple of control_freq")
    return out


def _parse_camera(c: dict, idx: int) -> CameraSpec:
    where = f"cameras[{idx}]"
    require_keys(c, CAMERA_KEYS, where)
    name = c.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{where}.name: required non-empty string")
    where = f"cameras[{name}]"
    for k in ("width", "height", "K", "T_world_camera"):
        if k not in c:
            raise ConfigError(f"{where}.{k}: required")
    w = _int(c["width"], f"{where}.width", 1)
    h = _int(c["height"], f"{where}.height", 1)
    dist = c.get("distortion")
    if dist is not None and any(float(x) != 0.0 for x in dist):
        raise ConfigError(f"{where}.distortion: non-zero distortion is not supported; undistort images and use the rectified K")
    return CameraSpec(name, w, h, K_field(c["K"], f"{where}.K", w, h),
                      se3_field(c["T_world_camera"], f"{where}.T_world_camera"))


def load_config(path: str | Path, *, check_files: bool = True) -> ObjectConfig:
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"config not found: {cfg_path}")
    d = read_mapping(cfg_path)
    require_keys(d, TOP_LEVEL_KEYS, "config")
    ver = d.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise ConfigError(f"schema_version: expected {SCHEMA_VERSION}, got {ver!r}")
    for k in ("object_name", "urdf", "T_world_root", "links"):
        if k not in d:
            raise ConfigError(f"{k}: required")
    base = cfg_path.parent
    urdf = resolve_path(base, d["urdf"], "urdf", check_files)

    links_d = d["links"]
    if not isinstance(links_d, dict) or not links_d:
        raise ConfigError("links: expected a non-empty mapping link_name -> {ply, T_link_ply}")
    links = []
    seen_plys: dict[Path, str] = {}
    for name, ld in links_d.items():
        where = f"links.{name}"
        require_keys(ld, LINK_KEYS, where)
        if "ply" not in ld:
            raise ConfigError(f"{where}.ply: required")
        if "T_link_ply" not in ld:
            raise ConfigError(f"{where}.T_link_ply: required (state identity explicitly if the PLY is already in the link frame)")
        ply = resolve_path(base, ld["ply"], f"{where}.ply", check_files)
        if ply in seen_plys:
            raise ConfigError(f"{where}.ply: same file already bound to link {seen_plys[ply]!r}")
        seen_plys[ply] = name
        links.append(LinkBinding(str(name), ply, se3_field(ld["T_link_ply"], f"{where}.T_link_ply")))

    cams_d = d.get("cameras", [])
    if not isinstance(cams_d, list):
        raise ConfigError("cameras: expected a list")
    cams = tuple(_parse_camera(c, i) for i, c in enumerate(cams_d))
    names = [c.name for c in cams]
    if len(set(names)) != len(names):
        raise ConfigError(f"cameras: duplicate names {names}")

    iq = d.get("initial_q", {}) or {}
    if not isinstance(iq, dict):
        raise ConfigError("initial_q: expected mapping joint_name -> value")
    iq = {str(k): float(v) for k, v in iq.items()}

    fixed = d.get("fixed_base", True)
    if not isinstance(fixed, bool):
        raise ConfigError("fixed_base: expected true/false")
    name = d["object_name"]
    if not isinstance(name, str) or not name:
        raise ConfigError("object_name: required non-empty string")

    cfg = ObjectConfig(
        source=cfg_path, schema_version=ver, object_name=name, urdf=urdf, fixed_base=fixed,
        T_world_root=se3_field(d["T_world_root"], "T_world_root"),
        links=tuple(links), cameras=cams, initial_q=iq,
        render=_parse_render(d.get("render", {}) or {}),
        viewer=_parse_viewer(d.get("viewer", {}) or {}),
        sim=_parse_sim(d.get("sim", {}) or {}),
    )
    if check_files:
        from ..assets.urdf import check_bindings

        check_bindings(cfg)
    return cfg
