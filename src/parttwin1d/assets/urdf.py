"""Static URDF description used for validation, inspection and overlays.

This is **not** a kinematics engine: live link poses always come from the
SAPIEN/PhysX articulation (``parttwin1d.sim``). We parse the XML only to
(a) validate config bindings before starting PhysX, (b) report joint/collision
metadata, and (c) draw joint axes / collision overlays in the joint's child
link frame (URDF: the child link frame coincides with the joint frame).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..config.validate import ConfigError


@dataclass(frozen=True)
class CollisionShape:
    kind: str  # box | cylinder | sphere | mesh
    T_link_shape: np.ndarray
    size: tuple = ()  # box: (x,y,z); cylinder: (radius, length); sphere: (radius,)
    mesh: str | None = None
    scale: tuple = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class URDFLink:
    name: str
    mass: float | None
    collisions: tuple[CollisionShape, ...] = ()


@dataclass(frozen=True)
class URDFJoint:
    name: str
    type: str  # revolute | continuous | prismatic | fixed | floating | planar
    parent: str
    child: str
    T_parent_joint: np.ndarray
    axis: np.ndarray  # unit vector in the joint (= child link) frame
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    damping: float = 0.0
    friction: float = 0.0
    mimic: str | None = None

    @property
    def movable(self) -> bool:
        return self.type in ("revolute", "continuous", "prismatic") and self.mimic is None

    @property
    def unit(self) -> str:
        return "m" if self.type == "prismatic" else "rad"


@dataclass(frozen=True)
class URDFModel:
    path: Path
    name: str
    links: dict = field(default_factory=dict)  # name -> URDFLink
    joints: dict = field(default_factory=dict)  # name -> URDFJoint

    @property
    def root_links(self) -> list[str]:
        children = {j.child for j in self.joints.values()}
        return [n for n in self.links if n not in children]

    def joint_for_child(self, link: str) -> URDFJoint | None:
        for j in self.joints.values():
            if j.child == link:
                return j
        return None


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is None:
        return T
    xyz = [float(v) for v in el.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.attrib.get("rpy", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()  # URDF: fixed-axis roll, pitch, yaw
    T[:3, 3] = xyz
    return T


def _f(el, attr):
    if el is None or attr not in el.attrib:
        return None
    return float(el.attrib[attr])


def parse_urdf(path: str | Path) -> URDFModel:
    path = Path(path)
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ConfigError(f"URDF {path}: XML parse error: {exc}") from exc
    if root.tag != "robot":
        raise ConfigError(f"URDF {path}: root element must be <robot>")
    links: dict[str, URDFLink] = {}
    for le in root.findall("link"):
        name = le.attrib["name"]
        if name in links:
            raise ConfigError(f"URDF {path}: duplicate link {name!r}")
        mass_el = le.find("inertial/mass")
        cols = []
        for ce in le.findall("collision"):
            g = ce.find("geometry")
            T = _origin(ce.find("origin"))
            if g is None or len(g) == 0:
                continue
            s = g[0]
            if s.tag == "box":
                cols.append(CollisionShape("box", T, tuple(float(v) for v in s.attrib["size"].split())))
            elif s.tag == "cylinder":
                cols.append(CollisionShape("cylinder", T, (float(s.attrib["radius"]), float(s.attrib["length"]))))
            elif s.tag == "sphere":
                cols.append(CollisionShape("sphere", T, (float(s.attrib["radius"]),)))
            elif s.tag == "mesh":
                sc = tuple(float(v) for v in s.attrib.get("scale", "1 1 1").split())
                cols.append(CollisionShape("mesh", T, (), s.attrib.get("filename"), sc))
        links[name] = URDFLink(name, _f(mass_el, "value"), tuple(cols))
    joints: dict[str, URDFJoint] = {}
    for je in root.findall("joint"):
        name = je.attrib["name"]
        if name in joints:
            raise ConfigError(f"URDF {path}: duplicate joint {name!r}")
        jt = je.attrib["type"]
        parent = je.find("parent").attrib["link"]
        child = je.find("child").attrib["link"]
        for ln in (parent, child):
            if ln not in links:
                raise ConfigError(f"URDF {path}: joint {name!r} references unknown link {ln!r}")
        axis_el = je.find("axis")
        axis = np.array([float(v) for v in (axis_el.attrib.get("xyz", "1 0 0") if axis_el is not None else "1 0 0").split()])
        n = np.linalg.norm(axis)
        if jt in ("revolute", "continuous", "prismatic") and n < 1e-9:
            raise ConfigError(f"URDF {path}: joint {name!r} has a zero axis")
        axis = axis / n if n > 0 else axis
        lim = je.find("limit")
        dyn = je.find("dynamics")
        mim = je.find("mimic")
        joints[name] = URDFJoint(
            name, jt, parent, child, _origin(je.find("origin")), axis,
            _f(lim, "lower"), _f(lim, "upper"), _f(lim, "effort"), _f(lim, "velocity"),
            _f(dyn, "damping") or 0.0, _f(dyn, "friction") or 0.0,
            mim.attrib.get("joint") if mim is not None else None,
        )
    model = URDFModel(path, root.attrib.get("name", path.stem), links, joints)
    if len(model.root_links) != 1:
        raise ConfigError(f"URDF {path}: expected exactly one root link, found {model.root_links}")
    for j in model.joints.values():
        if j.type in ("revolute", "prismatic"):
            if j.lower is None or j.upper is None:
                raise ConfigError(f"URDF {path}: joint {j.name!r} ({j.type}) needs <limit lower upper>")
            if not j.lower <= j.upper:
                raise ConfigError(f"URDF {path}: joint {j.name!r} has lower > upper")
    return model


def check_bindings(cfg) -> URDFModel:
    """Validate config link bindings and initial_q against the URDF."""
    model = parse_urdf(cfg.urdf)
    absent = [b.link for b in cfg.links if b.link not in model.links]
    if absent:
        raise ConfigError(f"links: {absent} not in URDF {cfg.urdf.name}; URDF links: {sorted(model.links)}")
    movable = {n: j for n, j in model.joints.items() if j.movable}
    for jn, q in cfg.initial_q.items():
        if jn not in movable:
            raise ConfigError(f"initial_q.{jn}: not a movable URDF joint; movable: {sorted(movable)}")
        j = movable[jn]
        if j.lower is not None and not (j.lower - 1e-9 <= q <= j.upper + 1e-9):
            raise ConfigError(f"initial_q.{jn}={q} outside URDF limits [{j.lower}, {j.upper}] {j.unit}")
    return model
