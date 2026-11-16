"""Debug overlays in the viser scene. Overlays only — Gaussians are never replaced.

Scene graph (viser hierarchical names)::

    /world_axes                        world frame (debug)
    /debug_grid                        DEBUG-ONLY floor grid (not a 3DGS environment)
    /cameras/<name>                    calibrated camera frusta (OpenCV c2w)
    /twin/<link>                       pose := snapshot.T_world_link[link]  (SAPIEN)
        /axes                          link frame
        /label                         link name (+ joint q readout)
        /joint_axis, /hinge            joint whose child is <link> (URDF: joint frame = child frame)
        /ply/bbox                      Gaussian AABB in the PLY frame, placed by T_link_ply
        /collision/<i>                 URDF collision primitives (wireframe)

Only the per-link parent poses change per snapshot, so every overlay follows
the PhysX link pose exactly.
"""
from __future__ import annotations

import numpy as np

from ..geometry.cameras import vertical_fov_from_K
from ..geometry.se3 import quat_from_matrix

OVERLAY_KINDS = ("link_frames", "joint_axes", "labels", "bounding_boxes", "collision", "camera_frusta",
                 "world_axes", "debug_grid")


def _box_segments(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.stack([c[a], c[b]]) for a, b in edges])


def _circle(r, z, n=24):
    t = np.linspace(0, 2 * np.pi, n + 1)
    p = np.stack([r * np.cos(t), r * np.sin(t), np.full_like(t, z)], -1)
    return np.stack([p[:-1], p[1:]], 1)


def _shape_segments(shape) -> np.ndarray | None:
    if shape.kind == "box":
        h = np.asarray(shape.size) / 2
        return _box_segments(-h, h)
    if shape.kind == "cylinder":
        r, length = shape.size
        segs = [_circle(r, -length / 2), _circle(r, length / 2)]
        for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            p = np.array([r * np.cos(a), r * np.sin(a), 0])
            segs.append(np.array([[p + [0, 0, -length / 2], p + [0, 0, length / 2]]]))
        return np.concatenate(segs)
    if shape.kind == "sphere":
        r = shape.size[0]
        c = _circle(r, 0)
        Rx = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        Ry = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        return np.concatenate([c, c @ Rx.T, c @ Ry.T])
    return None


class OverlayManager:
    def __init__(self, server, urdf_model, link_bindings, canonical, cameras, *, label_q: bool = True):
        self.server = server
        self.urdf = urdf_model
        self.bindings = list(link_bindings)
        self.cameras = list(cameras)
        self.label_q = label_q
        self.handles: dict[str, list] = {k: [] for k in OVERLAY_KINDS}
        self.link_nodes: dict[str, object] = {}
        self.labels: dict[str, object] = {}
        sc = server.scene
        self.handles["world_axes"].append(sc.add_frame("/world_axes", axes_length=0.15, axes_radius=0.004))
        g = sc.add_grid("/debug_grid", width=4, height=4, cell_size=0.1, section_size=1.0, visible=False)
        self.handles["debug_grid"].append(g)
        for cam in self.cameras:
            T = cam.T_world_camera
            h = sc.add_camera_frustum(f"/cameras/{cam.name}", fov=vertical_fov_from_K(cam.K, cam.height),
                                      aspect=cam.width / cam.height, scale=0.12, thickness=0.004,
                                      color=(40, 90, 200), wxyz=quat_from_matrix(T[:3, :3]), position=T[:3, 3])
            self.handles["camera_frusta"].append(h)
            self.handles["labels"].append(sc.add_label(f"/cameras/{cam.name}/label", cam.name, position=(0, 0, 0)))
        for b, cg in zip(self.bindings, canonical):
            link = b.link
            base = f"/twin/{link}"
            self.link_nodes[link] = sc.add_frame(base, show_axes=False)
            self.handles["link_frames"].append(sc.add_frame(f"{base}/axes", axes_length=0.08, axes_radius=0.0035))
            T_lp = b.T_link_ply
            lo, hi = cg.aabb()
            # label at the top of the link's Gaussian AABB (link frame), so links
            # sharing an origin (e.g. a hinge at the base origin) do not overlap
            centre_link = T_lp[:3, :3] @ ((lo + hi) / 2) + T_lp[:3, 3]
            top_link = centre_link + np.array([0.0, 0.0, 0.5 * abs((T_lp[:3, :3] @ (hi - lo))[2])])
            lab = sc.add_label(f"{base}/label", link, position=top_link)
            self.labels[link] = lab
            self.handles["labels"].append(lab)
            ply_node = sc.add_frame(f"{base}/ply", show_axes=False, wxyz=quat_from_matrix(T_lp[:3, :3]),
                                    position=T_lp[:3, 3])
            self.handles["bounding_boxes"].append(ply_node)
            self.handles["bounding_boxes"].append(
                sc.add_line_segments(f"{base}/ply/bbox", _box_segments(lo, hi), (250, 170, 30), thickness=0.003))
            ul = urdf_model.links.get(link)
            for i, s in enumerate(ul.collisions if ul else ()):
                segs = _shape_segments(s)
                if segs is None:
                    continue
                T = s.T_link_shape
                self.handles["collision"].append(
                    sc.add_line_segments(f"{base}/collision/{i}", segs, (0, 170, 170), thickness=0.002,
                                         wxyz=quat_from_matrix(T[:3, :3]), position=T[:3, 3]))
            j = urdf_model.joint_for_child(link)
            if j is not None and j.type != "fixed":
                a = j.axis / np.linalg.norm(j.axis)
                seg = np.array([[-0.25 * a, 0.45 * a]])
                self.handles["joint_axes"].append(
                    sc.add_line_segments(f"{base}/joint_axis", seg, (255, 0, 200), thickness=0.006))
                self.handles["joint_axes"].append(
                    sc.add_icosphere(f"{base}/hinge", radius=0.012, color=(255, 0, 200)))
        # Bounding boxes / collision off by default.
        self.set_visible("bounding_boxes", False)
        self.set_visible("collision", False)

    def set_visible(self, kind: str, visible: bool) -> None:
        for h in self.handles[kind]:
            h.visible = bool(visible)

    def update(self, snapshot) -> None:
        with self.server.atomic():
            for link, node in self.link_nodes.items():
                T = snapshot.T_world_link[link]
                node.position = T[:3, 3]
                node.wxyz = quat_from_matrix(T[:3, :3])
            if self.label_q:
                for j in snapshot.joints:
                    uj = self.urdf.joints.get(j.name)
                    if uj is not None and uj.child in self.labels:
                        self.labels[uj.child].text = f"{uj.child}\n{j.name} = {j.q:+.4f} {j.unit}"
