"""
Developer tool: build each robot's collision capsules from its 3D model files.

Each robot link is wrapped in 1-4 capsules (line segment + radius) that
contain EVERY vertex of that link's mesh (fast first check), plus its convex
hull (precise second check against obstacle boxes). They are saved, in each link's own
DH frame, to <robot mesh folder>/collision_capsules.json, which the robot
classes load automatically. Run this again if a mesh or a robot's DH changes:

    py -3.11 robot_controller/make_collision_capsules.py

Needs: trimesh, pycollada  (py -3.11 -m pip install trimesh pycollada)
Only this script needs them - the simulation itself just reads the JSON.
"""
import json
import os
import sys

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _segment_point_distances(P, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip((P - a) @ ab / L2, 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _fit_one(P):
    """Smallest-volume capsule (from a few candidates) containing all points P."""
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    axis = Vt[0]
    t = (P - c) @ axis
    t0, t1 = t.min(), t.max()
    best = None
    for shrink in np.linspace(0.0, 0.5, 11):              # pull the ends in
        span = t1 - t0
        a = c + axis * (t0 + shrink * span / 2)
        b = c + axis * (t1 - shrink * span / 2)
        r = _segment_point_distances(P, a, b).max()         # radius = covers everything
        vol = np.pi * r ** 2 * np.linalg.norm(b - a) + 4 / 3 * np.pi * r ** 3
        if best is None or vol < best[3]:
            best = (a, b, r, vol)
    return best


def _capsule_volume(r, a, b):
    return np.pi * r ** 2 * np.linalg.norm(b - a) + 4 / 3 * np.pi * r ** 3


def _kmeans(P, n, rng, iters=30):
    C = P[rng.choice(len(P), n, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        C_new = np.array([P[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(n)])
        if np.allclose(C_new, C):
            break
        C = C_new
    return lab


def fit_capsules(P, max_capsules=4, seed=0):
    """Wrap points P in 1..max_capsules capsules that contain all of them.

    Tries splitting along the main axis and k-means clustering, and keeps the
    candidate with the smallest total volume (a small penalty per extra capsule
    keeps collision checks fast)."""
    rng = np.random.default_rng(seed)
    c = P.mean(axis=0)
    axis = np.linalg.svd(P - c, full_matrices=False)[2][0]
    t = (P - c) @ axis
    groupings = []
    for n in range(1, max_capsules + 1):
        edges = np.linspace(t.min(), t.max(), n + 1)
        groupings.append([(t >= edges[k]) & (t <= edges[k + 1]) for k in range(n)])
        if n >= 2:
            for _ in range(4):
                lab = _kmeans(P, n, rng)
                groupings.append([lab == k for k in range(n)])
    best = None
    for groups in groupings:
        caps, total = [], 0.0
        for sel in groups:
            if sel.sum() < 4:
                continue
            a, b, r, vol = _fit_one(P[sel])
            caps.append((a, b, r))
            total += vol
        total *= 1.0 + 0.08 * (len(caps) - 1)
        if best is None or total < best[1]:
            best = (caps, total)
    return best[0]


def simplified_hull(P, n_points=200):
    """Convex hull of P reduced to ~n_points corners (farthest-point sampling).
    Returns vertices, unique face normals and 'pad': the largest distance any
    point of the full hull lies outside the simplified one."""
    full = trimesh.convex.convex_hull(P)
    V = full.vertices
    if len(V) > n_points:
        idx = [int(np.argmax(np.linalg.norm(V - V.mean(0), axis=1)))]
        d = np.linalg.norm(V - V[idx[0]], axis=1)
        for _ in range(n_points - 1):
            i = int(np.argmax(d))
            idx.append(i)
            d = np.minimum(d, np.linalg.norm(V - V[i], axis=1))
        small = trimesh.convex.convex_hull(V[idx])
    else:
        small = full
    # signed distance of every full-hull vertex to each simplified face plane
    N = small.face_normals
    offs = np.einsum('ij,ij->i', N, small.triangles[:, 0, :])
    outside = (V @ N.T - offs).max(axis=1)
    pad = float(max(0.0, outside.max()))
    normals = np.unique(np.round(N, 6), axis=0)
    return {'vertices': np.round(small.vertices, 6).tolist(),
            'normals': normals.tolist(), 'pad': pad}


def build(robot, mesh_dir, mesh_names):
    links = []
    hulls = []
    for k, name in enumerate(mesh_names):
        mesh = trimesh.load(os.path.join(mesh_dir, name + '.dae'), force='mesh')
        V = np.c_[mesh.vertices, np.ones(len(mesh.vertices))]
        P = (robot._relation_matrices[k] @ V.T).T[:, :3]      # vertices in DH frame k
        caps = fit_capsules(P)
        links.append([[a.tolist(), b.tolist(), float(r)] for a, b, r in caps])
        # Convex hull for the precise second-stage check against obstacle boxes,
        # simplified to ~200 corners; 'pad' = how far the true hull can poke out
        hull = simplified_hull(P)
        hulls.append(hull)
        print(f"  link {k} ({name}): {len(caps)} capsule(s), radii "
              + ", ".join(f"{r*100:.1f}cm" for _, _, r in caps)
              + f"; hull {len(hull['vertices'])} pts, pad {hull['pad']*1000:.1f}mm")
    out = os.path.join(mesh_dir, 'collision_capsules.json')
    with open(out, 'w') as f:
        json.dump({'frame': 'each list entry k is in DH frame k (frame 0 = base)',
                   'links': links, 'hulls': hulls}, f)
    print(f"  saved {out}")


if __name__ == "__main__":
    from DOBOT_CR5 import DOBOT_CR5
    from OMRON_TM5 import OMRON_TM5_700
    from MOTOMAN_GP4 import MOTOMAN_GP4

    for cls in (DOBOT_CR5, OMRON_TM5_700, MOTOMAN_GP4):
        robot = cls()
        print(robot.name)
        build(robot, robot._collision_mesh_dir, robot._collision_mesh_names)
