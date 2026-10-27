"""
Load a robot's mesh-fitted collision capsules (made by make_collision_capsules.py).

After loading, robot.collision_link_capsules[k] is a list of (p0, p1, radius)
in DH frame k, which collision_handler uses instead of plain lines between
joint centres. If the JSON file is missing, collision checking falls back to
the old joint-centre lines.
"""
import json
import os

import numpy as np


def load_collision_capsules(robot, mesh_dir, mesh_names):
    robot._collision_mesh_dir = mesh_dir
    robot._collision_mesh_names = mesh_names
    path = os.path.join(mesh_dir, 'collision_capsules.json')
    if not os.path.exists(path):
        print(f"Note: {path} not found - using simple joint-line collision model")
        robot.collision_link_capsules = None
        robot.collision_link_hulls = None
        return
    with open(path) as f:
        data = json.load(f)
    robot.collision_link_capsules = [
        [(np.array(a, float), np.array(b, float), float(r)) for a, b, r in link]
        for link in data['links']
    ]
    # Convex hulls per link: a LIST of (vertices (n,3), face normals (m,3), pad metres)
    # in DH frame k (a list, so a mounted gripper can add its own hull to the last link)
    robot.collision_link_hulls = [
        [(np.array(h['vertices'], float), np.array(h['normals'], float), float(h.get('pad', 0.0)))]
        for h in data.get('hulls', [])
    ] or None
